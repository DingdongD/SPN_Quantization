# W4A4 Activation Histogram Design

## Objective

Profile every activation quantization boundary in the official CSPN, DySPN,
NLSPN, and CompletionFormer models on the fixed 64-sample NYU calibration set.
The profiler must distinguish activation outliers from local quantization error,
show how RGB and sparse-depth inputs use the available codes, and preserve the
exact W4A4 deployment policy used by the strict evaluation.

The profiler is diagnostic only. It must not retrain a model, update a
checkpoint, change model outputs, or introduce a different quantization policy.

## Evaluation Identity

The run reuses the model builders, checkpoints, dataset preprocessing, sample
seed, sample indices, Conv-BN preparation, LayerNorm boundaries, merge adapters,
activation bit overrides, propagation adapters, and quantizer definitions from
the strict W4A4 evaluation.

The nominal W4A4 policy includes semantic exceptions already present in the
strict evaluation:

- weights are signed symmetric W4 with one scale per output channel;
- ordinary signed activations use symmetric A4;
- nonnegative and ReLU activations use unsigned A4;
- raw sparse-depth inputs use unsigned A8;
- CSPN's combined four-channel RGBD input uses unsigned A8 because RGB and
  sparse depth share the first convolution boundary;
- confidence uses unsigned A8;
- propagation coefficients use signed Q13 INT16 and are normalized after
  quantization;
- CompletionFormer concat-convolution inputs retain their configured
  per-channel activation scales.

Ground-truth dense depth remains FP32 and is never passed through a quantizer.

## Statistical Definitions

For a quantization site with reference activation `x` and dequantized value
`Q(x)`, the profiler accumulates:

- local error energy: `sum((x - Q(x)) ** 2)`;
- signal energy: `sum(x ** 2)`;
- SQNR: `10 * log10(signal_energy / error_energy)`;
- zero ratio: exact zeros in the reference activation divided by element count;
- zero-code ratio: zero quantization codes divided by element count;
- saturation ratio: codes at either representable endpoint divided by element
  count;
- spatial-tail ratios: `p99.9 / p99`, `p99.99 / p99`, and `max / p99.99`;
- channel imbalance: maximum channel absolute maximum divided by the median
  channel absolute maximum.

Group error-energy share is reported as a diagnostic aggregation only. It is
not described as outlier prevalence, latency share, parameter share, or strict
causal attribution to endpoint RMSE.

## Coverage

Coverage is driven by the actual activation entries in the W4A4 quantization
manifest rather than by a hardcoded module-type list. Every manifest activation
site must have one histogram record for every observed call index. Shared
modules called multiple times are recorded as separate sites.

The profiler covers input, output, ReLU, LayerNorm, merge, and other semantic
boundaries owned by the strict quantization path. A missing manifest site is a
hard error. A site that exists but is not executed by all samples records its
actual update count and fails validation when the count differs from the
expected model contract.

Input-specific records are added for:

- CSPN `conv1_1` RGB channels 0-2;
- CSPN `conv1_1` sparse-depth channel 3;
- DySPN, NLSPN, and CompletionFormer RGB stem inputs;
- DySPN, NLSPN, and CompletionFormer sparse-depth stem inputs.

These records expose branch range mismatch without changing the first-layer
operator.

## Two-Pass Streaming Collection

Before the two profiling passes, the existing strict calibration procedure runs
on the fixed 64 samples and freezes the W4A4 quantizers. This prerequisite is
not a histogram pass and remains identical to the strict evaluation.

### Pass 1: Profiled-Flow Range Collection

The first profiling pass runs the fixed 64 samples through the configured W4A4
model. At each QDQ boundary, the recorder receives the local reference tensor,
dequantized tensor, integer codes, and frozen quantizer. A bounded deterministic
sample per site computes p75, p90, p99, p99.9, and p99.99. Exact element counts,
zero counts, extrema, local error energy, and channel maxima are accumulated
without sampling. The model is restored and configured identically before the
second pass.

### Pass 2: Histogram Accumulation

The second pass reruns the same samples in the same order. Hooks observe the
tensor immediately before and after each real QDQ boundary and accumulate fixed
histogram bins online. Raw activations are not retained after each hook call.

Each site records four distributions:

1. signed reference activation, using symmetric bins derived from the frozen
   representable range and explicit underflow/overflow bins;
2. normalized magnitude `abs(x) / p99`, using common log2 bins so tails can be
   compared across layers;
3. quantization error `x - Q(x)`, using symmetric bins derived from the local
   quantization scale;
4. integer code occupancy over the quantizer's exact code domain.

Reference zeros are stored in a dedicated bucket so sparse tensors do not hide
the nonzero distribution. Counts use 64-bit integers. Histogram edges and
summary values use float64 on the host.

## Output Contract

Results are written below:

`profile_logs/nyu_w4a4_activation_histograms_64/`

Each model directory contains:

- `histogram_data.npz`: bin edges and counts for every site;
- `histogram_index.csv`: array keys, module, call index, kind, group, bit width,
  signedness, granularity, scale description, update count, and element count;
- `outlier_summary.csv`: percentiles, tail ratios, channel imbalance, SQNR,
  zero ratios, saturation ratio, and local/group error-energy shares;
- `all_sites_histograms.pdf`: paginated plots for every site;
- `critical_layers.png`: highest-error and strongest-tail sites;
- `rgb_depth_input_histograms.png`: RGB and sparse-depth ranges, zero mass,
  quantization bins, and code occupancy;
- `group_outlier_distribution.png`: site prevalence and severity by semantic
  model group;
- `metadata.json`: checkpoint hash, source identity, dataset root, sample
  indices, seed, quantization policy, bin definitions, and site counts.

The root output directory additionally contains a four-model comparison figure
and a CSV summary using identical metrics and group names.

## Plotting Rules

Plots use Arial when available, with Liberation Sans and DejaVu Sans as explicit
font fallbacks. Figures have no decorative title, use non-rotated model labels,
place grid lines below plotted data, and use bars and lines at higher z-order.

The paginated PDF uses stable axes per distribution type:

- signed values use a symmetric linear or symlog axis as appropriate;
- normalized magnitudes use a log2 x-axis and logarithmic count axis;
- errors use symmetric bins centered at zero;
- code occupancy uses the exact integer code values;
- zero mass appears as a separate annotated bar.

Critical-layer figures select sites independently by local error-energy share,
lowest SQNR, spatial-tail ratio, and channel imbalance. Selection criteria are
written to the summary CSV rather than encoded only in plotting code.

## Validation

Unit tests are written before implementation and cover:

- signed and unsigned code occupancy;
- explicit zero and saturation buckets;
- deterministic streaming accumulation across batches;
- per-tensor and per-channel quantizer scales;
- shared-module call indexing;
- RGB and sparse-depth channel splitting;
- NPZ/CSV round-trip consistency;
- manifest coverage failure when a site is missing;
- plot and PDF generation from a minimal synthetic profile.

Runtime validation requires:

- every W4A4 activation manifest site to appear in the histogram index;
- each histogram count sum to the recorded reference element count;
- each code-occupancy count sum to the quantized element count;
- all histogram edges and summary values to be finite except ratios whose
  denominator is explicitly zero;
- recomputed SQNR, zero-code ratio, and saturation ratio to match the strict
  layer metrics within documented numerical tolerance;
- all four model smoke runs to complete on CUDA before the 64-sample run;
- every PDF to have at least one page and every PNG to contain nonblank pixels.

The complete run is rejected rather than partially reported when coverage,
count conservation, model identity, checkpoint identity, or sample identity
does not match the strict evaluation.

## Non-Goals

This work does not change bit allocation, add clipping, apply SmoothQuant or
AWQ, retrain checkpoints, alter propagation arithmetic, benchmark latency, or
store raw activation tensors. Those experiments consume the histogram outputs
in a later sensitivity-allocation stage.
