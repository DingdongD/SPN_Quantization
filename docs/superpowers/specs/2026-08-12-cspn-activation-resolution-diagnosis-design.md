# CSPN Activation Resolution Diagnosis and Optimization Design

## Objective

Determine why the official CSPN model loses accuracy under strict W4A4 and
identify the smallest activation-quantization change that improves the fixed
64-sample NYU result without changing the trained FP model or the existing
propagation contract.

The primary hypothesis is resolution loss rather than clipping: a MinMax A4
scale is enlarged by cross-channel range imbalance, so semantically useful
small activations collapse to zero. The experiment must distinguish this from
natural sparsity, isolated element outliers, weight error, and residual-branch
scale mismatch before applying an optimization.

SVD or channel rotation is not part of the main experiment. It may be revisited
only if group-wise and per-channel activation quantization reach a clear
accuracy plateau while channel imbalance remains high.

## Fixed Evaluation Contract

- Model: official CSPN ResNet-18 architecture.
- Checkpoint: converged `cspn_iter24/best.pt`.
- Dataset: real NYU Depth V2 data.
- Calibration: the existing deterministic 128-sample training subset.
- Evaluation: the existing fixed 64-sample evaluation indices.
- Weights: signed symmetric W4, per output channel.
- Ordinary signed activations: symmetric integer A4.
- ReLU activations: unsigned integer A4.
- Guidance head: FP32 and excluded from ordinary CNN quantization.
- Propagation affinity and state: A8.
- Propagation coefficients: signed INT16 Q13.
- Propagation products and reduction: INT32 accumulation before A8
  requantization.
- Bias: FP32 for group-wise activation experiments where no unique input scale
  exists.
- Conv-BN folding: performed before calibration and weight quantization.

Every compared configuration uses the same checkpoint, calibration indices,
evaluation indices, preprocessing, propagation settings, and prediction metric
implementation. A configuration with any non-finite prediction is invalid.

## Stage 1: Activation Resolution Diagnosis

### Site Coverage

Collect every activation edge owned by the existing CSPN semantic adapter and
hardware-aligned instrumentor:

- encoder convolution inputs and outputs;
- encoder ReLU outputs;
- decoder convolution inputs and outputs;
- decoder ReLU outputs;
- structural concat branches and outputs;
- structural residual/shortcut branches and add outputs;
- initial depth head input and output.

The sparse depth input and guidance/affinity/propagation tensors remain
separately identified. Their natural sparsity must not be interpreted as
activation collapse.

### Tensor Metrics

For reference activation `x`, quantized activation `xq`, and integer code `q`,
record:

```text
elements
reference_zero_rate
quantized_zero_rate
new_zero_rate
zero_collapse_error_energy
rounding_error_energy
clipping_error_energy
signal_energy
total_error_energy
sqnr_db
p75
p99
p99_9
p99_99
maximum_abs
tail_ratio_p99_99_over_p99
channel_rms_imbalance
effective_code_count
saturation_rate
```

The central metric is conditional new-zero rate:

```text
new_zero_rate = count(x != 0 and q == 0) / count(x != 0)
```

The error partition uses mutually exclusive element masks:

- zero-collapse: `x != 0 and q == 0`;
- clipping: the unrounded code is outside `[qmin, qmax]`;
- rounding: all remaining elements.

Each partition reports squared error energy. Their sum must equal total QDQ
error energy within floating-point summation tolerance.

Channel imbalance is computed from accumulated channel RMS values:

```text
channel_rms_imbalance = max(channel_rms) / mean(channel_rms)
```

### Per-Channel Metrics

For every NCHW activation site, write one row per channel containing:

```text
site
channel
rms
maximum_abs
p99
p99_9
p99_99
reference_zero_rate
new_zero_rate
zero_collapse_error_energy
rounding_error_energy
clipping_error_energy
sqnr_db
error_energy_share
```

Percentiles use the existing bounded deterministic sampler. The same metric
schema is collected separately for the 128 calibration samples and 64
evaluation samples, with a required `split` field. Exact element counts and
error energies are accumulated within each split.

### Attribution Configurations

Run five global configurations under otherwise identical settings:

1. `FP32`
2. `PA_ONLY`: FP weights and activations with fixed A8/Q13 propagation.
3. `W4_ONLY`: W4 weights, FP activations, and fixed A8/Q13 propagation.
4. `A4_ONLY`: FP weights, A4 activations, and fixed A8/Q13 propagation.
5. `W4A4_RTN`: W4 weights, A4 activations, and fixed A8/Q13 propagation.

`PA_ONLY - FP32` isolates the propagation contract. Relative to `PA_ONLY`, the
W4-only and A4-only deltas isolate weight and activation effects under the same
propagation arithmetic. The W4/A4 interaction is computed as:

```text
W4A4_RTN - W4_ONLY - A4_ONLY + PA_ONLY
```

The diagnostic report must not infer causality from total tensor error alone.

### Outputs

```text
profile_logs/nyu_cspn_activation_resolution/
  cspn/metadata.json
  cspn/config_manifest.csv
  cspn/sample_metrics.csv
  cspn/activation_resolution_metrics.csv
  cspn/activation_channel_metrics.csv
  cspn/block_attribution_metrics.csv
  cspn/merge_branch_metrics.csv
  analysis/error_source_summary.csv
  analysis/sensitive_activation_sites.csv
```

`sensitive_activation_sites.csv` is based on measured single-site intervention,
not a hand-built composite score. Candidate sites are selected only from the
largest calibration-split zero-collapse error contributors, then each candidate
is changed from A4 to A8 alone on calibration block outputs. The table records
the actual block MSE and SQNR delta. Evaluation metrics are not used to select
candidates or thresholds.

## Stage 2: Group-A4 Sweep

The sweep compares:

```text
Tensor A4
Group128 A4
Group64 A4
Group32 A4
Group16 A4
Group8 A4
Per-channel A4
```

Group activation scales are independent contiguous channel-group scales. ReLU
groups use unsigned A4; signed sites use symmetric A4. Only group sizes that
divide the site channel count are valid for that site. Per-channel is group size
one and is reported explicitly.

The first experiment applies one granularity globally to all ordinary CSPN CNN
activation sites. The second experiment applies group/per-channel A4 only to
the calibration-sensitive sites while other sites remain tensor A4.

Group selection uses calibration block-output MSE, then block-output SQNR. The
fixed 64-sample evaluation is used once for every declared sweep configuration
and is never used to select a group size.

Record the fraction of activation elements and sites using each granularity so
accuracy is not reported without its scale-storage cost.

## Stage 3: Residual-Aware Quantization

Use the existing CSPN structural merge adapter to test decoder residual adds,
starting with the sites identified by Stage 1.

For each selected add:

- shortcut/base branch: A8 with its own scale;
- residual/update branch: A4 with its own scale;
- branch products: integer codes retained independently;
- merge: each branch is requantized from its own scale into the declared A8
  output scale with a fixed-point multiplier, then added in INT32;
- merge output: A8 before the following ReLU or convolution.

The baseline for each site is the existing shared A4 merge contract. The
experiment records branch RMS ratio, branch new-zero rate, update-to-base energy
ratio, merge output SQNR, and block-output error.

Residual-aware activation is enabled only at measured sensitive decoder adds.
Encoder residual blocks, guidance logits, depth values, affinity, and propagation
state are not changed in this stage.

## Stage 4: Learned Activation Scale

Learn one scale per selected tensor/group while keeping integer A4 codes and the
same signedness. Initialize from MinMax and optimize on the fixed calibration
set for block-output reconstruction loss. The scale is the only optimized
parameter; model weights and biases are frozen.

Compare MinMax and learned scale on:

- new-zero rate;
- zero-collapse error energy;
- clipping error energy;
- activation and block-output SQNR;
- end-to-end depth metrics.

The learned scale may trade a bounded amount of clipping for improved
resolution. A result is rejected if clipping dominates total activation error
or if calibration improvement does not transfer to the fixed evaluation set.

## Deferred Work

Zero-aware BRECQ is evaluated only if Group-A4, residual-aware activation, and
learned scale identify a remaining reconstruction gap at a specific block. Its
loss must use the measured zero-collapse mask and task/block sensitivity rather
than total zero-code rate.

SVD-derived rotation is evaluated only if:

- channel RMS imbalance remains high;
- Group8 and per-channel A4 stop improving block and end-to-end error; and
- the candidate transform can be represented as an orthogonal update that
  preserves FP equivalence.

Truncated low-rank SVD reconstruction is excluded because it introduces FP
approximation error and confounds the quantization comparison.

## Evaluation Metrics

End-to-end metrics:

- RMSE;
- MAE;
- AbsRel;
- iRMSE;
- flat-region RMSE;
- boundary-region RMSE;
- non-finite prediction ratio.

Quantization metrics:

- new-zero rate and error energy;
- clipping and rounding error energy;
- tensor and per-channel SQNR;
- block-output MSE and SQNR;
- branch scale and RMS ratios;
- A8 activation element/site fraction;
- number of activation scales.

Prediction comparison uses the same fixed samples for GT, FP32, RTN W4A4,
best Group-A4, and the best later-stage configuration.

## Decision Rules

1. If A4-only accounts for most of the FP32-to-W4A4 degradation, continue the
   activation-resolution path. If W4-only dominates, stop and report that the
   hypothesis is not supported.
2. If Group/Per-channel reduces new-zero and block error with a corresponding
   RMSE improvement, classify the site as channel-imbalance dominated.
3. If branch-independent A8/A4 improves a decoder add beyond Group-A4, classify
   it as residual-update collapse.
4. If learned scale improves Group-A4 while introducing bounded clipping,
   classify the remaining error as global resolution loss.
5. Do not continue to Zero-aware BRECQ or SVD rotation without the corresponding
   measured precondition.

The target is to beat the strict Group-A4 RMSE of `0.426802 m`. An optimization
is considered useful only when it also reduces the diagnosed error mechanism;
an isolated RMSE fluctuation without matching activation/block evidence is not
accepted as support for the method.

## Implementation Constraints

- Extend existing `QuantSpec`, semantic-site, merge, recorder, and hardware
  instrumentor APIs; do not create a parallel quantization framework.
- Configuration fields are accessed directly and dictionaries use indexed
  access where values are required.
- Do not add fallback behavior, path probing, broad exception handling, hashes,
  or compatibility wrappers.
- Keep runtime outputs under the project `profile_logs` directory, not `/tmp`.
- Preserve the user's existing unstaged change in
  `tests/test_qdrop_reconstruction.py`.
- Use tests for metric partitioning, group quantization, attribution ownership,
  residual integer merge, and fixed experiment configuration before running the
  real GPU study.
