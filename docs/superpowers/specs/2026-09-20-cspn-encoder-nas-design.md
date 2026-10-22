# CSPN Encoder NAS Experiment Design

## Goal

Determine whether the converged NYU CSPN ResNet18 encoder can be replaced by a
smaller and faster encoder without practically meaningful accuracy loss. The
formal FP32 non-inferiority margin is 2% relative RMSE. The existing reference
checkpoint has RMSE 0.1481478 m, so 0.1511107 m is the search-screening limit.

Among non-inferior candidates, retain the Pareto frontier for full-model
parameter count and measured end-to-end latency on one NVIDIA A100-SXM4-40GB at
batch size 1. The primary deployment candidate is the lowest-median-latency
model; candidates within 1% latency are ordered by parameter count.

## Fixed Baseline Contract

The control is the local converged CSPN ResNet18 model with:

- 228 x 304 RGB plus sparse-depth input;
- 500 valid sparse-depth samples;
- ResNet18 BasicBlock depths `[2, 2, 2, 2]` and widths
  `[64, 128, 256, 512]`;
- the current decoder, depth head, guidance head, and skip locations;
- 24 CSPN propagation iterations using the current `8sum` implementation;
- the existing NYU preprocessing and metric definitions; and
- FP32 inference with TF32 disabled.

The existing `cspn_iter24/best.pt` result is a search sanity reference, not the
only statistical control. Final claims use newly trained paired baseline and
candidate runs under the same protocol.

## Scope

The first experiment searches the encoder only. It does not search decoder
channels, heads, CSPN normalization, CSPN iteration count, input resolution,
quantization bit width, kernels, or activation functions. This isolates whether
ResNet18 encoder capacity is necessary. Joint encoder/decoder or propagation
search is a later experiment and must not be mixed into this result.

## Search Space

Each candidate retains four spatial stages and output stride 32.

- Stem width: 32, 48, or 64 channels.
- Stage 1 contains 1 or 2 BasicBlocks.
- Each of stages 2 through 4 uses one of three modes: projection-only
  downsample, one BasicBlock, or two BasicBlocks. Projection-only is the
  canonical zero-residual-block representation.
- Each stage independently uses a channel multiplier of 0.5, 0.75, or 1.0
  relative to its baseline width, rounded to a multiple of 16.
- Widths must be non-decreasing across stages.
- Candidate skip and bottleneck outputs pass through learned 1 x 1 adapters to
  the fixed decoder interfaces of 64, 64, 128, and 512 channels.

Adapters are part of the candidate and count toward parameters and latency.
The unmodified ResNet18 is a control point inside the same model factory.
Invalid channel combinations and candidates with no parameter or latency
advantage over the control are rejected before training.

## Model Boundaries

The implementation separates four responsibilities:

1. An encoder specification validates depths, widths, stage modes, and channel
   alignment and has a stable serialized representation.
2. An encoder factory builds the candidate and its fixed decoder adapters.
3. A weight-transfer helper initializes compatible tensors from a trained R18
   checkpoint by deterministic prefix-channel selection; unmatched adapter tensors use
   the repository's standard initialization.
4. A runner trains, evaluates, benchmarks, and records one immutable candidate
   specification without embedding search policy in the model code.

The existing CSPN model remains available unchanged so old checkpoints and
quantization tools continue to load.

## Data Isolation

Architecture search must not read the official 654-image NYU validation split.
The 6,700-image training split is deterministically partitioned with a recorded
seed into 6,030 search-training and 670 search-development samples. The current
CSV and HDF5 files contain no scene identifier, so this is explicitly a
sample-level split and the exact indices are stored in the run manifest.

The search-development result selects one deployment architecture before any
new run reads the official validation split. That architecture and the R18
controls then use all 6,700 training samples for the fixed final protocol. The
official validation split is evaluated without candidate or checkpoint
selection on it.

## Search Procedure

The search has four stages:

1. Enumerate legal candidates, count encoder and full-model parameters, and
   benchmark their untrained FP32 latency. Remove candidates dominated by R18
   on both parameter count and latency.
2. Run single-axis sensitivity candidates to identify stages that cannot be
   reduced without large proxy-RMSE loss.
3. Apply successive halving at 5, 15, and 40 epochs using one fixed seed. At
   each rung, retain `max(8, ceil(N / 4))` candidates ordered by Pareto rank and
   crowding distance over RMSE, full-model parameters, and median latency.
4. Select the deployment architecture using search-development RMSE and
   repeatable A100 latency only: among candidates within the 2% proxy margin,
   choose minimum median latency and break latency ties within 1% by minimum
   full-model parameters. Do not use official-validation results to choose it.

Search candidates inherit compatible channels from the R18 checkpoint and are
fine-tuned. This experiment therefore proves that the smaller architecture can
preserve a trained CSPN's performance after compression and fine-tuning; it
does not claim identical from-scratch optimization behavior.

## Final Training and Statistical Test

Train the R18 control and the selected deployment candidate with three matched
seeds. Candidate run `s` is initialized from control run `s`; both use the same
data order, augmentation seed, optimizer family, schedule, epoch count, and
metric implementation where their contracts permit it.

For each seed and each of the 654 official-validation samples, record R18 and
candidate RMSE. Use a deterministic paired hierarchical bootstrap that samples
training seeds and then validation samples, with 10,000 replicates. Let
`delta = RMSE_candidate - RMSE_R18` and let the non-inferiority margin be 2% of
the paired R18 RMSE. FP32 non-inferiority passes only when the one-sided 95%
upper confidence bound for `delta` is below the margin. Also report MAE,
ABS_REL, DELTA1.25, per-seed results, and the unrounded confidence interval.

The absolute 0.1511107 m value is a search-screening threshold. The formal final
threshold is derived from the newly trained paired controls, so training drift
cannot make the comparison artificially easy or hard.

## A100 Latency Protocol

The primary benchmark is PyTorch eager FP32 on one A100-SXM4-40GB:

- input shape `1 x 4 x 228 x 304`;
- `eval()` and inference mode;
- TF32 disabled and the same cuDNN settings for all candidates;
- one exclusive, idle GPU with device, driver, PyTorch, CUDA, clock,
  temperature, and power state recorded;
- 200 warm-up iterations;
- 1,000 synchronized CUDA Event measurements per repetition; and
- five repetitions.

Report median and p95 end-to-end latency, encoder-only latency, peak allocated
CUDA memory, encoder parameters, and full-model parameters. The median of the
five per-repetition medians is the search objective. A benchmark is invalid if
the GPU is shared by another compute process, clock state changes outside the
recorded tolerance, any output is non-finite, or output shapes differ.

## W8A8 Verification

FP32 is the formal NAS acceptance gate. After the architecture is fixed, apply
the repository's existing W8A8 quantization path to both R18 and the selected
candidate on identical calibration and evaluation indices. Report quantized
RMSE, FP32-relative degradation, non-finite counts, and prediction agreement.

W8A8 failure does not invalidate the FP32 encoder-capacity result, but it marks
the candidate as not deployment-ready and prevents claims of quantized latency
or accuracy preservation. Fake-QDQ timing is not reported as INT8 hardware
speedup; actual INT8 latency requires a backend that executes integer kernels.

## Failure Handling and Resumption

Each run writes its validated candidate specification, source revision,
environment, dataset-manifest hash, seed, metrics, and checkpoint atomically.
Runs resume only when all immutable fields match. A mismatch creates a new run
rather than silently reusing state.

Non-finite loss or gradients, invalid output range, wrong output shape, missing
samples, checkpoint mismatch, or an invalid latency environment terminates the
candidate with a machine-readable failure reason. Failed candidates remain in
the search table and are not converted to poor finite scores.

## Tests and Acceptance

Unit tests cover specification validation, candidate construction, output and
skip shapes, exact parameter counting, deterministic enumeration, deterministic
weight transfer, split reproducibility, Pareto filtering, resume rejection, and
paired-bootstrap calculations. Integration tests run an R18 control and a
reduced candidate through forward/backward, one short training epoch,
checkpoint resume, evaluation, and latency measurement.

The experiment is complete when it produces:

- a reproducible search manifest and results table;
- the full Pareto frontier, including failed candidates;
- three independently trained matched-seed R18/candidate pairs;
- the formal FP32 non-inferiority report;
- A100 latency and parameter reports under the fixed protocol;
- W8A8 secondary validation; and
- checkpoints and configuration metadata needed to reproduce every reported
  result.
