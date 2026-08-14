# CSPN Stem Mixed-Precision and Branch-Aware Quantization Design

## Objective

Measure whether the RGB/sparse-depth scale mismatch at the official CSPN
`conv1_1` stem materially contributes to W4A4 depth-completion error. The
experiment compares four stem contracts while holding the checkpoint,
calibration identities, evaluation identities, propagation path, and all
non-stem quantization settings fixed.

## Scope

The experiment covers the official CSPN ResNet-18 NYU model with 24 propagation
steps and the converged `cspn_iter24/best.pt` checkpoint. It uses the existing
stratified 128-sample NYU training calibration set and fixed 64-sample NYU
validation set. No training, QAT, AdaRound, BRECQ, QDrop, SmoothQuant, rotation,
or evaluation-driven calibration is enabled.

The stem is the official `conv1_1` 4-to-64, 7-by-7, stride-2 convolution. Its
input channels are RGB channels 0 through 2 and sparse depth channel 3. The raw
stem output is also retained by the official model as `skip4`, so every stem
configuration must define both the main encoder path and this skip path.

## Compared Configurations

All non-stem ordinary layers use static contiguous Group-8 W4A4 MinMax. Sites
whose channel count is not divisible by eight retain the existing tensor-scale
contract. Guidance remains FP, and propagation retains the existing
propagation-aware A8/INT16-Q13/INT32 implementation.

1. `STRICT_W4A4`: `conv1_1` uses per-output-channel symmetric W4 weights and
   one unsigned static tensor A4 input scale shared by RGB and sparse depth.
   Existing A4 boundaries after the stem remain unchanged.
2. `STEM_W8A8`: `conv1_1` uses per-output-channel symmetric W8 weights and one
   unsigned static tensor A8 input scale. The raw `conv1_1`/`skip4` output and
   the stem ReLU output use A8 before entering the otherwise W4A4 network.
3. `STEM_FP16`: `conv1_1` executes through the PyTorch CUDA convolution with
   FP16 input and weights, then casts its output to FP32. The experiment does
   not infer an undocumented accumulator format from this reference path. The
   raw `conv1_1`/`skip4` output and the BN/ReLU main path remain floating until
   their first downstream non-stem A4 boundaries. No stem activation or weight
   QDQ is allowed.
4. `STEM_BRANCH_A4`: RGB and sparse depth use independent unsigned static A4
   MinMax scales. The original `conv1_1` weight tensor uses one shared
   per-output-channel symmetric W4 scale, then is split into RGB and depth
   input-channel slices. Each branch performs an integer partial convolution.
   Partial INT32 accumulators are requantized to the coarser per-output-channel
   accumulator scale

   `s_acc,o = max(s_rgb * s_w,o, s_depth * s_w,o)`

   before INT32 addition. `conv1_1` has no bias. The combined result is
   dequantized to FP32 and enters the same downstream A4 boundaries as
   `STRICT_W4A4`.

`STEM_BRANCH_A4` changes only the first convolution input and accumulation
contract. It does not assign separate output scales to RGB and depth after they
have been mixed by the convolution.

## Calibration and Execution

The runner reuses the persisted stratified calibration manifest. It rejects a
different identity set, duplicate indices, overlap with evaluation identities,
or a checkpoint/source mismatch. MinMax ranges are observed on the FP32 model:

- one merged RGBD range for `STRICT_W4A4` and `STEM_W8A8`;
- one RGB range and one sparse-depth range for `STEM_BRANCH_A4`;
- no stem range for `STEM_FP16`;
- the existing Group-8 ranges for every non-stem activation site.

Every configuration starts from a freshly loaded official checkpoint. The
runner must not restore weights or quantizers from a preceding configuration.
All predictions and diagnostics must be finite; violations terminate the run.

## Metrics

The primary metric is aggregate RMSE over the fixed 64 validation samples.
Secondary end-to-end metrics are MAE, AbsRel, iRMSE, flat-region RMSE, and
boundary-region RMSE.

Stem diagnostics include:

- RGB and sparse-depth reference range, scale, SQNR, zero ratio, new-zero rate,
  saturation rate, and clipping-error share;
- RGB and depth partial-convolution output MSE and SQNR;
- combined `conv1_1` output MSE/SQNR;
- stem ReLU output MSE/SQNR;
- `skip4` input error at `gud_up_proj_layer4`;
- per-sample RMSE delta relative to `STRICT_W4A4`.

The deployment summary reports stem FP16, W8A8, W4A4, and A4 activation element
fractions, plus weight storage and activation-scale counts. These are logical
operation and storage counts, not fused-kernel latency claims.

## Analysis and Acceptance

The report must answer three questions without attributing correlation as
causality:

1. Does `STEM_W8A8` recover accuracy relative to `STRICT_W4A4`?
2. How much additional recovery does `STEM_FP16` provide over `STEM_W8A8`?
3. Does strict branch-aware A4 recover a material fraction of the W8A8 gain
   without promoting stem precision?

A configuration is considered an accuracy improvement only when aggregate
RMSE decreases and at least 33 of 64 samples improve. Regional and inverse-depth
regressions remain explicitly reported even when the primary criterion passes.
No method is declared deployment-superior without measured CUDA kernel latency.

## Artifacts

The run writes under
`profile_logs/nyu_cspn_stem_mixed_precision_branch_a4_64`:

- `aggregate_metrics.csv` and `sample_metrics_64.csv`;
- `stem_activation_metrics.csv` and `stem_partial_conv_metrics.csv`;
- `precision_coverage.csv`;
- prediction payloads for all four configurations;
- one JSON manifest containing source identities, checkpoint hash, calibration
  identities, evaluation identities, quantization contracts, and artifact
  hashes;
- a concise Markdown results report.

Prediction images are not required by this experiment. Existing visualization
tools can consume the persisted prediction payloads without changing inference.

## Verification

Unit tests cover branch splitting, A4 code generation, shared W4 weight codes,
integer partial convolution, Q31 requantization, overflow rejection, exact FP32
equivalence when quantization is disabled, configuration isolation, and strict
manifest validation. Integration tests use a small CSPN-shaped model before the
full CUDA/NYU run. The full repository test suite must pass before results are
reported.
