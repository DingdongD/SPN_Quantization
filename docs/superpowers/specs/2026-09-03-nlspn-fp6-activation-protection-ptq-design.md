# NLSPN FP6 Activation Protection PTQ Design

## Objective

Measure whether selective FP8 activation protection and branch-independent
scales reduce the remaining NLSPN error under an FP6-dominant post-training
quantization configuration. The experiment isolates each protection before
combining them. It does not train or fine-tune the model.

The primary reference is the existing official NLSPN ResNet-34 checkpoint on
the fixed NYU protocol:

- FP32 pooled RMSE: `0.1508939417473765` m.
- Complete propagation semantic subgraph: FP16.

The previous initial-depth boundary artifacts under
`nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v1` through `v4` do not define the
new baseline. Their concat consumers skipped weight quantization and received
two activation QDQ operations. This experiment first establishes a corrected
`BASE` with one input QDQ and effective FP6 weights.

Success requires at least one valid candidate with pooled RMSE below the newly
measured corrected `BASE`. Every candidate must also produce finite,
nonnegative depth and preserve the official propagation structure and
iteration count.

## Fixed Protocol

- Model: official `NLSPNModel`, ResNet-34, 18 propagation iterations.
- Checkpoint:
  `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18/best.pt`.
- Calibration: existing stratified 128-sample training subset.
- Evaluation: existing fixed 64-sample validation subset.
- Ordinary weights: scaled FP6 E3M2, per-output-channel scale.
- Ordinary activations: scaled FP6 E3M2, static calibrated owner scale unless
  promoted or assigned branch-independent scales below.
- Initial-depth weights: FP6 E3M2.
- `id_dec0.0` and `id_dec1.0` input activations: FP8 E4M3FN.
- Propagation-owned modules, affinity, offset, confidence, normalization,
  sparse-depth injection, and recurrent state: FP16.

Quantization is fake-quant QDQ. Convolution consumes dequantized floating
tensors; bit averages describe format storage and traffic, not a native CUDA
FP6 or FP8 arithmetic kernel.

## Initial-Depth Ownership Correction

The existing FP concat adapter marks `id_dec0.0` and `id_dec1.0` as fully
externally owned. The generic instrumentor consequently omits their weights,
while the adapter quantizes branches and output. The previous boundary runner
then installs another input quantizer, which produces two input QDQ operations.

The corrected experiment uses one owner for each operation:

- disable the concat adapter execution path for the initial-depth consumers;
- clear active external input and output ownership for those consumers;
- let the ordinary FP instrumentor quantize each initial-depth weight once;
- install exactly one common or branch-independent input quantizer;
- do not quantize `id_dec0.0` or `id_dec1.0` output directly.

For branch-independent candidates, splitting the already concatenated tensor
at the verified channel boundary is numerically identical to quantizing the
two branches immediately before concatenation because `_concat` only crops
the decoder branch and concatenates along the channel dimension. The raw RGB
and sparse-depth inputs are protected contract signals, so stem protection is
implemented once at the first ordinary boundary, `conv2.0.conv1`, after the
48-channel RGB stem and 16-channel depth stem are concatenated.

## Candidate Matrix

All candidates start from `BASE`. A module promotion changes its input
activation format only; its weight remains FP6.

| Candidate | Additional protection |
| --- | --- |
| `BASE` | None |
| `DEPTH_A8` | First ordinary stem boundary with RGB FP6 and depth FP8, independently scaled |
| `RGB_A8` | First ordinary stem boundary with RGB FP8 and depth FP6, independently scaled |
| `STEM_A8` | First ordinary stem boundary with both branches FP8 and independently scaled |
| `EARLY_A8` | `conv2.0.conv1`, `conv2.0.conv2`, and `conv3.0.downsample.0` inputs at FP8 |
| `STEM_EARLY_A8` | `STEM_A8` plus `EARLY_A8` |
| `STEM_EARLY_ID_BRANCH` | `STEM_EARLY_A8` plus independent scales for the two branches entering `id_dec0.0` and `id_dec1.0` |
| `FULL_BRANCH_AWARE` | `STEM_EARLY_ID_BRANCH` plus independent scales for the two branches entering `dec4.0`, `dec3.0`, and `dec2.0` |

The official branch layouts are fixed and must be checked against the loaded
module shapes:

| Consumer | Decoder branch | Encoder branch | Total channels |
| --- | ---: | ---: | ---: |
| `conv2.0.conv1` | RGB stem: 48 | depth stem: 16 | 64 |
| `dec4.0` | 256 | 512 | 768 |
| `dec3.0` | 128 | 256 | 384 |
| `dec2.0` | 64 | 128 | 192 |
| `id_dec1.0` | 64 | 64 | 128 |
| `id_dec0.0` | 64 | 64 | 128 |

Each branch is calibrated independently and QDQ is applied before
concatenation. The consuming convolution receives the concatenated dequantized
branches. Guidance and confidence head concatenations are not modified because
they belong to the fixed FP16 propagation semantic subgraph.

## Sparse-Depth Accounting

Scaled floating-point quantization maps exact zero to exact zero. A separate
mask-preserving operator would therefore duplicate existing behavior and is
not added. The depth-stem diagnostic instead distinguishes:

- native-zero ratio: reference values equal to zero;
- new-zero ratio: nonzero reference values mapped to zero;
- saturation ratio;
- nonzero-reference SQNR.

Calibration maxima remain computed by the same static calibrated owner
observer. Zeros are not removed from, or assigned a special fallback in, the
calibration path.

## Measurements

For every candidate, write:

- pooled RMSE and mean per-sample RMSE;
- delta and relative loss against FP32 and `BASE`;
- pooled MAE, AbsRel, and iRMSE;
- weighted average weight and activation bits;
- finite, nonnegative, paired-forward reproducibility checks;
- activation SQNR, native-zero, new-zero, and saturation ratios for every
  promoted owner;
- per-branch maximum, p99, RMS, SQNR, new-zero ratio, and saturation ratio for
  every branch-aware consumer;
- MSE and SQNR for initial depth, normalized affinity, offset, confidence,
  final prediction, and every recurrent propagation state.

The result root contains `summary.csv`, `module_diagnostics.csv`,
`branch_diagnostics.csv`, `propagation_signal_metrics.csv`,
`propagation_state_metrics.csv`, `pareto.csv`, and `manifest.json`. The
manifest records the exact assignment, checkpoint identity, sample indices,
format definitions, propagation exclusions, branch layouts, QDQ call counts,
and effective weight-format checks.

## Selection Rule

The primary selection orders valid candidates by pooled RMSE. If two results
differ by less than `0.0001` m, prefer the candidate with lower average
activation bits. The Pareto table retains every candidate not dominated in
pooled RMSE, average weight bits, and average activation bits.

The experiment must not infer causality from local SQNR alone. A protection is
reported as beneficial only when its paired end-to-end pooled RMSE improves.
If no candidate beats `BASE`, the result is recorded as a negative PTQ result
and no protection is silently retained.

After the corrected artifacts pass schema, assignment, reproducibility,
finite-output, effective-weight, and single-QDQ validation, remove the invalid
`v1` through `v4` initial-depth boundary artifact directories. The result
document records their removal and the ownership defect that invalidated them.

## Testing And Execution

Unit tests cover the exact candidate matrix, assignment immutability, official
branch channel validation, independent-scale behavior, sparse zero accounting,
bit-budget calculation, result selection, propagation ownership exclusion,
effective initial-depth weight quantization, and exactly one input QDQ per
forward. The focused test suite runs in both the repository Python environment
and the official Python 3.7 NLSPN environment where compatible.

The end-to-end run must use the configured CUDA device, load the official DCN
extension, reject an existing output directory, and fail directly on missing
modules, shapes, calibration rows, or non-finite output. No fallback model,
device, quantizer, sample set, or propagation mode is allowed.

QAT and multi-level distillation are explicitly deferred until this PTQ matrix
identifies the protections that provide measured end-to-end benefit.
