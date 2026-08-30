# NLSPN Propagation Float-Dtype Ablation

## Goal

Measure how much of NLSPN's W8A8 loss is caused by quantized propagation
signals and recurrent propagation state, while keeping the ordinary model
quantization fixed at W8A8.

## Scope

The first experiment targets the official NLSPN model and its converged NYU
checkpoint. It uses the existing 128-sample calibration set and the fixed
64-sample evaluation set. The official model structure, preprocessing,
propagation iteration count, and CUDA extensions remain unchanged.

The ordinary quantization set contains every contract-owned `Conv2d`,
`ConvTranspose2d`, and `Linear` module at W8A8. BatchNorm, LayerNorm, ReLU,
softmax, interpolation, sparse anchors, and propagation signals are not
ordinary weight modules.

## Configurations

Four paired configurations are evaluated:

1. `PA_W8A8`: existing propagation-aware W8A8, including A8 propagation
   signals, Q13 affinity coefficients, and quantized recurrent state.
2. `W8A8_FP32_PROP`: ordinary modules remain W8A8; guidance, offset, initial
   depth, affinity, confidence, propagation state, and propagation arithmetic
   remain FP32.
3. `W8A8_BF16_STATE`: the propagation state is stored as BF16 between rounds;
   offset sampling, affinity normalization, anchor injection, and accumulation
   are FP32. Propagation inputs are FP32.
4. `W8A8_FP16_STATE`: the same contract as BF16, with FP16 state storage.

Affinity normalization and center-coefficient recomputation stay FP32 in the
float configurations. No FP16/BF16 path is enabled for a custom operator until
the operator accepts the dtype explicitly; unsupported dtype execution is a
failed experiment rather than an automatic cast or fallback.

## Metrics

The primary metric is pooled RMSE over valid depth pixels. The result also
records mean per-sample RMSE, MAE, AbsRel, iRMSE, non-finite and non-positive
prediction ratios, and paired sample deltas against FP32.

Propagation diagnostics record per-signal RMSE, SQNR, cosine similarity, zero
rate, saturation rate, and per-iteration state error for guidance, offset,
initial depth, affinity, confidence, and propagation state.

The report compares every configuration against both FP32 and PA-W8A8. The
FP32 propagation recovery is attributed to propagation only when all
propagation inputs are also FP32; changing state dtype alone is reported as a
state-storage ablation.

## Acceptance and failure rules

- The paired evaluation identities must be identical across all configurations.
- All four configurations must use the same official checkpoint and 128-sample
  calibration metadata.
- No NaN/Inf or non-positive prediction is accepted as a successful result.
- A BF16/FP16 custom-kernel dtype error is recorded as a failed configuration.
- The runner writes no result row for a configuration that did not complete.
- No fallback dtype, alternate operator, or numerical recovery is allowed.

## Outputs

Results are written below:

`profile_logs/nyu_nlspn_propagation_dtype_ablation_64/nlspn/`

The directory contains the immutable protocol, configuration manifest,
sample-level metrics, pooled summary, propagation diagnostics, and prediction
payloads for completed configurations. Existing unified PA results are read
only and are not overwritten.
