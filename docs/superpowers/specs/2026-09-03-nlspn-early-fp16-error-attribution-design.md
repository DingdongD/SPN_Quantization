# NLSPN Early FP16 Error Attribution Design

## Objective

Determine whether replacing the three `EARLY_A8` activation boundaries with
IEEE FP16 reduces the remaining NLSPN error, and separate residual early-block
activation error from FP6 weight error. This is a PTQ fake-quantization
ablation. It does not retrain or fine-tune the model.

## Fixed Protocol

- Official NLSPN ResNet-34 checkpoint and DCN extension.
- Existing stratified 128-sample NYU train calibration set.
- Existing fixed 64-sample NYU validation set.
- Eighteen propagation iterations with the complete propagation operator in
  FP16.
- Ordinary weights and activations remain FP6 E3M2.
- `id_dec0.0` and `id_dec1.0` input activations remain FP8 E4M3FN.
- The corrected single-QDQ ownership path remains active: concat execution
  adapter disabled, ordinary weights quantized exactly once, and no direct
  initial-depth output QDQ.

IEEE FP16 means an explicit `FP32 -> IEEE FP16 -> FP32` QDQ at the selected
weight or activation boundary. It is not a bypass and is reported as 16-bit
storage and traffic.

## Early Group

The early group contains the same boundaries selected by the preceding
experiment:

- `conv2.0.conv1` input;
- `conv2.0.conv2` input;
- `conv3.0.downsample.0` input.

The RGB/depth branch-aware stem path is not enabled in this experiment because
it did not improve on common `EARLY_A8` quantization.

## Candidate Matrix

| Candidate | Early weights | Early activations |
| --- | --- | --- |
| `EARLY_W6A8` | FP6 | all three FP8 |
| `EARLY_W6A16_CONV2_0_CONV1` | FP6 | only `conv2.0.conv1` FP16; other two FP8 |
| `EARLY_W6A16_CONV2_0_CONV2` | FP6 | only `conv2.0.conv2` FP16; other two FP8 |
| `EARLY_W6A16_CONV3_0_DOWNSAMPLE` | FP6 | only `conv3.0.downsample.0` FP16; other two FP8 |
| `EARLY_W6A16` | FP6 | all three FP16 |
| `EARLY_W16A8` | selected module weights FP16 | all three FP8 |
| `EARLY_W16A16` | selected module weights FP16 | all three FP16 |

For `EARLY_W16A8` and `EARLY_W16A16`, FP16 weight protection applies to the
three named convolution modules only. All remaining ordinary weights stay
FP6.

## Measurements

Each candidate runs twice on the same concatenated 64-sample evaluation batch.
The prediction, initial depth, guidance, offset, normalized affinity,
confidence, and all 18 propagation states must be bit-exact between forwards.

Record:

- pooled RMSE, MAE, AbsRel, iRMSE, and mean sample RMSE;
- delta and relative loss against FP32 and `EARLY_W6A8`;
- weighted average weight and activation bits;
- effective format and changed-element count for each early weight;
- activation native-zero, new-zero, saturation, and nonzero SQNR diagnostics;
- MSE, MAE, maximum error, and SQNR for propagation-entry signals and every
  propagation state.

## Attribution Rules

- `EARLY_W6A16 - EARLY_W6A8` measures recoverable early activation error.
- `EARLY_W16A8 - EARLY_W6A8` measures recoverable early weight error.
- `EARLY_W16A16 - EARLY_W6A16 - EARLY_W16A8 + EARLY_W6A8` measures the
  weight-activation interaction.
- Single-site A16 deltas rank the three activation boundaries by end-to-end
  task impact; local SQNR alone does not determine the ranking.
- Residual loss of `EARLY_W16A16` against FP32 is attributed to later
  encoder, decoder, guidance, confidence, initial-depth, FP16 propagation, and
  their interactions. It is not assigned to one module without an additional
  ablation.

## Outputs And Validation

Write a new immutable result root containing `summary.csv`,
`sample_metrics.csv`, `module_diagnostics.csv`,
`effective_weight_metrics.csv`, `propagation_signal_metrics.csv`,
`propagation_state_metrics.csv`, `attribution.csv`, and `manifest.json`.

The run fails directly on an existing output root, missing owner, wrong module
shape, unsupported format, non-finite or non-positive depth, changed sample
identity, non-reproducible signal, wrong propagation iteration count, or an
effective weight format that differs from the candidate. Existing FP6/A8
artifacts remain unchanged.
