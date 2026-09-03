# NLSPN Decoder, Guidance, And Initial-Depth FP16 Ablation Design

## Objective

Localize the residual NLSPN PTQ error after early-layer FP16 protection by
measuring the independent and cumulative task impact of the shared decoder,
guidance decoder, and initial-depth decoder. This is a PTQ fake-quantization
ablation. It does not retrain or fine-tune the model.

## Fixed Protocol

- Official NLSPN ResNet-34 checkpoint and DCN extension.
- Existing stratified 128-sample NYU train calibration set.
- Existing fixed 64-sample NYU validation set.
- Eighteen propagation iterations with the complete propagation operator in
  FP16.
- `EARLY_W16A16` is the fixed baseline: `conv2.0.conv1`,
  `conv2.0.conv2`, and `conv3.0.downsample.0` use explicit IEEE FP16 weight
  and input QDQ.
- Ordinary unselected weights and activations remain FP6 E3M2.
- `id_dec0.0` and `id_dec1.0` input activations remain FP8 E4M3FN until the
  initial-depth decoder group is selected for FP16 protection.
- The corrected single-QDQ ownership path remains active: concat execution
  adapter disabled, ordinary weights quantized exactly once, and no direct
  initial-depth output QDQ.

IEEE FP16 protection means joint explicit weight and input
`FP32 -> IEEE FP16 -> FP32` QDQ. It is not a bypass.

## Module Groups

| Group | Modules |
| --- | --- |
| `dec5` | `dec5.0` |
| `dec4` | `dec4.0` |
| `dec3` | `dec3.0` |
| `dec2` | `dec2.0` |
| `guidance` | `gd_dec1.0` |
| `initial_depth` | `id_dec1.0`, `id_dec0.0` |

## Candidate Matrix

The experiment contains 12 unique candidates. The isolated `dec5` candidate
is also the first cumulative prefix and is evaluated only once.

| Candidate | Additional FP16-protected group(s) |
| --- | --- |
| `EARLY_W16A16_BASE` | none |
| `ISO_DEC5_W16A16` | `dec5` |
| `ISO_DEC4_W16A16` | `dec4` |
| `ISO_DEC3_W16A16` | `dec3` |
| `ISO_DEC2_W16A16` | `dec2` |
| `ISO_GUIDANCE_W16A16` | `guidance` |
| `ISO_INITIAL_DEPTH_W16A16` | `initial_depth` |
| `PREFIX_DEC5_DEC4_W16A16` | `dec5`, `dec4` |
| `PREFIX_DEC5_DEC4_DEC3_W16A16` | `dec5`, `dec4`, `dec3` |
| `PREFIX_SHARED_DECODER_W16A16` | `dec5` through `dec2` |
| `PREFIX_SHARED_GUIDANCE_W16A16` | shared decoder and `guidance` |
| `PREFIX_SHARED_GUIDANCE_INITIAL_W16A16` | all six groups |

## Attribution

- Isolated recovery for group `g` is
  `RMSE(EARLY_W16A16_BASE) - RMSE(ISO_g)`.
- The first cumulative recovery uses `ISO_DEC5_W16A16`, avoiding a duplicate
  run.
- Each later cumulative marginal recovery is the preceding prefix RMSE minus
  the current prefix RMSE.
- Total recovery is the baseline RMSE minus
  `PREFIX_SHARED_GUIDANCE_INITIAL_W16A16` RMSE.
- Shared-decoder interaction is the total shared-decoder recovery minus the
  sum of isolated `dec5` through `dec2` recoveries.
- Downstream interaction is the full cumulative recovery minus the sum of all
  six isolated recoveries.

Positive recovery means lower pooled RMSE. A local FP16 substitution is not
classified as beneficial from signal SQNR alone.

## Measurements

Each candidate runs twice on the same concatenated 64-sample evaluation
batch. Prediction, initial depth, guidance, offset, normalized affinity,
confidence, and all 18 propagation states must be bit-exact between forwards.

Record pooled RMSE, MAE, AbsRel, iRMSE, mean sample RMSE, deltas against FP32
and the fixed baseline, average W/A bits, effective selected weight formats,
activation zero/saturation/SQNR diagnostics, propagation-entry signal error,
and per-iteration propagation-state error.

## Outputs And Validation

Write a new immutable result root containing `summary.csv`,
`sample_metrics.csv`, `module_diagnostics.csv`,
`effective_weight_metrics.csv`, `propagation_signal_metrics.csv`,
`propagation_state_metrics.csv`, `isolated_attribution.csv`,
`cumulative_attribution.csv`, and `manifest.json`.

The run fails directly on an existing output root, missing owner, unsupported
format, non-finite or non-positive depth, changed sample identity,
non-reproducible signal, wrong propagation iteration count, or an effective
selected weight format that differs from the candidate. Existing result roots
remain unchanged.
