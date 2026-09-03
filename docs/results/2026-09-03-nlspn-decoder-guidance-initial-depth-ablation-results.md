# NLSPN Decoder, Guidance, And Initial-Depth FP16 Ablation Results

## Protocol

The run uses the official NLSPN ResNet-34 checkpoint and DCN extension, the
fixed 128-sample NYU train calibration set, the fixed 64-sample validation
set, and 18 propagation iterations in FP16. `EARLY_W16A16` is fixed in every
candidate. Ordinary tensors remain FP6, and initial-depth inputs remain FP8
unless the complete initial-depth decoder is promoted to explicit IEEE FP16
weight and input QDQ. No training or QAT is performed.

The FP32 pooled RMSE is `0.150921` m. The fixed `EARLY_W16A16` baseline is
`0.152919` m (`+1.324%`). All 12 candidates completed two bit-exact forwards
and passed effective-format, finite-output, positive-depth, sample-identity,
and 18-state checks.

## Isolated Results

| Additional FP16 group | Pooled RMSE (m) | Recovery vs baseline (m) | Samples improved / worsened | Average W/A bits |
| --- | ---: | ---: | ---: | ---: |
| None | 0.152919 | 0.000000 | - | 6.494 / 7.593 |
| `dec5` | 0.152954 | -0.000035 | - | 6.620 / 7.606 |
| `dec4` | 0.153068 | -0.000148 | - | 6.860 / 7.672 |
| `dec3` | 0.152825 | +0.000094 | 46 / 18 | 6.854 / 7.750 |
| `dec2` | 0.152975 | -0.000055 | - | 7.214 / 7.907 |
| `gd_dec1` | 0.153056 | -0.000137 | - | 6.974 / 8.431 |
| `id_dec1 + id_dec0` | **0.152185** | **+0.000735** | **59 / 5** | **6.981 / 8.934** |

The initial-depth decoder recovers `36.8%` of the residual pooled-RMSE gap
between `EARLY_W16A16` and FP32. `dec3` provides a much smaller positive
contribution. The other isolated FP16 substitutions regress task RMSE.

## Cumulative Results

| Cumulative prefix | Pooled RMSE (m) | Marginal recovery (m) |
| --- | ---: | ---: |
| `dec5` | 0.152954 | -0.000035 |
| `dec5 -> dec4` | 0.153124 | -0.000170 |
| `dec5 -> dec3` | 0.152992 | +0.000133 |
| `dec5 -> dec2` | 0.153152 | -0.000161 |
| `+ gd_dec1` | 0.153330 | -0.000178 |
| `+ initial-depth` | 0.152511 | +0.000819 |

The complete prefix recovers `0.000408` m relative to the baseline but is
worse than protecting initial depth alone while using substantially more
precision (`W9.035/A10.337`). The shared-decoder interaction is
`-0.000088` m and the complete downstream interaction is `-0.000045` m.
Promoting all decoder modules therefore is not justified by task accuracy.

## Error Attribution

| Configuration | Initial-depth MSE | Guidance MSE | Offset MSE | Prediction MSE |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.020814 | 0.261627 | 0.175127 | 0.000981 |
| `dec3` only | 0.020607 | 0.259615 | 0.174058 | 0.000962 |
| Initial-depth only | **0.018166** | 0.261627 | 0.175127 | 0.000788 |
| Shared decoder | 0.018543 | 0.255918 | 0.171359 | 0.000973 |
| Shared + guidance | 0.018543 | 0.251906 | 0.168405 | 0.000990 |
| All groups | 0.016138 | 0.251906 | 0.168405 | **0.000777** |

Shared-decoder and guidance protection move every measured propagation-entry
signal closer to FP32, yet task RMSE worsens. Their FP6 perturbations partly
cancel the checkpoint's residual task error, so signal SQNR or FP32-distance
alone cannot select their precision. In contrast, initial-depth protection
reduces prediction MSE against FP32 and task RMSE consistently across most
samples.

Propagation still contracts the error: for initial-depth-only protection,
state MSE falls from `0.066675` at iteration 1 to `0.000788` at iteration 18.
The dominant recoverable residual is therefore injected by the
`id_dec1/id_dec0` initial-depth decoder before propagation, not accumulated by
the FP16 propagation loop. The next useful ablation should separate weight
and activation protection inside these two modules; promoting the whole
shared or guidance decoder is not supported by this result.

Validated artifacts are stored under
`profile_logs/nyu_nlspn_decoder_guidance_initial_ablation_64_v1`.
