# Four-Model FP6 Evaluation

The evaluation uses the existing 128-sample calibration split, 64-sample
evaluation split, official model checkpoints, and the unified FP16
propagation protocol. Ordinary model tensors use static calibrated scaled
floating-point quantization. The added format is finite scaled FP6 E3M2 with
32 non-negative code values: three subnormal values and seven normal exponent
levels. Propagation state, affinity, offset, confidence, and gates remain
outside the ordinary FP quantizer.

## Uniform Results

`pooled_rmse` and `relative_fp_loss` are read from each model manifest.

| Model | FP8W/FP8A | FP6W/FP6A | FP4W/FP6A | FP6W/FP4A | FP4W/FP4A |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSPN | 0.195746 (+1.01%) | 0.209187 (+7.94%) | 0.243550 (+25.67%) | 0.279111 (+44.02%) | 0.296917 (+53.21%) |
| DySPN | 0.143229 (+0.99%) | 0.142536 (+0.50%) | 0.143673 (+1.31%) | 0.530101 (+273.78%) | 0.642926 (+353.34%) |
| NLSPN | 0.158965 (+5.35%) | 0.189209 (+25.39%) | 0.198093 (+31.28%) | 1.008600 (+568.42%) | 0.944900 (+526.20%) |
| CompletionFormer | 0.140670 (+0.36%) | 0.142555 (+1.70%) | 0.165262 (+17.90%) | 0.503082 (+258.91%) | 0.731919 (+422.17%) |

All rows are valid and use `propagation_dtype=fp16`. The results are stored
under `profile_logs/nyu_four_model_fp6_uniform_fp16prop_v2`.

## Three-Level Allocation

The staged allocator starts every ordinary tensor at FP4, satisfies the
configured FP8 floor for boundary groups, and then promotes each remaining
unit through `FP4 -> FP6 -> FP8` using the largest feasible marginal score
reduction per cost. Propagation remains FP16.

| Model | FP4/FP6/FP8 grouped pooled RMSE | Relative FP loss | Average W | Average A |
| --- | ---: | ---: | ---: | ---: |
| CSPN | 0.282329 | +45.69% | 5.914 | 6.496 |
| DySPN | 0.152607 | +7.61% | 5.991 | 6.489 |
| NLSPN | 0.707240 | +368.70% | 5.899 | 6.490 |
| CompletionFormer | 0.333829 | +138.16% | 5.815 | 6.471 |

The three-level allocation improves the previous FP4/FP8 grouped policy for
DySPN and NLSPN, but is worse for CSPN and CompletionFormer. This is an
important result: adding FP6 is not itself a solution. Under the same global
budget, the greedy staged allocator can replace an FP8 assignment with several
FP6 assignments and lose protection at a nonlinear boundary. The assignment
must therefore use downstream and propagation-entry sensitivity, not only the
local FP4-to-FP6 or FP6-to-FP8 score.

## Error Source

At uniform FP6, the sum of the task-gradient weighted quantization error from
activation rows accounts for the following fraction of the combined weight
and activation score:

| Model | Activation share of score |
| --- | ---: |
| CSPN | 80.8% |
| DySPN | 93.3% |
| NLSPN | 91.8% |
| CompletionFormer | 95.3% |

This agrees with the mixed-format ablations: FP4 activation is much more
damaging than FP4 weight for DySPN, NLSPN, and CompletionFormer. CSPN is less
separable because its decoder and propagation-entry features have stronger
interaction.

The highest FP6 activation task scores are concentrated in:

- CSPN: `conv1_1` and the `gud_up_proj_layer4` decoder path.
- DySPN: `base.conv2.0.conv1`, followed by the RGB/depth stem and `base.dec2.0`.
- NLSPN: `conv2.0.conv1`, `conv2.0.conv2`, `conv1_dep`, and the `dec4` path.
- CompletionFormer: `backbone.conv1.0`, `backbone.conv1_dep`, transformer
  embedding layers, and early decoder inputs.

The task score is

`S_i(b) = sum(abs(g_i * (x_i - Q_b(x_i))))`.

It is more useful than local output MSE because it weights an error by its
effect on the depth loss. Local MSE alone is not sufficient: CompletionFormer
has very large decoder feature MSE values whose task gradients are near zero,
while NLSPN has large encoder task scores despite moderate local MSE.

The remaining limitation is cross-layer propagation. The current score is a
first-order, single-module metric; it does not model skip-connection scale
mismatch, initial-depth/fusion interaction, or repeated nonlinear sensitivity.
The next allocation metric should therefore combine task-gradient error with
the measured downstream output perturbation and propagation-entry gain, while
keeping propagation itself in FP16.
