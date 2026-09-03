# NLSPN Early FP16 Error Attribution Results

## Protocol

The run uses the official NLSPN ResNet-34 checkpoint and DCN extension, the
fixed 128-sample NYU train calibration set, the fixed 64-sample validation
set, and 18 propagation iterations in FP16. Ordinary tensors remain FP6,
initial-depth inputs remain FP8, and selected early tensors use explicit
`FP32 -> IEEE FP16 -> FP32` QDQ. No training or QAT is performed.

The FP32 pooled RMSE measured in the same process is `0.150921` m. Every
candidate completed two bit-exact forwards and passed effective-format,
finite-output, positive-depth, sample-identity, and propagation-state checks.

## Results

| Configuration | Pooled RMSE (m) | Delta vs FP32 | Delta vs W6A8 | Average W/A bits |
| --- | ---: | ---: | ---: | ---: |
| `EARLY_W6A8` | 0.153473 | +1.691% | 0.000000 | 6.000 / 6.587 |
| `EARLY_W6A16_CONV2_0_CONV1` | 0.153515 | +1.719% | +0.000043 | 6.000 / 6.922 |
| `EARLY_W6A16_CONV2_0_CONV2` | 0.153490 | +1.702% | +0.000017 | 6.000 / 6.922 |
| `EARLY_W6A16_CONV3_0_DOWNSAMPLE` | 0.153740 | +1.868% | +0.000267 | 6.000 / 6.922 |
| `EARLY_W6A16` | 0.153862 | +1.948% | +0.000389 | 6.000 / 7.593 |
| `EARLY_W16A8` | 0.153905 | +1.977% | +0.000432 | 6.494 / 6.587 |
| **`EARLY_W16A16`** | **0.152919** | **+1.324%** | **-0.000554** | **6.494 / 7.593** |

Replacing `EARLY_A8` with `EARLY_A16` alone does not improve task accuracy.
All three single-site A16 substitutions also regress, with
`conv3.0.downsample.0` producing the largest increase.

## Attribution

The early activation-only recovery is `-0.000389` m and the early weight-only
recovery is `-0.000432` m; negative recovery means a regression. Protecting
both produces a `0.000554` m improvement over `EARLY_W6A8`. The RMSE-domain
weight-activation interaction term is `-0.001375` m, showing that the combined
improvement is an interaction rather than the sum of two independently useful
protections.

Early A16 increases activation SQNR from about 31.5 dB to 73.6-73.7 dB and
reduces new-zero ratios to at most `2.61e-7`. It also reduces final prediction
MSE against FP32 from `0.001112` to `0.001003`, yet task RMSE increases. The A8
perturbation therefore partly cancels other model/quantization errors on this
evaluation set. Tensor SQNR and distance to the FP32 prediction are not valid
standalone bit-allocation objectives here.

`EARLY_W16A16` reduces initial-depth MSE against FP32 to `0.02081`, but guidance
and offset MSE remain `0.26163` and `0.17513`. These are the largest remaining
propagation-entry errors. State MSE decreases from `0.06696` at iteration 1 to
`0.000981` at iteration 18, so propagation contracts rather than accumulates
the measured error.

The remaining `0.001998` m pooled-RMSE gap between `EARLY_W16A16` and FP32 is
therefore predominantly injected outside the protected early group, before
propagation. The next strict localization boundary is the shared decoder
(`dec5` through `dec2`) followed by `gd_dec1` and the initial-depth decoder.
Those modules require separate task-level ablations before assigning the
residual to one layer.

Validated artifacts are stored under
`profile_logs/nyu_nlspn_early_fp16_error_attribution_64_v1`.
