# NLSPN Selective Channel Smoothing

This experiment uses the unified 128-sample calibration and 64-sample
evaluation sets. Propagation remains FP16. Ordinary tensors use FP6 E3M2,
while `id_dec0.0` and `id_dec1.0` weights and inputs use FP8. Channel
smoothing is tested only on `conv2.0.conv1`, `conv2.0.conv2`,
`conv3.0.conv1`, and `conv6.0`.

## End-to-End Results

| Configuration | Pooled RMSE | Relative FP loss |
| --- | ---: | ---: |
| Protected FP6 baseline | 0.162533 | +7.71% |
| Conv2 alpha 0.50 | 0.163260 | +8.20% |
| Conv3 alpha 0.50 | 0.162958 | +8.00% |
| Conv6 alpha 0.50 | 0.162558 | +7.73% |
| All alpha 0.25 | 0.163010 | +8.03% |
| All alpha 0.50 | 0.163424 | +8.30% |
| All alpha 0.75 | 0.161933 | +7.32% |

All candidates are finite, reproducible, and propagation-valid. The full
artifact is stored under
`profile_logs/nyu_nlspn_selective_channel_smoothing_fp6_v1`.

## Range And Quantization Changes

The best alpha 0.75 configuration changes the selected activation ranges as
follows:

| Module | Channel imbalance before | After | FP6 zero ratio before | After | Activation SQNR before | After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `conv2.0.conv1` | 10.71 | 1.99 | 8.10% | 2.60% | 25.26 dB | 25.63 dB |
| `conv2.0.conv2` | 3.67 | 1.45 | 46.78% | 45.91% | 25.05 dB | 25.64 dB |
| `conv3.0.conv1` | 3.69 | 1.52 | 19.28% | 17.89% | 25.41 dB | 25.83 dB |
| `conv6.0` | 12.36 | 1.94 | 33.96% | 30.57% | 24.14 dB | 25.56 dB |

The transformation removes most channel imbalance without materially
degrading FP6 weight SQNR, but the pooled RMSE improvement is only 0.000600 m.
Individual Conv2 and Conv3 smoothing is slightly worse than the protected
baseline, so local range or SQNR improvement does not predict task gain.

The much larger improvement comes from protecting the initial-depth path:
uniform FP6W/FP6A previously produced pooled RMSE 0.189209, while the protected
FP6 baseline produces 0.162533. This confirms that activation outliers are a
secondary numerical contributor. The dominant error source is the task gain
at `id_dec0/1` and the interaction between upstream encoder quantization and
the propagation entry.
