# NLSPN FP6 Activation Protection Results

## Protocol

The experiment uses the official NLSPN ResNet-34 model and checkpoint, 18
propagation iterations, the fixed stratified 128-sample NYU train calibration
set, and the fixed 64-sample validation set. Ordinary convolution weights and
activations use scaled FP6 E3M2 fake quantization. Initial-depth inputs use FP8
E4M3FN, and the complete propagation operator remains FP16. No training or QAT
is performed.

The corrected evaluator disables the old concat adapter, clears external
ownership, applies initial-depth weight QDQ through the ordinary instrumentor,
and installs exactly one common or branch-aware input QDQ. Both candidate
forwards were bit-exact. The FP32 pooled RMSE under the same batched execution
is `0.150921` m.

## End-To-End Results

| Configuration | Pooled RMSE (m) | Delta vs FP32 (m) | Relative loss | Delta vs BASE (m) | Average W/A bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| BASE | 0.155148 | +0.004226 | +2.800% | 0.000000 | 6.000 / 6.335 |
| DEPTH_A8 | 0.155816 | +0.004895 | +3.243% | +0.000668 | 6.000 / 6.356 |
| RGB_A8 | 0.154592 | +0.003671 | +2.432% | -0.000555 | 6.000 / 6.398 |
| STEM_A8 | 0.154049 | +0.003128 | +2.072% | -0.001098 | 6.000 / 6.419 |
| EARLY_A8 | **0.153473** | **+0.002551** | **+1.691%** | **-0.001675** | **6.000 / 6.587** |
| STEM_EARLY_A8 | 0.153668 | +0.002746 | +1.820% | -0.001480 | 6.000 / 6.587 |
| STEM_EARLY_ID_BRANCH | 0.153645 | +0.002724 | +1.805% | -0.001502 | 6.000 / 6.587 |
| FULL_BRANCH_AWARE | 0.153549 | +0.002628 | +1.741% | -0.001598 | 6.000 / 6.587 |

`EARLY_A8` is selected by pooled RMSE and the `0.0001` m tie rule. It protects
the common inputs of `conv2.0.conv1`, `conv2.0.conv2`, and
`conv3.0.downsample.0`. Branch-independent scales do not improve on this
candidate, so they are not retained.

## Error Attribution

The effective FP6 checks confirm that `id_dec0.0` changes 1,151 of 1,152
weights and `id_dec1.0` changes 73,673 of 73,728 weights. Their weight SQNR is
24.91 dB and 25.62 dB, respectively. This rules out the earlier accidental
floating-point initial-depth weights.

Compared with BASE, `EARLY_A8` reduces initial-depth MSE from `0.06209` to
`0.03145`, guidance MSE from `0.53977` to `0.34240`, and affinity MSE from
`6.40e-5` to `4.23e-5`. Final prediction MSE against FP32 falls from `0.001885`
to `0.001112`.

Propagation does not amplify the observed error monotonically. BASE state MSE
falls from `0.09744` at iteration 1 to `0.001885` at iteration 18;
`EARLY_A8` falls from `0.07689` to `0.001112`. The dominant error is therefore
injected by the encoder and prediction-head inputs before propagation, while
input preservation and normalized propagation contract it over subsequent
iterations.

At the RGB/depth stem boundary, FP6 creates new zero codes for 1.69% of
nonzero RGB features and 5.90% of nonzero depth features. Protecting depth
alone worsens end-to-end RMSE, while RGB-only and common early protection
improve it. This confirms that local zero-code or SQNR improvement is not a
sufficient task-level selection criterion.

## Cleanup

The old `nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v1` through `v4`
artifact roots were removed. Those runs skipped `id_dec0.0/id_dec1.0` weight
quantization through external concat ownership and applied a second input QDQ,
so their W6/W8 labels and RMSE values were not valid comparisons.

The validated artifacts are under
`profile_logs/nyu_nlspn_fp6_activation_protection_64_v1`. They describe QDQ
fake quantization and format traffic budgets, not native FP6/FP8 CUDA kernels.
