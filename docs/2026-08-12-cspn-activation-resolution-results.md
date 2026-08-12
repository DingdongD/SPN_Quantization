# CSPN Activation Resolution Results

## Protocol

- Model: official CSPN ResNet-18, 24 propagation iterations.
- Checkpoint: converged `cspn_iter24/best.pt`.
- Data: NYU Depth V2, 128 fixed training samples for calibration and the
  existing fixed 64-sample validation subset for evaluation.
- Weights: signed symmetric per-output-channel W4.
- Activations: A4 with tensor, contiguous-group, or channel scales. ReLU
  outputs use unsigned A4; signed boundaries use symmetric A4.
- Propagation: affinity, confidence, offset, and state use A8; coefficients
  use signed INT16 Q13 and INT32 accumulation.
- Exclusions: the guidance head and bias remain FP32.
- Coverage: 69 ordinary QDQ sites and two externally owned decoder
  boundaries. Structural merge and propagation statistics are recorded by
  their owning adapters instead of being duplicated as ordinary QDQ sites.

Configuration and sensitive-site selection used calibration block-output MSE
and SQNR. Scale factors were selected by direct downstream block MSE, with the
factor as the deterministic tie-breaker. Evaluation RMSE was used only to
reject a calibration-selected scale configuration that did not transfer.

## End-to-End Results

| Configuration | RMSE (m) | MAE (m) | AbsRel | iRMSE | Flat RMSE (m) | Boundary RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.166932 | 0.067154 | 0.022361 | 0.022990 | 0.115248 | 0.416973 |
| PA only | 0.175221 | 0.082805 | 0.029492 | 0.026161 | 0.125171 | 0.423577 |
| W4 only | 0.203795 | 0.117773 | 0.042022 | 0.032471 | 0.158178 | 0.446227 |
| A4 only | 0.417413 | 0.325058 | 0.155111 | 0.101304 | 0.397733 | 0.538718 |
| RTN W4A4 | 0.439167 | 0.330944 | 0.150979 | 0.099180 | 0.419371 | 0.563534 |
| Hybrid Group128 | 0.430961 | 0.325712 | 0.148014 | 0.097656 | 0.410437 | 0.559222 |
| Hybrid Group64 | 0.437981 | 0.333111 | 0.153246 | 0.099709 | 0.418468 | 0.560272 |
| Hybrid Group32 | 0.405914 | 0.302502 | 0.137338 | 0.094989 | 0.383538 | 0.550302 |
| Hybrid Group16 | 0.369955 | 0.272886 | 0.123937 | 0.088699 | 0.344567 | 0.535366 |
| Hybrid Group8 | 0.313318 | 0.228845 | 0.097618 | 0.072664 | 0.281930 | 0.501738 |
| Per-channel | **0.287500** | **0.206191** | 0.082030 | 0.068337 | **0.253407** | **0.486298** |
| Selective channel | 0.369686 | 0.281619 | 0.114307 | 0.088188 | 0.343905 | 0.531498 |
| Shared-A4 merge | 1.651565 | 1.517410 | 0.628087 | 402114.072754 | 1.644531 | 1.722686 |
| Residual A4/A8 merge | 0.470958 | 0.378844 | 0.146967 | 0.116209 | 0.450461 | 0.615636 |
| Calibrated scale | 0.311916 | 0.225373 | 0.087123 | 0.083957 | 0.275250 | 0.520867 |

Per-channel A4 reduces RMSE by 34.54% relative to tensor RTN W4A4 and is the
accepted configuration. It requires 13,380 activation scales instead of 71.
The group configurations are explicitly hybrid: sites whose channels are not
divisible by the requested group size retain tensor scales. Hybrid Group128
uses tensor scales for 75.69% of activation elements. Hybrid Group16 and
Group8 use group scales for 98.89% of activation elements and require 837 and
1,673 scales respectively. Hybrid Group8 reaches 0.313318 m and is the
practical intermediate point when per-channel scale storage is too high.

## Error Attribution

The RTN RMSE deltas relative to FP32 are:

- propagation: +0.008289 m;
- W4 weights: +0.028574 m;
- A4 activations: +0.242192 m;
- weight/activation interaction: -0.006820 m.

Activation quantization is therefore the dominant source. The two
rotation-owned boundaries account for 22.48% of total recorded activation
error energy. After excluding those boundaries, ordinary decoder QDQ sites
account for 70.91% and encoder sites account for 29.09%.

Tensor A4 maps 39.65% of originally nonzero activation values to zero. The
aggregate activation SQNR is 12.13 dB, and zero collapse contributes 53.75%
of activation error energy. Per-channel A4 lowers the new-zero rate to 19.41%
and raises SQNR to 17.76 dB. Clipping error energy remains negligible at
0.036%, so the dominant failure is insufficient resolution rather than
clipping.

The strongest calibration-sensitive sites are:

1. the stem `relu#0` output;
2. `gud_up_proj_layer4.sc_conv1` output;
3. `gud_up_proj_layer2.sc_conv1` output;
4. `layer1.0.relu#1` output.

The critical tensor-level tail ratios are moderate while their new-zero rates
are high. These aggregate tensor statistics support channel imbalance and
low-energy feature collapse as the dominant observed mechanism; they do not
by themselves rule out outliers within individual channels.

## Rejected Extensions

The two selected decoder add branches have calibration base-to-update RMS
ratios of 0.98 to 1.00. They do not satisfy the assumed small-update/large-base
structure. On evaluation, residual A4/A8 quantization maps 52.89% and 90.85%
of nonzero A4 update-branch values to zero at the selected adds and degrades
RMSE to 0.470958 m. The A8 shortcut/base branches add no new zeros, but this
does not recover the collapsed update branches. Shared-A4 merge is
substantially worse and is rejected. Split-local branch zero rates and
merge-output SQNR are recorded in `merge_branch_metrics.csv`; calibration
distribution ratios retain a `calibration_` prefix.

Coordinate-calibrated scales optimize each owner's direct downstream block
and lower evaluation new-zero rate to 16.83%, but evaluation RMSE is
0.311916 m versus 0.287500 m for its per-channel base. Boundary RMSE also
increases from 0.486298 to 0.520867 m. The selected factors are 0.625 for the
two encoder ReLU owners and 0.5 for both decoder shortcut projections. The
scale configuration is rejected by the declared transfer rule; it is not
reported as a learned or LSQ method.

## Artifacts

Runtime outputs are stored outside Git under:

```text
/workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/
```

The directory contains per-sample and aggregate depth metrics, exact
activation error partitions, per-channel diagnostics, block attribution,
propagation statistics, split-scoped merge statistics, selection tables, and
64 prediction payloads each for FP32, RTN W4A4, per-channel W4A4, and the
rejected calibrated-scale configuration. Every payload includes RGB, sparse
depth, GT, FP32 prediction, quantized prediction, and absolute error.

Reproduce the run with:

```bash
PYTHONPATH=. python scripts/run_nyu_cspn_activation_resolution.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --sample-metrics /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4/cspn/sample_metrics.csv \
  --data-root /workspace/CSPN/cspn_pytorch \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution \
  --device cuda:0 --seed 20260812 --calibration-samples 128 \
  --sample-capacity 256 --candidate-sites 8 --sensitive-sites 4 \
  --fold-max-error 0.05
```
