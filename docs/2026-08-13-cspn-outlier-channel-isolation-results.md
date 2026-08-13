# CSPN Outlier Channel Isolation Results

## Experiment contract

The corrected experiment uses the official CSPN ResNet-18 architecture with
24 propagation iterations and the converged checkpoint
`cspn_iter24/best.pt`. It uses 128 fixed NYU training samples for calibration
and 64 fixed NYU validation samples for evaluation with seed `20260812`.
Weights are per-output-channel symmetric W4, eligible nonnegative activations
are unsigned Group-8 A4, bias and guidance remain FP32, and propagation keeps
the existing A8/Q13/INT32 contract.

OCI is applied at the first local QDQ owner. Harm is collected with the
contiguous W4A4 baseline active and immediately before each local QDQ. The
discarded implementation applied OCI at consumer inputs after producer/ReLU
QDQ had already rounded values to zero; those results were deleted and are not
used below.

## Boundary audit

The corrected calibration found 1,424 contiguous groups across 61 eligible
activation sites. All 776 consumer-input groups had zero positive harm because
their candidate values had already been quantized upstream. All 647 positive
candidates came from ReLU-output QDQ owners. This verifies that isolation must
act at the producer/ReLU quantization boundary to recover zero-collapsed
values.

The four highest calibration-ranked candidates were:

| Rank | QDQ owner | Group | Channel | Rescued energy |
|---:|---|---:|---:|---:|
| 1 | `relu#0` | 2 | 18 | 268,674.37 |
| 2 | `relu#0` | 7 | 62 | 123,433.32 |
| 3 | `relu#0` | 6 | 52 | 80,821.47 |
| 4 | `gud_up_proj_layer4.relu#0` | 2 | 20 | 53,093.73 |

The first three groups are in the encoder stem ReLU owner; the fourth is in
the deepest decoder up-projection ReLU owner. OCI changes no weight, channel
order, or channel count. Across all 50 quantized weight tensors, reconstruction
statistics are exactly invariant between the contiguous and OCI budgets.

## End-to-end results

| Configuration | Extra scales | RMSE | MAE | Boundary RMSE | Flat RMSE |
|---|---:|---:|---:|---:|---:|
| Contiguous Group8 | 0 | 0.313318 | 0.228845 | 0.501738 | 0.281930 |
| OCI-1 | 1 | 0.313184 | 0.228899 | 0.498496 | 0.282090 |
| OCI-2 | 2 | 0.311814 | 0.227725 | 0.500846 | 0.280336 |
| OCI-4 | 4 | **0.311092** | **0.227001** | **0.498766** | **0.279648** |
| OCI-8 | 8 | 0.317225 | 0.232833 | 0.501440 | 0.286872 |
| OCI-16 | 16 | 0.322406 | 0.237091 | 0.507040 | 0.292085 |
| OCI-32 | 32 | 0.336579 | 0.250445 | 0.512282 | 0.308108 |
| OCI-64 | 64 | 0.406315 | 0.311613 | 0.557527 | 0.383862 |
| OCI-all | 647 | 0.440713 | 0.334915 | 0.579868 | 0.420454 |

OCI-4 reduces mean RMSE by 0.002226, or 0.71%, and improves 37 of 64 paired
samples. It uses four additional activation scales across two QDQ sites. No
configuration produced a non-finite prediction.

## Error decomposition

OCI-4 reduces aggregate activation new-zero elements from 194,188,719 to
189,960,005 and total activation error energy from 9,760,851.81 to
9,515,182.36. At `relu#0` alone, new-zero elements fall from 21,857,947 to
17,436,809, zero-collapse energy falls from 515,445.22 to 271,325.78, and SQNR
rises from 7.78 dB to 10.06 dB. This is direct evidence that the selected
outlier channels had been increasing the Group8 step enough to erase smaller
nonzero activations.

The benefit is not monotonic. OCI-8 has slightly lower aggregate activation
error energy than OCI-4, but worse endpoint RMSE. Larger budgets also increase
downstream consumer-input rounding error and propagation MSE; OCI-all reduces
ReLU zero collapse while increasing total activation error to 10,380,802.73
and propagation mean MSE from `2.43945e-4` to `2.55754e-4`. Therefore rescued
activation energy is a useful first-stage zero-collapse score, but it does not
encode downstream Jacobian sensitivity or propagation amplification.

## Conclusion

The corrected experiment validates outlier channel isolation, not channel
splitting. A small selective budget can recover activations and modestly
improve CSPN W4A4 accuracy. Applying OCI broadly is harmful. The next selection
criterion should combine calibration rescue with local block-output error or a
propagation-aware sensitivity term, while keeping the fixed `1+7` scale
contract and calibration-only selection.

Artifacts are stored in
`profile_logs/nyu_cspn_outlier_channel_isolation_64/`.
