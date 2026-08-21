# CSPN Mixed Task-Aware QAT Results

## Result

The CSPN W4-dominant, mixed A4/A6/A8 task-aware QAT configuration passed the
predeclared fixed-64 acceptance protocol. It uses the official CSPN ResNet-18
model with 24 propagation iterations, the fixed P3/T3 W4/W8 weight assignment,
static contiguous Group8 activation quantization, FP32 guidance, and the
propagation-aware A8/Q13 contract.

No precision promotion, FP fallback, numerical recovery, or evaluation-set
selection was used.

## Selected Precision

The 128-sample train-calibration search evaluated 179 joint candidates and
selected `MIXED_A8_8_8_6_6`:

| Block | Activation bits |
| --- | ---: |
| Encoder stem | 8 |
| Encoder layer 1 | 8 |
| Encoder layer 2 | 8 |
| Decoder layer 4 | 6 |
| Initial-depth head | 6 |
| Remaining ordinary activation owners | 4 |

The selected assignment has activation-element fractions A4 `0.253497`, A6
`0.534249`, and A8 `0.212255`. Its independently recomputed weighted average
is `147402752 / 24909568 = 5.917515390070193` bits. The fixed P3/T3 weight
assignment averages `4.169608968254615` bits by weight elements; W8 accounts
for `4.240224%` of weight elements and `46.325040%` of weight MACs.

## Training

Training used all 6,700 official NYU training samples. Early stopping used 590
validation samples after excluding the fixed 64 evaluation identities. The
run stopped on `validation_plateau` at epoch 28 with `patience=6` and a minimum
relative improvement of `0.001`.

| Measurement | Value |
| --- | ---: |
| Best epoch | 22 |
| Best validation RMSE | 0.159893739286333 m |
| Final epoch validation RMSE | 0.159929346279807 m |
| Canonical parametrization keys | 0 |

The canonical hard checkpoint is `mixed_static/best.pt`; the QAT optimizer
state remains separately available in `mixed_static/best_qat.pt`.

## Hard-Path Evaluation

Every row below was produced from a fresh official model instance. The full
validation set contains 654 samples. Fixed-64 RMSE is the mean of the 64
per-sample RMSE values used by the acceptance protocol.

| Configuration | Full RMSE (m) | Fixed-64 RMSE (m) | Average W bits | Average A bits |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.148247853852 | 0.159791981234 | 32.000000 | 32.000000 |
| Uniform W6A6 | 0.173413191210 | 0.183695553632 | 6.000000 | 6.000000 |
| P3/T3 | 0.163062110289 | 0.175442385940 | 4.169609 | 6.986013 |
| Mixed task-aware QAT | **0.161019651374** | **0.171399163760** | **4.169609** | **5.917515** |

Mixed QAT improves fixed-64 RMSE by `0.004043222180 m` over P3/T3 and by
`0.012296389872 m` over uniform W6A6 while meeting the average activation
budget. Its fixed-64 RMSE remains `0.011607182526 m` above FP32.

## Acceptance And Independent Verification

All declared gates passed:

| Gate | Measured | Limit |
| --- | ---: | ---: |
| Fixed-64 RMSE | 0.171399163760 m | <= 0.175 m |
| Average activation bits | 5.917515390070 | <= 6.0 |
| Non-finite prediction ratio | 0 | 0 |
| Non-positive prediction ratio | 0 | 0 |
| Sparse-anchor maximum error | 0 | 0 |
| Coefficient-sum maximum error | 0 | 0 |
| Contraction violation ratio | 0 | 0 |

RMSE was independently recomputed from all 256 saved prediction payloads
using the repository's float64 accumulation definition. The recomputed values
for FP32, W6A6, P3/T3, and Mixed QAT match `sample_metrics_64.csv` exactly;
the maximum absolute difference is `0`. The independently recomputed
activation numerator, denominator, and average also match the acceptance JSON
exactly.

## Artifacts

The artifact root is
`profile_logs/nyu_cspn_mixed_task_aware_qat`.

- Search: `search/selected_assignment.json`, `search/cost_basis.json`, and
  `search/candidate_metrics.csv`.
- Training: `mixed_static/best.pt`, `mixed_static/best_qat.pt`,
  `mixed_static/metrics.csv`, and `mixed_static/run_summary.json`.
- Evaluation: `evaluation/aggregate_metrics.csv`,
  `evaluation/sample_metrics_64.csv`, `evaluation/acceptance_report.json`, and
  `evaluation/evaluation_manifest.json`.
- Prediction comparison: `evaluation/figures/mixed_prediction_details.png`
  and `evaluation/figures/mixed_predictions_64.png`, with matching PDF files.

Key SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Official checkpoint | `482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855` |
| Precision config | `e476f2400c9afc337df28ae99f70248733a2b00c3ba231d9a1deb4ad75761e76` |
| Combined calibration metadata | `4c709b30e013ae43f038c18e01c43e054cb1acb71774b465a5761ee8db4498a0` |
| Selected assignment | `811dc5307b242a57c3fa230eda0cbd3859d53a9389c241a94888f2a8d331669b` |
| Cost basis | `4b1f27fe0ee3dfbeb5440286bb72bf2e511f3f6bf0fa6eeff4b1b8ea7056cccc` |
| Canonical Mixed QAT checkpoint | `078613d6e5e24efca93abbcfc69e1483d22ab6c608302ec48bccb3260f6a914e` |
| Acceptance report | `8a0dc16d2995c371737acb844bd17ecbb499f96c86830467e2c7692243c9d3d7` |

