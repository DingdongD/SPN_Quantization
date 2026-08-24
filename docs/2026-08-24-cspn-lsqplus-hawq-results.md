# CSPN LSQ+ and HAWQ Results

## Protocol

- Model: official CSPN ResNet-18 with 24 propagation steps.
- Checkpoint: `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt`.
- Calibration: 128 stratified NYU train samples.
- Evaluation: one shared fixed set of 64 NYU validation samples.
- Guidance: FP32.
- Propagation: A8 affinity/confidence/offset/state, Q13 coefficients, INT32 accumulation.
- Training: at most 30 epochs with patience 6; evaluation uses each method's best validation checkpoint.

All evaluated model loads reported no missing or unexpected checkpoint keys. The only ignored checkpoint key is the unused official `post_process_layer.sum_conv.weight`.

## Fixed-64 Results

| Configuration | RMSE (m) | Delta FP32 (m) | Relative RMSE | MAE (m) | AbsRel | iRMSE |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.187659 | 0.000000 | 0.00% | 0.064922 | 0.021802 | 0.026301 |
| PA-RTN W4A4 | 0.335449 | +0.147790 | +78.75% | 0.245582 | 0.106399 | 474.794038 |
| PA-RTN W6A6 | 0.203897 | +0.016238 | +8.65% | 0.102987 | 0.038818 | 0.036211 |
| LSQ+ W4A4 | 0.266277 | +0.078618 | +41.89% | 0.171049 | 0.067001 | 0.055647 |
| LSQ+ W6A6 | 0.198875 | +0.011216 | +5.98% | 0.095696 | 0.035111 | 0.032879 |
| HAWQ Mixed<=6 | 0.273392 | +0.085733 | +45.69% | 0.113232 | 0.036648 | 0.032620 |
| Mixed task-aware QAT | 0.190818 | +0.003160 | +1.68% | 0.090952 | 0.033359 | 0.031671 |

PA-RTN W4A4 produced one non-positive pixel out of 4,435,968 evaluated pixels. The near-zero prediction makes its inverse-depth metric invalid as a useful comparison, although its depth-domain RMSE remains finite. All other configurations had zero non-positive and non-finite predictions.

## Convergence

| Method | Best epoch | Best validation RMSE (m) | Stop reason |
|---|---:|---:|---|
| LSQ+ W4A4 | 29 | 0.240001 | 30-epoch limit |
| LSQ+ W6A6 | 30 | 0.165358 | 30-epoch limit |
| HAWQ Mixed<=6 | 14 | 0.201002 | 6-epoch validation plateau |

LSQ+ improves both uniform baselines, but the gain is much larger at 6 bit. Learned activation step and offset do not recover the information removed by uniform A4 at sensitive encoder-decoder boundaries.

## HAWQ Allocation

The corrected block-diagonal Hutchinson trace is positive for every measured weight module. The selected allocation is:

| Block | Bits |
|---|---:|
| Encoder stem | 8 |
| Encoder layer 1 | 6 |
| Encoder layer 2 | 6 |
| Encoder layer 3 | 4 |
| Encoder layer 4 | 6 |
| Decoder layer 1 | 4 |
| Decoder layer 2 | 4 |
| Decoder layer 3 | 4 |
| Decoder layer 4 | 6 |
| Initial depth | 8 |

The resulting average precision is 4.797 weight bits and 5.998 activation bits. Its fixed-64 RMSE is worse than uniform LSQ+ W6A6 because the HAWQ objective ranks weight perturbation with weight Hessian traces, while the tied activation assignment does not directly model activation quantization error. Assigning A4 to encoder layer 3 and three decoder blocks is therefore too aggressive for CSPN depth completion.

## Artifacts

- Metrics: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_lsqplus_hawq_formal/evaluation/aggregate_metrics.csv`
- Relative loss: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_lsqplus_hawq_formal/evaluation/relative_fp_loss.csv`
- Prediction details: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_lsqplus_hawq_formal/evaluation/figures/prediction_details.png`
- Prediction contact sheet: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_lsqplus_hawq_formal/evaluation/figures/prediction_contact_sheet.png`
- RMSE comparison: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_lsqplus_hawq_formal/evaluation/figures/quantization_rmse_comparison.png`
- HAWQ traces and assignment: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_lsqplus_hawq_formal/hawq_trace`
