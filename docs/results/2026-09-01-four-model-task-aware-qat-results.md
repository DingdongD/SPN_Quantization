# Four-Model Task-Aware QAT Results

## Protocol

- Official CSPN, DySPN, NLSPN, and CompletionFormer checkpoints.
- Propagation kept in FP16 and excluded from integer QAT.
- Ordinary Conv/Linear modules use the existing task-aware LSQ+ controller.
- 128 train-split calibration samples and the fixed ordered 64-sample NYU
  evaluation set.
- Twenty QAT epochs; the best finite and positive validation checkpoint is
  used for the reported metrics.

The W4 configuration is uniform W4A4. The W5 and W6 configurations use the
existing sensitivity-derived assignment under the corresponding nominal
budget; their realized average bits are recorded below rather than rounded to
the nominal label.

## Best QAT Checkpoints

| Model | FP32 pooled RMSE | Configuration | Pooled RMSE | Relative loss | Average W/A bits | Best epoch |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CSPN | 0.193794 | W4A4 | 0.489359 | +152.51% | 4.00 / 4.00 | 20 |
| CSPN | 0.193794 | W5A5 allocation | 0.273358 | +41.06% | 5.00 / 5.00 | 20 |
| CSPN | 0.193794 | W6A6 allocation | 0.226074 | +16.66% | 5.99 / 5.94 | 20 |
| DySPN | 0.141821 | W4A4 | 0.351915 | +148.14% | 4.00 / 4.00 | 20 |
| DySPN | 0.141821 | W5A5 allocation | 0.194317 | +37.02% | 4.99 / 4.99 | 20 |
| DySPN | 0.141821 | W6A6 allocation | 0.184567 | +30.14% | 5.99 / 6.00 | 18 |
| NLSPN | 0.150894 | W4A4 | 0.321946 | +113.36% | 4.00 / 4.00 | 19 |
| NLSPN | 0.150894 | W5A5 allocation | 0.266309 | +76.49% | 5.00 / 5.00 | 19 |
| NLSPN | 0.150894 | W6A6 allocation | 0.251391 | +66.60% | 5.99 / 6.00 | 20 |
| CompletionFormer | 0.140168 | W4A4 | 0.227919 | +62.60% | 4.00 / 4.00 | 20 |
| CompletionFormer | 0.140168 | W5A5 allocation | 0.203766 | +45.37% | 5.00 / 5.00 | 20 |
| CompletionFormer | 0.140168 | W6A6 allocation | 0.184460 | +31.60% | 6.00 / 6.00 | 20 |

## Interpretation

Task-aware QAT makes the previously invalid NLSPN W4/W5 results finite and
positive under the FP16 propagation contract. It also substantially improves
the existing sensitivity-allocated PTQ results for CSPN W5/W6, DySPN W5, and
CompletionFormer W5. It is not universally better: CSPN W4, DySPN W4, and
CompletionFormer W6 remain worse than their corresponding PTQ measurements.

Therefore, QAT improves propagation stability and selected low-bit operating
points, but the current controller is not yet a proof of an optimal mixed
precision allocation. The next optimization target is the propagation-aware
task loss and layer assignment around initial-depth, decoder fusion, and model
specific propagation inputs.

Results are stored under
`profile_logs/nyu_four_model_unified_fp16_task_aware_qat_64_v2/`.
