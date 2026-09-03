# W6 Boundary Bit Allocation Results

## Protocol

- NYU evaluation set: the same 64 paired samples used by the existing model
  comparisons.
- Calibration set: the existing 128-sample metadata.
- Ordinary Conv/Linear modules use independent weight and activation budgets.
- Propagation loop remains FP32 and propagation projection remains W8A8.
- A module is protected at W8 only when its direct single-module W6A6 endpoint
  increases pooled RMSE by more than 0.02 m. Other modules remain at W6 unless
  the protected modules require W4 demotions to satisfy the average budget.
- The budget is 6.0 weighted average bits for both weights and activations.

## Results

| Model | FP32 | W8A8 | W6A6 | W6-boundary mixed | Mixed vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSPN | 0.193794 | 0.194676 | 0.199229 | 0.199229 | +2.80% |
| DySPN | 0.141821 | 0.141876 | 0.149459 | 0.149459 | +5.39% |
| NLSPN | 0.150921 | 0.159787 | 0.677532 | 0.446676 | +195.97% |
| CompletionFormer | 0.140186 | 0.141636 | 0.154394 | 0.155270 | +10.76% |

The measured outputs are under:

`/workspace/SPN_Quantization/profile_logs/nyu_task_gradient_w6_boundary_ablation_128`

The reported mixed assignments satisfy both average budgets and all endpoint
validity checks. The largest W6-sensitive modules in NLSPN are `id_dec0.0`,
`conv2.0.conv1`, and `conv2.0.conv2`; they are retained at W8. This reduces the
NLSPN endpoint from 0.677532 m to 0.446676 m, but does not make W6A6 acceptable.

## Decision

The W6 boundary is useful as a protection rule, but it is not a guarantee of
accuracy. CSPN and DySPN need no W4/W8 redistribution at this budget. NLSPN
needs a propagation-aware method or a larger protected precision region; a
plain W6-centered RTN allocation remains insufficient. A gradient-only W8/W4
exchange was tested separately and rejected because it increased endpoint RMSE
substantially through cross-module interactions.
