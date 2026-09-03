# Task-Gradient Ablation and Mixed Precision

## Protocol

- Use the official NYU checkpoint and the existing 128-sample calibration metadata.
- Evaluate 64 paired validation samples for every candidate.
- Use W8A8 ordinary modules with an FP32 propagation loop as the reference.
- Keep the propagation projection at W8A8; propagation semantic tensors are not
  included in the ordinary CNN budget.
- Run ranked single-module endpoint ablations at the selected boundary. For the
  W6-boundary study, this is W6A6: a module changes its own weight and mapped
  module-input activation owners only.
- Protect an ablated module at W8 when its endpoint delta exceeds 0.02 m or the
  endpoint is numerically invalid.
- Use W6A6 as the precision boundary: protected units start at W8, all other
  units start at W6, and only the least sensitive non-protected units are
  demoted to W4 when a separate budget requires it.
- Allocate ordinary weights and activations independently over 4, 6, and 8 bit
  using task-gradient marginal loss per saved weighted cost. The W8 protection
  decision comes from direct W6A6 endpoint ablation. An infeasible protected
  boundary is reported explicitly.

## Cost And Metrics

Weight budget is MAC-weighted over contract-owned Conv/Linear modules. Activation
budget is element-weighted over contract activation owners. The final endpoint is
reported with pooled RMSE, mean per-sample RMSE, finite/reproducible flags, and
the actual weighted W/A averages.

## Measured Runs

The earlier 4.9/5.9 independent-budget W4-ablation runs are stored under:

`/workspace/SPN_Quantization/profile_logs/nyu_task_gradient_ablation_128_w4a4_budget_4.9_5.9`

The published model-specific runs are:

- `cspn_top16`
- `dyspn`
- `nlspn_top20`
- `completionformer`

The W6-boundary rerun is stored under:

`/workspace/SPN_Quantization/profile_logs/nyu_task_gradient_boundary_w6_128`

It includes `FP32_REFERENCE`, `W8A8_FP32_PROP`, `W6A6_FP32_PROP`, and the
W6-boundary mixed endpoint for each model.

Their propagation projection is W8A8 and their propagation loop is FP32.

## Interpretation

The W6-boundary study uses the direct W6A6 endpoint ablations. This is required
because a W4A4 endpoint can overstate or misidentify the modules that must be
protected at W8 when the operating point is W6A6. The measured W6A6 baseline and
the resulting conservative mixed endpoint are reported separately for every
model. A separate exploratory W8/W4 exchange based only on gradient scores was
rejected after endpoint validation because cross-module interactions caused
large regressions.
