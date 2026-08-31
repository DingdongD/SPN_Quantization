# Unified FP16 Propagation and Task-Aware Mixed-Precision Quantization

## Goal

Unify CSPN, DySPN, NLSPN, and CompletionFormer under one fair NYU quantization protocol: keep each model's complete propagation semantic subgraph in FP16, and independently allocate W/A bits from `{4, 6, 8}` over the remaining ordinary encoder, decoder, and non-propagation head modules using task-gradient sensitivity under separate weight and activation budgets.

## Scope

The experiment covers the official converged checkpoints and the existing 128-sample stratified calibration set with the existing fixed 64-sample evaluation identities. It changes the active evaluation protocol and allocation path; it does not change official model topology, propagation iteration count, checkpoint weights, dataset preprocessing, or the existing integer PA implementation used for historical comparisons.

## Unified Protocol

### Propagation ownership

Propagation ownership is model-specific but the precision policy is shared:

- CSPN: guidance/initial-depth propagation heads, propagation projection, affinity normalization, state update, and sparse-depth anchor injection.
- DySPN: initial-depth/guidance/confidence heads, dynamic propagation projection, offset/affinity construction, confidence gate, state update, and sparse-depth anchor injection.
- NLSPN: `id_dec*`, `gd_dec*`, `cf_dec*`, `prop_layer.conv_offset_aff`, offset construction, confidence sampling, affinity normalization, deformable propagation, state update, and sparse-depth anchor injection.
- CompletionFormer: corresponding depth/guidance/confidence heads, `prop_layer.conv_offset_aff`, attention-independent propagation signals, affinity normalization, deformable propagation, state update, and sparse-depth anchor injection.

Propagation-owned tensors and modules are excluded from integer W/A allocation and execute in FP16. No propagation input may first pass through W4/W6/W8 QDQ. Propagation coefficient normalization and center residual construction remain mathematically constrained by the official model semantics.

### Ordinary quantization

All non-propagation Conv2d, ConvTranspose2d, and Linear modules use the existing hardware-aligned QDQ implementation. Candidate bits are exactly `{4, 6, 8}`. Weight and activation allocation are independent. Existing branch-aware concat ownership remains active where the model contract declares it; it must not change the propagation FP16 boundary.

### Sensitivity and allocation

For each ordinary module `l` and candidate bit `b`, compute weight sensitivity:

\[
S^W_{l,b}=\sum\left|g^W_l\left(W_l-Q_b(W_l)\right)\right|.
\]

For each ordinary activation site, compute activation sensitivity from the captured task gradient and activation QDQ error:

\[
S^A_{l,b}=\sum\left|g^A_l\left(X_l-Q_b(X_l)\right)\right|.
\]

Start every ordinary site at W4/A4. Promote one site at a time through `4 -> 6 -> 8`, selecting the largest sensitivity reduction per additional weight or activation storage cost while enforcing separate average W and A budgets. The selected assignment is then measured end to end on all 64 paired validation samples. A candidate is invalid if it violates either budget, has non-finite output, is non-reproducible, or violates the model's output and anchor invariants.

### Metrics and artifacts

Every model produces:

- FP32 and unified FP16-propagation baseline rows;
- task-gradient sensitivity tables for weights and activations;
- selected W/A assignment and exact average bit costs;
- per-module ownership and propagation precision manifest;
- pooled RMSE, mean-sample RMSE, MAE, AbsRel, and iRMSE;
- finite/non-positive/anchor validity fields;
- candidate Pareto rows indexed by weight and activation budget.

Historical integer propagation results remain outside the new comparison table and are labeled as legacy protocol artifacts.

## Implementation Boundaries

- Reuse `spn_quant.propagation` adapters and add one explicit FP16 configuration path shared by all four adapters.
- Extend the existing task sensitivity/allocation modules instead of introducing a second quantizer framework.
- Make ownership validation strict: unknown modules, missing propagation sites, overlapping ownership, missing gradients, and incomplete cost coverage raise immediately.
- Do not add fallback model paths, silent exception recovery, implicit default budgets, or hidden protocol changes.

## Verification

Unit tests must cover FP16 propagation ownership for all four model adapters, exclusion of propagation sites from ordinary allocation, exact W/A budget accounting, deterministic greedy selection, and rejection of incomplete sensitivity data. Integration validation must run the official Python/CUDA environment on the fixed 64-sample identities and compare FP32, uniform W8A8 with FP16 propagation, and selected mixed-precision assignments.
