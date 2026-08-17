# CSPN Task-Sensitive Mixed-Bit Allocation Design

## Objective

Find the lowest-NYU-RMSE layerwise weight and activation precision assignment
for the official CSPN ResNet-18 model under separate W4-equivalent and
A4-equivalent average-bit budgets. Searchable precisions are 2, 4, 6, and 8
bits. The search is post-training quantization: it does not train model
parameters, optimize rounding, or change the official architecture.

The result must answer whether task-sensitive redistribution can outperform
uniform W4A4 by spending W6/W8 or A6/A8 on sensitive paths and financing that
precision with W2 or A2 on insensitive paths.

## Fixed Model and Data Contract

- Architecture: official CSPN ResNet-18 with 24 propagation iterations.
- Checkpoint: `output/nyu_converged_baselines/cspn_iter24/best.pt`.
- Calibration: persisted stratified 128-sample NYU train subset.
- Evaluation: persisted fixed 64-sample NYU validation subset, seed
  `20260812`.
- Candidate selection uses only the 128 calibration samples. Validation
  results never change a bit assignment, search order, or tie break.
- Every measured candidate starts from a fresh checkpoint load and receives
  an independent static calibration pass over the same sample identities.
- BatchNorm folding, dataset preprocessing, sparse-depth construction, and
  depth metrics remain identical to the existing CSPN evaluation runners.

## Quantization Contract

Searchable weight and activation bits are

`B = {2, 4, 6, 8}`.

The existing quantization semantics remain fixed while the bit count changes:

- Conv weights use static symmetric per-output-channel quantization and the
  existing rounding rule.
- Signed activations use static symmetric contiguous Group-8 MinMax.
- ReLU outputs and other proven nonnegative activation owners use static
  unsigned contiguous Group-8 MinMax.
- Decoder skip and upsample branches retain separate activation owners and
  independent scales.
- Bias remains FP32 under the existing folded-Conv execution contract.
- No percentile calibration, clipping search, dynamic activation range,
  SmoothQuant, rotation, AdaRound, BRECQ, QDrop, QAT, or FP4 is enabled.

The existing signed-symmetric formula uses `qmax = 2^(bits - 1) - 1` and
`qmin = -qmax`. Signed INT2 therefore has the explicit three-level integer code
`{-1, 0, 1}` stored in two bits. The 2-bit unsigned representation uses
`{0, 1, 2, 3}`. Neither convention changes during the search.

Guidance, affinity, confidence, anchor consistency, and propagation state are
outside ordinary-layer allocation. They retain the propagation-aware contract:
FP32 guidance, A8 affinity/confidence/state, INT16 Q13 coefficients, and INT32
accumulation. No invalid ordinary candidate is repaired by raising these bits
or falling back to FP32.

## Search Ownership

The first search level contains ten blocks:

1. stem;
2. encoder layer1;
3. encoder layer2;
4. encoder layer3;
5. encoder layer4;
6. decoder layer1;
7. decoder layer2;
8. decoder layer3;
9. decoder layer4;
10. initial-depth head.

Each block jointly selects one pair `(weight_bits, activation_bits)` from
`B x B`. All registered Conv weights and activation owners in the block use
that pair. Shared activation owners are configured and charged exactly once.
Blocks selected for refinement are later split into their exact Conv modules
and activation owners without changing ownership semantics.

## Independent Average-Bit Budgets

Weight cost is Conv-MAC weighted over searchable ordinary Conv modules. For
Conv module `l`, let `M_l` be its measured MAC count and `b_w,l` its assigned
weight precision:

`average_weight_bits = sum_l(M_l * b_w,l) / sum_l(M_l)`.

Activation cost is element weighted over searchable ordinary activation
owners. For registered activation owner `s`, let `N_s` be its measured
calibration element count and `b_a,s` its assigned precision:

`average_activation_bits = sum_s(N_s * b_a,s) / sum_s(N_s)`.

Every Stage-2-or-later retained and evaluated joint allocation must satisfy
both

`average_weight_bits <= 4`

and

`average_activation_bits <= 4`.

The constraints use exact integer MAC and element totals before division.
Weight and activation costs are not merged into one scalar. Because the
objective is minimum RMSE, unused budget is legal; no candidate is forced to
spend bits that do not improve calibration RMSE.

## Stage 1: Single-Block Task Sensitivity

Uniform W4A4 is measured once. For each of the ten blocks, evaluate the other
15 pairs in `B x B` while every other searchable block remains W4A4. This
produces 151 measured configurations rather than repeating the same baseline
ten times.

These single-block probes measure task sensitivity and may individually exceed
an average-bit budget when the probed block uses W6/W8 or A6/A8. They are not
deployable allocation candidates. Their exact excess or slack is recorded and
only Stage-2-or-later joint assignments are required to satisfy both budgets.

For block `l` and pair `(w, a)`, primary sensitivity is

`S_l,w,a = calibration_RMSE_l,w,a - calibration_RMSE_W4A4`.

The primary quantity is aggregate end-to-end depth RMSE. The following remain
explicit diagnostics and deterministic tie breaks, in order:

1. boundary-region RMSE;
2. final propagation-output MSE against the FP32 reference;
3. lower exact weight cost;
4. lower exact activation cost;
5. canonical block and bit tuple.

No undocumented weighted combination of these metrics is used. Each
configuration also records MAE, AbsRel, iRMSE, flat-region RMSE, block-output
MSE/SQNR, per-iteration propagation error, zero and saturation ratios, and
configured precision coverage.

## Stage 2: Budget-Constrained Joint Beam Search

The single-block table initializes an additive task-loss estimate, but it is
only a candidate generator. Interactions are resolved by real end-to-end
calibration inference.

The deterministic Beam has width 512. It expands blocks in canonical model
order and retains assignments according to:

1. estimated calibration RMSE;
2. estimated propagation-output MSE;
3. weight-budget slack;
4. activation-budget slack;
5. canonical bit tuple.

Partial states are retained only when their minimum possible remaining cost
can satisfy both final budgets. Promotion to W6/W8 or A6/A8 is therefore
financed by W2 or A2 assignments in the same complete candidate; no evaluated
candidate exceeds either budget.

After all ten blocks are assigned, exact duplicate assignments are removed and
dominated states are pruned. The first 128 budget-feasible assignments are
measured with fresh model loads and full 128-sample calibration inference.
Their measured RMSE, rather than the additive estimate, determines the current
best assignment.

## Stage 3: Budget-Preserving Local Search

Starting from the best measured block assignment, generate all canonical
paired moves that change one source and one destination precision by one
two-bit step. A move may alter weight bits, activation bits, or both, but the
result must satisfy both exact budgets. Duplicate assignments are removed
before inference.

Measure all valid neighbors on the calibration set. Replace the current
assignment only when calibration RMSE strictly decreases; resolve equal RMSE
using the Stage-1 tie-break order. Run at most three local-search rounds and
stop after the first round without improvement. Search order cannot depend on
validation metrics.

## Stage 4: Layer and Activation-Owner Refinement

Rank blocks by the measured RMSE increase caused by the least expensive valid
two-bit demotion from the current assignment. Select the four most sensitive
blocks, resolving ties in canonical block order.

Split those blocks into their registered Conv weight modules and activation
owners. Keep all other blocks fixed. A deterministic width-128 Beam applies
the same separate budgets and task-sensitivity ordering to module/owner bit
assignments. Shared skip owners remain unique. The first 128 refined
assignments are measured on calibration data, and the minimum-calibration-RMSE
assignment becomes the final fixed policy.

## Final Evaluation and Comparisons

The final allocation is frozen before validation inference. Evaluate it once
on the fixed 64 validation samples and compare it against:

- FP32;
- uniform W4A4 under the same quantization contract;
- the measured P3/T3 mixed W8A8 context;
- the best all-ordinary-W4 selective-W4A8 candidate `ACT_MASK_15`.

Report RMSE, MAE, AbsRel, iRMSE, flat-region RMSE, boundary-region RMSE,
nonfinite ratio, nonpositive valid-depth ratio, exact average weight bits,
exact average activation bits, W2/W4/W6/W8 MAC fractions, and
A2/A4/A6/A8 element fractions.

Generate prediction payloads for all 64 validation samples for FP32, uniform
W4A4, P3/T3, and the final allocation. Plot generation is downstream of CSV
and payload persistence; plots cannot recompute or select metrics.

## Validity and Failure Rules

A candidate is invalid when any of the following occurs:

- a Stage-2-or-later joint allocation exceeds either exact average-bit budget;
- configured bits differ from the assignment;
- a required weight module or activation owner is missing or duplicated;
- metrics or predictions contain NaN or Inf;
- any valid prediction pixel is nonpositive;
- propagation coefficient-sum error is nonzero;
- contraction violation rate is nonzero;
- anchor consistency error is nonzero;
- operation shapes, checkpoint load, fold manifest, or sample identities
  change between candidates.

Invalid candidates are recorded with their explicit failure reason and are
excluded from ranking. They are not retried with modified bits, skipped in
coverage counts, or replaced by a fallback policy. Existing output roots are
never overwritten. Results are first written to a sibling `.incomplete`
directory and published atomically only after the complete audit passes.

## Outputs

The immutable result root is

`profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64`.

It contains:

- single-block sample, aggregate, regional, block, propagation, operation,
  and precision-coverage CSVs;
- exact weight-MAC and activation-element cost bases;
- Beam expansion, pruning, predicted-score, and measured-candidate tables;
- local-search rounds and accepted-move table;
- module/owner refinement candidates and final assignment table;
- calibration and validation metrics;
- per-bit MAC and activation-element fractions;
- prediction payloads for the four comparison configurations;
- an artifact manifest with checkpoint, source, protocol, sample-identity,
  assignment, and output hashes.

The required summary figures are bit allocation by block, calibration RMSE
versus both independent budgets, per-bit MAC/activation fractions, and the
64-sample prediction comparison. Figure labels use Arial, remain horizontal,
omit titles, and place grids behind plotted data.

## Testing and Verification

Tests are written before implementation and cover:

- exact `{2, 4, 6, 8}` configuration of weight and activation sites;
- signed and unsigned 2/4/6/8-bit QDQ contracts;
- exact independent budget arithmetic using integer cost totals;
- 151 deterministic single-block configurations;
- shared-owner deduplication;
- Beam determinism, feasibility lower bounds, width limits, dominance pruning,
  and canonical tie breaks;
- proof that validation data cannot influence search;
- budget-preserving local moves and three-round termination;
- deterministic top-four block refinement;
- invalid-candidate recording without fallback;
- output schemas, immutable publication, artifact hashing, and CLI use outside
  the repository root.

Before CUDA evaluation, focused tests and the complete repository suite pass.
After evaluation, an independent audit verifies every candidate's configured
bits, exact budgets, sample counts, finite metrics, propagation invariants,
assignment uniqueness, prediction coverage, and artifact hashes. Figures are
visually inspected for clipping, overlap, font, grid order, and tick rotation.
