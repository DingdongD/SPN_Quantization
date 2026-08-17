# CSPN Selective W4A8 Boundary Search Design

## Objective

Find the lowest-cost selective W4A4/W4A8 activation policy for the official
CSPN ResNet-18 model while keeping aggregate NYU RMSE at or below 0.1773 m.
The search starts from the measured mixed-precision result that encoder and
decoder activation precision, rather than uniformly higher weight precision,
dominates W4A4 recovery.

All ordinary Conv weights remain Group-8 W4. The only weight exception is
`gud_up_proj_layer5.conv1`, the initial-depth head, which remains W8 because
the prior fixed-sample experiment measured W8A4 and W8A8 at the same RMSE and
showed that its input activation was insensitive. This fixed exception costs
approximately 0.0012% normalized added bit-elements.

## Alternatives

Three search strategies were considered:

1. Hierarchical unit factorial followed by calibration-ranked boundary
   elimination. This captures non-additive unit interactions and then removes
   unnecessary A8 edges without using validation results to choose the
   elimination order. This is the selected approach.
2. Unit-only factorial search. This is rigorous and inexpensive but leaves all
   boundaries inside a selected unit at A8, so it cannot find the minimum A8
   set.
3. Validation-driven boundary greedy search. This needs fewer calibration
   diagnostics but is order-dependent and directly overfits the fixed
   64-sample evaluation set.

## Immutable Quantization Contract

- Architecture: official CSPN ResNet-18 with 24 propagation iterations.
- Checkpoint: `output/nyu_converged_baselines/cspn_iter24/best.pt`.
- Calibration: persisted stratified 128-sample NYU train set.
- Evaluation: persisted fixed 64-sample NYU validation set, seed `20260812`.
- Ordinary weights: static symmetric per-output-channel W4.
- Initial-depth weight: static symmetric per-output-channel W8.
- Activations: static contiguous Group-8 MinMax A4 or A8 at explicitly
  registered owners.
- Bias: FP32 with the existing folded-Conv execution contract.
- Guidance: FP32.
- Propagation: A8 affinity/confidence/state, INT16 Q13 coefficients, and INT32
  accumulation.
- Disabled methods: QAT, AdaRound, BRECQ, QDrop, SmoothQuant, rotation,
  clipping search, percentile calibration, dynamic ranges, and FP4.

Every measured candidate starts from a fresh official checkpoint load and
observes the same calibration identities. Candidate state, observer values,
hooks, and quantized tensors are never reused across configurations.

## Precision Ownership

Stage 1 has four activation units:

1. `stem`: merged RGBD input owned by `CSPNStemController`, root ReLU output,
   and retained skip4 boundary;
2. `encoder_layer1`: every registered layer1 Conv input, ReLU output, residual
   boundary, and output owner;
3. `encoder_layer2`: every registered layer2 Conv input, ReLU output, residual
   boundary, and output owner;
4. `decoder_layer4`: every registered official `gud_up_proj_layer4` input,
   branch output, ReLU output, skip input, and output owner.

The unit registries reuse the exact official owner identities already
validated by `cspn_encoder_prefix`. A candidate's A8 owner set is the stable
set union of selected units. Shared owners, including skip4, are configured
and charged once. Every owner outside the set is A4. The initial-depth input
remains A4 in the primary search.

The stem controller gains one explicit `STEM_W4A8` contract: W4 stem weight
and A8 merged RGBD input. Root ReLU and skip4 remain independently owned by
the existing generic and rotation-boundary controllers. No configuration is
inferred from a name or module prefix.

## Stage 1: Unit Factorial

Evaluate the complete `2^4 = 16` Cartesian product of A4/A8 unit states. Names
encode a four-bit mask only for stable identity; the manifest stores the
descriptive selected-unit and owner lists used for interpretation.

The runner also evaluates two explicit context configurations:

- strict W4A4, including the initial-depth weight at W4;
- the prior P3/T3 W8A8 precision contract.

These context rows are not candidates for the selective-W4A8 winner. The
all-A4 primary candidate differs from strict W4A4 only by the fixed W8
initial-depth weight.

For every primary candidate, record end-to-end, regional, block-output,
propagation, operation, activation-cost, and exact configured-precision rows.
A Stage-1 anchor candidate must satisfy `RMSE <= 0.1773`, finite outputs, no
non-positive valid prediction pixels, and the propagation constraints defined
for the final winner. Select the anchor lexicographically from those candidates:
lowest normalized added bit-element cost, then lowest A8 activation-element
fraction, then lowest RMSE, then canonical candidate name. If no Stage-1
candidate is feasible, the experiment fails and Stage 2 does not run; it does
not substitute the lowest-RMSE candidate.

## Stage 2: Boundary Elimination

Let `A` be the exact A8 owner set of the Stage-2 anchor. For every owner
`o in A`, construct one calibration-only candidate `A - {o}`. Each candidate
uses a fresh model and the same 128 calibration identities. It records:

- final propagation-output MSE against the FP32 calibration reference;
- output MSE and SQNR for the owner block and every downstream block;
- per-iteration propagation MSE, saturation, zeroing, contraction, coefficient
  sum, and anchor diagnostics;
- A8 activation elements and normalized bit-element cost saved by demotion.

Define the primary demotion score as

`score(o) = max(0, E_prop(A - {o}) - E_prop(A)) / saved_cost(o)`,

where `E_prop` is total final propagation-output squared error divided by its
element count, and `saved_cost` is the positive normalized bit-element cost
removed by demoting `o`. Lower score is less harmful per unit cost saved.
Ties are resolved by lower downstream block-MSE increase, larger saved cost,
then canonical owner order. SQNR and per-iteration metrics remain diagnostics
and do not enter an undocumented composite score.

Sort owners once by this calibration-only rule. Construct a cumulative path by
demoting one additional owner at each step. The order is frozen before any
Stage-2 validation inference. Evaluate every cumulative path point on the
fixed 64 validation samples so that the minimum feasible A8 set is not skipped.
Validation results never change the path order.

## Winner and Safety Constraints

A Stage-2 candidate is feasible only when all conditions hold:

- aggregate RMSE is at most 0.1773 m;
- all predictions and metrics are finite;
- no valid prediction pixel is non-positive;
- propagation coefficient-sum maximum error is zero;
- contraction violation rate is zero;
- anchor maximum error is zero;
- prediction rerun reproduces the first-pass aggregate RMSE exactly.

Choose the winner lexicographically from feasible Stage-1 and Stage-2
candidates: lowest normalized added bit-element cost, lowest A8
activation-element fraction, lowest RMSE, then canonical name. W8 Conv MAC
fraction and A8 element fraction are logical coverage metrics, not measured
latency or energy.

## Outputs

Write a new immutable result root at
`profile_logs/nyu_cspn_selective_w4a8_boundary_search_64` containing:

- Stage-1 aggregate, sample, regional, block, propagation, operation, and
  precision-coverage CSVs;
- Stage-2 single-demotion calibration metrics and frozen ranking CSV;
- cumulative path aggregate, sample, block, propagation, and cost CSVs;
- feasible set and normalized-cost/A8-fraction Pareto CSVs;
- prediction payloads for strict W4A4, fixed-head all-A4, Stage-1 anchor,
  final winner, and P3/T3 context without duplicate reruns;
- RMSE-by-unit-state matrix, boundary sensitivity, normalized-cost Pareto, and
  A8-fraction Pareto PNG/PDF plots generated only from persisted CSVs;
- a manifest containing exact owner contracts, stage transition, frozen
  ranking, checkpoint/source/protocol hashes, sample identities, fold metadata,
  prediction selection, and every artifact hash.

Plots use Arial, omit titles, keep tick labels horizontal, and place grids
behind data. Plot generation cannot recompute model metrics.

## Failure Contract

The runner raises on an existing output root, checkpoint or source mismatch,
changed fold manifest, split-aware calibration/evaluation identity overlap,
incomplete unit or owner registry, duplicate ownership, missing or extra
W8/A8 sites, invalid stem contract, non-positive saved cost, incomplete
16-candidate coverage, operation-shape drift, incomplete 64-sample coverage,
non-finite values, propagation invariant failure, prediction-rerun mismatch,
or artifact hash mismatch. It does not skip candidates, infer missing fields,
replace the search anchor, or fall back to another precision policy.

## Testing and Verification

Tests are written before implementation and cover:

- complete and deterministic 16-mask generation;
- fixed initial-depth W8 ownership and all-other-weight W4 validation;
- exact activation-owner unions and shared-owner cost deduplication;
- explicit `STEM_W4A8` integer and contract behavior;
- Stage-1 anchor threshold and lexicographic selection;
- single-demotion construction, score calculation, deterministic tie breaks,
  and cumulative monotonic cost reduction;
- proof that validation metrics cannot alter the frozen ranking;
- feasibility constraints, winner selection, Pareto logic, output schemas,
  immutable output handling, and CLI use outside the repository root.

Focused tests and the complete repository suite run before CUDA evaluation.
After evaluation, an independent audit verifies candidate and sample counts,
configured precision, finite predictions, propagation invariants, Pareto
non-domination, prediction coverage, and every manifest hash. All plots are
visually inspected for clipping, overlap, font, grid order, and tick rotation.
