# CSPN Encoder-Prefix and Sensitive-Tail W8A8 Design

## Objective

Measure how much of the official CSPN Group-8 W4A4 degradation can be
recovered by retaining an increasing encoder prefix in Group-8 W8A8, and
determine how that prefix interacts with the previously identified decoder-4
and initial-depth sensitivities. The primary search uses only binary W4A4 and
W8A8 unit precision. Earlier W4A8 and W8A4 measurements remain diagnostic
context and are not candidates in this search.

## Prior Evidence

The fixed 64-sample decoder search measured strict W4A4 RMSE at 0.342178 m.
Decoder-4 W4A8 and W8A8 reached 0.286173 m and 0.284687 m, respectively, so
decoder-4 sensitivity is primarily activation-driven. Initial-depth W8A4 and
W8A8 both reached 0.335302 m, so its measured benefit is weight-driven. The
independent stem experiment found that stem W8A8 improved local SQNR but did
not improve end-to-end RMSE when every downstream ordinary unit remained
W4A4. The joint experiment tests whether downstream W8A8 retention preserves
encoder recovery that was previously requantized away.

## Alternatives

Three search strategies were considered:

1. Full prefix-by-tail factorial search. Evaluate every encoder prefix with
   every sensitive-tail state. This directly measures interactions and is the
   selected design.
2. Staged search. Evaluate encoder prefixes first, then combine only the best
   prefixes with the tail states. This is cheaper but can miss a prefix whose
   value appears only after decoder recovery.
3. Greedy unit promotion. Add whichever unit gives the largest immediate RMSE
   gain. This uses fewer runs but is order-dependent and does not answer the
   requested strict-prefix question.

The complete factorial contains only 24 configurations and is small enough to
run without sacrificing interaction coverage.

## Immutable Evaluation Contract

- Use the official CSPN ResNet-18 architecture with 24 propagation steps.
- Load `cspn_iter24/best.pt` independently for every configuration.
- Use the persisted stratified 128-sample NYU train calibration identities and
  fixed 64-sample NYU validation identities with seed `20260812`.
- Use static contiguous Group-8 MinMax quantization for ordinary CNN weights
  and activations. Only the selected bit width changes.
- Keep guidance in FP32 and propagation at A8/INT16-Q13/INT32.
- Do not train or enable QAT, AdaRound, BRECQ, QDrop, SmoothQuant, rotation,
  clipping search, dynamic ranges, or evaluation-driven calibration.
- Keep the official forward path, tensor shapes, propagation iteration count,
  checkpoint tensors, and calibration/evaluation identities unchanged.

## Precision Units

The encoder prefix contains five ordered units:

1. `stem`: official `conv1_1` plus its stem input and output boundaries;
2. `encoder_layer1`: official `layer1` BasicBlocks;
3. `encoder_layer2`: official `layer2` BasicBlocks;
4. `encoder_layer3`: official `layer3` BasicBlocks;
5. `encoder_layer4`: official `layer4` BasicBlocks.

The two sensitive tail units are:

- `decoder_layer4`: official `gud_up_proj_layer4`;
- `initial_depth`: official `gud_up_proj_layer5`.

Each non-stem unit contains every executed Conv weight under its exact official
module and every registered activation boundary assigned to that unit. Unit
registries are explicit and are validated against the executed official model
after calibration. Missing, extra, duplicate, or unobserved modules and owners
are errors; prefix matching is not used as an evaluation fallback.

The stem remains exclusively controlled by `CSPNStemController`. In a W8A8
prefix, the controller uses its `STEM_W8A8` contract for the 4-to-64 stem Conv
weight and RGBD input. The root ReLU output and retained `skip4` edge are also
A8. In the strict prefix they use `STRICT_W4A4`. The generic instrumentor must
not own the stem Conv weight or stem input.

`skip4` is a shared boundary: stem W8A8 requires its retained output to be A8,
and decoder-4 W8A8 requires its side input to be A8. The candidate activation
set is a set union, so this edge is configured and charged once when either or
both units require it. Other producer-consumer boundaries are A8 inside a
selected unit and are requantized to A4 when they enter an unselected unit.

## Configuration Matrix

The six cumulative encoder prefixes are:

| Prefix | W8A8 encoder units |
|---|---|
| `P0` | none |
| `P1` | stem |
| `P2` | stem, layer1 |
| `P3` | stem, layer1, layer2 |
| `P4` | stem, layer1, layer2, layer3 |
| `P5` | stem, layer1, layer2, layer3, layer4 |

The four tail states are:

| Tail | W8A8 tail units |
|---|---|
| `T0` | none |
| `T1` | decoder4 |
| `T2` | initial-depth |
| `T3` | decoder4, initial-depth |

Every Cartesian-product pair is evaluated, producing 24 configurations named
`PREFIX_P<index>__TAIL_T<index>`. The manifest stores descriptive prefix and
tail unit lists; result interpretation must not parse semantics from the name.
All ordinary units outside those lists remain W4A4.

## Configuration and Evaluation Flow

For each candidate:

1. load a fresh official checkpoint and reproduce the same Conv-BN fold
   manifest;
2. build fresh generic, rotation-boundary, propagation, and stem controllers;
3. observe the same 128 calibration samples and freeze all ranges;
4. configure the stem contract, exact W8 weight overrides, and exact A8 owner
   promotions;
5. compare the actual configured W8/A8 sets with the candidate contract;
6. evaluate the same 64 validation identities against a shared FP32 reference;
7. release every hook and controller before loading the next candidate.

The evaluation pass records predictions only for configurations selected after
the aggregate and Pareto tables are complete. A prediction rerun must reproduce
the first-pass aggregate RMSE exactly before its payloads are accepted.

## Metrics

For every configuration, record:

- sample and aggregate RMSE, MAE, AbsRel, iRMSE, flat RMSE, and boundary RMSE;
- paired RMSE wins and delta against `P0/T0`;
- per-block output MSE and SQNR for stem, encoder layers, decoder layers,
  initial depth, and propagation;
- initial-depth and per-iteration propagation error diagnostics;
- effective precision and operation counts for every executed Conv;
- W8 weight-MAC fraction, W8 weight-element fraction, A8 activation-element
  fraction, and normalized added bit-element cost.

RMSE is the primary accuracy objective. iRMSE remains a required diagnostic
because a few non-positive or near-zero predictions can expose failures hidden
by aggregate RMSE. Pareto membership does not waive finite-value, minimum-depth,
or inverse-depth diagnostics.

## Cost Accounting

The stem Conv is included in the Conv operation and weight basis even though it
is not owned by the generic instrumentor. Activation accounting includes the
stem RGBD input and every observed generic or rotation boundary. Shared owners
are deduplicated before summation.

For weight elements `W_m` and activation-edge elements `A_e`, the normalized
added bit-element cost is

`C = (sum_m((b_w,m - 4) W_m) + sum_e((b_a,e - 4) A_e)) /
     (4 (sum_m W_m + sum_e A_e))`.

Because every promoted unit is W8A8, the added term is one base-width element
for each promoted weight or activation element. W8 Conv MAC fraction is
reported separately as the operator-oriented hardware axis; it is not claimed
to be measured latency.

## Interaction Analysis

For encoder prefix `P` and tail state `T`, report

`interaction(P,T) = R(P,T) - R(P,T0) - R(P0,T) + R(P0,T0)`.

A negative value indicates that the joint configuration recovers more RMSE
than the two independent changes predict. A positive value indicates
interference, loss of beneficial error cancellation, or another non-additive
regression. The report includes the full interaction matrix rather than using
the interaction value as a configuration-selection shortcut.

## Pareto and Outputs

Write a new immutable result root at
`profile_logs/nyu_cspn_encoder_prefix_joint_w8a8_64` containing:

- aggregate and 64-sample metric CSVs;
- regional, block-error, propagation, operation-count, activation-cost, and
  precision-coverage CSVs;
- the 6-by-4 RMSE and interaction matrices;
- non-dominated tables for normalized cost and W8 Conv MAC fraction;
- an Arial RMSE heatmap, normalized-cost Pareto plot, and W8-MAC Pareto plot;
- prediction payloads for strict W4A4, the cheapest improving point, all
  normalized-cost Pareto points, and the full `P5/T3` anchor;
- a manifest with exact configuration contracts, source/checkpoint/protocol
  hashes, sample identities, fold metadata, and artifact hashes;
- a concise measured-result report.

Plots have no title, place grids behind data, and do not rotate tick labels.
The plotting scripts consume persisted CSVs and never recompute model results.

## Failure Contract

The runner terminates on an existing output directory, checkpoint or source
mismatch, changed fold manifest, changed sample identities, candidate registry
mismatch, duplicate quantization ownership, missing or extra W8/A8 sites,
operation-shape drift, incomplete 64-sample coverage, non-finite predictions or
metrics, prediction-rerun mismatch, or artifact-hash mismatch. It does not skip
configurations, substitute modules, infer missing fields, or fall back to a
different precision contract.

## Verification

Tests are written before implementation and cover exact unit registries,
prefix and Cartesian-product generation, stem/generic ownership separation,
shared-boundary union and cost deduplication, configured precision validation,
interaction values, both Pareto definitions, protocol validation, deterministic
prediction selection, output schemas, and CLI use outside the repository root.
Focused tests and the complete repository suite must pass before CUDA
evaluation. The final audit recomputes row counts, unique identities, finite
metrics, Pareto non-domination, prediction coverage, and every artifact hash.
