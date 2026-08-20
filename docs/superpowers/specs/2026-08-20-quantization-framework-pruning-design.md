# Quantization Framework Pruning Design

## Objective

Reduce the active quantization surface and generated artifact footprint to the
methods that remain useful for deployment or continued research. Remove the
implementation, tests, command-line options, and large generated outputs for
methods whose unified NYU evaluations show no useful accuracy retention.

The cleanup must preserve reproducibility at the conclusion level. Final
metrics and the reason for retiring each method will be consolidated into a
tracked inventory document before ignored experiment outputs are deleted.

## Starting Point

The cleanup branch is based on `main` commit `0824819`. Its clean baseline is:

```text
1359 passed, 8 subtests passed, 1 warning
```

The working branch that existed before this cleanup contains unrelated user
changes. Those changes and the `external/DySPN` submodule state are outside the
cleanup scope.

## Framework Inventory

### Retained active frameworks

The following capabilities remain supported and tested:

- uniform RTN with per-output-channel symmetric weights;
- hardware-aligned static QDQ with Conv-BN folding, signed and unsigned
  activation contracts, INT32 bias scaling, and explicit Add/Concat
  requantization;
- semantic and edge-aware quantization for CSPN, DySPN, NLSPN, and
  CompletionFormer;
- propagation-aware SPN quantization with quantize-then-normalize affinity,
  signed INT16 Q13 coefficients, A8 confidence/offset/state, INT32 conceptual
  accumulation, and sparse-anchor restoration;
- per-tensor and contiguous Group8 activation quantization, including the
  dynamic Group8 path used by QAT;
- W8A8, W4A8, and mixed-precision P3/T3 deployment configurations;
- AdaRound and BRECQ weight reconstruction because their W4A8 results remain
  useful;
- QDrop reconstruction, especially the uniformly finite W6A6 result;
- static and dynamic Group8 W4A4 QAT;
- task-sensitive bit allocation and encoder/decoder sensitivity tools;
- CompletionFormer attention Q/K/V, QK/AV, and Concat scale contracts;
- MinMax calibration and the 128-sample stratified calibration selection;
- activation histogram, im2col, channel, spatial-token, and kernel-offset
  diagnostics.

### Retired frameworks

The following methods are removed from active code rather than hidden behind
deprecated or fallback options:

- LogNP and selective LogNP activation companding and compensation;
- random, Hadamard, and selective channel rotation;
- SmoothQuant activation-to-weight migration;
- AWQ-style clipping in the CNN PTQ runner;
- percentile and histogram-MSE activation calibration policies;
- FP4 E2M1 activation quantization and its strict evaluation workflow;
- scale-aware channel permutation;
- RMS/Max clustering, Snake spread, and outlier-channel isolation/splitting
  experiments.

Activation outlier measurement itself is retained. Only the retired mitigation
and grouping strategies are removed.

### Retired configurations in retained frameworks

Some frameworks are retained, but specific configurations are no longer
deployment candidates:

- uniform RTN W4A4;
- AdaRound W4A4;
- BRECQ W4A4;
- QDrop W4A4 as a strict deployment result;
- RTN and BRECQ W6A6 for CSPN.

The QDrop W4A4 implementation remains because it is shared with QDrop W6A6
and is still a useful diagnostic. The failed W4A4 generated contracts and raw
predictions are removed.

## Evidence For Retirement

All RMSE values are metres.

| Method or configuration | Reference | Best relevant result | Decision |
| --- | ---: | ---: | --- |
| SmoothQuant W4-only | 0.204621 | 0.204136 | 0.24% gain is not material |
| SmoothQuant W4A4 Group8 | 0.347124 | 0.359289 | Worse than Group8 baseline |
| SmoothQuant W4A4 Group16 | 0.417303 | 0.407825 | Small gain but 144% above FP32 |
| Rotation Group-W4A4 | 0.426802 | 0.430580 | Best rotation is worse |
| Scale-aware Group8 | 0.313318 | 0.316218 | Worse than contiguous Group8 |
| Percentile P99.9 | 0.313318 | 1.354118 | Severe regression |
| Percentile P99.99 | 0.313318 | 0.490240 | Severe regression |
| Histogram-MSE | 0.313318 | 0.988717 | Severe regression |
| CSPN RTN W4A4 | FP32 0.158092 | strict `inf` | 64/64 invalid |
| CSPN BRECQ W4A4 | FP32 0.158092 | strict `inf` | 64/64 invalid |
| Four-model W4-E2M1 | model FP32 | none accepted | No model met retention criterion |

The all-model strict study also found that no RTN, AdaRound, or BRECQ W4A4
configuration met the preservation criterion. AdaRound and BRECQ are retained
solely because W4A8 remains effective, including accepted results on CSPN,
NLSPN, and CompletionFormer and an accepted BRECQ result on DySPN.

## Code Changes

### Dedicated modules and entry points

Delete dedicated retired modules, runners, plotters, shell entry points, and
their tests. This includes the LogNP, FP4 E2M1, rotation, SmoothQuant,
scale-aware grouping, and outlier isolation files.

### Shared quantization code

Remove retired options from shared modules instead of leaving inert branches:

- `scripts/hardware_aligned_quantization.py` will support uniform activation
  quantization only; LogNP, SmoothQuant, AWQ clipping, and percentile override
  state will be removed;
- `scripts/run_nyu_rtn_quantization.py` will stop advertising or constructing
  `lognp` and `outlier` backends;
- `spn_quant/specs.py` will remove `lognp` and `smooth` transforms and retired
  observer choices;
- non-rotating boundary ownership used by retained activation-resolution and
  mixed-precision workflows will move to a neutral boundary representation
  before `spn_quant/rotation.py` is deleted;
- retained Group8 and QAT code must not import scale-aware permutation helpers;
- strict AdaRound/BRECQ reconstruction and strict W4A8 evaluation remain
  operational after FP4-specific orchestration is removed.

No compatibility aliases, deprecated CLI choices, silent fallback, or
exception-based downgrade will be introduced. A removed configuration must
fail naturally as an unknown option.

### Documentation

Create `docs/2026-08-20-quantization-framework-inventory.md` as the active
source of truth. It will list supported methods, retired methods, final metrics,
artifact locations retained after cleanup, and rerun commands for active
baselines.

Remove method-specific implementation plans and result pages for retired
frameworks after their conclusions are represented in the inventory. Keep
cross-method result reports where they are still the evidence for retained
AdaRound, BRECQ, QDrop, W4A8, QAT, or mixed-precision decisions.

## Artifact Cleanup

Generated files under `profile_logs/` are ignored by Git and are cleaned only
after the tracked inventory is committed and its metrics are checked against
the source CSV files.

Delete these retired experiment roots:

- `nyu_cspn_rotation_w4a4`;
- `nyu_cspn_smoothquant_group`;
- `nyu_cspn_smoothquant_group_stratified128`;
- `nyu_cspn_static_calibration_group8`;
- `nyu_cspn_scale_aware_group8`;
- `nyu_cspn_outlier_grouping_64`;
- `nyu_cspn_outlier_channel_isolation_64`;
- `nyu_cspn_selective_w4a8_boundary_search_64.incomplete`;
- `nyu_strict_w4a4_fp4_evaluation`.

Within `nyu_cspn_unified_brecq_qdrop_w4a4_w6a6_64`, retain aggregate CSVs and
figures but remove W4A4 reconstruction contracts and raw prediction trees.
Remove CSPN BRECQ W6A6 reconstruction and raw prediction outputs while keeping
the QDrop W6A6 contracts and evaluation, which are the active W6A6 result.

The older `nyu_propagation_aware_quantization` root may be deleted only after a
manifest-level comparison proves that
`nyu_propagation_aware_quantization_unified` supersedes it. If that proof does
not hold, both roots remain.

Active QAT checkpoints, P3/T3 assignments, stratified calibration data,
W4A8 reconstruction contracts, CompletionFormer joint quantization results,
activation histograms, and im2col diagnostics remain.

Expected reclaimed storage before optional duplicate removal is approximately
4.5 GB.

## Validation

The cleanup is complete only when all of the following hold:

1. Repository search finds no active import, CLI option, or configuration for
   LogNP, E2M1, rotation, SmoothQuant, AWQ clipping, percentile/histogram
   calibration, scale-aware grouping, Snake spread, or outlier isolation.
2. Focused retained-framework tests cover RTN, hardware-aligned QDQ,
   propagation-aware quantization, Group8, QAT, QDrop, AdaRound/BRECQ strict
   reconstruction, W4A8, mixed precision, and CompletionFormer attention.
3. The full test suite passes from the same clean environment used for the
   baseline.
4. The active inventory metrics match the source CSVs before deletion.
5. Deleted artifact paths no longer exist, retained paths pass their existing
   manifest/hash audits, and before/after disk usage is recorded.
6. `git status` contains only intentional cleanup changes and no modifications
   to submodules or unrelated work.

## Delivery

Implement on `cleanup/quantization-framework-pruning`, review the diff and test
evidence, then merge into `main`. Artifact deletion is applied to the shared
`/workspace/SPN_Quantization/profile_logs` directory only after code validation
passes.
