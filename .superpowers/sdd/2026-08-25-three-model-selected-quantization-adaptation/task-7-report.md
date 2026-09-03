# Task 7 Report: Unified Formal Evaluation and Aligned Predictions

## Status

Implemented the unified formal evaluator and plotter for the exact ordered
configuration matrix:

1. FP32
2. RTN W8A8
3. RTN W4A4
4. QDrop W6A6
5. BRECQ W6A6
6. HAWQ mixed<=6
7. LSQ++ W6A6
8. LSQ++ W4A4
9. mixed task-aware QAT
10. P3/T3 mixed PTQ

The generic path is model-contract driven and contains no CSPN module names,
bit assignments, dataset wrappers, or output-tuple assumptions. The legacy
CSPN evaluator now delegates sample metric arithmetic to the generic metric
helper while retaining its existing CLI and result schema.

## Artifact Contract

`scripts/evaluate_nyu_selected_quantization.py` consumes one immutable formal
artifact index per model. The index binds:

- the exact ordered ten-method matrix and method artifact kinds;
- the exact ordered 64 validation identities and split-qualified SHA-256;
- the completed Task 4 selected PTQ matrix and its five strict manifests;
- terminal Task 6 HAWQ, LSQ++, and mixed-QAT checkpoints;
- the Task 5 HAWQ assignment and trace linkage;
- one shared Task 3 P3/T3 assignment for P3/T3 PTQ and mixed QAT;
- explicit graph folding and CompletionFormer joint calibration settings;
- byte fingerprints for every primary and supporting artifact.

The evaluator independently derives the train calibration and validation
identity hashes. Missing files, changed bytes, incomplete PTQ matrices,
incorrect method order, missing HAWQ/P3 links, artifact-kind changes, and
model or sample identity mismatches fail before model evaluation.

## Hard Deployment Evaluation

- FP32 loads only the configured official checkpoint.
- RTN W8A8/W4A4 and P3/T3 rebuild the declared calibrated hard-QDQ graph,
  compare graph and activation manifests, compare every rematerialized state
  tensor with the Task 4 hard checkpoint, and strictly reload that checkpoint.
- QDrop and BRECQ load exact activation and weight contracts, replay the fixed
  calibration identities, validate graph folding, compare every rematerialized
  tensor with the Task 4 hard checkpoint, and strictly reload it.
- HAWQ, LSQ++, and mixed-QAT require a terminal completed Task 6 checkpoint.
  Their canonical masters and method state are validated, hard weights and
  frozen qparams are derived, and a fresh official model is materialized.
  The live fake-quant training model is closed without being evaluated.

Every quantized forward validates propagation invariants. Every prediction,
GT map, RGB input, and sparse-depth input must be finite and shape-aligned.
The formal run manifest is written only after all 64 predictions, per-method
metrics, cost rows, and diagnostics have been published.

## Metrics and Costs

Pooled RMSE is computed from global squared-error sum divided by global valid
pixel count. Mean per-sample RMSE is computed separately and labeled as a
diagnostic. The aggregate tables also contain pooled MAE, AbsRel, iRMSE,
non-positive counts, and raw global accumulators.

`relative_fp_loss.csv` records absolute and percentage degradation relative to
FP32 for pooled RMSE, mean per-sample RMSE, pooled MAE, pooled AbsRel, and
pooled iRMSE. `cost_table.csv` uses the validated model-specific P3/T3 cost
basis to report MAC-weighted average weight bits, traffic-weighted average
activation bits, and W4/W6/W8/FP32 shares.

Diagnostics retain prediction finite/non-positive ratios, propagation rows,
per-tensor error/SQNR rows exposed by hard PTQ instrumentors, and weighted
saturation and zero-code ratios when the deployment exposes code statistics.

## Prediction Figures

`scripts/plot_nyu_selected_quantization.py` validates every NPZ field and the
model, method, sample, and evaluation identities. RGB, sparse depth, GT, and
valid masks must be byte-aligned across all methods for a sample.

Each sample panel includes RGB, sparse depth, GT, FP32, and all nine selected
quantized predictions, followed by an absolute-error row. All depth panels in
one sample use one shared depth range; all error panels use one shared error
range. The summary figure labels pooled RMSE as primary and mean per-sample
RMSE as diagnostic.

## TDD Evidence

Initial red phase:

```text
PYTHONPATH=. python -m pytest -q \
  tests/test_evaluate_nyu_selected_quantization.py \
  tests/test_plot_nyu_selected_quantization.py
Result: 2 collection errors.
Expected failure: both new modules were missing.
```

Subsequent focused red phases demonstrated missing global-SSE aggregation,
missing artifact-index loading, acceptance of an incomplete PTQ matrix,
acceptance of a missing HAWQ trace link, missing multi-metric FP32-relative
loss, and missing pooled-RMSE plotting. Each failed for the named behavior
before its focused implementation passed.

Final required Task 7 suite:

```text
PYTHONPATH=. python -m pytest -q \
  tests/test_evaluate_nyu_selected_quantization.py \
  tests/test_plot_nyu_selected_quantization.py \
  tests/test_evaluate_nyu_cspn_lsqplus_hawq.py
Result: 20 passed in 5.18s.
```

## Verification

Task 3-7 artifact, deployment, CompletionFormer, and evaluation integration:

```text
Result: 170 passed in 14.76s.
```

Full default-environment suite, with the two official Python 3.7-only tests
deselected:

```text
Result: 1461 passed, 2 skipped, 2 deselected, 1 warning,
16 subtests passed in 275.50s.
```

The warning is the existing PyTorch `meshgrid` deprecation warning in
`tests/test_nlspn_temporal_residual.py`.

Official Python 3.7 model-contract tests:

```text
Result: 2 passed in 9.08s.
```

Task 7 and legacy CSPN evaluator tests under Python 3.7:

```text
Result: 20 passed in 3.51s.
```

All changed Python sources and tests pass Python 3.11 and Python 3.7
`py_compile`. Both new CLIs import and expose help under Python 3.7.

## Self Review

- exact ordered method coverage includes FP32 and cannot be extended by CLI;
- pooled and sample-mean RMSE cannot be conflated in tables or figures;
- all prediction exports carry and validate model/method/sample identities;
- formal summaries reject missing, incomplete, or changed method runs;
- PTQ evaluation consumes hard Task 4 states and contracts;
- QAT evaluation never forwards the live fake-quant training model;
- P3/T3 and HAWQ provenance is validated through the Task 5-6 loaders;
- CompletionFormer attention/concat owners remain contract-owned;
- generic code contains no CSPN-specific site, precision, or output logic;
- no broad exception handler, fallback lookup, temporary-directory path,
  or subagent dispatch was added.

## Operational Limit

Formal CUDA evaluation and the 64-sample image export were not launched in
Task 7. The configured root
`/workspace/SPN_Quantization/profile_logs/nyu_three_model_selected_quantization`
does not exist in this workspace, so completed Task 4-6 runtime artifacts are
not available to execute. The production evaluator fails closed on that
condition. Task 8 remains responsible for creating the artifact index and
launching each method in its configured official-model environment and CUDA
device.

## Commit

Commit message: `feat: add unified selected quantization evaluation`.

## Fix Round 1/5

### Review Findings Resolved

- Cross-method publication now receives `FormalAggregation` directly. Every
  retained run metric is required to have the exact prediction-derived fields,
  finite values, and values; every retained cost row is recomputed from the
  strict assignment and model-relative cost basis and then required to match
  exactly. The selected summary and cost CSV are written from those
  authoritative rows, not copied manifest metrics or costs.
- The aggregate and plot CLIs now require the formal artifact index. Its exact
  resolved path and SHA-256 are persisted in every formal run, and the SHA-256
  is persisted in every prediction NPZ. Aggregation reloads the index, verifies
  all indexed primary and supporting files, and rejects any run whose index,
  primary artifact, supporting artifacts, diagnostics path, or prediction root
  differs from the indexed formal root.
- Prediction loading now revalidates finite RGB and sparse depth, RGB `[0, 1]`,
  sparse-depth nonnegativity, and the declared 10-meter NYU ceiling. The
  ceiling is aligned with RGB, sparse depth, GT, and masks across methods.
  All Task 7 and legacy CSPN alignment comparisons use strict finite equality;
  no `equal_nan=True` path remains.
- Every method now captures contract-block outputs and semantic initial-depth
  and per-iteration propagation states against a fresh FP32 model on the same
  sample tensors. Rows persist model, method, sample, iteration, owner, owner
  kind, element count, signal/error energy, MSE, and finite SQNR. Summary
  validation requires exact 64-sample coverage, stable block owners, one
  initial-depth row per sample, and contiguous propagation iterations.
- Frozen QAT activation quantizers and exact materialized QAT weight grids now
  report code-zero and saturation counts. CompletionFormer hard joint PTQ
  quantizers report the same statistics. Quantized runs with empty or invalid
  hard-code diagnostics fail publication.
- The absolute pooled-iRMSE degradation field is now
  `pooled_irmse_delta_inverse_m`; only AbsRel retains a dimensionless delta.

### Focused Reproductions

The added tests reproduce and reject:

- non-finite/out-of-range RGB;
- negative or above-contract sparse depth;
- stale prediction artifact-index fingerprints;
- aggregate CLI invocation without `--artifact-index`;
- fabricated finite run metrics and costs;
- stale same-model formal-run index fingerprints;
- empty QAT hard-code diagnostics;
- mismatched semantic propagation iteration counts;
- missing QAT hard-controller code statistics.

Generic block-capture tests cover structured module outputs, and adapter tests
confirm task capture remains active after semantic quantization ownership is
delegated to the hard deployment controller.

### Fix Verification

Focused Task 7, CSPN compatibility, adapter, PTQ, and QAT regressions:

```text
Result: 148 passed in 14.19s.
```

Task 7 and generic QAT/PTQ tests under Python 3.7:

```text
Result: 58 passed in 7.18s.
```

Official Python 3.7 contracts:

```text
NLSPN and CompletionFormer: 2 passed.
DySPN: not runnable in that environment because `einops` is not installed;
DySPN remains covered by the Python 3.11 suite.
```

Full supported Python 3.11 suite, excluding only the two tests that explicitly
assert Python 3.7:

```text
Result: 1476 passed, 2 skipped, 2 deselected, 1 warning,
16 subtests passed in 302.44s.
```

The warning remains the existing PyTorch `meshgrid` deprecation warning.

### Fix Self Review

- final metrics originate only from reloaded aligned prediction payloads;
- final costs originate only from recomputation and exact manifest validation;
- all publication inputs are bound to one immutable artifact-index fingerprint;
- stale equivalent-model paths and unrelated prediction roots are rejected;
- PTQ and QAT diagnostics use hard deployment forwards and paired FP32
  references, never the live fake-quant training forward;
- diagnostics cannot publish with missing samples, owners, iterations, or code
  statistics;
- generic block capture depends only on the model quantization contract and
  contains no CSPN-specific assumptions;
- no broad exception handler, fallback route, `/tmp` path, or subagent dispatch
  was added.

## Fix Round 2/5

### Re-review Findings Resolved

- Final cost publication now consumes an `IndexedCostContracts` value bound to
  the exact formal artifact-index SHA-256. The cost basis is reconstructed by
  the strict P3/T3 assignment loader and checked against the strict HAWQ trace
  cost basis. RTN, QDrop, and BRECQ assignments are reconstructed only after
  validating each indexed strict PTQ manifest and its fingerprinted deployment
  artifacts. P3/T3, HAWQ, LSQ++, and mixed-QAT assignments are independently
  reconstructed from their method contracts and compared exactly with the
  indexed terminal artifacts. Formal-run assignment, basis, and cost fields
  must equal these indexed values before publication.
- Prediction reload now requires the persisted `sparse_depth_max_m` scalar to
  be finite and exactly equal to the fixed `NYU_SPARSE_DEPTH_MAX_M` value of
  `10.0`. Sparse samples are validated against that fixed constant, never a
  payload-defined ceiling.
- Every hard diagnostic source now implements an explicit zero-state
  `counter_snapshot()` contract. There is no fallback to cumulative
  `statistics()`. Formal evaluation snapshots immediately before and after
  each candidate forward and emits only the integer deltas for calls, elements,
  zero codes, and saturated codes. Negative/reset counters, non-advancing
  owners, malformed counts, and owner/source drift fail closed.
- Diagnostics persist one exact ordered owner contract containing
  `source_index`, `owner`, and `owner_kind`. Validation requires every sample
  to contain exactly that ordered coverage and rejects all non-counter rows.
  Exact QDrop sites use their full activation-site identity so same-block Q/K/V
  owners remain distinct. Joint CompletionFormer owners remain active across
  QDrop bind/configure and are included in the zero-state contract.

### Focused Red-Green Reproductions

The added tests first reproduced and then rejected:

- an unbound or stale-index cost contract supplied to summary publication;
- a source exposing only cumulative `statistics()` instead of
  `counter_snapshot()`;
- hardware-aligned and exact-QDrop sources without zero-state snapshots;
- cumulative counter reset, owner drift, and source-index drift;
- non-counter rows hidden alongside valid quantization rows;
- joint QDrop owners disappearing during configure;
- ambiguous exact-QDrop owner labels; and
- a sparse payload that raised its own ceiling to 100 m and stored a 50 m
  sample.

### Fix Verification

Task 7 evaluator, plotter, and CSPN compatibility regressions:

```text
Result: 42 passed in 6.82s.
```

Selected QAT, HAWQ, LSQ++, controller, and CSPN regressions:

```text
Result: 149 passed in 13.32s.
```

Selected PTQ, P3/T3, RTN, hardware-aligned, QDrop, and BRECQ regressions:

```text
Result: 252 passed, 4 subtests passed in 12.49s.
```

All changed Python sources and tests pass Python 3.11 `py_compile`, and
`git diff --check` is clean. A Python 3.7 interpreter is not installed in this
worktree environment, so the prior round's Python 3.7 verification could not
be repeated.

### Fix Self Review

- final bit costs have no formal-run-manifest source of authority;
- the fixed sparse-depth domain cannot be widened by an NPZ payload;
- every published code count is a delta from exactly one sample forward;
- exact ordered source and owner coverage is checked at capture and reload;
- QAT weight counters remain excluded from runtime per-sample activation rows;
- no broad exception handler, execution fallback, `/tmp` path, or subagent
  dispatch was added;
- official CUDA evaluation remains unavailable because the indexed Task 4-6
  runtime artifacts described in the operational limit are absent.
