# Task 3 Report: Generic Allocation Registry and Model-Relative P3/T3 Search

## Status

Completed. Architecture-neutral allocation records, budget audits, dominance
pruning, sensitivity ranking, beam search, local search, and refinement search
now live in `spn_quant/mixed_precision.py`. The CSPN module retains its exact
topology and historical constructors while forwarding generic behavior through
compatibility aliases and wrappers.

The new model P3/T3 runner builds all candidates from
`QuantizationModelContract.blocks`, `prefix_groups`, and `tail_groups`. It
selects only from measured per-sample results and does not contain an accuracy
estimator.

## Implementation

- `spn_quant/mixed_precision.py`
  - Defines strict generic `AllocationRegistry`, model-tagged `BitAssignment`,
    existing allocation/search records, and `P3T3SearchResult` evidence.
  - Builds registries from `QuantizationModelContract` plus a complete
    `CostBasis`; missing or extra weight-MAC and activation-element rows fail.
  - Uses `registry.blocks` for every generic traversal. It contains no CSPN
    block names, topology tables, or protected-layer constants.
  - Preserves the architecture-neutral CSPN allocation algorithms with block
    order supplied by the registry.
- `spn_quant/cspn_task_sensitive_bits.py`
  - Retains `BLOCK_ORDER`, `P3_T3_PROTECTED_BLOCKS`, and exact CSPN ownership
    tables.
  - Keeps every prior public class/function import available.
  - Keeps the historical two-field `AllocationRegistry` constructor and the
    two tuple payloads on `BitAssignment`; CSPN-generated assignments retain an
    empty model tag so existing equality and JSON payloads are unchanged.
  - Implements `build_registry`, `p3_t3_assignment`, and the activation-search
    seed as CSPN compatibility wrappers.
- `scripts/run_nyu_model_p3t3_search.py`
  - Generates uniform W4A4, every single-block W8A8 promotion, every contract
    prefix, every nonempty tail-group combination, and every prefix/tail
    interaction without embedding architecture names.
  - Accepts a measured evaluator and requires exact paired sample coverage.
    Every row supplies squared error, valid pixels, per-sample RMSE, prediction
    finiteness, propagation validity, and reproducibility.
  - Recomputes pooled RMSE as `sqrt(total squared error / total valid pixels)`
    and labels mean per-sample RMSE separately.
  - Defines normalized weight cost as
    `sum(weight_bits * MACs) / (base_weight_bits * sum(MACs))` and normalized
    activation cost analogously with activation elements. The two budgets are
    independent and explicit inputs.
  - Finds the non-dominated stable prefix frontier, selects the smallest
    geometric knee, then selects the minimum-pooled-RMSE stable interaction at
    or below both normalized budgets.
  - Binds the search to `NYUModelRuntime` and
    `build_model_quantization_contract`; the injected hard-deployment evaluator
    receives the runtime, official model, contract, and generic registry.
  - Writes `p3_t3_assignment.json` with the selected tuple assignment, all
    candidate metrics, sample RMSEs, paired differences, validity, cost basis,
    formulas, precision values, and budgets.

## TDD Evidence

Initial red phase:

```text
PYTHONPATH=. python -m pytest -q tests/test_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py
Result: 2 collection errors.
Expected failures: spn_quant.mixed_precision and
scripts.run_nyu_model_p3t3_search did not exist.
```

Strict generic registry red phase:

```text
PYTHONPATH=. python -m pytest -q tests/test_mixed_precision.py::test_generic_registry_does_not_infer_missing_contract_identity
Result: 1 failed.
Failure: generic AllocationRegistry inferred omitted block/model identity.
```

Runtime lifecycle red phase:

```text
PYTHONPATH=. python -m pytest -q tests/test_run_nyu_model_p3t3_search.py::test_runtime_search_builds_the_official_contract_and_closes_runtime
Result: 1 failed.
Failure: successful runtime search did not close NYUModelRuntime.
```

Measured-row strictness red phase:

```text
PYTHONPATH=. python -m pytest -q tests/test_run_nyu_model_p3t3_search.py::test_search_rejects_malformed_measured_rows
Result: 2 failed.
Failures: a string propagation-valid flag and RMSE/SSE disagreement were
accepted.
```

Artifact-evidence red phase:

```text
PYTHONPATH=. python -m pytest -q tests/test_run_nyu_model_p3t3_search.py::test_assignment_artifact_persists_measured_evidence_and_tuple_payload
Result: 1 failed.
Failure: cost definition, denominators, budgets, and cost basis were absent.
```

Each red case was followed by the minimal implementation and a focused green
rerun before proceeding.

## Verification

Required Task 3 and CSPN regressions:

```text
PYTHONPATH=. python -m pytest -q tests/test_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py tests/test_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py
Final result: 58 passed, 5 subtests passed in 107.65s.
```

Every test module importing the legacy CSPN allocator:

```text
PYTHONPATH=. python -m pytest -q tests/test_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py tests/test_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_mixed_activation_search.py tests/test_plot_nyu_cspn_task_sensitive_bits.py tests/test_evaluate_nyu_cspn_group_a4_qat.py
Result: 84 passed, 5 subtests passed in 110.13s.
```

The first complete Python 3.11 suite run exposed one new test path that was
relative to process CWD. A prior integration test can leave a different CWD,
so the artifact test failed to create `tests/.model_p3t3_output`. The test now
anchors its repository-local output to `Path(__file__).resolve().parent` and
passes when launched from `/workspace`.

Full split-environment verification:

```text
PYTHONPATH=. python -m pytest -q --deselect=tests/test_official_model_quantization_contracts.py::test_official_nlspn_builder_strictly_loads_checkpoint_and_contract --deselect=tests/test_official_model_quantization_contracts.py::test_official_completionformer_builder_strictly_loads_checkpoint_and_contract
Result: 1325 passed, 2 skipped, 2 deselected, 1 warning, 16 subtests passed in 286.34s.

TORCH_LIB=/opt/conda/envs/completionformer-py37/lib/python3.7/site-packages/torch/lib LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" SPN_EXTERNAL_ROOT=/workspace/external_depth_completion_models COMPLETIONFORMER_ROOT=/workspace/CompletionFormer PYTHONPATH=. /opt/conda/envs/completionformer-py37/bin/python -m pytest -q tests/test_official_model_quantization_contracts.py::test_official_nlspn_builder_strictly_loads_checkpoint_and_contract tests/test_official_model_quantization_contracts.py::test_official_completionformer_builder_strictly_loads_checkpoint_and_contract
Result: 2 passed in 10.49s.
```

The warning is the existing PyTorch `meshgrid` deprecation warning in
`test_nlspn_temporal_residual.py`.

Additional checks:

```text
python -m py_compile spn_quant/mixed_precision.py spn_quant/cspn_task_sensitive_bits.py scripts/run_nyu_model_p3t3_search.py
Result: exit 0.

git diff --check
Result: exit 0 with no output.

Public CSPN surface audit
Result: all 27 prior public classes/functions are present; missing=().

Architecture coupling audit
Result: no CSPN/stem/encoder-layer/decoder-layer/initial-depth names in the
generic allocator or model P3/T3 runner.
```

## Self Review

Reviewed the complete task diff against the Task 3 brief and design:

- contract-owned blocks and search groups are the only topology inputs;
- pooled RMSE is derived from measured SSE and valid pixels;
- sample RMSE, paired differences, validity, reproducibility, and propagation
  status are persisted for every candidate;
- normalized W/A costs and budgets are independent and explicit;
- no synthetic accuracy estimator, model fallback, dictionary fallback,
  broad exception handler, or `/tmp` output was introduced;
- all prior CSPN public imports and tuple assignment payloads remain available;
- no subagents were dispatched.

No unresolved code finding remains.

## Commit

Planned commit message: `feat: generalize model-specific P3 T3 search`.

## Concern

The approved normalized P3/T3 W/A budget values are explicit arguments to the
search, but Task 2's shared JSON schema does not yet contain those two fields.
The later selected-method launcher must supply approved values explicitly; it
must not infer defaults. Formal CUDA search execution also depends on the hard
deployment evaluator supplied by the selected PTQ runner in Task 4.

## Fix Round 1/5 - Executable Runner and JSON-Safe Invalid Evidence

This section supersedes the preceding concern about a future Task 4 evaluator.
Task 3 now owns a complete executable P3/T3 search path and has no Task 4
runtime dependency.

### Review Findings Reproduced

The new tests were written before implementation. The first repository-aware
red run was:

```text
PYTHONPATH=. pytest -q tests/test_run_nyu_model_p3t3_search.py
Result: 3 failed, 8 passed in 8.91s.
```

The failures reproduced both findings and the direct-execution symptom:

- a retained non-finite single-block candidate raised `ValueError: Out of
  range float values are not JSON compliant: inf` in the strict writer;
- `RunnerDependencies` and `run_cli` did not exist, so no selected-config to
  artifact path was available;
- direct `python scripts/run_nyu_model_p3t3_search.py` returned 0 with no work.

An initial invocation through the installed `pytest` entry point failed during
collection because the worktree root was absent from `sys.path`. Rerunning with
the repository's established `PYTHONPATH=.` environment produced the intended
three red behavior failures above; no code change was made for that environment
condition.

### Production Implementation

`scripts/run_nyu_model_p3t3_search.py` now provides a strict argparse entry
point. Every direct run must explicitly provide:

- selected experiment config, model name, and configured CUDA device;
- independent normalized weight and activation budgets;
- `module,macs` weight cost rows and `site,role,elements` activation cost rows;
- an existing output directory;
- an explicit Conv-BN folding choice and fold error threshold;
- explicit CompletionFormer joint calibration factors and limits.

There are no parser defaults for required experiment inputs. The runner loads
the exact selected model entry, rejects a device mismatch, loads the model's
`p3_t3_mixed_ptq` precision attributes, builds `NYUModelRuntime`, builds the
official `QuantizationModelContract`, validates exact cost coverage through the
generic registry, executes the measured search, and exclusively creates
`p3_t3_assignment.json` in the requested directory.

`HardDeploymentP3T3Evaluator` is the production evaluator. It:

- loads the configured train and validation datasets through
  `NYUModelRuntime`;
- requires the persisted 128-sample `32_tail_96_kmedoids` calibration metadata
  and exact fixed evaluation identities;
- prepares the official model through `prepare_hardware_model`;
- calibrates `HardwareAlignedInstrumentor` and the established propagation
  adapter on the persisted training identities;
- admits weighted modules only through `contract.weight_modules`, protects
  propagation projection outputs, and translates contract activation sites to
  exact hardware QDQ boundaries;
- applies every candidate's explicit weight and activation tuple assignment as
  RTN bit overrides;
- uses `CompletionFormerJointAdapter` plus hardware-aligned symmetric QDQ for
  contract-owned Q/K/V and independent concat-branch sites;
- configures the existing fixed-point propagation API and validates state,
  coefficient-sum, contraction, and sparse-anchor invariants;
- evaluates every configured validation identity twice, requiring exact paired
  predictions for reproducibility;
- returns only measured SSE, valid-pixel counts, RMSE, prediction-finite,
  propagation-valid, and reproducibility evidence to the generic search.

The evaluator has no synthetic accuracy estimator and no placeholder or empty
success path. Dependency injection is restricted to the runtime, contract, and
evaluator construction boundaries used by the controlled end-to-end test;
production dependencies instantiate `NYUModelRuntime`,
`build_model_quantization_contract`, and `HardDeploymentP3T3Evaluator`.

CompletionFormer's configured interpreter is Python 3.7. The production joint
site path exposed two existing uses of `str.removesuffix`, which is unavailable
there. Both were replaced by equivalent suffix slicing in
`spn_quant/adapters/completionformer_joint.py`; the full joint-adapter tests and
a Python 3.7 runner import check pass.

### Non-Finite Evidence Contract

Invalid candidates remain in `P3T3SearchResult`, but publication now converts
every non-finite pooled, mean-sample, per-sample, and paired-difference metric
to JSON `null` and includes `metrics_finite: false`. The writer retains
`allow_nan=False`, so `NaN`, `Infinity`, and `-Infinity` cannot enter the
artifact. The reviewer reproduction now writes successfully while a different
stable finite interaction remains selected.

The uniform baseline must itself be stable and finite because all paired
differences depend on it. The writer independently requires the selected
candidate's aggregate, per-sample, paired, and cost metrics to be finite.

### Test Evidence

Focused runner and CompletionFormer adapter tests:

```text
PYTHONPATH=. pytest -q tests/test_run_nyu_model_p3t3_search.py tests/test_completionformer_joint_adapter.py
Result: 25 passed in 9.16s.
```

The runner tests include:

- the exact reviewer non-finite candidate reproduction and strict JSON token
  checks;
- non-finite baseline rejection;
- both established propagation anchor signal names;
- full selected-config/cost/search/write orchestration through controlled
  dependency factories with all CLI arguments explicit;
- direct script execution with missing arguments returning argparse status 2.

Runtime/config/RTN integration regressions:

```text
PYTHONPATH=. pytest -q tests/test_run_nyu_model_p3t3_search.py tests/test_completionformer_joint_adapter.py tests/test_experiment_config.py tests/test_nyu_model_runtime.py tests/test_run_nyu_rtn_quantization.py
Result: 78 passed, 2 skipped in 12.28s.
```

Final required generic and CSPN regression command on the final tree:

```text
PYTHONPATH=. python -m pytest -q tests/test_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py tests/test_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py
Result: 64 passed, 5 subtests passed in 113.18s.
```

Python 3.7 executable-path compatibility:

```text
TORCH_LIB=/opt/conda/envs/completionformer-py37/lib/python3.7/site-packages/torch/lib LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONPATH=. /opt/conda/envs/completionformer-py37/bin/python -c 'from scripts import run_nyu_model_p3t3_search as runner; runner.build_parser()'
Result: exit 0.
```

Static checks:

```text
python -m py_compile scripts/run_nyu_model_p3t3_search.py
git diff --check
Result: both exit 0.
```

Strict style audit found no dictionary `.get`, exception handler, `/tmp`, CSPN
architecture name, fallback, or synthetic estimator in the generic runner.
The CSPN compatibility modules were not changed in this fix round, and the
required CSPN regressions confirm their public imports and tuple payloads remain
intact. No subagents were dispatched.

### Self Review and Remaining Concern

The complete fix diff was reviewed for model topology leakage, exact ownership,
calibration/evaluation identity enforcement, paired metric validity, independent
W/A cost formulas and budgets, JSON compliance, direct execution behavior,
runtime cleanup, and Python 3.7 syntax/runtime compatibility. Two self-review
findings were corrected before final verification: protected weighted children
are now excluded by exact contract membership, and NLSPN/CompletionFormer's
`anchor_injection` invariant is accepted alongside DySPN's `anchor` invariant.

No formal 128-calibration by 64-evaluation CUDA search was launched during this
fix round because the selected config's three calibration metadata files are
not currently present under the configured output root. The runner fails
explicitly when those required inputs are absent; unit and integration evidence
uses controlled runtime/evaluator factories only at declared dependency
boundaries.
