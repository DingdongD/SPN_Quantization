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
