# SPN Three-Mapper Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible BF16 validation that compares LLMCompass, TCM/AccelForge, and FactorFlow on all array-mappable workloads from CSPN, DySPN, NLSPN, and CompletionFormer while excluding propagation operations completely.

**Architecture:** A common environment-neutral GEMM contract feeds three isolated backend workers. The orchestrator deduplicates identical GEMMs, runs microbenchmarks and four anchor architectures, expands results back to every original dense layer, and reports absolute native metrics plus backend-normalized architecture rankings; it never treats native cost-model numbers as directly interchangeable. CACTI-calibrated SRAM properties and the existing LPDDR4 configuration remain the shared hardware source of truth.

**Tech Stack:** Python 3.11 for the orchestrator, an isolated Python 3.12 environment for AccelForge/TCM, pytest, LLMCompass, FactorFlow, CACTI-calibrated JSON, CSV/JSON artifacts, matplotlib.

---

### Task 1: Dense-only workload contract

**Files:**
- Create: `spn_quant/hardware/mapper_validation.py`
- Test: `tests/test_mapper_validation.py`

- [x] **Step 1: Write the failing dense-workload tests**

```python
def test_dense_instances_exclude_propagation():
    profile = WorkloadProfile(..., layers=(
        LayerWorkload.linear("dense", (1, 8), (1, 4), execution_index=0),
        LayerWorkload.non_mappable("prop", "propagation", (1,), (1,), 7, 1),
    ))
    rows = dense_gemm_instances(profile)
    assert [row.name for row in rows] == ["dense"]


def test_unique_gemms_preserve_all_original_instances():
    rows = (...two equal GEMMs at different execution indices...)
    unique, uses = deduplicate_gemms(rows)
    assert len(unique) == 1
    assert len(uses) == 2
```

- [x] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_mapper_validation.py`

Expected: import failure for `spn_quant.hardware.mapper_validation`.

- [x] **Step 3: Implement immutable common records and dense-only lowering**

Implement `DenseInstance`, `UniqueGemm`, `BackendResult`, `dense_gemm_instances()`, `deduplicate_gemms()`, utilization validation, and model aggregation. Reject any backend result whose MAC count differs from the common GEMM or whose utilization lies outside `(0, 1]`.

- [x] **Step 4: Run the tests and verify GREEN**

Run: `pytest -q tests/test_mapper_validation.py tests/test_hardware_workload.py`

Expected: all tests pass.

### Task 2: FactorFlow backend worker

**Files:**
- Create: `spn_quant/hardware/factorflow_adapter.py`
- Create: `scripts/mapper_backends/run_factorflow.py`
- Test: `tests/test_factorflow_adapter.py`

- [x] **Step 1: Write failing tests for architecture translation and result parsing**

```python
def test_factorflow_architecture_uses_bf16_entries_and_physical_mesh():
    spec = factorflow_architecture(point, assumptions)
    assert spec.array_height == point.array.height
    assert spec.array_width == point.array.width
    assert spec.activation_entries == point.memory.activation_buffer_bytes // 2
    assert spec.weight_entries == point.memory.weight_buffer_bytes // 2


def test_factorflow_result_rejects_wrong_mac_count():
    with pytest.raises(ValueError, match="MAC count"):
        parse_factorflow_result(workload, {"macs": workload.macs - 1, ...})
```

- [x] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_factorflow_adapter.py`

Expected: missing adapter import.

- [x] **Step 3: Implement the adapter and isolated worker**

Construct a FactorFlow hierarchy with separate activation and weight SRAM levels, two spatial fanout levels matching array height/width, BF16 values, FP32 accumulation, the configured DRAM/SRAM bandwidth ceilings, and calibrated access/MAC energies. Run `engine.run_engine()` with deterministic single-process settings and emit JSON containing latency cycles, energy, EDP, spatial utilization, search time, MOPs, and mapping factors.

- [x] **Step 4: Run unit and real upstream smoke tests**

Run: `pytest -q tests/test_factorflow_adapter.py`

Run: `(cd .venv/upstreams/factorflow && /opt/conda/bin/python test.py)`

Expected: adapter tests and FactorFlow upstream tests pass.

### Task 3: TCM/AccelForge backend worker

**Files:**
- Create: `spn_quant/hardware/tcm_adapter.py`
- Create: `scripts/mapper_backends/run_tcm.py`
- Create: `requirements-mapper-validation.txt`
- Test: `tests/test_tcm_adapter.py`

- [x] **Step 1: Write failing tests for AccelForge workload/architecture generation**

```python
def test_tcm_workload_is_one_unfused_bf16_matmul():
    payload = tcm_workload(GemmWorkload("g", "linear", 32, 64, 16))
    assert payload["workload"]["bits_per_value"] == {"All": 16}
    assert len(payload["workload"]["einsums"]) == 1


def test_tcm_architecture_carries_capacity_and_throughput():
    payload = tcm_architecture(point, assumptions)
    assert payload["activation_buffer_bytes"] == point.memory.activation_buffer_bytes
    assert payload["weight_buffer_bytes"] == point.memory.weight_buffer_bytes
```

- [x] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_tcm_adapter.py`

Expected: missing adapter import.

- [x] **Step 3: Implement YAML generation, TCM execution, and result parsing**

Generate one-Einsum AccelForge specs, force all tensors into outer memory to disable fusion, set `max_fused_loops = 1`, use the upstream O3 default to optimize EDP over the perfectly-factorized mapspace, and parse the best mapping's latency, dynamic energy, EDP, spatial usage, memory occupancy, mapspace counts, and mapper runtime. Keep TCM in a separate Python 3.12 environment so its dependencies do not alter the SPN evaluation environment. Preserve an O4 imperfect-loop microbenchmark as a tail-shape sensitivity check rather than mixing it into the O3 full-model ranking.

- [x] **Step 4: Install the pinned backend and run a real microbenchmark**

Run: `conda create -p .venv/accelforge python=3.12`

Run: `.venv/accelforge/bin/pip install -r requirements-mapper-validation.txt`

Run: `.venv/accelforge/bin/python scripts/mapper_backends/run_tcm.py --self-test`

Expected: one BF16 GEMM returns positive cycles/energy and utilization no greater than one.

### Task 4: Three-backend validation orchestrator

**Files:**
- Create: `scripts/run_spn_mapper_validation.py`
- Create: `configs/spn_mapper_validation_bf16_lpddr4.json`
- Test: `tests/test_run_spn_mapper_validation.py`

- [x] **Step 1: Write failing artifact and propagation-exclusion tests**

```python
def test_validation_artifacts_cover_three_backends_and_no_propagation(tmp_path):
    output = run(config, workloads, tmp_path / "report", executors=fake_executors)
    coverage = read_csv(output / "workload_coverage.csv")
    assert sum(int(r["included_layer_count"]) for r in coverage) == 486
    assert sum(int(r["excluded_propagation_count"]) for r in coverage) == 4
    assert not any(r["kind"] == "propagation" for r in read_csv(output / "layer_results.csv"))
    assert {r["backend"] for r in read_csv(output / "backend_status.csv")} == {
        "llmcompass", "tcm", "factorflow"}
```

- [x] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_run_spn_mapper_validation.py`

Expected: missing runner import.

- [x] **Step 3: Implement cached execution, expansion, validation, and reports**

Run three analytical microbenchmarks first, then the four anchor points `balanced__32x32`, `balanced__64x64`, `aggressive__64x64`, and `aggressive__128x128`. Cache by the fully serialized backend request, including revision, mapper policy, architecture, GEMM, and cost assumptions; expand unique results to original layers; and write the architecture manifest, backend status, workload coverage, unique-GEMM results, layer results, model summaries, rankings, validation metadata, and comparison plot. Normalize latency/energy/EDP within each backend before comparing architecture ranks.

- [x] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_run_spn_mapper_validation.py tests/test_mapper_validation.py tests/test_factorflow_adapter.py tests/test_tcm_adapter.py`

Expected: all tests pass and the synthetic artifact validator returns `SPN_THREE_MAPPER_VALIDATION_OK`.

### Task 5: Execute the real four-model validation

**Files:**
- Create: `reports/generated/spn_three_mapper_validation_bf16_lpddr4/analysis/*`
- Modify: `reports/README.md`
- Modify: `README.md`

- [x] **Step 1: Pin and initialize upstream sources**

Initialize FactorFlow revision `40bcf3109377f12a63c8e8ba3a01df2208977bf3` and AccelForge revision `7312a7d915625552bf6eb1db577941f8317b6ac7`, recording both revisions in validation metadata.

- [x] **Step 2: Run microbenchmark gates**

Run: `python scripts/run_spn_mapper_validation.py --config configs/spn_mapper_validation_bf16_lpddr4.json --mode microbench`

Expected: all backends pass MAC conservation, positive-cycle, physical-utilization, and capacity checks.

- [x] **Step 3: Run all four models on four anchor architectures**

Run: `python scripts/run_spn_mapper_validation.py --config configs/spn_mapper_validation_bf16_lpddr4.json --mode anchors`

Expected: 486 dense layers reconstructed from 115 cross-model unique workload signatures for each successful backend; zero propagation rows enter layer or model summaries.

- [x] **Step 4: Validate artifacts and document conclusions**

Run: `python scripts/run_spn_mapper_validation.py --validate reports/generated/spn_three_mapper_validation_bf16_lpddr4/analysis`

Expected: `SPN_THREE_MAPPER_VALIDATION_OK`.

Document backend-native absolute results separately from normalized cross-backend architecture rankings, including mapper runtime and model confidence.

### Task 6: Final regression verification

**Files:**
- Test: all hardware DSE and mapper-validation tests

- [x] **Step 1: Run focused regression tests**

Run: `pytest -q tests/test_hardware_workload.py tests/test_array_dse.py tests/test_memory_dse.py tests/test_joint_dse.py tests/test_mapper_validation.py tests/test_factorflow_adapter.py tests/test_tcm_adapter.py tests/test_run_spn_mapper_validation.py`

Expected: all pass.

- [x] **Step 2: Check worktree scope**

Run: `git status --short`

Expected: only mapper-validation files plus the pre-existing joint-DSE changes are modified; unrelated user changes remain untouched.

No commit is created because this worktree already contains active, uncommitted joint-DSE work that must remain reviewable as one coherent series.
