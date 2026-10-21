# SPN LLMCompass Array DSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible LLMCompass-based workflow that extracts one-image NYU workloads from CSPN, DySPN, NLSPN, and CompletionFormer, sweeps systolic-array shapes, and reports whether each shape is efficient per layer and per model.

**Architecture:** Each official model is executed once in its already-declared Conda environment and exported to an environment-neutral workload JSON. A base-environment analyzer lowers dense Conv2d, ConvTranspose2d, Linear, and CompletionFormer attention products to GEMMs, calls the pinned LLMCompass systolic-array cycle model, and aggregates utilization/cycle/coverage results. Irregular propagation and deformable sampling remain explicit non-array operations so the report never presents dense-array estimates as whole-model latency.

**Tech Stack:** Python 3.7-compatible workload exporter, Python 3.11 analyzer, PyTorch forward hooks, LLMCompass ISCA_AE commit `62321b1ee28ddbdba8a2eb7475d7caa30f75e8be`, Scale-Sim 2.0.2, pandas, matplotlib, pytest.

---

## File map

- `.gitmodules`: pin the upstream LLMCompass repository as `external/LLMCompass`.
- `requirements-llmcompass.txt`: pin the one missing simulator dependency without changing model environments.
- `configs/spn_array_dse_bf16.json`: declare BF16 data type, candidate shapes, comparison groups, dataflow, and report thresholds.
- `spn_quant/hardware/workload.py`: immutable workload schema, validation, and Conv/Linear/attention lowering rules.
- `spn_quant/hardware/collector.py`: framework hooks plus model-aware attention and propagation metadata collection.
- `spn_quant/hardware/llmcompass_adapter.py`: isolated import of pinned upstream code and cycle/utilization simulation.
- `spn_quant/hardware/array_dse.py`: per-layer simulation, coverage accounting, Pareto analysis, and shared-array ranking.
- `scripts/export_nyu_spn_workload.py`: execute one official model in its configured environment and write JSON.
- `scripts/run_spn_array_dse.py`: analyze one or four exported workloads and write CSV/JSON/plots.
- `scripts/run_four_model_array_dse.sh`: issue the four environment-correct export commands and one base-environment analysis command.
- `tests/test_hardware_workload.py`: schema and lowering unit tests.
- `tests/test_hardware_collector.py`: hook and model-specific derived-operation unit tests.
- `tests/test_llmcompass_adapter.py`: upstream integration and cycle-model tests.
- `tests/test_array_dse.py`: aggregation, fairness, Pareto, and recommendation tests.
- `tests/test_export_nyu_spn_workload.py`: exporter configuration and serialization tests.
- `tests/test_run_spn_array_dse.py`: CLI artifact contract tests.
- `docs/2026-10-21-spn-array-dse-requirements.md`: requirements, assumptions, interfaces, open hardware questions, and risk register.
- `reports/generated/spn_array_dse_bf16/`: generated workload manifests, detailed rows, summaries, plots, and provenance.

### Task 1: Record requirements and pin LLMCompass

**Files:**
- Create: `docs/2026-10-21-spn-array-dse-requirements.md`
- Modify: `.gitmodules`
- Create: `requirements-llmcompass.txt`
- Test: `tests/test_llmcompass_adapter.py`

- [ ] **Step 1: Write the failing provenance test**

```python
from pathlib import Path


def test_llmcompass_submodule_and_dependency_are_pinned():
    root = Path(__file__).resolve().parents[1]
    modules = (root / ".gitmodules").read_text(encoding="utf-8")
    requirements = (root / "requirements-llmcompass.txt").read_text(
        encoding="utf-8")
    assert 'path = external/LLMCompass' in modules
    assert 'branch = ISCA_AE' in modules
    assert requirements.strip() == "scalesim==2.0.2"
```

- [ ] **Step 2: Run the test and verify the missing pin fails**

Run: `/opt/conda/bin/python -m pytest -q tests/test_llmcompass_adapter.py::test_llmcompass_submodule_and_dependency_are_pinned`

Expected: FAIL because the submodule entry and requirements file do not exist.

- [ ] **Step 3: Add the pinned upstream and dependency**

Run:

```bash
git submodule add -b ISCA_AE https://github.com/PrincetonUniversity/LLMCompass.git external/LLMCompass
git -C external/LLMCompass checkout 62321b1ee28ddbdba8a2eb7475d7caa30f75e8be
/opt/conda/bin/python -m pip install scalesim==2.0.2
```

Create `requirements-llmcompass.txt` with exactly:

```text
scalesim==2.0.2
```

Create the requirements document with these fixed assumptions: NYU batch one and 228x304 model input; BF16 operands with one MAC per PE per cycle; output-stationary dataflow; per-array compute cycles are analytical/model cycles, not measured wall time; irregular propagation and deformable sampling are excluded from array latency and included in coverage accounting. Record open questions for target process node, frequency, number of arrays, SRAM sizes/bandwidth, DRAM bandwidth, throughput/latency target, area/power budgets, and whether a separate sparse/vector engine is allowed.

- [ ] **Step 4: Re-run the provenance test**

Run: `/opt/conda/bin/python -m pytest -q tests/test_llmcompass_adapter.py::test_llmcompass_submodule_and_dependency_are_pinned`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitmodules external/LLMCompass requirements-llmcompass.txt docs/2026-10-21-spn-array-dse-requirements.md tests/test_llmcompass_adapter.py
git commit -m "build: pin LLMCompass for SPN array analysis"
```

### Task 2: Define the environment-neutral workload schema

**Files:**
- Create: `spn_quant/hardware/__init__.py`
- Create: `spn_quant/hardware/workload.py`
- Create: `tests/test_hardware_workload.py`

- [ ] **Step 1: Write failing schema and lowering tests**

```python
from spn_quant.hardware.workload import (
    LayerWorkload, conv2d_to_gemms, conv_transpose2d_to_gemms,
    linear_to_gemms,
)


def test_conv2d_group_lowering_preserves_macs():
    layer = LayerWorkload.conv2d(
        "group", [1, 32, 28, 38], [1, 64, 28, 38],
        kernel=(3, 3), stride=(1, 1), groups=4)
    gemms = conv2d_to_gemms(layer)
    assert [(g.m, g.n, g.k, g.repetitions) for g in gemms] == [
        (28 * 38, 16, 8 * 3 * 3, 4)]
    assert sum(g.macs for g in gemms) == layer.macs


def test_transposed_conv_uses_input_stationary_gemm_shape():
    layer = LayerWorkload.conv_transpose2d(
        "up", [1, 64, 14, 19], [1, 32, 28, 38],
        kernel=(4, 4), stride=(2, 2), groups=1)
    gemm, = conv_transpose2d_to_gemms(layer)
    assert (gemm.m, gemm.n, gemm.k) == (14 * 19, 32 * 4 * 4, 64)
    assert gemm.macs == layer.macs


def test_linear_flattens_all_leading_dimensions():
    layer = LayerWorkload.linear("fc", [2, 7, 64], [2, 7, 96])
    gemm, = linear_to_gemms(layer)
    assert (gemm.m, gemm.n, gemm.k) == (14, 96, 64)
```

- [ ] **Step 2: Run and verify import failure**

Run: `/opt/conda/bin/python -m pytest -q tests/test_hardware_workload.py`

Expected: FAIL with `ModuleNotFoundError: spn_quant.hardware`.

- [ ] **Step 3: Implement strict dataclasses and lowering**

Implement frozen `LayerWorkload` and `GemmWorkload` dataclasses. Require positive dimensions/MACs, accepted kinds (`conv2d`, `conv_transpose2d`, `linear`, `attention_qk`, `attention_av`, `propagation`, `deformable_sampling`), and an explicit `array_mappable` flag. Use:

```python
# Conv2d, once per group
M = output_h * output_w
N = output_channels // groups
K = (input_channels // groups) * kernel_h * kernel_w

# ConvTranspose2d, once per group; col2im/scatter is separate
M = input_h * input_w
N = (output_channels // groups) * kernel_h * kernel_w
K = input_channels // groups

# Linear
M = product(input_shape[:-1])
N = output_features
K = input_features
```

Store `repetitions=groups` rather than creating duplicate rows. Serialization must sort keys, include schema version 1, and preserve model/source/config/checkpoint identities.

- [ ] **Step 4: Run schema tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_hardware_workload.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/hardware tests/test_hardware_workload.py
git commit -m "feat: define SPN hardware workload schema"
```

### Task 3: Collect executed dense layers and derived attention/propagation operations

**Files:**
- Create: `spn_quant/hardware/collector.py`
- Create: `tests/test_hardware_collector.py`

- [ ] **Step 1: Write failing execution-hook tests**

```python
import torch
import torch.nn as nn
from spn_quant.hardware.collector import WorkloadCollector


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 12, 3, padding=1, groups=2)
        self.fc = nn.Linear(12, 5)

    def forward(self, x):
        y = self.conv(x)
        return self.fc(y.mean((-2, -1)))


def test_collector_records_executed_shapes_once():
    model = Toy().eval()
    with WorkloadCollector(model, model_name="dyspn") as collector:
        model(torch.zeros(1, 8, 10, 14))
    rows = {row.name: row for row in collector.layers()}
    assert rows["conv"].input_shape == (1, 8, 10, 14)
    assert rows["conv"].output_shape == (1, 12, 10, 14)
    assert rows["conv"].groups == 2
    assert rows["fc"].input_shape == (1, 12)
    assert rows["fc"].output_shape == (1, 5)
```

Use these concrete derived-operation assertions after constructing the fake
attention and propagation modules in the test fixture:

```python
derived = {row.kind: row for row in collector.layers()
           if row.kind in {"attention_qk", "attention_av", "propagation"}}
assert (derived["attention_qk"].m, derived["attention_qk"].n,
        derived["attention_qk"].k) == (56, 14, 32)
assert (derived["attention_av"].m, derived["attention_av"].n,
        derived["attention_av"].k) == (56, 32, 14)
assert derived["attention_qk"].repetitions == 4
assert derived["attention_av"].repetitions == 4
assert derived["propagation"].iterations == 18
assert derived["propagation"].neighborhood_size == 8
assert derived["propagation"].array_mappable is False
```

- [ ] **Step 2: Run and verify collector failure**

Run: `/opt/conda/bin/python -m pytest -q tests/test_hardware_collector.py`

Expected: FAIL because `WorkloadCollector` is missing.

- [ ] **Step 3: Implement the collector**

Register forward hooks only on leaf `nn.Conv2d`, `nn.ConvTranspose2d`, and `nn.Linear` modules. Record every executed invocation in order, including repeated module calls. Detect deformable convolution owners by module class/name and add a separate non-array sampling row without double-counting the dense weight MACs. For CompletionFormer attention, derive QK/AV from executed input shapes and module attributes; fail with an actionable error if head count or spatial-reduction dimensions cannot be determined. Add one propagation row from configured iterations and the model's neighborhood/kernel attributes.

- [ ] **Step 4: Run collector tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_hardware_collector.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/hardware/collector.py tests/test_hardware_collector.py
git commit -m "feat: collect executed SPN layer workloads"
```

### Task 4: Adapt the pinned LLMCompass cycle model

**Files:**
- Create: `spn_quant/hardware/llmcompass_adapter.py`
- Modify: `tests/test_llmcompass_adapter.py`

- [ ] **Step 1: Write failing cycle and utilization tests**

```python
from spn_quant.hardware.llmcompass_adapter import ArrayShape, simulate_gemm
from spn_quant.hardware.workload import GemmWorkload


def test_llmcompass_cycle_result_has_bounded_effective_utilization():
    result = simulate_gemm(
        GemmWorkload("x", "conv2d", 1024, 64, 288, 1),
        ArrayShape(16, 16), mac_per_cycle=1, dataflow="os")
    assert result.backend == "llmcompass"
    assert result.cycles > 0
    assert 0 < result.effective_utilization <= 1
    assert result.macs == 1024 * 64 * 288


def test_narrow_output_exposes_wide_array_underutilization():
    work = GemmWorkload("head", "conv2d", 4096, 1, 64, 1)
    narrow = simulate_gemm(work, ArrayShape(16, 8), 1, "os")
    wide = simulate_gemm(work, ArrayShape(8, 16), 1, "os")
    assert narrow.effective_utilization > wide.effective_utilization
```

- [ ] **Step 2: Run and verify adapter failure**

Run: `/opt/conda/bin/python -m pytest -q tests/test_llmcompass_adapter.py -k 'cycle or narrow'`

Expected: FAIL because the adapter is missing.

- [ ] **Step 3: Implement isolated upstream invocation**

Resolve `external/LLMCompass` relative to the repository, prepend it only while importing `software_model.matmul.Matmul`, and verify the checked-out Git revision. Call `Matmul.simulate_systolic_array_cycle_count()` with an empty indexed pandas lookup table inside a temporary working directory containing `systolic_array_model/temp`; this prevents generated Scale-Sim files from dirtying the submodule. Compute effective utilization as:

```python
effective_utilization = total_macs / (
    cycles * array.height * array.width * mac_per_cycle)
```

Clamp only a floating-point excess smaller than `1e-12`; otherwise raise. Multiply cycles and MACs by `repetitions`, and record whether LLMCompass used its analytical branch or Scale-Sim fallback.

- [ ] **Step 4: Run adapter tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_llmcompass_adapter.py`

Expected: PASS with no modified files under `external/LLMCompass`.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/hardware/llmcompass_adapter.py tests/test_llmcompass_adapter.py
git commit -m "feat: adapt LLMCompass systolic cycle model"
```

### Task 5: Implement fair array-shape exploration and recommendations

**Files:**
- Create: `configs/spn_array_dse_bf16.json`
- Create: `spn_quant/hardware/array_dse.py`
- Create: `tests/test_array_dse.py`

- [ ] **Step 1: Write failing aggregation tests**

```python
from spn_quant.hardware.array_dse import aggregate_candidate, pareto_frontier


def test_aggregate_weights_utilization_by_useful_macs(simulated_rows):
    summary = aggregate_candidate(simulated_rows)
    expected = sum(r.macs for r in simulated_rows) / sum(
        r.cycles * r.array_pe_count for r in simulated_rows)
    assert summary.effective_utilization == expected


def test_pareto_rejects_more_pes_and_more_cycles(candidate_summaries):
    frontier = pareto_frontier(candidate_summaries)
    assert {(row.height, row.width) for row in frontier} == {
        (8, 8), (16, 16), (32, 16)}
```

Add a shared-ranking test that gives each model equal weight, regardless of its raw MAC count, and a coverage test that excludes propagation cycles from dense-array latency while retaining its MAC/operation estimate in `array_coverage_fraction`.

- [ ] **Step 2: Run and verify DSE failure**

Run: `/opt/conda/bin/python -m pytest -q tests/test_array_dse.py`

Expected: FAIL because the DSE module is missing.

- [ ] **Step 3: Add BF16 candidates and comparison policy**

Create a configuration containing:

```json
{
  "format_version": 1,
  "dtype": "bf16",
  "word_bytes": 2,
  "mac_per_cycle": 1,
  "dataflow": "os",
  "array_shapes": [[8,8],[8,16],[16,8],[8,32],[16,16],[32,8],[16,32],[32,16],[32,32],[16,64],[64,16],[64,64],[128,128]],
  "iso_pe_groups": {
    "128": [[8,16],[16,8]],
    "256": [[8,32],[16,16],[32,8]],
    "512": [[16,32],[32,16]],
    "1024": [[32,32],[16,64],[64,16]]
  },
  "low_utilization_threshold": 0.25
}
```

- [ ] **Step 4: Implement metrics and decision rules**

For every candidate/model report total dense MACs, LLMCompass cycles, effective utilization, ideal-cycle overhead, low-utilization MAC fraction, low-utilization layer fraction, array-mappable coverage, and cycles per useful MAC. Build two decisions: best aspect ratio within each equal-PE group, and a PE-count/cycle Pareto frontier. The shared recommendation is the highest geometric mean of per-model normalized effective utilization; break ties by lower PE count, then lower cycle count. Never label a candidate sign-off-ready while any hardware budget remains open.

- [ ] **Step 5: Run DSE tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_array_dse.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/spn_array_dse_bf16.json spn_quant/hardware/array_dse.py tests/test_array_dse.py
git commit -m "feat: rank systolic array shapes for SPN workloads"
```

### Task 6: Add model-environment exporter CLI

**Files:**
- Create: `scripts/export_nyu_spn_workload.py`
- Create: `tests/test_export_nyu_spn_workload.py`

- [ ] **Step 1: Write failing CLI/config tests**

```python
from scripts.export_nyu_spn_workload import runtime_args, select_sample_index


def test_exporter_uses_first_fixed_evaluation_identity(model_spec):
    assert select_sample_index(model_spec) == model_spec["evaluation_indices"][0]


def test_runtime_args_preserve_model_identity(model_spec):
    args = runtime_args(model_spec, "cuda:2")
    assert args.model == model_spec["model"]
    assert args.device == "cuda:2"
    assert args.propagation_iterations == model_spec["propagation_iterations"]
```

- [ ] **Step 2: Run and verify exporter failure**

Run: `/opt/conda/bin/python -m pytest -q tests/test_export_nyu_spn_workload.py`

Expected: FAIL because the script is missing.

- [ ] **Step 3: Implement one-model export**

Use `NYUModelRuntime` and the model entry from `configs/four_model_unified_fp16_task_aware.json`. Load the fixed first evaluation sample, enter `WorkloadCollector`, execute exactly one inference under `torch.no_grad()`, validate a finite `[1,1,228,304]` prediction, then write schema-versioned JSON with checkpoint hash, config hash, model class, interpreter, torch version, CUDA device, sample index, input shapes, ordered layer rows, mapped MAC total, irregular MAC/operation total, and coverage.

- [ ] **Step 4: Run exporter tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_export_nyu_spn_workload.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_nyu_spn_workload.py tests/test_export_nyu_spn_workload.py
git commit -m "feat: export official SPN inference workloads"
```

### Task 7: Add analyzer CLI and artifact contract

**Files:**
- Create: `scripts/run_spn_array_dse.py`
- Create: `scripts/run_four_model_array_dse.sh`
- Create: `tests/test_run_spn_array_dse.py`

- [ ] **Step 1: Write the failing artifact test**

```python
def test_analyzer_writes_complete_artifact_set(tmp_path, workload_paths, config):
    output = run(config, workload_paths, tmp_path / "report")
    assert {path.name for path in output.iterdir()} == {
        "candidate_ranking.csv", "layer_results.csv", "metadata.json",
        "model_summary.csv", "pareto_frontier.csv",
        "shape_utilization.png", "workload_coverage.csv"
    }
```

Use these explicit completeness and overwrite assertions:

```python
summary = pandas.read_csv(output / "model_summary.csv")
layers = pandas.read_csv(output / "layer_results.csv")
candidate_count = len(config["array_shapes"])
assert len(summary) == 4 * candidate_count
assert set(summary["model"]) == {
    "cspn", "dyspn", "nlspn", "completionformer"}
expected_layer_rows = sum(profile.mappable_gemm_count
                          for profile in workload_paths.values()) * candidate_count
assert len(layers) == expected_layer_rows
metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
assert set(metadata["input_sha256"]) == set(workload_paths)
with pytest.raises(FileExistsError):
    run(config, workload_paths, output)
```

- [ ] **Step 2: Run and verify CLI failure**

Run: `/opt/conda/bin/python -m pytest -q tests/test_run_spn_array_dse.py`

Expected: FAIL because the analyzer is missing.

- [ ] **Step 3: Implement deterministic output**

Write CSVs with fixed field order and stable sorting by model, array PE count, height, width, and layer execution index. Plot per-model and geometric-mean effective utilization versus shape. Record upstream commit, Scale-Sim version, repo revision, dirty state, all input hashes, config, formulas, limitations, and open hardware questions in `metadata.json`.

The shell wrapper must invoke CSPN and DySPN with `/opt/conda/bin/python`; invoke NLSPN and CompletionFormer with `/opt/conda/envs/completionformer-py37/bin/python` and their existing `PYTHONPATH`/`LD_LIBRARY_PATH`; then invoke the analyzer with `/opt/conda/bin/python`.

- [ ] **Step 4: Run CLI tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_run_spn_array_dse.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_spn_array_dse.py scripts/run_four_model_array_dse.sh tests/test_run_spn_array_dse.py
git commit -m "feat: orchestrate four-model SPN array DSE"
```

### Task 8: Execute the four-model study and validate results

**Files:**
- Create: `reports/generated/spn_array_dse_bf16/*`
- Modify: `docs/2026-10-21-spn-array-dse-requirements.md`

- [ ] **Step 1: Initialize model submodules or use declared official roots**

Run: `git submodule update --init external/DySPN external/NLSPN_ECCV20 external/CompletionFormer`

Expected: all three submodules match the commits in `.gitmodules` with a clean status.

- [ ] **Step 2: Run four exports and the DSE**

Run: `bash scripts/run_four_model_array_dse.sh reports/generated/spn_array_dse_bf16`

Expected: four validated workload JSON files plus all seven analyzer artifacts.

- [ ] **Step 3: Validate artifact arithmetic independently**

Run:

```bash
/opt/conda/bin/python -m pytest -q \
  tests/test_hardware_workload.py \
  tests/test_hardware_collector.py \
  tests/test_llmcompass_adapter.py \
  tests/test_array_dse.py \
  tests/test_export_nyu_spn_workload.py \
  tests/test_run_spn_array_dse.py
```

Expected: all task-scoped tests PASS.

Run a separate validation script embedded in `run_spn_array_dse.py --validate-only` that recomputes every row's MACs, effective utilization, model totals, geometric means, and Pareto dominance from the emitted CSVs. Expected terminal line: `SPN_ARRAY_DSE_VALIDATION_OK`.

- [ ] **Step 4: Interpret without overclaiming**

Update the requirements report with: best aspect ratio for each equal-PE group; per-model best; shared best; layer families causing underutilization; dense-array coverage; CompletionFormer attention impact; and whether one shared array is acceptable. State confidence `MEDIUM` for dense GEMM cycle/utilization comparisons and `LOW` for whole-model latency until propagation/deformable/vector and memory-system calibration are supplied.

- [ ] **Step 5: Check repository hygiene**

Run: `git diff --check && git status --short && git -C external/LLMCompass status --short`

Expected: no whitespace errors and a clean LLMCompass submodule; generated reports are the only intentional output files beyond source/test/docs changes.

- [ ] **Step 6: Commit**

```bash
git add reports/generated/spn_array_dse_bf16 docs/2026-10-21-spn-array-dse-requirements.md
git commit -m "docs: report four-model BF16 array shape study"
```

