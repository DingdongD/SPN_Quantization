# Strict W4A4 and FP4 Reconstruction Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate whether fixed strict AdaRound and BRECQ W4 contracts preserve the four official NYU depth-completion models under matched uniform A4 and scaled E2M1 activation contracts.

**Architecture:** Reuse the existing strict weight-contract replay and FP4V activation backend without changing either model weights or model architectures. Add a fail-closed experiment orchestrator, a method-aware aggregation module, and dedicated plots that keep the matched FP4V comparison separate from the all-A4 integer stress baseline.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, Bash, pytest, official CSPN/DySPN/NLSPN/CompletionFormer CUDA environments.

---

## File Structure

- Create `scripts/strict_w4a4_fp4_evaluation.py`: experiment constants, acceptance decisions, paired statistics, and result-root validation helpers.
- Create `scripts/analyze_strict_w4a4_fp4_evaluation.py`: validate all method/model outputs and write aggregate CSV/Markdown reports.
- Create `scripts/plot_strict_w4a4_fp4_evaluation.py`: render RMSE, retention, paired-delta, prediction, error, activation, propagation, and stress figures.
- Create `scripts/run_strict_w4a4_fp4_evaluation.sh`: smoke/full orchestration across methods, models, CUDA environments, primary FP4V configurations, and the integer stress baseline.
- Create `tests/test_strict_w4a4_fp4_evaluation.py`: pure analysis and fail-closed validation tests.
- Create `tests/test_plot_strict_w4a4_fp4_evaluation.py`: panel ordering, deterministic sample selection, and nonfinite rendering tests.
- Create `tests/test_strict_w4a4_fp4_shell_contract.py`: shell matrix and environment-contract tests.
- Modify `tests/test_semantic_edge_runner.py`: prove strict contracts compose with the FP4 backend without activation overrides.
- Modify `docs/2026-08-06-strict-reconstruction-deployment.md`: document the new evaluation command and output boundary.

### Task 1: Define the Strict Evaluation Contract

**Files:**
- Create: `scripts/strict_w4a4_fp4_evaluation.py`
- Create: `tests/test_strict_w4a4_fp4_evaluation.py`

- [ ] **Step 1: Write failing tests for the immutable matrix and acceptance rule**

```python
from scripts.strict_w4a4_fp4_evaluation import (
    METHOD_ORDER,
    PRIMARY_CONFIGS,
    STRESS_CONFIGS,
    performance_decision,
)


def test_matrix_is_fixed_and_keeps_stress_separate():
    assert METHOD_ORDER == ("rtn", "adaround", "brecq")
    assert PRIMARY_CONFIGS == (
        "FP32", "FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8")
    assert STRESS_CONFIGS == ("FP32", "HW_W4A4_full")


def test_performance_decision_requires_all_three_conditions():
    accepted = performance_decision(
        fp32_rmse=1.0, quant_rmse=1.08, rtn_rmse=1.09,
        nonfinite_samples=0, nonfinite_pixels=0)
    assert accepted["status"] == "preserved"

    degraded = performance_decision(
        fp32_rmse=1.0, quant_rmse=1.11, rtn_rmse=1.12,
        nonfinite_samples=0, nonfinite_pixels=0)
    assert degraded["status"] == "rejected_fp32_degradation"

    regression = performance_decision(
        fp32_rmse=1.0, quant_rmse=1.08, rtn_rmse=1.07,
        nonfinite_samples=0, nonfinite_pixels=0)
    assert regression["status"] == "rejected_rtn_regression"

    invalid = performance_decision(
        fp32_rmse=1.0, quant_rmse=1.02, rtn_rmse=1.03,
        nonfinite_samples=1, nonfinite_pixels=10)
    assert invalid["status"] == "rejected_nonfinite"
```

- [ ] **Step 2: Run the focused test and verify that the module is missing**

Run:

```bash
python -m pytest -q tests/test_strict_w4a4_fp4_evaluation.py
```

Expected: FAIL with `ModuleNotFoundError: scripts.strict_w4a4_fp4_evaluation`.

- [ ] **Step 3: Implement constants and the explicit acceptance decision**

```python
#!/usr/bin/env python3
"""Contracts and statistics for strict W4A4 and FP4 evaluation."""

from __future__ import division, print_function

import numpy as np


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
METHOD_ORDER = ("rtn", "adaround", "brecq")
PRIMARY_CONFIGS = (
    "FP32", "FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8")
STRESS_CONFIGS = ("FP32", "HW_W4A4_full")
STRICT_METHODS = {
    "adaround": "adaround_strict",
    "brecq": "brecq_strict",
}
MAX_RELATIVE_RMSE_DEGRADATION = 0.10


def performance_decision(fp32_rmse, quant_rmse, rtn_rmse,
                         nonfinite_samples, nonfinite_pixels):
    values = np.asarray(
        [fp32_rmse, quant_rmse, rtn_rmse], dtype=np.float64)
    if not np.isfinite(values).all() or float(fp32_rmse) <= 0.0:
        raise ValueError("finite positive RMSE values are required")
    relative = (float(quant_rmse) - float(fp32_rmse)) / float(fp32_rmse)
    if int(nonfinite_samples) != 0 or int(nonfinite_pixels) != 0:
        status = "rejected_nonfinite"
    elif relative > MAX_RELATIVE_RMSE_DEGRADATION:
        status = "rejected_fp32_degradation"
    elif float(quant_rmse) > float(rtn_rmse):
        status = "rejected_rtn_regression"
    else:
        status = "preserved"
    return {
        "relative_rmse_degradation": relative,
        "delta_vs_rtn": float(quant_rmse) - float(rtn_rmse),
        "status": status,
    }
```

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest -q tests/test_strict_w4a4_fp4_evaluation.py`

Expected: PASS.

- [ ] **Step 5: Commit the contract module**

```bash
git add scripts/strict_w4a4_fp4_evaluation.py \
  tests/test_strict_w4a4_fp4_evaluation.py
git commit -m "feat: define strict W4A4 FP4 evaluation contract"
```

### Task 2: Verify Strict Weight Contracts Compose With FP4

**Files:**
- Modify: `tests/test_semantic_edge_runner.py`
- Modify: `scripts/run_nyu_edge_quantization.py`

- [ ] **Step 1: Add a failing test for strict FP4 metadata and activation ownership**

```python
import json

import torch

from scripts import run_nyu_edge_quantization as edge


def test_strict_fp4_replay_keeps_weight_only_manifest(tmp_path):
    contract = tmp_path / "strict_deployment_contract.pt"
    torch.save({
        "format_version": 1,
        "strict": 1,
        "method": "adaround_strict",
        "targets": ["conv"],
        "weight_contracts": {"conv": {"bits": 4}},
    }, contract)
    manifest = tmp_path / "strict_reconstruction_manifest.json"
    manifest.write_text(json.dumps({
        "strict": 1,
        "method": "adaround_strict",
        "weight_bits": 4,
        "activation_bits": 0,
        "activation_manifest": [],
        "targets": ["conv"],
        "deployment_contract": str(contract),
    }), encoding="utf-8")

    loaded = edge.load_reconstruction_manifest(str(manifest))

    assert loaded["strict"] == 1
    assert loaded["weight_bits"] == 4
    assert loaded["activation_bits"] == 0
    assert loaded["method"] == "adaround_strict"
```

Add a second assertion to the existing runner-installation fixture that a
`HardwareAlignedInstrumentor` created for the FP4 backend is wrapped by
`StrictContractInstrumentor` and receives the exact contract.

- [ ] **Step 2: Run the test and capture the exact failure**

Run:

```bash
python -m pytest -q \
  tests/test_semantic_edge_runner.py::test_strict_fp4_replay_keeps_weight_only_manifest
```

Expected: FAIL until the fixture and explicit metadata contract are present.

- [ ] **Step 3: Add explicit activation-policy metadata to strict replay**

In `write_json_with_semantics`, extend only the existing reconstruction block:

```python
payload["reconstruction"] = {
    "manifest": reconstruction["path"],
    "method": reconstruction["method"],
    "weight_bits": reconstruction["weight_bits"],
    "activation_bits": reconstruction["activation_bits"],
    "activation_policy": "evaluation_backend_owned",
    "targets": reconstruction["targets"],
    "strict_deployment_contract": reconstruction[
        "strict_contract_path"],
    "exact_weight_contract": 1,
}
```

Do not add a fallback for legacy activation manifests. A strict reconstruction
manifest with any activation override continues to raise `ValueError`.

- [ ] **Step 4: Run strict edge tests**

Run:

```bash
python -m pytest -q tests/test_semantic_edge_runner.py \
  tests/test_edge_runner.py tests/test_strict_reconstruction.py
```

Expected: PASS.

- [ ] **Step 5: Commit strict FP4 composition**

```bash
git add scripts/run_nyu_edge_quantization.py \
  tests/test_semantic_edge_runner.py
git commit -m "fix: declare activation ownership in strict FP4 replay"
```

### Task 3: Add Fail-Closed Result Validation

**Files:**
- Modify: `scripts/strict_w4a4_fp4_evaluation.py`
- Modify: `tests/test_strict_w4a4_fp4_evaluation.py`

- [ ] **Step 1: Write fixtures for aligned and mismatched result roots**

Create fixture helpers that write `metadata.json`, `sample_metrics.csv`,
`semantic_a8_boundaries.csv`, `fp4_manifest.csv`, and prediction NPZ files for
all three methods. Add tests that reject:

```python
def test_validation_rejects_checkpoint_mismatch(result_root):
    metadata = read_json(
        result_root / "primary" / "brecq" / "cspn" / "metadata.json")
    metadata["model_provenance"]["checkpoint_sha256"] = "different"
    write_json(
        result_root / "primary" / "brecq" / "cspn" / "metadata.json",
        metadata)

    with pytest.raises(ValueError, match="checkpoint provenance mismatch"):
        validate_result_root(result_root, expected_samples=2)


def test_validation_rejects_semantic_boundary_mismatch(result_root):
    path = result_root / "primary" / "adaround" / "nlspn" / \
        "semantic_a8_boundaries.csv"
    rows = read_csv(path)
    rows[0]["bits"] = "4"
    write_csv(path, rows)

    with pytest.raises(ValueError, match="semantic A8 boundary mismatch"):
        validate_result_root(result_root, expected_samples=2)
```

Also test missing predictions, unequal evaluation indices, wrong E2M1 execution,
non-exact strict contracts, unexpected reconstruction methods, and stress
metadata without int32 bias.

- [ ] **Step 2: Run the validation tests and verify failure**

Run: `python -m pytest -q tests/test_strict_w4a4_fp4_evaluation.py`

Expected: FAIL because `validate_result_root` does not exist.

- [ ] **Step 3: Implement strict metadata and payload validation**

Add these public entry points:

```python
def validate_result_root(root, expected_samples):
    root = Path(root)
    reference = {}
    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            primary = validate_primary_result(
                root / "primary" / method / model,
                method, model, expected_samples)
            stress = validate_stress_result(
                root / "stress" / method / model,
                method, model, expected_samples)
            identity = {
                field: primary["model_provenance"][field]
                for field in provenance_fields(model)
            }
            if model not in reference:
                reference[model] = {
                    "identity": identity,
                    "calibration_indices": primary["calibration_indices"],
                    "evaluation_indices": primary["evaluation_indices"],
                    "semantic_boundaries": read_csv(
                        root / "primary" / method / model /
                        "semantic_a8_boundaries.csv"),
                }
            validate_against_reference(
                reference[model], primary, stress, identity,
                root / "primary" / method / model)
    return reference
```

Use direct dictionary indexing for every required field. Require primary
`quant_backend == "fp4"`, primary execution
`float_e2m1_qdq_integer_normalization_reference`, stress
`quant_backend == "hardware"`, and stress execution
`hardware_aligned_qdq`. Require
`metadata["hardware_alignment"]["bias_contract"]` to equal
`int32 scale=sx*sw[o]` for stress results and `fp32_isolation` for primary
results.

For AdaRound and BRECQ, require:

```python
reconstruction = metadata["reconstruction"]
if int(reconstruction["exact_weight_contract"]) != 1:
    raise ValueError("non-exact strict weight contract")
if reconstruction["method"] != STRICT_METHODS[method]:
    raise ValueError("strict reconstruction method mismatch")
if reconstruction["activation_policy"] != "evaluation_backend_owned":
    raise ValueError("strict activation ownership mismatch")
```

RTN metadata must not contain a reconstruction block.

- [ ] **Step 4: Run validation and existing FP4 tests**

Run:

```bash
python -m pytest -q tests/test_strict_w4a4_fp4_evaluation.py \
  tests/test_fp4_activation_validation.py \
  tests/test_run_nyu_rtn_quantization.py
```

Expected: PASS.

- [ ] **Step 5: Commit result validation**

```bash
git add scripts/strict_w4a4_fp4_evaluation.py \
  tests/test_strict_w4a4_fp4_evaluation.py
git commit -m "feat: validate strict W4A4 FP4 result roots"
```

### Task 4: Implement Method-Aware Aggregation

**Files:**
- Create: `scripts/analyze_strict_w4a4_fp4_evaluation.py`
- Modify: `scripts/strict_w4a4_fp4_evaluation.py`
- Modify: `tests/test_strict_w4a4_fp4_evaluation.py`

- [ ] **Step 1: Write failing aggregation and bootstrap tests**

```python
def select_one(rows, **expected):
    matches = [
        row for row in rows
        if all(row[key] == value for key, value in expected.items())]
    assert len(matches) == 1
    return matches[0]


def test_aggregation_compares_each_method_with_same_format_rtn(result_root):
    tables = analyze_result_root(
        result_root, expected_samples=2,
        bootstrap_resamples=500, bootstrap_seed=20260806)

    row = select_one(
        tables["summary"], model="cspn", method="adaround",
        config="FP4V_W4E2M1")
    assert row["delta_vs_rtn"] == pytest.approx(
        row["mean_rmse"] -
        select_one(
            tables["summary"], model="cspn", method="rtn",
            config="FP4V_W4E2M1")["mean_rmse"])

    paired = select_one(
        tables["paired"], model="cspn", method="brecq",
        comparison="e2m1_minus_a4")
    assert paired["samples"] == 2
    assert paired["ci_lower"] <= paired["mean_difference"]
    assert paired["mean_difference"] <= paired["ci_upper"]
```

Add tests for AdaRound-minus-RTN and BRECQ-minus-RTN under A4, E2M1, and A8.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest -q tests/test_strict_w4a4_fp4_evaluation.py`

Expected: FAIL because `analyze_result_root` is missing.

- [ ] **Step 3: Implement paired statistics and aggregate tables**

Use one generic paired helper:

```python
def paired_rmse_difference(left, right, resamples, seed):
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("paired RMSE arrays must have identical shape")
    differences = left_values - right_values
    generator = np.random.RandomState(int(seed))
    indices = generator.randint(
        0, differences.size,
        size=(int(resamples), differences.size))
    means = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "samples": int(differences.size),
    }
```

`analyze_result_root` first calls `validate_result_root`, then creates:

- `summary`: primary FP32/A4/E2M1/A8 aggregates and acceptance status;
- `paired`: method-minus-RTN and E2M1-minus-A4 comparisons;
- `activation_groups`: weighted SQNR, zero, saturation, and nonfinite rates;
- `propagation_steps`: mean state RMSE by iteration;
- `stress`: `HW_W4A4_full` aggregates and deltas.

The CLI writes:

```text
strict_w4a4_fp4_summary.csv
strict_w4a4_fp4_paired.csv
strict_w4a4_fp4_activation_groups.csv
strict_w4a4_fp4_propagation_steps.csv
strict_w4a4_integer_stress.csv
strict_w4a4_fp4_report.md
```

- [ ] **Step 4: Run analyzer tests and CLI help**

Run:

```bash
python -m pytest -q tests/test_strict_w4a4_fp4_evaluation.py
python scripts/analyze_strict_w4a4_fp4_evaluation.py --help
```

Expected: tests PASS and help exits 0.

- [ ] **Step 5: Commit aggregation**

```bash
git add scripts/analyze_strict_w4a4_fp4_evaluation.py \
  scripts/strict_w4a4_fp4_evaluation.py \
  tests/test_strict_w4a4_fp4_evaluation.py
git commit -m "feat: aggregate strict W4A4 FP4 evaluations"
```

### Task 5: Add Deterministic Prediction And Diagnostic Figures

**Files:**
- Create: `scripts/plot_strict_w4a4_fp4_evaluation.py`
- Create: `tests/test_plot_strict_w4a4_fp4_evaluation.py`

- [ ] **Step 1: Write failing tests for panels and sample selection**

```python
def make_rows(values):
    rows = []
    for sample_index, rmses in values.items():
        for rank, rmse in enumerate(rmses):
            rows.append({
                "sample_index": str(sample_index),
                "series": "series_%d" % rank,
                "RMSE": str(rmse),
            })
    return rows


def test_prediction_panels_include_all_weight_methods_and_formats():
    assert plotting.prediction_panels() == (
        ("GT", "GT"),
        ("FP32", "FP32"),
        ("rtn:FP4V_W4A4", "RTN A4"),
        ("rtn:FP4V_W4E2M1", "RTN E2M1"),
        ("adaround:FP4V_W4A4", "AdaRound A4"),
        ("adaround:FP4V_W4E2M1", "AdaRound E2M1"),
        ("brecq:FP4V_W4A4", "BRECQ A4"),
        ("brecq:FP4V_W4E2M1", "BRECQ E2M1"),
    )


def test_visual_sample_uses_largest_finite_cross_method_spread():
    rows = make_rows({11: [0.2, 0.4], 17: [0.1, 0.9], 23: [0.5, 0.6]})
    assert plotting.select_visual_index(rows) == 17
```

Add the existing invalid-GT and magenta nonfinite rendering assertions.

- [ ] **Step 2: Run the plotting tests and verify failure**

Run: `python -m pytest -q tests/test_plot_strict_w4a4_fp4_evaluation.py`

Expected: FAIL because the plotting module is missing.

- [ ] **Step 3: Implement plotting with fixed presentation contracts**

Use Arial with Liberation Sans and DejaVu Sans fallbacks, no figure titles,
unrotated model labels, grid `zorder=0`, data `zorder=3`, and common depth/error
ranges. Render:

```text
strict_w4a4_fp4_rmse.png
strict_w4a4_fp4_retention.png
strict_w4a4_fp4_paired.png
strict_w4a4_fp4_predictions.png
strict_w4a4_fp4_errors.png
strict_w4a4_integer_stress.png
strict_w4a4_fp4_activation_<model>.png
strict_w4a4_fp4_propagation_<model>.png
```

`select_visual_index` must reject unequal sample sets and ignore a sample only
when at least one compared RMSE is nonfinite. If every sample has a nonfinite
method, raise `ValueError` instead of selecting a fallback.

- [ ] **Step 4: Run plotting tests and render fixture figures**

Run:

```bash
python -m pytest -q tests/test_plot_strict_w4a4_fp4_evaluation.py
```

The plotting tests create a complete result fixture under `tmp_path`, invoke
the CLI entry point, and assert that all required PNG files are nonempty.
Expected: tests PASS.

- [ ] **Step 5: Commit plotting**

```bash
git add scripts/plot_strict_w4a4_fp4_evaluation.py \
  tests/test_plot_strict_w4a4_fp4_evaluation.py
git commit -m "feat: plot strict W4A4 FP4 comparisons"
```

### Task 6: Add Four-Model GPU Orchestration

**Files:**
- Create: `scripts/run_strict_w4a4_fp4_evaluation.sh`
- Create: `tests/test_strict_w4a4_fp4_shell_contract.py`

- [ ] **Step 1: Write failing shell-contract tests**

```python
def test_runner_uses_edge_backend_for_every_quantized_run():
    script = script_text()
    assert 'scripts/run_nyu_edge_quantization.py' in script
    assert '--merge-policy independent' in script
    assert '--quant-backend fp4' in script
    assert '--quant-backend hardware' in script


def test_runner_uses_64_calibration_and_evaluation_samples():
    script = script_text()
    assert 'CALIBRATION_SAMPLES=64' in script
    assert 'EVALUATION_SAMPLES=64' in script


def test_runner_requires_all_eight_strict_manifests():
    script = script_text()
    assert 'adaround_strict/strict_reconstruction_manifest.json' in script
    assert 'brecq_strict/strict_reconstruction_manifest.json' in script
    assert '[[ -f "$manifest" ]]' in script
```

- [ ] **Step 2: Run the shell tests and verify failure**

Run: `python -m pytest -q tests/test_strict_w4a4_fp4_shell_contract.py`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement explicit smoke/full orchestration**

The script requires these environment variables without defaults:

```bash
: "${SPN_DATA_ROOT:?}"
: "${SPN_EXTERNAL_ROOT:?}"
: "${COMPLETIONFORMER_ROOT:?}"
: "${STRICT_RECONSTRUCTION_ROOT:?}"
: "${STRICT_REFERENCE_ROOT:?}"
: "${STRICT_W4A4_FP4_OUTPUT_ROOT:?}"
: "${CSPN_PYTHON:?}"
: "${DYSPN_PYTHON:?}"
: "${NLSPN_PYTHON:?}"
: "${COMPLETIONFORMER_PYTHON:?}"
: "${CSPN_GPU:?}"
: "${DYSPN_GPU:?}"
: "${NLSPN_GPU:?}"
: "${COMPLETIONFORMER_GPU:?}"
```

Use `CUDA_VISIBLE_DEVICES="${GPUS[$model]}"` and always pass
`--device cuda:0` so the official DCN extensions use visible device zero.

For each method/model, run the primary backend into
`$OUTPUT_ROOT/primary/$method` with:

```bash
--quant-backend fp4
--config-names FP32 FP4V_W4A4 FP4V_W4E2M1 FP4V_W4A8
--export-prediction-configs FP32 FP4V_W4A4 FP4V_W4E2M1 FP4V_W4A8
```

Run the stress backend into `$OUTPUT_ROOT/stress/$method` with:

```bash
--quant-backend hardware
--config-names FP32 HW_W4A4_full
--export-prediction-configs FP32 HW_W4A4_full
```

Pass `--reconstruction-manifest` only for AdaRound and BRECQ. Pass the original
checkpoint in every case. Use the RTN strict-W4A8 `sample_metrics.csv` only as
the immutable ordered evaluation-index source.

After all model processes finish, invoke the analyzer and plotter. Do not use
`--append`, and reject an output root that already contains any method/model
result directory.

- [ ] **Step 4: Run shell tests and syntax validation**

Run:

```bash
python -m pytest -q tests/test_strict_w4a4_fp4_shell_contract.py
bash -n scripts/run_strict_w4a4_fp4_evaluation.sh
```

Expected: PASS.

- [ ] **Step 5: Commit orchestration**

```bash
git add scripts/run_strict_w4a4_fp4_evaluation.sh \
  tests/test_strict_w4a4_fp4_shell_contract.py
git commit -m "feat: orchestrate strict W4A4 FP4 evaluation"
```

### Task 7: Run A Two-Sample CUDA Smoke Evaluation

**Files:**
- Generated: `profile_logs/nyu_strict_w4a4_fp4_smoke/`

- [ ] **Step 1: Verify environments and extension visibility**

Run `torch.cuda.is_available()` and one official-model forward in each selected
environment. For NLSPN and CompletionFormer, expose one physical GPU and use
`cuda:0` inside the process.

Expected: all four forwards complete and produce finite depth tensors.

- [ ] **Step 2: Run the smoke matrix**

Run:

```bash
SPN_DATA_ROOT=/workspace/CSPN/cspn_pytorch \
SPN_EXTERNAL_ROOT=/workspace/SPN_Quantization/.worktrees/pr5-strict-reconstruction/external \
COMPLETIONFORMER_ROOT=/workspace/SPN_Quantization/.worktrees/pr5-strict-reconstruction/external/CompletionFormer \
STRICT_RECONSTRUCTION_ROOT=/workspace/SPN_Quantization/profile_logs/nyu_strict_w4a8_reconstruction_current \
STRICT_REFERENCE_ROOT=/workspace/SPN_Quantization/profile_logs/nyu_strict_w4a8_evaluation/rtn \
STRICT_W4A4_FP4_OUTPUT_ROOT=/workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_smoke \
CSPN_PYTHON=python DYSPN_PYTHON=python \
NLSPN_PYTHON=/opt/conda/envs/completionformer-py37/bin/python \
COMPLETIONFORMER_PYTHON=/opt/conda/envs/completionformer-py37/bin/python \
CSPN_GPU=0 DYSPN_GPU=1 NLSPN_GPU=2 COMPLETIONFORMER_GPU=3 \
bash scripts/run_strict_w4a4_fp4_evaluation.sh smoke
```

Expected: 48 primary method/model/config runs and 24 stress
method/model/config runs complete with two evaluation samples each. This is 36
primary quantized combinations plus 12 primary FP32 combinations, and 12 stress
quantized combinations plus 12 stress FP32 combinations. No CUDA
illegal-memory-access, contract mismatch, or nonfinite process failure occurs.

- [ ] **Step 3: Validate smoke outputs and inspect figures**

Run the analyzer independently with `--expected-samples 2`, verify all NPZ
counts, inspect every PNG with `view_image`, and confirm no text overlap or blank
panels.

- [ ] **Step 4: Remove smoke artifacts after validation**

Delete only `profile_logs/nyu_strict_w4a4_fp4_smoke` after recording the smoke
status. Do not delete strict reconstruction contracts or the W4A8 reference.

### Task 8: Run The Formal 64-Sample Evaluation

**Files:**
- Generated: `profile_logs/nyu_strict_w4a4_fp4_evaluation/`

- [ ] **Step 1: Run the complete matrix into a new root**

Use the Task 7 environment command with:

```bash
STRICT_W4A4_FP4_OUTPUT_ROOT=/workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation \
bash scripts/run_strict_w4a4_fp4_evaluation.sh full
```

Expected: four models, three weight methods, four primary configurations, and
two stress configurations complete for the fixed 64 validation samples.

- [ ] **Step 2: Verify exact result counts**

For every primary method/model directory, require 64 rows and 64 NPZ payloads
for FP32, A4, E2M1, and A8. For every stress method/model directory, require 64
rows and 64 NPZ payloads for FP32 and `HW_W4A4_full`.

Expected totals: 3,072 primary prediction payloads and 1,536 stress prediction
payloads.

- [ ] **Step 3: Rerun aggregation from the canonical root**

Run:

```bash
python scripts/analyze_strict_w4a4_fp4_evaluation.py \
  --root /workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation/analysis \
  --expected-samples 64 --bootstrap-resamples 10000 \
  --bootstrap-seed 20260806
```

Expected: validation passes before any aggregate file is replaced.

- [ ] **Step 4: Regenerate and inspect formal figures**

Run the plotter against the canonical root. Inspect every generated image at
original resolution and verify the prediction/error panels use the same selected
sample for all methods in a model.

- [ ] **Step 5: Record the conclusion without changing the criterion**

Report each model/config as `preserved` or the explicit rejection status.
Report E2M1 as a float QDQ reference. If standard AdaRound/BRECQ fails, preserve
the result and propose activation-aware reconstruction as a separately named
follow-up; do not tune thresholds after seeing metrics.

### Task 9: Documentation And Final Verification

**Files:**
- Modify: `docs/2026-08-06-strict-reconstruction-deployment.md`
- Modify: `README.md`

- [ ] **Step 1: Document the experiment boundary and commands**

Add links to the design, implementation plan, formal runner, aggregate report,
and prediction figures. State that the matched FP4V group uses FP32 bias and A8
sensitive boundaries, while `HW_W4A4_full` is a separate integer stress result.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
python -m pytest -q tests
git diff --check
bash -n scripts/run_strict_w4a4_fp4_evaluation.sh
```

Expected: all tests pass, no whitespace errors, and shell syntax exits 0.

- [ ] **Step 3: Re-run final artifact validation**

Run the analyzer one final time against the canonical 64-sample result root and
verify the report and image hashes are stable across repeated aggregation.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/2026-08-06-strict-reconstruction-deployment.md
git commit -m "docs: report strict W4A4 FP4 evaluation"
```

- [ ] **Step 5: Review repository status**

Confirm only intentional pre-existing worktree changes remain. Do not include
generated profile artifacts or unrelated user files in source commits.
