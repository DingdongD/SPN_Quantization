# Task-Gradient Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure task-gradient-weighted W/A quantization sensitivity for the four official SPN models and validate it with propagation-aware and end-to-end metrics.

**Architecture:** Add a standalone analyzer that loads the existing official model runtime and calibration identity, records module outputs and gradients during the official task loss, computes W4/W6/W8 fake-quant perturbation scores, and runs one-module deployment ablations through the existing hardware-aligned quantization path. Propagation signals are reported separately and are never silently merged into ordinary CNN activation scores.

**Tech Stack:** Python, PyTorch autograd, existing `NYUModelRuntime`, hardware-aligned QDQ, propagation adapters, CSV/JSON artifacts, pytest.

---

### Task 1: Define sensitivity data contracts and unit tests

**Files:**
- Create: `spn_quant/task_sensitivity.py`
- Test: `tests/test_task_sensitivity.py`

- [ ] **Step 1: Write failing tests** for score formulas, separate weight/activation budgets, propagation metric aggregation, and deterministic module ordering.
- [ ] **Step 2: Run `python -m pytest -q tests/test_task_sensitivity.py` and verify the new API is absent.
- [ ] **Step 3: Implement small pure functions** for gradient-weighted error, normalized score, marginal bit loss, and propagation metric aggregation. Inputs use explicit mappings and raise on missing keys.
- [ ] **Step 4: Run the focused test and verify it passes.

### Task 2: Implement calibration gradient and signal capture

**Files:**
- Modify: `spn_quant/task_sensitivity.py`
- Modify: `spn_quant/propagation/adapters.py`
- Test: `tests/test_task_sensitivity.py`

- [ ] **Step 1: Add tests** covering module output retention, finite gradient checks, and propagation records for initial depth, affinity, offset, confidence, and every recurrent state.
- [ ] **Step 2: Implement a capture context** that registers hooks only for the contract modules, runs `model.train(False)` with gradient tracking, and computes the official runtime task loss over exactly 128 calibration identities.
- [ ] **Step 3: Extend the existing adapters with explicit captured tensors** for propagation signals without changing their forward math or quantization behavior.
- [ ] **Step 4: Compute W4/W6/W8 parameter and activation perturbations using the existing quantizer conventions: per-output-channel symmetric weights and static per-tensor MinMax activations.
- [ ] **Step 5: Run focused tests and verify no propagation or ordinary-module coverage mismatch is accepted.

### Task 3: Add unified four-model CLI and artifact schema

**Files:**
- Create: `scripts/run_nyu_task_gradient_sensitivity.py`
- Create: `configs/nyu_task_gradient_sensitivity.json`
- Test: `tests/test_run_nyu_task_gradient_sensitivity.py`

- [ ] **Step 1: Write CLI contract tests** for required model/config/calibration arguments, fixed model order, exact 128 calibration identities, and output schema.
- [ ] **Step 2: Implement the CLI using the existing model runtime and official checkpoints; do not add alternate loaders, fallback paths, or implicit defaults.
- [ ] **Step 3: Emit `module_sensitivity.csv`, `propagation_signal_sensitivity.csv`, `end_to_end_ablation.csv`, `gradient_rankings.json`, and `sensitivity_summary.json`.
- [ ] **Step 4: Include model, module, role, tensor size, W/A bit, gradient score, normalized score, output error, SQNR, zero/saturation/sign-flip ratios, pooled SSE delta, nonfinite/nonpositive counts, and propagation-step metrics.
- [ ] **Step 5: Run unit and CLI contract tests.

### Task 4: Run the four official-model analysis and validate artifacts

**Files:**
- Create: `/workspace/SPN_Quantization/profile_logs/nyu_task_gradient_sensitivity_128/`

- [ ] **Step 1: Run the analyzer with the persisted stratified 128-sample train calibration identity and the existing fixed evaluation identity.
- [ ] **Step 2: Verify all four official models have identical metric definitions and complete ordinary/propagation coverage.
- [ ] **Step 3: Independently recompute pooled SSE/RMSE from per-sample records and compare with the summary.
- [ ] **Step 4: Check that no result contains NaN/Inf, missing module rows, duplicate module identities, or mixed calibration identities.
- [ ] **Step 5: Summarize model-specific sensitive modules and identify candidates for later W/A bit allocation; do not generate a mixed-precision deployment assignment in this phase.
