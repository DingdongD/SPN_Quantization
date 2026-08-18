# CSPN Uniform W6A6 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute a strictly comparable uniform W6A6 CSPN validation against W4A4, P3/T3, FINAL, and FP32.

**Architecture:** Extend the existing task-sensitive validation candidate tuple rather than adding a second quantization path. Keep search and quantization contracts unchanged, then extend the artifact auditor and prediction renderer to require the fifth configuration.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, pytest, official CSPN CUDA model

---

### Task 1: Add the W6A6 Validation Contract

**Files:**
- Modify: `tests/test_run_nyu_cspn_task_sensitive_bits.py`
- Modify: `scripts/run_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 1: Write the failing orchestration assertion**

Change the expected validation names to:

```python
("FP32", "UNIFORM_W4A4", "UNIFORM_W6A6",
 "CONTEXT_P3_T3_W8A8", "FINAL")
```

Also assert that `UNIFORM_W6A6` equals
`allocation.uniform_assignment(registry, 6, 6)`.

- [ ] **Step 2: Verify the focused test fails**

Run:
`python -m pytest -q tests/test_run_nyu_cspn_task_sensitive_bits.py::SearchOrchestrationTest::test_orchestration_runs_fixed_phases_and_freezes_before_validation`

Expected: failure because W6A6 is absent.

- [ ] **Step 3: Add the candidate**

Insert:

```python
ValidationCandidate(
    "UNIFORM_W6A6", allocation.uniform_assignment(registry, 6, 6)),
```

between W4A4 and P3/T3.

- [ ] **Step 4: Verify runner tests pass**

Run: `python -m pytest -q tests/test_run_nyu_cspn_task_sensitive_bits.py`

Expected: all tests pass.

### Task 2: Extend Audit and Prediction Rendering

**Files:**
- Modify: `tests/test_plot_nyu_cspn_task_sensitive_bits.py`
- Modify: `scripts/plot_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 1: Write failing five-configuration fixture assertions**

Require `UNIFORM_W6A6` in `VALIDATION_CONFIGS`, create 64 matching prediction
payloads for it, and require five validation rows.

- [ ] **Step 2: Verify plot tests fail**

Run: `python -m pytest -q tests/test_plot_nyu_cspn_task_sensitive_bits.py`

Expected: failure because the production configuration tuple has four entries.

- [ ] **Step 3: Extend production constants and renderer**

Use:

```python
VALIDATION_CONFIGS = (
    "FP32", "UNIFORM_W4A4", "UNIFORM_W6A6",
    "CONTEXT_P3_T3_W8A8", "FINAL")
```

Render six columns labelled `GT`, `FP32`, `W4A4`, `W6A6`, `P3/T3`, and
`Final` with the same per-sample GT color limits.

- [ ] **Step 4: Verify focused tests pass**

Run:
`python -m pytest -q tests/test_plot_nyu_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py`

Expected: all tests pass.

### Task 3: Execute and Report the Strict Comparison

**Files:**
- Modify: `docs/2026-08-18-cspn-task-sensitive-mixed-bit-allocation-results.md`
- Generate: `profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64/`

- [ ] **Step 1: Restore the published root as resumable staging**

Rename only the exact result root to `.incomplete`. Preserve `phase_cache`,
delete the old five-output summary files through normal overwrite, and resume
the existing production command with `--resume-incomplete`.

- [ ] **Step 2: Regenerate plots and audit**

Run:

```bash
python scripts/plot_nyu_cspn_task_sensitive_bits.py \
  --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64
```

Expected: five configurations, 64 identities each, finite metrics, exact
budget, and matching hashes.

- [ ] **Step 3: Update the measured report**

Add the W6A6 metrics, its exact `6.0/6.0` average ordinary-CNN bit widths, and
the measured RMSE gap to P3/T3. Do not infer similarity without reporting both
absolute and relative differences.

- [ ] **Step 4: Verify code and artifacts**

Run:

```bash
python -m pytest -q
git diff --check
```

Expected: zero failures and a clean diff check.
