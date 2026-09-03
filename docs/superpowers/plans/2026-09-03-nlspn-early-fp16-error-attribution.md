# NLSPN Early FP16 Error Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit IEEE FP16 QDQ and measure whether early activation or weight protection recovers NLSPN FP6 error.

**Architecture:** Extend the existing scaled floating-point quantizer factory with one native IEEE FP16 QDQ implementation. Reuse the corrected NLSPN activation-protection evaluator with a dedicated candidate matrix and immutable result root, preserving one QDQ owner and FP16 propagation.

**Tech Stack:** Python 3.7/3.11, PyTorch, official NLSPN/DCN, pytest, CSV/JSON.

---

### Task 1: Add Explicit IEEE FP16 QDQ

**Files:**
- Modify: `spn_quant/fp_formats.py`
- Modify: `tests/test_fp_formats.py`

- [ ] Write a failing test that requires `make_quantizer("fp16_ieee", ...)` to equal `tensor.half().float()`, report 16 bits, preserve shape, and expose zero/saturation accounting.
- [ ] Run `PYTHONPATH=. pytest -q tests/test_fp_formats.py -k fp16_ieee` and confirm the unsupported-format failure.
- [ ] Add `FPFormatSpec("fp16_ieee", 16, 65504.0)` and an `IEEEFP16Quantizer` whose QDQ is an explicit FP16 cast without calibration scaling.
- [ ] Run the focused test and the complete `tests/test_fp_formats.py` suite.

### Task 2: Add The Early Attribution Matrix

**Files:**
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] Write failing tests for the exact seven-candidate order and exact early weight/activation formats.
- [ ] Require the baseline to be the measured `EARLY_W6A8` assignment, with only the three early inputs changed by later candidates.
- [ ] Implement FP16 weight and activation assignments, weighted format budgets, and effective-weight checks without changing propagation ownership.
- [ ] Run the candidate and ownership tests.

### Task 3: Add Strict Attribution Artifacts

**Files:**
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] Write failing tests for activation, weight, interaction, and residual attribution formulas.
- [ ] Add `run_early_fp16_attribution()` using the existing paired capture, depth aggregation, propagation signal metrics, and effective weight diagnostics.
- [ ] Write `summary.csv`, `sample_metrics.csv`, `module_diagnostics.csv`, `effective_weight_metrics.csv`, `propagation_signal_metrics.csv`, `propagation_state_metrics.csv`, `attribution.csv`, and `manifest.json`.
- [ ] Add the `early-fp16-attribution` CLI choice and run schema tests.

### Task 4: Verify And Run Official CUDA Evaluation

**Files:**
- Create: `docs/results/2026-09-03-nlspn-early-fp16-error-attribution-results.md`
- Create runtime output: `profile_logs/nyu_nlspn_early_fp16_error_attribution_64_v1/`

- [ ] Run Python 3.11 and official Python 3.7 focused regressions, `py_compile`, and `git diff --check`.
- [ ] Verify GPU 1 and the official Python 3.7 DCN extension.
- [ ] Run all seven candidates with the fixed 128/64 protocol.
- [ ] Validate finite metrics, paired reproducibility, 18 states, exact sample identities, expected effective formats, and artifact schemas.
- [ ] Document the measured RMSE deltas and error-source conclusion without deleting the preceding activation-protection result.
