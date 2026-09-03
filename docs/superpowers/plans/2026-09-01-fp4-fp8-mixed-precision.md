# FP4/FP8 Mixed-Precision Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict FP4-E2M1 and FP8-E4M3FN fake quantization to the existing four-model quantization framework and evaluate mixed precision under the unified FP16 propagation protocol.

**Architecture:** Add format-specific scalar fake quantizers and contract-driven format assignments. Reuse the existing four-model runtime, activation-owner contract, calibration metadata, propagation exclusion, evaluation metrics, and manifest structure. The first implementation is PTQ/fake-quant only; it does not modify the QAT controller.

**Tech Stack:** Python, PyTorch, existing `spn_quant` contracts, official CUDA model environments, pytest.

---

### Task 1: Define format quantizer contracts

**Files:**
- Create: `spn_quant/fp_formats.py`
- Test: `tests/test_fp_formats.py`

- [ ] **Step 1: Write failing tests** for exact FP4 E2M1 nearest-code behavior, FP8 E4M3FN conversion, positive finite scales, symmetric clipping, and rejection of unsupported formats.
- [ ] **Step 2: Run `PYTHONPATH=. pytest -q tests/test_fp_formats.py` and verify the missing-module failure.
- [ ] **Step 3: Implement `FP4E2M1Quantizer` and `FP8E4M3FNQuantizer` with explicit format names, calibration, `quantize_with_codes`, saturation/zero statistics, and no fallback.
- [ ] **Step 4: Run the format tests and verify all pass.

### Task 2: Integrate format assignments with ordinary model contracts

**Files:**
- Create: `spn_quant/fp_mixed_precision.py`
- Modify: `spn_quant/model_contracts.py`
- Test: `tests/test_fp_mixed_precision.py`

- [ ] **Step 1: Write failing tests** for full FP4, full FP8, FP4-weight/FP8-activation, and sensitivity-promoted activation assignments; assert propagation owners are absent.
- [ ] **Step 2: Run the tests and verify the missing assignment/controller failure.
- [ ] **Step 3: Implement strict assignment validation, per-output-channel weight format application, per-owner activation format application, weighted format fractions, and propagation exclusion.
- [ ] **Step 4: Run the tests and verify all pass.

### Task 3: Add the unified four-model evaluation runner

**Files:**
- Create: `scripts/run_nyu_four_model_fp4_fp8.py`
- Create: `configs/four_model_fp4_fp8_mixed.json`
- Test: `tests/test_run_nyu_four_model_fp4_fp8.py`

- [ ] **Step 1: Write failing tests** for exact model order, FP16 propagation, 128/64 sample counts, configuration matrix, source checkpoint validation, and strict output collision behavior.
- [ ] **Step 2: Run the tests and verify the missing-runner failure.
- [ ] **Step 3: Implement calibration, sensitivity-ranked mixed activation selection, model preparation, ordinary-module fake quantization, 64-sample evaluation, and JSON/CSV manifests.
- [ ] **Step 4: Run runner unit tests and compile the runner.

### Task 4: Execute and verify the four-model experiment

**Files:**
- Output: `profile_logs/nyu_four_model_fp4_fp8_mixed_<date>/`
- Report: `docs/results/2026-09-01-four-model-fp4-fp8-results.md`

- [ ] **Step 1: Run the runner in each pinned model environment with the configured external roots.
- [ ] **Step 2: Verify four models, six configurations, 128 calibration identities, 64 evaluation identities, FP16 propagation, and finite/positive status.
- [ ] **Step 3: Compute pooled RMSE, relative FP32 loss, weighted format fractions, saturation, zero-code, and SQNR comparisons.
- [ ] **Step 4: Write the result report and run the complete focused regression suite.
