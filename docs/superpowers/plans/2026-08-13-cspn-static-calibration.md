# CSPN Group-A4 Static Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare MinMax, P99.9, P99.99, and Histogram-MSE static activation calibration on official CSPN W4A4 Group-8.

**Architecture:** Add one focused histogram calibration module with vectorized grouped updates and deterministic threshold derivation. Connect it to the existing hardware instrumentor and CSPN rotation controller through explicit FP-reference calibration hooks and explicit range overrides, then reuse the audited CSPN evaluation path in a dedicated runner.

**Tech Stack:** Python, PyTorch, CUDA, NumPy, Matplotlib, pytest.

---

### Task 1: Grouped histogram calibration core

**Files:**
- Create: `spn_quant/static_calibration.py`
- Create: `tests/test_static_calibration.py`

- [x] Write failing tests for one-update grouped bin counts, multi-update accumulation, non-finite rejection, and exact zero entities.
- [x] Run `python -m pytest tests/test_static_calibration.py -q` and verify the module is missing.
- [x] Implement `GroupedHistogramObserver` with one combined group/bin `torch.bincount` per update and strict channel/range validation.
- [x] Write failing tests for P99.9/P99.99 boundary selection, signed/unsigned Histogram-MSE, and larger-threshold tie breaking.
- [x] Implement normalized histogram error matrices and threshold derivation for `minmax`, `percentile_p999`, `percentile_p9999`, and `hist_mse`.
- [x] Run the focused tests and commit the calibration core.

### Task 2: Strict activation-site integration

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `spn_quant/rotation.py`
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`
- Modify: `tests/test_hardware_aligned_quantization.py`
- Modify: `tests/test_rotation.py`

- [x] Write failing tests proving FP observation forwards ordinary input/output/ReLU references to a calibration recorder without QDQ.
- [x] Add explicit set/clear calibration-recorder APIs and call them only in observe mode.
- [x] Write failing tests proving group range overrides configure both ordinary activation quantizers and rotation-boundary quantizers exactly.
- [x] Add strict range-override coverage and shape validation. Do not add fallback to MinMax when an override is missing.
- [x] Run hardware, rotation, and CSPN activation-resolution regression tests and commit integration.

### Task 3: CSPN static-calibration runner

**Files:**
- Create: `scripts/run_nyu_cspn_static_calibration.py`
- Create: `tests/test_run_nyu_cspn_static_calibration.py`

- [x] Write failing tests for the exact four-configuration matrix, 71-site/1673-scale contract, static-only settings, threshold manifest, and exact sample/prediction coverage.
- [x] Implement first-pass MinMax observation and second-pass 2048-bin histogram collection over the same 128 ordered samples.
- [x] Derive ordinary and rotation overrides for each method and run all four W4A4 Group-8 configurations through the existing block, regional, propagation, and activation diagnostics.
- [x] Export strict CSV/JSON artifacts and prediction payloads without using evaluation metrics during calibration.
- [x] Run runner contract tests and commit the runner.

### Task 4: Real CUDA experiment and analysis

**Files:**
- Create: `scripts/plot_cspn_static_calibration.py`
- Create: `tests/test_plot_cspn_static_calibration.py`
- Create: `docs/2026-08-13-cspn-static-calibration-results.md`

- [x] Run the official checkpoint with 128 real NYU calibration samples and fixed 64-sample evaluation on an available A100.
- [x] Audit 256 finite sample rows, exact sample identities, threshold coverage, 71 activation sites, 1673 scales, and prediction payload coverage.
- [x] Generate RMSE, activation-error composition, and prediction/error comparison figures from completed artifacts.
- [x] Record whether any non-MinMax method improves RMSE without material regression in secondary and propagation metrics.
- [x] Run the complete repository test suite, commit code/report, and leave the unrelated QDrop test edit untouched.
