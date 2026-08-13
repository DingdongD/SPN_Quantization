# CSPN SmoothQuant Group-A4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement strict SmoothQuant plus Group-A4 for CSPN and evaluate its activation benefit, W4 cost, and end-to-end accuracy.

**Architecture:** Extend the hardware instrumentor with explicit transformed activation ranges compatible with tensor/channel/group QuantSpec. Add a dedicated CSPN runner that reuses the audited model, dataset, propagation, diagnostics, and prediction contracts while leaving the existing 15-configuration study unchanged.

**Tech Stack:** Python, PyTorch, NumPy, CUDA, Matplotlib, pytest.

---

### Task 1: Group-aware SmoothQuant ranges

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [x] Write a failing test that combines a group QuantSpec with SmoothQuant
  channel maxima and asserts one transformed maximum per contiguous group.
- [x] Run the test and verify the current scalar maximum fails the group count
  contract.
- [x] Implement transformed channel maxima aggregation according to the
  QuantSpec granularity and pass the resulting range to the quantizer.
- [x] Verify tensor, group, and channel SmoothQuant range tests pass.

### Task 2: CSPN SmoothQuant experiment contracts

**Files:**
- Create: `scripts/run_nyu_cspn_smoothquant_group.py`
- Create: `tests/test_run_nyu_cspn_smoothquant_group.py`
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`

- [x] Write failing tests for the exact configuration matrix, eligible input
  modules, transformed group ranges, calibration-only selection, and strict
  output coverage.
- [x] Add a public configuration path to the existing CSPN quantized setup that
  accepts SmoothQuant maxima and alpha without changing existing configurations.
- [x] Implement the dedicated runner using the same checkpoint/data/propagation
  setup and output schemas as the activation-resolution runner.
- [x] Verify all runner contract tests pass.

### Task 3: Real CUDA evaluation and report

**Files:**
- Create: `scripts/plot_cspn_smoothquant_group.py`
- Create: `tests/test_plot_cspn_smoothquant_group.py`
- Create: `docs/2026-08-13-cspn-smoothquant-group-results.md`

- [x] Run 128-sample calibration and all declared configurations on the fixed
  64-sample NYU evaluation subset.
- [x] Audit finite outputs, sample identity, activation/weight diagnostics, and
  prediction payload coverage.
- [x] Generate RMSE versus scale-count and prediction/error comparison figures.
- [x] Record whether activation SQNR/new-zero improvement exceeds transformed
  W4 degradation, then run the full test suite and commit source and report.
