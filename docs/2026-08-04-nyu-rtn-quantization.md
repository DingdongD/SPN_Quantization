# NYU RTN Quantization Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible 64-sample FP32 comparison and module-level W8A8/W4A4 RTN sensitivity analysis for four converged NYU depth-completion models.

**Architecture:** A framework-independent QDQ engine instruments existing Conv2d and Linear modules with hooks, preserving all official model structures and custom CUDA operators. Architecture-aware grouping, paired FP32/quantized diagnostics, and propagation adapters separate feature quantization damage from recurrent state damage.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, pytest, official CSPN/DySPN/NLSPN/CompletionFormer implementations

---

### Task 1: Random-64 Contact Sheet

**Files:**
- Create: `scripts/plot_nyu_random64_predictions.py`
- Create: `tests/test_plot_nyu_random64_predictions.py`

- [ ] **Step 1: Write failing tests for model-label ordering, complete index matching, and the 16 by 4 sample-block layout.**
- [ ] **Step 2: Run `pytest -q tests/test_plot_nyu_random64_predictions.py` and verify failures are caused by the missing plotting module.**
- [ ] **Step 3: Implement CSV/NPZ discovery, strict shared-index validation, metric aggregation, and the five-panel sample-block renderer.**
- [ ] **Step 4: Run the focused tests and verify they pass.**
- [ ] **Step 5: Render PNG and PDF contact sheets from `profile_logs/nyu_prediction_random64_fp32/predictions/` and verify dimensions and nonblank pixels.**

### Task 2: RTN-QDQ Core

**Files:**
- Create: `scripts/rtn_quantization.py`
- Create: `tests/test_rtn_quantization.py`

- [ ] **Step 1: Write failing tests for symmetric per-output-channel weight RTN, asymmetric activation RTN, calibration freezing, saturation accounting, and exact FP32 bypass.**
- [ ] **Step 2: Run `pytest -q tests/test_rtn_quantization.py` and verify the expected import failure.**
- [ ] **Step 3: Implement observers, QDQ functions, accumulated error statistics, and hook-based Conv2d/Linear instrumentation compatible with both active PyTorch environments.**
- [ ] **Step 4: Run the focused tests and verify numerical expectations at 8 and 4 bits.**

### Task 3: Model Grouping and Signal Diagnostics

**Files:**
- Create: `scripts/nyu_quantization_analysis.py`
- Create: `tests/test_nyu_quantization_analysis.py`

- [ ] **Step 1: Write failing tests for disjoint architecture-aware module groups and tensor diagnostics including SQNR, cosine, sign flips, affinity-neighbor changes, and offset endpoint error.**
- [ ] **Step 2: Run `pytest -q tests/test_nyu_quantization_analysis.py` and verify failures identify missing behavior.**
- [ ] **Step 3: Implement model configuration loading through `train_nyu_iteration_sweep.py`, deterministic train calibration sampling, validation sampling, grouping rules, paired output extraction, and regional depth metrics.**
- [ ] **Step 4: Run focused tests and inspect group manifests to ensure every eligible Conv2d/Linear belongs to exactly one group.**

### Task 4: Propagation-State Stress Adapters

**Files:**
- Create: `scripts/propagation_quantization.py`
- Create: `tests/test_propagation_quantization.py`

- [ ] **Step 1: Write failing toy-loop tests proving that state QDQ is applied after every iteration and that disabled adapters reproduce FP32 exactly.**
- [ ] **Step 2: Run `pytest -q tests/test_propagation_quantization.py` and verify the expected failures.**
- [ ] **Step 3: Implement adapters for CSPN, DySPN, and the shared NLSPN-style propagation module without modifying external repositories.**
- [ ] **Step 4: Run toy tests and one-sample CUDA smoke tests for all four official models.**

### Task 5: Quantization Experiment and Plots

**Files:**
- Create: `scripts/run_nyu_rtn_quantization.py`
- Create: `scripts/plot_nyu_rtn_quantization.py`
- Create: `tests/test_plot_nyu_rtn_quantization.py`

- [ ] **Step 1: Write failing tests for result-table schemas and deterministic plot ordering.**
- [ ] **Step 2: Implement experiment orchestration for FP32, full W8A8/W4A4, group-only W8A8/W4A4, and full quantization plus propagation-state QDQ.**
- [ ] **Step 3: Add resumable per-model JSON/CSV outputs and quantized prediction NPZ exports.**
- [ ] **Step 4: Implement accuracy-delta, layer-damage, regional-error, and propagation-drift figures using Arial-compatible typography.**
- [ ] **Step 5: Run the four-model CUDA analysis, render all plots, and summarize which signal classes dominate degradation.**

### Task 6: Verification

**Files:**
- Modify only files created by Tasks 1-5 if verification exposes defects.

- [ ] **Step 1: Run all new focused tests in the base environment.**
- [ ] **Step 2: Run compatibility tests in `completionformer-py37`.**
- [ ] **Step 3: Run existing repository tests and Python compilation checks.**
- [ ] **Step 4: Verify all CSV row counts, the shared 64-index manifest, image dimensions, nonblank image pixels, and absence of active experiment processes.**
- [ ] **Step 5: Record final metric deltas and evidence-based conclusions in the completion report.**
