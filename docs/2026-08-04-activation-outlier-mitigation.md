# Activation Outlier and Mitigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile activation tails and evaluate percentile, SmoothQuant, and AWQ-style mitigations on four official NYU depth-completion models.

**Architecture:** Add a bounded deterministic activation sampler with exact per-channel maxima, reuse the hardware PTQ preparation and dataset path, and keep mitigation runs as separate configurations. Aggregate results into auditable CSV files and static Matplotlib figures.

**Tech Stack:** PyTorch, CUDA, NumPy, pytest, Matplotlib.

---

### Task 1: Distribution Statistics

**Files:**
- Create: `scripts/activation_outlier_analysis.py`
- Create: `tests/test_activation_outlier_analysis.py`

- [ ] Write failing tests for monotonic percentiles, tail ratios, deterministic
  bounded sampling, and exact per-input-channel maxima.
- [ ] Run `python -m pytest -q tests/test_activation_outlier_analysis.py` and
  confirm failure because the module is absent.
- [ ] Implement `BoundedActivationSampler`, percentile row generation, and
  channel-outlier statistics.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Model Profiler and Occupancy

**Files:**
- Modify: `scripts/activation_outlier_analysis.py`
- Modify: `tests/test_activation_outlier_analysis.py`

- [ ] Write failing toy-model tests for call-indexed Conv/Linear input/output
  hooks and occupancy shares summing to one.
- [ ] Implement `ActivationOutlierProfiler`, hardware model preparation reuse,
  CSV output, and occupancy aggregation from layer quantization metrics.
- [ ] Run focused tests and `python -m py_compile`.
- [ ] Commit the profiler implementation and tests.

### Task 3: Four-Model Distribution Run

**Files:**
- Output: `profile_logs/nyu_activation_outliers/<model>/activation_percentiles.csv`
- Output: `profile_logs/nyu_activation_outliers/encoder_occupancy.csv`

- [ ] Run CSPN-24 and DySPN-9 with base Python on an available CUDA device.
- [ ] Run NLSPN-12 and CompletionFormer-6 in `completionformer-py37` with the
  custom CUDA extension visible as `cuda:0`.
- [ ] Verify each model uses 128 identical calibration indices and all
  percentile rows are monotonic and finite.

### Task 4: Mitigation Primitives

**Files:**
- Create: `scripts/outlier_mitigation_quantization.py`
- Create: `tests/test_outlier_mitigation_quantization.py`
- Modify: `scripts/hardware_aligned_quantization.py`

- [ ] Write failing tests for percentile clipping, SmoothQuant FP32
  equivalence, SmoothQuant scale shape, and AWQ-style clipping.
- [ ] Implement percentile quantizer construction, local SmoothQuant parameter
  transforms, restoration, and per-output-channel W4 clipping.
- [ ] Integrate optional mitigation policies into the hardware instrumentor
  without changing existing MinMax behavior.
- [ ] Run existing and new quantization tests, then commit.

### Task 5: Fixed-64 Mitigation Evaluation

**Files:**
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`
- Output: `profile_logs/nyu_activation_outliers/<model>/mitigation_metrics.csv`

- [ ] Add distinct configuration names for percentile, SmoothQuant alpha
  sweep, and AWQ clipping controls.
- [ ] Run the ranked candidate set first on one sample and reject unexpected
  nonfinite behavior.
- [ ] Run the retained candidates on all fixed 64 samples for four models.
- [ ] Verify row counts, indices, and prediction exports.

### Task 6: Plots and Findings

**Files:**
- Create: `scripts/plot_activation_outlier_analysis.py`
- Create: `tests/test_plot_activation_outlier_analysis.py`
- Output: `profile_logs/nyu_activation_outliers/activation_outlier_findings.md`

- [ ] Write failing aggregation tests for severity ranking and mitigation
  comparison ordering.
- [ ] Implement encoder occupancy, percentile-tail, channel-outlier, and RMSE
  plots with Arial-compatible font settings.
- [ ] Generate the findings report, inspect all plots, and state deployment
  limitations of local SmoothQuant scaling.
- [ ] Run both Python-environment test suites and `git diff --check`.

