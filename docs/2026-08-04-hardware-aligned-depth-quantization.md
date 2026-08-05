# Hardware-Aligned Depth Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a standard-backend-aligned W4A4/W4A8 PTQ simulator for four official NYU depth-completion models.

**Architecture:** Add a focused hardware quantization module for signed/unsigned activation QDQ, Conv-BN folding, shared branch requantization, and INT32 bias QDQ. Integrate it as separate experiment configurations so existing unfused RTN results remain reproducible.

**Tech Stack:** PyTorch, CUDA QDQ simulation, NumPy, pytest, Matplotlib.

---

### Task 1: Integer Quantization Primitives

**Files:**
- Create: `scripts/hardware_aligned_quantization.py`
- Create: `tests/test_hardware_aligned_quantization.py`

- [ ] Write failing tests asserting W4 signed activation codes use `[-7, 7]`,
  ReLU activation codes use `[0, 15]`, weight scales are per output channel,
  and bias QDQ uses `sx * sw[o]` with INT32 rounding.
- [ ] Run `python -m pytest -q tests/test_hardware_aligned_quantization.py`
  and confirm failures are caused by missing primitives.
- [ ] Implement `SymmetricActivationQuantizer`,
  `UnsignedActivationQuantizer`, `symmetric_weight_qdq`, and
  `int32_bias_qdq` with saturation and SQNR statistics.
- [ ] Re-run the test file and confirm all primitive tests pass.

### Task 2: Conv-BN Folding Before Calibration

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [ ] Write a failing toy-model test that records one Conv-BN pair, folds it,
  checks FP32 equivalence within `1e-5`, and verifies observers are empty before
  the folded model is calibrated.
- [ ] Implement execution-based Conv-BN pair discovery, module replacement,
  folded-pair manifest rows, and an error for ambiguous producers.
- [ ] Verify the fold test and primitive tests pass.

### Task 3: ReLU and Shared Merge Quantization

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`
- Create: `scripts/hardware_merge_adapters.py`
- Create: `tests/test_hardware_merge_adapters.py`

- [ ] Write failing tests for explicit post-ReLU unsigned QDQ and two unequal
  branches sharing one MinMax scale before Add and Concat.
- [ ] Implement call-indexed ReLU observers and `SharedMergeQuantizer`.
- [ ] Implement model-family adapters for CSPN UpProj Cat/Add, DySPN `_concat`,
  NLSPN `_concat`, and CompletionFormer `_concat`; adapters expose expected and
  installed merge counts.
- [ ] Verify adapters fail closed when expected merge sites are absent.

### Task 4: Hardware-Aligned Instrumentor

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [ ] Write a failing Conv-BN-ReLU-Conv test that checks operation order:
  fold, observe, freeze, then configure.
- [ ] Implement `HardwareAlignedInstrumentor` with Conv/Linear pre/post hooks,
  explicit ReLU hooks, folded bias QDQ, statistics, manifest export, and clean
  restoration between configurations.
- [ ] Confirm folded FP32 and disabled instrumentor output equivalence.

### Task 5: Experiment Driver Integration

**Files:**
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`

- [ ] Write failing tests for `HW_W4A8_full` and `HW_W4A4_full` configuration
  metadata and safe append behavior.
- [ ] Add `--quant-backend hardware` setup that performs dry-run discovery and
  folding before the 128-sample calibration loop.
- [ ] Persist hardware manifest, folded-equivalence error, bias scales, merge
  ranges, and the existing metric tables under distinct configuration names.
- [ ] Run one-sample CSPN preflight and reject nonfinite output.

### Task 6: Plot and Report Integration

**Files:**
- Modify: `scripts/plot_nyu_rtn_quantization.py`
- Modify: `tests/test_plot_nyu_rtn_quantization.py`

- [ ] Write failing aggregation tests that preserve existing RTN rows and add
  hardware-aligned W4A8/W4A4 rows in deterministic order.
- [ ] Add RMSE, signal SQNR, regional error, and prediction panels comparing
  unfused and hardware-aligned results.
- [ ] Ensure nonfinite rates remain explicit in plots and CSV files.

### Task 7: Four-Model Evaluation and Verification

**Files:**
- Output: `profile_logs/nyu_hardware_aligned_quantization/`

- [ ] Run CSPN-24, DySPN-9, NLSPN-12, and CompletionFormer-6 on the fixed 64
  validation samples and 128 calibration samples.
- [ ] Verify each model/config has 64 unique rows and prediction files and that
  all metadata indices match.
- [ ] Run `python -m pytest -q tests` and
  `conda run -n completionformer-py37 python -m pytest -q tests`.
- [ ] Run `git diff --check`, inspect every generated plot, and summarize which
  accuracy changes come from BN folding, shared merge scales, W4 weights, and
  A4 activations.
