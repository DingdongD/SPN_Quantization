# CSPN Outlier Channel Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose contiguous Group-8 outlier harm and evaluate calibration-selected `1+7` unsigned A4 outlier channel isolation on official CSPN.

**Architecture:** A focused OCI module computes per-group outlier candidates and applies explicit per-channel scale overrides without changing channel order or weights. The existing grouped activation quantizer accepts declared channel-scale overrides, while a dedicated runner collects calibration-only harm statistics, selects cumulative OCI budgets, and reuses the strict CSPN 64-sample evaluation path.

**Tech Stack:** Python, PyTorch, CUDA, NumPy, pytest.

---

### Task 1: OCI primitives and grouped quantizer support

**Files:**
- Create: `spn_quant/outlier_channel_isolation.py`
- Create: `tests/test_outlier_channel_isolation.py`
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [x] **Step 1: Write failing primitive tests**

Test a known Group8 where channel 3 is the maximum. Assert exact outlier and
second maximum, thresholds `M/30`, independent outlier scale `M/15`, shared
victim scale `M_second/15`, unchanged channel order, and rejection of invalid
group size, non-finite maxima, duplicate declarations, or a declared channel
outside its contiguous group.

- [x] **Step 2: Run primitive tests and verify RED**

```bash
python -m pytest tests/test_outlier_channel_isolation.py -q
```

Expected: import failure because the OCI module does not exist.

- [x] **Step 3: Implement minimal OCI records and scale construction**

Implement frozen candidate and declaration records plus functions that build
one candidate per contiguous Group8 and replace the selected channel scale in
original order. Do not add permutation, splitting, padding, or fallback logic.

- [x] **Step 4: Write failing instrumentor tests**

Assert that an explicit OCI declaration changes only the declared local QDQ
owner scales, including ReLU outputs, preserves unsigned A4 codes and channel
order, leaves W4 weights identical, and rejects non-Group8 specs, signed sites,
unknown modules, and SmoothQuant/permutation combinations.

- [x] **Step 5: Integrate explicit OCI declarations**

Add `activation_isolations` to instrumentor configuration. Build the ordinary
GroupedActivationQuantizer first, then replace its expanded channel scales
with the strict `1+7` scale vector through a dedicated channel-scale uniform
quantizer. Preserve all existing behavior when the explicit mapping is empty.

- [x] **Step 6: Run focused tests and commit**

```bash
python -m pytest tests/test_outlier_channel_isolation.py \
  tests/test_hardware_aligned_quantization.py -q
```

Commit only Task 1 files.

### Task 2: Calibration harm accumulation and budget selection

**Files:**
- Create: `scripts/run_nyu_cspn_outlier_channel_isolation.py`
- Create: `tests/test_run_nyu_cspn_outlier_channel_isolation.py`
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`
- Modify: `tests/test_run_nyu_cspn_activation_resolution.py`

- [x] **Step 1: Write failing harm-accumulator tests**

Use two synthetic NCHW batches and verify, for every candidate, exact
`H_o`, rescued element count, rescued energy, affected victim count, per-victim
rescue rate, and `M_o/M_second`. Exclude exact zeros and the outlier channel.

- [x] **Step 2: Implement the calibration-only accumulator**

Record element and squared-energy counts in the interval
`M_second/30 <= x < M_o/30`. Serialize one candidate row and one victim row per
positive-harm relation. Direct dictionary indexing and explicit configuration
fields are required.

- [x] **Step 3: Write failing budget-selection tests**

Verify stable ranking by rescued energy, rescued count, module, group index,
and channel. Verify cumulative budgets contain zero, fixed small prefixes, and
all positive-harm candidates without using evaluation metrics.

- [x] **Step 4: Add OCI configuration plumbing**

Add the explicit isolation tuple to CSPN configurations and pass it directly
to the instrumentor. Ordinary configurations declare an empty tuple. There is
no runtime default or inferred declaration.

- [x] **Step 5: Run runner contract tests and commit**

```bash
python -m pytest tests/test_run_nyu_cspn_outlier_channel_isolation.py \
  tests/test_run_nyu_cspn_activation_resolution.py -q
```

Commit only Task 2 files.

### Task 3: Real CSPN calibration and CUDA evaluation

**Files:**
- Create: `profile_logs/nyu_cspn_outlier_channel_isolation_64/`
- Create: `docs/2026-08-13-cspn-outlier-channel-isolation-results.md`

- [x] **Step 1: Run fixed real-data calibration**

Use the official converged `cspn_iter24/best.pt`, seed `20260812`, the existing
128 calibration identities, contiguous Group8 MinMax, and no evaluation-driven
selection. Write candidate and victim harm tables before evaluation starts.

- [x] **Step 2: Evaluate deterministic OCI budgets**

Run the contiguous baseline and cumulative isolation budgets on the fixed 64
NYU evaluation identities with W4A4, FP32 bias/guidance, and A8/Q13/INT32
propagation. Write aggregate, sample, activation, block, propagation, manifest,
and metadata artifacts.

- [x] **Step 3: Audit artifacts and analyze outcomes**

Verify exact sample identity coverage, finite predictions, candidate ranking,
scale overhead, activation error closure, unchanged W4 reconstruction, and no
permutation or channel-count change. Report RMSE versus isolated-channel and
additional-scale fractions and identify whether high `H_o` predicts endpoint
improvement.

- [x] **Step 4: Run full regression verification and commit**

```bash
python -m pytest -q
git diff --check -- . ':(exclude)tests/test_qdrop_reconstruction.py'
```

Commit implementation, tests, and result report while leaving the unrelated
`tests/test_qdrop_reconstruction.py` modification untouched.
