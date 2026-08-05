# Activation Bit Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate sensitivity-driven W4 mixed-A4/A8 configurations for the official CSPN, DySPN, NLSPN, and CompletionFormer checkpoints.

**Architecture:** Extend the hardware-aligned QDQ instrumentor with per-module activation-bit overrides while retaining W4 per-output-channel weights and per-tensor activations. Select candidate modules from measured W4A4 input SQNR, evaluate single-site and grouped A8 upgrades on the fixed 64 validation samples, then rank configurations by invalid-rate reduction and RMSE gain per additional activation bit traffic. Propagation remains an FP32 island in the baseline; separate A8/A16 state experiments measure whether that island can be reduced.

**Tech Stack:** Python, PyTorch forward hooks, NumPy, Matplotlib, unittest, existing NYU quantization runner.

---

### Task 1: Per-Module Activation Bits

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [ ] Add failing tests showing that one selected module uses A8 while other modules remain A4.
- [ ] Add a failing test showing that a fused ReLU follows its producer module's activation override.
- [ ] Add `activation_bit_overrides` to `HardwareAlignedInstrumentor.configure`.
- [ ] Record effective activation bits in the hardware manifest.
- [ ] Run the focused hardware quantization tests in base and `completionformer-py37`.

### Task 2: Mixed-Precision Configuration Builder

**Files:**
- Create: `scripts/activation_bit_allocation.py`
- Create: `tests/test_activation_bit_allocation.py`
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`

- [ ] Add failing tests for finite lowest-SQNR candidate selection with module de-duplication.
- [ ] Add failing tests for baseline, single-site A8, head A8, group A8, top-k A8, full A8, and state A8/A16 configurations.
- [ ] Add the `mixed` backend and sensitivity-source CLI option to the runner.
- [ ] Persist each configuration's overridden modules and state precision in `mixed_precision_configs.csv`.
- [ ] Run focused runner and allocation tests.

### Task 3: Allocation Cost and Ranking

**Files:**
- Modify: `scripts/activation_bit_allocation.py`
- Modify: `tests/test_activation_bit_allocation.py`

- [ ] Add failing tests for finite/nonfinite pixel aggregation.
- [ ] Add failing tests for incremental activation-bit traffic and benefit-per-cost ranking.
- [ ] Implement ranking with invalid-rate reduction as the primary CSPN objective and pooled RMSE reduction for valid models.
- [ ] Emit `allocation_summary.csv`, `allocation_ranking.csv`, and a hardware-cost caveat for Conv/Linear-boundary traffic.

### Task 4: Four-Model Fixed-64 Evaluation

**Files:**
- Generate: `profile_logs/nyu_activation_bit_allocation/<model>/*`

- [ ] Run the official CSPN and DySPN checkpoints in the base environment.
- [ ] Run official NLSPN and CompletionFormer checkpoints in `completionformer-py37` on `cuda:0` so the official DCN extension is used.
- [ ] Verify common 128 calibration and fixed 64 evaluation indices.
- [ ] Verify every requested configuration has 64 sample rows.

### Task 5: Plots and Findings

**Files:**
- Create: `scripts/plot_activation_bit_allocation.py`
- Create: `tests/test_plot_activation_bit_allocation.py`
- Generate: `profile_logs/nyu_activation_bit_allocation/*.png`
- Generate: `profile_logs/nyu_activation_bit_allocation/activation_bit_allocation_findings.md`

- [ ] Add failing tests for configuration labels, cost normalization, and best-valid selection.
- [ ] Plot RMSE/nonfinite rate against added activation traffic and per-model Pareto frontiers.
- [ ] Plot selected mixed-precision module maps using Arial-compatible fonts, no title, and grid behind data.
- [ ] Document whether each model benefits from sparse A8 allocation and whether propagation state can leave FP32.

### Task 6: Verification and Cleanup

**Files:**
- Verify all modified scripts and generated formal outputs.

- [ ] Run all relevant tests in base and `completionformer-py37`.
- [ ] Inspect every generated figure.
- [ ] Remove preflight-only outputs while preserving formal fixed-64 results.
- [ ] Commit only files belonging to this feature.
