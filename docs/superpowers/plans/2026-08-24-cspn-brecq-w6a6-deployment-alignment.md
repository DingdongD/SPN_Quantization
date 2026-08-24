# CSPN BRECQ W6A6 Deployment Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct CSPN BRECQ W6A6 with deployment-identical A6 inputs and produce a strict fixed-64 NYU result without clipping or fallback.

**Architecture:** Reuse the existing joint QDrop block reconstructor with an explicit algorithm contract. QDrop uses probability 0.5; deterministic BRECQ uses probability 1.0 and exports the same exact weight/activation deployment schema under a distinct method identity.

**Tech Stack:** Python 3.11, PyTorch, CUDA, pytest, official CSPN NYU model, existing PA propagation simulator.

---

### Task 1: Lock Algorithm Semantics

**Files:**
- Modify: `tests/test_qdrop_reconstruction_runner.py`
- Modify: `tests/test_qdrop_reconstruction.py`
- Modify: `scripts/run_nyu_qdrop_reconstruction.py`

- [ ] Add failing tests requiring `--algorithm qdrop|brecq`, QDrop probability
  `0.5`, BRECQ probability `1.0`, and deterministic all-quantized input mixing.
- [ ] Run `python -m pytest -q tests/test_qdrop_reconstruction.py tests/test_qdrop_reconstruction_runner.py` and verify the new tests fail for the missing algorithm contract.
- [ ] Add direct algorithm validation and derive the reconstruction probability
  from the selected algorithm. Thread the algorithm through output labels,
  metadata, and optimizer configuration.
- [ ] Rerun the focused tests and commit the passing change.

### Task 2: Preserve Strict Method Identity

**Files:**
- Modify: `tests/test_qdrop_contract.py`
- Modify: `tests/test_edge_runner.py`
- Modify: `spn_quant/qdrop_contract.py`
- Modify: `scripts/run_nyu_edge_quantization.py`

- [ ] Add failing tests for distinct `qdrop_strict` and
  `brecq_joint_strict` contracts and reject method/runtime mismatches.
- [ ] Run `python -m pytest -q tests/test_qdrop_contract.py tests/test_edge_runner.py` and verify RED.
- [ ] Parameterize the existing joint contract builder and edge loader with
  the strict method while retaining the exact W6A6 activation contract.
- [ ] Rerun the focused tests and commit the passing change.

### Task 3: Separate Invalid-Depth Causes

**Files:**
- Modify: `tests/test_qdrop_w4a4_evaluation.py`
- Modify: `scripts/run_nyu_qdrop_w4a4.py`
- Modify: `scripts/run_nyu_qdrop_reconstruction.py`

- [ ] Add failing metric tests distinguishing non-finite pixels from finite
  predictions no greater than `1e-4`; both conditions must produce strict
  infinite RMSE.
- [ ] Run `python -m pytest -q tests/test_qdrop_w4a4_evaluation.py tests/test_qdrop_reconstruction_runner.py` and verify RED.
- [ ] Record `nonfinite_pixels`, `nonpositive_pixels`, `invalid_pixels`, and
  `prediction_min` without replacing prediction values.
- [ ] Rerun focused tests and commit the passing change.

### Task 4: Pilot The Deployment-Aligned Reconstruction

**Files:**
- Generated: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_brecq_w6a6_aligned_64/pilot/**`

- [ ] Run focused tests for reconstruction, contract replay, edge ownership,
  and PA propagation.
- [ ] Run a short CSPN W6A6 BRECQ pilot on an idle A100 using the persisted
  calibration and evaluation protocol.
- [ ] Verify the contract is W6A6, probability is 1.0, all target blocks are
  reconstructed in execution order, and validation diagnostics are finite.
- [ ] Use pilot evidence only to validate the path; do not report pilot RMSE as
  the formal result.

### Task 5: Formal Reconstruction And Fixed-64 Evaluation

**Files:**
- Generated: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_brecq_w6a6_aligned_64/formal/**`
- Create: `docs/2026-08-24-cspn-brecq-w6a6-aligned-results.md`

- [ ] Run 20,000-step deterministic BRECQ W6A6 reconstruction on the exact
  128-sample calibration set.
- [ ] Replay the exported contract with the existing PA evaluator on the exact
  fixed 64 samples.
- [ ] Audit sample identity, checkpoint hash, bit contract, method identity,
  invalid counters, and prediction files.
- [ ] Compare strict RMSE with PA-RTN W8A8 and retained QDrop W6A6, write the
  measured result document, and commit code plus documentation only.

### Task 6: Final Verification

**Files:**
- Modify only if a test exposes a regression.

- [ ] Run `python -m pytest -q`.
- [ ] Confirm no required process remains running and inspect `git diff` for
  unrelated changes.
- [ ] Record exact commands, test counts, artifact root, and Git commit in the
  final response.
