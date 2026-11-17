# Four-SPN RK3588 Compressed FP16 and INT8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure complete-forward RK3588 latency for the selected compressed CSPN, DySPN, NLSPN, and CompletionFormer topologies in both structural FP16 and RKNN-native W8A8 INT8, with propagation placement and numerical evidence.

**Architecture:** Freeze each selected architecture and state source before export. Produce one RKNN-safe FP16 graph and one calibration-derived W8A8 RKNN graph from the same ONNX topology; keep CPU fallback sampling in floating point as chosen by RKNN. Run fixed sample-386 controlled measurements and compare only candidates with identical input hashes and runtime versions.

**Tech Stack:** PyTorch/ONNX, ONNX Runtime, RKNN Toolkit2 2.5, RKNPU2 2.5, pytest, Orange Pi 5 Pro RK3588S.

**Status (2026-11-18):** Completed. The accepted five-process measurements,
numerical comparisons, and perf-detail evidence are published in
`reports/generated/rk3588_compressed_int8_20261117/`. The checklists below are
the original execution recipe and are retained as historical planning records.

---

### Task 1: Freeze the four deployment candidates

**Files:**
- Create: `spn_quant/rk3588_compressed_candidates.py`
- Create: `tests/test_rk3588_compressed_candidates.py`

- [ ] **Step 1: Write a failing inventory test** requiring exactly four models, declared NAS depths, structured candidate/state source, quantization source, and an explicit `rk3588_execution_precision` distinct from the research mixed-bit policy.
- [ ] **Step 2: Run** `PYTHONPATH=. pytest -q tests/test_rk3588_compressed_candidates.py` and confirm import failure.
- [ ] **Step 3: Implement immutable candidate records** for CSPN `drop_s4`, DySPN final standalone, NLSPN recovered `bridge62_s5hidden62_s4hidden_62p5pct`, and CompletionFormer final standalone. Fail closed on missing files and hashes.
- [ ] **Step 4: Run the test** and require all candidate sources to resolve.
- [ ] **Step 5: Commit** with `git commit -m "feat: freeze four compressed RK3588 candidates"`.

### Task 2: Export CSPN and NLSPN structural FP16 graphs

**Files:**
- Modify: `scripts/export_compressed_spn_onnx.py`
- Modify: `spn_quant/compressed_checkpoint.py`
- Create: `tests/test_export_four_compressed_spn_onnx.py`

- [ ] **Step 1: Write failing reconstruction tests** for CSPN baseline-state-plus-depth truncation and NLSPN recovered structured state.
- [ ] **Step 2: Run the focused tests** and confirm unsupported-model failures.
- [ ] **Step 3: Extend reconstruction and prediction wrappers** without changing DySPN/CompletionFormer behavior. Export fixed `rgb`, `dep` inputs and one `pred` output.
- [ ] **Step 4: Apply exact RKNN admission rewrites:** Sign replacement, ConvTranspose decomposition, and DCNv2-to-GridSample lowering. CSPN keeps its NPU convolutional propagation; NLSPN emits legacy and exact packed propagation graphs.
- [ ] **Step 5: Compare ONNX outputs to reconstructed PyTorch** and reject non-finite or tolerance-violating graphs.
- [ ] **Step 6: Commit** with `git commit -m "feat: export CSPN and NLSPN compressed graphs"`.

### Task 3: Build reproducible RKNN FP16 and W8A8 models

**Files:**
- Modify: `scripts/build_rk3588_spn_variants.py`
- Create: `scripts/materialize_rknn_calibration.py`
- Create: `tests/test_rknn_quantized_variants.py`

- [ ] **Step 1: Write failing tests** requiring `structural_fp16` and `structural_w8a8` variants, fixed 128-training-image calibration indices, two-input dataset rows, and separate research-policy versus executed-precision metadata.
- [ ] **Step 2: Run the tests** and confirm the W8A8 plan is absent.
- [ ] **Step 3: Materialize deterministic float32 calibration tensors** from the fixed training split and write SHA-256 hashes plus the RKNN dataset file.
- [ ] **Step 4: Build FP16** with `do_quantization=False` and W8A8 with `quantized_dtype="w8a8"`, `quantized_method="channel"`, and `do_quantization=True` using the same safe ONNX graph.
- [ ] **Step 5: Audit compiler output** for model hash, bytes, weight/internal memory, CPU/NPU placement, and any quantization fallback warning.
- [ ] **Step 6: Commit** with `git commit -m "feat: build four SPN RKNN INT8 variants"`.

### Task 4: Run controlled RK3588 measurements

**Files:**
- Modify: `scripts/summarize_rk3588_compressed_propagation.py`
- Modify: `tests/test_summarize_rk3588_compressed_propagation.py`

- [ ] **Step 1: Add failing protocol tests** for model/precision/candidate identity, five process indices, 150 samples per process, input hashes, runtime versions, and finite output.
- [ ] **Step 2: Run tests** and confirm incomplete evidence is rejected.
- [ ] **Step 3: Deploy hash-verified RKNN artifacts** and run a 2-warmup/3-timed smoke for every model/precision before controlled measurement.
- [ ] **Step 4: Run five independent processes** per accepted model/precision using 30 warmups and 3x50 timed forwards on cores 6--7 with two OpenMP threads.
- [ ] **Step 5: Capture separate perf-detail passes** and complete-forward output dumps; compare FP16 versus INT8 and packed versus legacy where applicable.
- [ ] **Step 6: Re-run tests** and commit with `git commit -m "feat: validate four compressed SPNs on RK3588"`.

### Task 5: Publish the corrected four-model report

**Files:**
- Modify: `reports/generated/rk3588_compressed_propagation_initial/EXPERIMENT.md`
- Modify: `reports/generated/rk3588_compressed_propagation_initial/initial_metrics.json`
- Add: `reports/generated/rk3588_compressed_propagation_initial/raw/{cspn,nlspn,dyspn,completionformer}/...`

- [ ] **Step 1: Report vanilla, structural FP16, and RKNN W8A8 separately** for all four models.
- [ ] **Step 2: State that RKNN W8A8 is not the original mixed W4/W6 research policy** and list all protected/fallback floating operators.
- [ ] **Step 3: Include latency, P05/P95, process CV, callback count/time, memory, output error, graph hashes, and raw evidence links.**
- [ ] **Step 4: Run all focused tests, JSON parsing, link checks, and `git diff --check`.**
- [ ] **Step 5: Commit** with `git commit -m "docs: report four compressed SPN RK3588 FP16 and INT8 results"`.
