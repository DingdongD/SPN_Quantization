# CSPN Inference Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and validate the fastest deployment configuration of the selected CSPN NAS model that remains non-inferior to the R18 baseline by the pre-specified 2% RMSE margin.

**Architecture:** Reuse the three fixed-epoch candidate checkpoints and evaluate inference-only CSPN iteration overrides on the untouched official validation set. Benchmark numerical execution modes independently from model quality, promote only finite and accurate configurations to strict A100 timing, and use the resulting bottleneck evidence to gate decoder NAS and true TensorRT INT8 work.

**Tech Stack:** PyTorch 2.7, CUDA 11.8, A100 CUDA-event timing, NYU Depth V2 official validation split, paired hierarchical bootstrap.

---

### Task 1: Inference-Time CSPN Iteration Override

**Files:**
- Modify: `scripts/evaluate_cspn_nas_official.py`
- Modify: `tests/test_evaluate_cspn_nas_official.py`

- [ ] Add a failing parser and model-loading test for `--cspn-steps`.
- [ ] Run `pytest tests/test_evaluate_cspn_nas_official.py -q` and confirm the new assertion fails.
- [ ] Thread an optional positive iteration override through `load_model`, `evaluate_run`, metadata, and CLI parsing.
- [ ] Run the focused test and commit the passing implementation.

### Task 2: Precision-Aware CUDA Benchmarking

**Files:**
- Modify: `spn_quant/nas/benchmark.py`
- Modify: `tests/test_cspn_nas_benchmark.py`
- Create: `scripts/benchmark_cspn_nas_inference.py`
- Create: `tests/test_benchmark_cspn_nas_inference.py`

- [ ] Add failing tests for accepted precision modes, state restoration, result schema, and invalid modes.
- [ ] Run the focused tests and verify failures are caused by missing precision support.
- [ ] Add `fp32`, `tf32`, `fp16`, and `bf16` execution contexts while preserving the strict-FP32 default.
- [ ] Add a checkpoint-aware CLI that benchmarks multiple CSPN step counts and emits atomic JSON.
- [ ] Run focused tests and commit.

### Task 3: Official Iteration Ablation

**Files:**
- Create: `output/cspn_encoder_nas_20260920/inference_acceleration/iteration_*/candidate.csv`
- Create: `output/cspn_encoder_nas_20260920/inference_acceleration/iteration_*/report/`

- [ ] Evaluate steps 8, 12, and 18 in parallel on separate A100s; reuse the completed step-24 CSV.
- [ ] Run the existing 10,000-replicate paired non-inferiority report for every step count.
- [ ] Reject any step count whose one-sided 95% upper confidence bound exceeds the 2% R18 margin.

### Task 4: A100 Numerical-Mode Latency

**Files:**
- Create: `output/cspn_encoder_nas_20260920/inference_acceleration/latency_screen.json`
- Create: `output/cspn_encoder_nas_20260920/inference_acceleration/latency_strict.json`

- [ ] Screen every finite precision/iteration combination with 200 warmups, 200 iterations, and 3 repeats.
- [ ] Compare each reduced-precision output with FP32 and reject non-finite or materially divergent results.
- [ ] Re-run surviving Pareto candidates with 200 warmups, 1,000 iterations, and 5 repeats.
- [ ] Record GPU environment and the user's accepted background-process condition.

### Task 5: Decoder and Deployment Gate

**Files:**
- Create: `output/cspn_encoder_nas_20260920/inference_acceleration/decision.json`
- Create: `output/cspn_encoder_nas_20260920/inference_acceleration/README.md`

- [ ] Select the lowest-latency configuration that passes official non-inferiority.
- [ ] Quantify residual encoder, decoder/head, and propagation latency shares.
- [ ] Record TensorRT/ONNX availability and do not claim INT8 speed without a real engine.
- [ ] Start decoder width search only if the validated inference configuration leaves the decoder as the dominant bottleneck.
- [ ] Run the full test suite, commit artifacts and code, and report measured results.
