# Jetson CPU-Only Four-SPN Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure strict CPU-only complete-forward latency, six-class CPU operator breakdown, and CPU/module energy for the four vanilla SPN models on Jetson AGX Xavier.

**Architecture:** Reuse the exact FP32 checkpoints and fixed NYU sample 386 from the GPU experiment while forcing `CUDA_VISIBLE_DEVICES=""`, `device=cpu`, eight Carmel CPU cores, one inter-op thread, and fixed MAXN/jetson_clocks state. Use monotonic wall timing for uninstrumented latency, PyTorch CPU profiler self time with explicit propagation/attention scopes for breakdown, and the existing external jtop sampler for synchronized INA3221 CPU-rail and module-total energy integration.

**Tech Stack:** Python 3.8, PyTorch 2.1 CPU/OpenMP, torchvision CPU deformable convolution, jtop/INA3221, pytest, JSON/Markdown evidence.

---

### Task 1: Admit and audit strict CPU execution

**Files:**
- Modify: `scripts/nyu_model_runtime.py`
- Modify: `tests/test_nyu_model_runtime.py`

- [ ] **Step 1: Write failing CPU-runtime tests**

Add tests proving that an explicitly configured CPU runtime accepts only a CPU device, validates a native CPU dispatch kernel when one is required, and preserves the existing explicit-CUDA checks for GPU runs.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `/opt/conda/bin/python -m pytest -q tests/test_nyu_model_runtime.py`

Expected: the CPU-runtime tests fail because the runtime currently requires CUDA.

- [ ] **Step 3: Implement device-specific backend validation**

Replace the CUDA-only build guard with a configured-device guard. For CUDA, keep explicit indexed-device activation and CUDA dispatch validation. For CPU, require `device == cpu`, import the configured compatibility module, and require a CPU dispatch kernel for native operators such as `torchvision::deform_conv2d`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `/opt/conda/bin/python -m pytest -q tests/test_nyu_model_runtime.py`

Expected: all runtime tests pass.

### Task 2: CPU latency, energy, and profiler primitives

**Files:**
- Create: `spn_quant/cpu_benchmark.py`
- Create: `tests/test_cpu_benchmark.py`
- Create: `scripts/benchmark_nyu_vanilla_cpu.py`

- [ ] **Step 1: Write failing timing and classification tests**

Test positive monotonic full-forward timing, exact completed-forward counting, strict rejection of CUDA tensors, propagation precedence, and the six exclusive categories: dense compute, attention, sampling, propagation, elementwise/norm, and layout/memory, plus an auditable residual.

- [ ] **Step 2: Run tests and verify RED**

Run: `/opt/conda/bin/python -m pytest -q tests/test_cpu_benchmark.py`

Expected: collection fails because the CPU benchmark module does not exist.

- [ ] **Step 3: Implement CPU-only primitives and worker**

Implement `perf_counter_ns` latency timing, eight-thread configuration, recursive CPU tensor auditing, synchronized idle/active energy phases, jtop sampler orchestration, and PyTorch profiler aggregation using operator self CPU time. Annotate `post_process_layer`, `dyspn_6_5`, and `prop_layer` as propagation; annotate CompletionFormer attention modules separately.

- [ ] **Step 4: Run focused and regression tests**

Run: `/opt/conda/bin/python -m pytest -q tests/test_cpu_benchmark.py tests/test_energy.py tests/test_nyu_model_runtime.py`

Expected: all tests pass.

### Task 3: Compatibility and controlled measurement

**Files:**
- Create: `reports/generated/spn_cross_platform_20261022/jetson_xavier/cpu_only/inventory.json`
- Create: `reports/generated/spn_cross_platform_20261022/jetson_xavier/cpu_only/controlled_repeats/*.json`
- Create: `reports/generated/spn_cross_platform_20261022/jetson_xavier/cpu_only/energy/*.jsonl`
- Create: `reports/generated/spn_cross_platform_20261022/jetson_xavier/cpu_only/breakdown/*.json`

- [ ] **Step 1: Deploy and smoke-test every model**

Require `torch.cuda.is_available() == false` inside the worker, all parameters/inputs/outputs on CPU, finite 228×304 output, strict checkpoint loading, and the CPU torchvision DCNv2 backend for all deformable-convolution models.

- [ ] **Step 2: Run a 1/2/4/8-thread pilot**

Measure each model briefly at fixed 2265.6 MHz and select the common thread count with the lowest geometric-mean latency. Record all pilot results; do not select separately per model.

- [ ] **Step 3: Run five independent latency/energy processes per model**

Use 30 warmups, three latency rounds, a minimum of 10 forwards per round, 15 s loaded idle, at least 60 s active complete forwards, and 50 ms requested power sampling. Record CPU-rail and module-total gross and idle-subtracted energy.

- [ ] **Step 4: Capture five profiled forwards per model**

Use CPU profiler self time and explicit scopes. Keep the profiler results separate from uninstrumented headline latency because instrumentation overhead changes wall time.

### Task 4: Aggregate and update the experiment report

**Files:**
- Create: `scripts/summarize_jetson_cpu_spn.py`
- Create: `tests/test_summarize_jetson_cpu_spn.py`
- Modify: `reports/generated/spn_cross_platform_20261022/EXPERIMENT.md`
- Modify: `reports/generated/spn_cross_platform_20261022/manifest.json`

- [ ] **Step 1: Write failing aggregation tests**

Test five-process Student-t confidence intervals, matching checkpoint/input hashes, CPU-only device audits, exclusive category sums, and matching sensor boundaries.

- [ ] **Step 2: Implement deterministic aggregation**

Aggregate independent process means for latency and energy, retain raw values and hashes, and validate that every breakdown assigns each profiled CPU operator exactly once.

- [ ] **Step 3: Update the existing Markdown report**

Add Carmel CPU topology, CPU-only controls, latency and energy tables, six-class CPU breakdown, profiler caveats, thread-scaling evidence, and a direct comparison with the existing Xavier GPU results.

- [ ] **Step 4: Run final verification**

Run the focused tests, deterministic re-aggregation, all evidence hashes, Markdown link validation, and `git diff --check` before completion.
