# Four-SPN Vanilla GPU Latency Implementation Plan

> **Execution note:** Implement in the existing `feature/spn-llmcompass-array-dse`
> worktree. Preserve all pre-existing uncommitted DSE changes.

**Goal:** Measure and compare reproducible, full-forward FP32 inference latency
for the official vanilla CSPN, DySPN, NLSPN, and CompletionFormer NYU models on
one physical A100 GPU.

**Measurement contract:** Batch size 1, official converged checkpoints, one
fixed NYU validation sample with shape 228x304, inputs resident on the GPU,
complete model forward including propagation/refinement, TF32 disabled, cuDNN
autotuning enabled before timed iterations, CUDA Event timing, and explicit
CUDA synchronization. Dataset decoding, host-to-device transfer, model loading,
and checkpoint loading are excluded.

**Architecture:** A Python-3.7-compatible worker loads and validates one model
using the existing `NYUModelRuntime`. A launcher invokes the worker sequentially
for all four models with their required conda interpreters on the same GPU, then
validates and aggregates per-model JSON into CSV, Markdown, and comparison
figures. Pure statistics/report helpers live in `spn_quant.gpu_latency` so they
can be unit-tested without a GPU.

---

### Task 1: Specify the benchmark/report contract with failing tests

**Files:**
- Create: `tests/test_gpu_latency.py`
- Create: `tests/test_run_nyu_vanilla_gpu_latency.py`

1. Test deterministic percentile/statistical summaries and invalid samples.
2. Test strict FP32 tensor auditing, including nested model inputs.
3. Test four-model result validation and stable summary ordering.
4. Test launcher command/environment construction for the two conda runtimes.
5. Run the focused tests and confirm they fail because implementation is absent.

### Task 2: Implement pure benchmark and report helpers

**Files:**
- Create: `spn_quant/gpu_latency.py`

1. Implement Python-3.7-compatible latency statistics.
2. Implement nested tensor traversal and strict FP32 auditing.
3. Implement per-model result contract validation.
4. Implement CSV and Markdown aggregation helpers.
5. Re-run focused tests until green.

### Task 3: Implement the official-model CUDA Event worker

**Files:**
- Create: `scripts/benchmark_nyu_vanilla_gpu_worker.py`

1. Load the selected model/checkpoint with `NYUModelRuntime`.
2. Load one fixed official validation identity and transfer it before timing.
3. Verify architecture, parameter/input dtype, prediction shape, and finiteness.
4. Warm up, synchronize, time complete forwards with CUDA Events, and record
   latency distribution, throughput, model size, peak allocated/reserved memory,
   GPU/software provenance, and checkpoint hash.
5. Write one atomic JSON result per model.

### Task 4: Implement the four-model launcher and checked-in config

**Files:**
- Create: `configs/four_spn_vanilla_gpu_latency.json`
- Create: `scripts/run_nyu_vanilla_gpu_latency.py`

1. Resolve repository-relative config paths and explicit model environments.
2. Check that the selected physical GPU is idle before launching.
3. Run CSPN, DySPN, NLSPN, and CompletionFormer sequentially on the same GPU.
4. Aggregate results into `latency_summary.csv`, `latency_report.md`, a raw
   manifest, and latency/memory comparison plots.
5. Reject incomplete, mixed-precision, mixed-GPU, or inconsistent benchmark
   outputs.

### Task 5: Run the real benchmark and verify artifacts

**Output:**
- Create: `reports/generated/spn_vanilla_gpu_latency_a100/`

1. Re-check GPU utilization and temperature.
2. Run 30 warmups plus 3x50 timed iterations per model on one A100.
3. Inspect per-model JSON and aggregate artifacts for all four models.
4. Run focused unit tests and relevant existing runtime/export tests.
5. Record limitations: one GPU SKU, batch 1, fixed input shape/sample, framework
   and custom-extension versions differ where required by official models.
