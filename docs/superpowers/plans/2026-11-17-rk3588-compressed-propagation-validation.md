# RK3588 Compressed SPN and Coarse Propagation Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-backed RK3588 comparison of compressed SPN models, an algebraically exact coarse-grained propagation lowering, and an explicitly non-equivalent NPU-friendly propagation candidate.

**Architecture:** The conservative candidate exports the verified NAS/structured-pruned model in FP16 and changes no propagation semantics. The balanced candidate packs every deformable-convolution kernel point into one wide `GridSample`, reducing CPU callback and NPU/CPU scheduling granularity while preserving exact bilinear sampling. The aggressive candidate constrains offsets to an NPU-supported fixed stencil and is reported separately because it requires retraining and full accuracy validation.

**Tech Stack:** PyTorch, ONNX, ONNX Runtime, torchvision deformable convolution reference, RKNN Toolkit2 2.5, RKNPU2 runtime 2.5, C++17, pytest.

---

## Fixed experiment contract

- Platform: Orange Pi 5 Pro, RK3588S, RKNPU2 at 1 GHz.
- Task: NYU Depth V2, batch 1, fixed `228 x 304` model input.
- Strict latency scope: synchronous `rknn_run + rknn_outputs_get`.
- Repeats: five independent processes; each process performs 30 warmups and three rounds of 50 timed forwards.
- Runtime placement: `RKNN_NPU_CORE_AUTO`, CPU affinity 6-7, two OpenMP threads.
- Candidate A: compressed FP16 graph, unchanged propagation lowering.
- Candidate B: candidate A plus packed-grid coarse propagation lowering.
- Candidate C: fixed-stencil NPU-friendly propagation prototype; never described as equivalent until retraining and all 654 validation samples pass the declared RMSE gate.
- Must report: output error, complete-forward latency, GridSample callback time and count, RKNN CPU/NPU placement, model weight memory, internal memory, and graph operator counts.
- Open questions that do not block A/B: native RKNN INT4/INT6 mapping, a vendor-supported custom NPU kernel API, and the acceptable RMSE budget for candidate C.

## File structure

- `spn_quant/rknn_graph.py`: exact packed-grid deformable-convolution lowering.
- `spn_quant/compressed_checkpoint.py`: reconstruct and validate independent compressed checkpoints.
- `scripts/export_compressed_spn_onnx.py`: export compressed FP16 ONNX graphs with fixed deployment inputs.
- `scripts/build_rk3588_spn_variants.py`: apply safe rewrites and compile A/B RKNN variants.
- `scripts/summarize_rk3588_compressed_propagation.py`: validate repeated measurements and build comparison tables.
- `tests/test_rknn_graph.py`: numerical and graph-structure tests for packed lowering.
- `tests/test_compressed_checkpoint.py`: checkpoint identity and model-reconstruction tests.
- `tests/test_summarize_rk3588_compressed_propagation.py`: aggregation and protocol-rejection tests.
- `reports/generated/rk3588_compressed_propagation_initial/EXPERIMENT.md`: measured results and evidence boundary.

### Task 1: Exact packed-grid propagation lowering

**Files:**
- Modify: `spn_quant/rknn_graph.py`
- Modify: `tests/test_rknn_graph.py`

- [ ] **Step 1: Write the failing packed-lowering test**

Add a test that builds one static one-channel 3x3 `ModulatedDeformConv2D`, calls the new packed lowering, and asserts one `GridSample` instead of nine:

```python
assert decompose_modulated_deform_conv_for_rknn(
    source, converted, pack_grid_samples=True) == 1
rewritten = onnx.load(converted)
assert sum(node.op_type == "GridSample" for node in rewritten.graph.node) == 1
```

The test must compare ONNX Runtime output against `torchvision.ops.deform_conv2d` with `rtol=1e-5, atol=2e-6`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_rknn_graph.py::test_packed_decompose_modulated_deform_conv_matches_torchvision
```

Expected: FAIL because `pack_grid_samples` is not accepted.

- [ ] **Step 3: Implement the minimal exact rewrite**

Extend the function signature:

```python
def decompose_modulated_deform_conv_for_rknn(
        source: Path, output: Path,
        pack_grid_samples: bool = False) -> int:
```

For packed mode, concatenate the per-point NHWC grids along output width, issue one `GridSample`, split the sampled NCHW result back into point tensors, and reuse the existing mask, scalar weight, accumulation, and bias operations. Do not change legacy mode.

- [ ] **Step 4: Verify GREEN and legacy regression**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_rknn_graph.py tests/test_rknn_grid_sample_cpp.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/rknn_graph.py tests/test_rknn_graph.py
git commit -m "feat: pack deformable sampling for RK3588"
```

### Task 2: Independent compressed-checkpoint reconstruction

**Files:**
- Create: `spn_quant/compressed_checkpoint.py`
- Create: `tests/test_compressed_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint-contract tests**

Define tests for `load_compressed_checkpoint(path)` that require:

```python
assert payload["format"] == "spn-independent-compressed-checkpoint"
assert payload["format_version"] == 1
assert payload["model"] == expected_model
assert verify_state_dict_sha256(payload["net"], payload["net_sha256"])
```

Add rejection tests for a wrong format, model mismatch, and altered state tensor.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_compressed_checkpoint.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement validation and reconstruction metadata**

Implement an immutable `CompressedCheckpoint` record containing model identity, NAS depths, structured candidate, quantization assignment, propagation dtype, state dictionary, and provenance. Reuse `screen_four_model_depth_nas.apply_depths` and `screen_structured_channel_nas.apply_structured_candidate`; reject state-dict key or shape mismatches.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_compressed_checkpoint.py tests/test_structured_channel_pruning.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/compressed_checkpoint.py tests/test_compressed_checkpoint.py
git commit -m "feat: validate independent compressed SPN checkpoints"
```

### Task 3: Export compressed FP16 ONNX graphs

**Files:**
- Create: `scripts/export_compressed_spn_onnx.py`
- Create: `tests/test_export_compressed_spn_onnx.py`

- [ ] **Step 1: Write failing CLI and manifest tests**

Test that the exporter rejects unsupported checkpoint formats and writes a manifest containing:

```python
{
    "model": "dyspn",
    "candidate": "compressed_fp16",
    "input_shapes": [[1, 3, 228, 304], [1, 1, 228, 304]],
    "propagation_semantics": "unchanged",
    "checkpoint_sha256": "...",
    "onnx_sha256": "...",
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_export_compressed_spn_onnx.py
```

Expected: import failure because the exporter does not exist.

- [ ] **Step 3: Implement the exporter**

Build the official model through `NYUModelRuntime`, apply checkpoint-declared depth and structured changes, load the independent state dictionary strictly, install the existing deformable-convolution ONNX symbolic, and export fixed-shape FP32 ONNX. Quantization metadata remains in the manifest; candidate A intentionally uses FP16 RKNN lowering because W4/W6 are not yet established as native RKNN kernels.

- [ ] **Step 4: Verify model-specific smoke exports**

Run DySPN and CompletionFormer exports from the two standardized local checkpoints, then use ONNX Runtime on sample 386 and compare with the reconstructed PyTorch output.

Expected: finite output and the configured export tolerance passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_compressed_spn_onnx.py tests/test_export_compressed_spn_onnx.py
git commit -m "feat: export compressed SPN deployment graphs"
```

### Task 4: Build and inspect RKNN A/B variants

**Files:**
- Create: `scripts/build_rk3588_spn_variants.py`
- Create: `tests/test_build_rk3588_spn_variants.py`

- [ ] **Step 1: Write failing variant-plan tests**

Require one source graph to produce exactly:

```python
("compressed_fp16_legacy_grid", "compressed_fp16_packed_grid")
```

Require graph audits to record deformable-node count, GridSample count, ConvTranspose count, and hashes.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_build_rk3588_spn_variants.py
```

Expected: import failure because the builder does not exist.

- [ ] **Step 3: Implement safe rewrite and RKNN compilation**

Apply the existing Sign and ConvTranspose rewrites to both candidates. Apply legacy DCN decomposition to A and packed decomposition to B. Compile with RKNN Toolkit2 2.5 for RK3588 with quantization disabled and write compiler logs plus hashes.

- [ ] **Step 4: Validate operator admission**

Require successful RKNN build, no native custom DCN nodes, expected GridSample count reduction, and no unexpected graph inputs or outputs.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_rk3588_spn_variants.py tests/test_build_rk3588_spn_variants.py
git commit -m "feat: build compressed RK3588 propagation variants"
```

### Task 5: Controlled RK3588 measurement and summary

**Files:**
- Create: `scripts/summarize_rk3588_compressed_propagation.py`
- Create: `tests/test_summarize_rk3588_compressed_propagation.py`
- Create: `reports/generated/rk3588_compressed_propagation_initial/EXPERIMENT.md`

- [ ] **Step 1: Write failing protocol-validation tests**

Reject missing processes, non-150 iteration processes, differing hashes, non-finite outputs, changed inputs, or mismatched runtime versions. Aggregate process means without pooling away process identity.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_summarize_rk3588_compressed_propagation.py
```

Expected: import failure because the summarizer does not exist.

- [ ] **Step 3: Run A/B on RK3588**

Copy models, fixed sample-386 inputs, benchmark binary, and hashes into a new dated board directory. For each model and candidate, run five independent processes using the fixed contract. Collect `RKNN_QUERY_PERF_DETAIL` in a separate pass.

- [ ] **Step 4: Validate numerical equivalence**

Compare candidate B against candidate A and the compressed PyTorch/ONNX reference. Record maximum absolute error, mean absolute error, RMSE, and task RMSE where ground truth is available. A/B differences must fit the FP16 export tolerance; otherwise mark B rejected.

- [ ] **Step 5: Generate the report**

Report latency mean/median/p05/p95, process-level coefficient of variation, callback time and count, NPU/CPU placement, memory, operator counts, and confidence. Keep vanilla results visible but do not mix their instrumented operator sums with complete-forward latency.

- [ ] **Step 6: Commit**

```bash
git add scripts/summarize_rk3588_compressed_propagation.py tests/test_summarize_rk3588_compressed_propagation.py reports/generated/rk3588_compressed_propagation_initial
git commit -m "docs: report RK3588 compressed propagation validation"
```

### Task 6: NPU-friendly fixed-stencil feasibility boundary

**Files:**
- Create: `scripts/analyze_fixed_stencil_propagation.py`
- Create: `tests/test_analyze_fixed_stencil_propagation.py`
- Modify: `reports/generated/rk3588_compressed_propagation_initial/EXPERIMENT.md`

- [ ] **Step 1: Write failing discretization tests**

Test offset projection to a declared stencil and require deterministic tie breaking, explicit out-of-range handling, and a measured offset-approximation error summary.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_analyze_fixed_stencil_propagation.py
```

Expected: import failure because the analyzer does not exist.

- [ ] **Step 3: Implement analysis-only projection**

Measure the fraction of offsets captured by 8-, 16-, and 25-point stencils and the induced one-step/final-depth error. Do not replace the production model or claim task equivalence.

- [ ] **Step 4: Record the decision**

Promote candidate C to retraining only if its error distribution is bounded enough to justify a new training experiment. Otherwise record it as rejected and retain candidate B.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_fixed_stencil_propagation.py tests/test_analyze_fixed_stencil_propagation.py reports/generated/rk3588_compressed_propagation_initial/EXPERIMENT.md
git commit -m "analysis: bound fixed-stencil propagation error"
```

## Final verification

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_rknn_graph.py \
  tests/test_rknn_grid_sample_cpp.py \
  tests/test_compressed_checkpoint.py \
  tests/test_export_compressed_spn_onnx.py \
  tests/test_build_rk3588_spn_variants.py \
  tests/test_summarize_rk3588_compressed_propagation.py \
  tests/test_analyze_fixed_stencil_propagation.py
```

Expected: all tests pass. Re-run report validation from raw JSON after deleting only generated summaries, never raw board evidence.
