# CompletionFormer Joint Integer Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict CompletionFormer W4A4 reference backend that jointly optimizes per-head integer attention and split-branch concat convolution scales, then evaluate it on the fixed 64-sample NYU set.

**Architecture:** Keep the official CompletionFormer source unchanged. Add reusable fixed-point primitives, focused Attention and split-concat controllers, and a model adapter that owns only Q/K/V and concat boundaries while the existing instrumentor owns ordinary Conv/Linear QDQ. Integrate explicit ablations into the existing NYU runner and persist manifests, local metrics, end-to-end metrics, and prediction comparisons.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu118, CUDA `torch._int_mm`, NumPy, Matplotlib, unittest/pytest, official CompletionFormer CUDA extension, NYU Depth V2.

---

## File Structure

- Create `spn_quant/integer_ops.py`: code generation, Q31 requantization, strict INT8-to-INT32 matrix multiplication, shifted-U8 correction, and INT8 im2col.
- Create `spn_quant/scale_search.py`: bounded calibration cache and deterministic coordinate search.
- Create `spn_quant/completionformer_attention.py`: per-head Q/K/V quantization, FP16 softmax, unsigned A8 probability, INT32 QK/AV, metrics, and manifests.
- Create `spn_quant/completionformer_concat.py`: branch quantization, split W4 convolution, partial-accumulator alignment, bias, output quantization, metrics, and manifests.
- Create `spn_quant/adapters/completionformer_joint.py`: official-module discovery, forward ownership, phase control, and restoration.
- Modify `spn_quant/adapters/__init__.py` and `scripts/hardware_aligned_quantization.py`: expose the adapter and externally-owned input boundaries.
- Modify `scripts/run_nyu_rtn_quantization.py`: add joint backend, ablations, output tables, and predictions.
- Create `scripts/plot_completionformer_joint_quantization.py` and `scripts/run_completionformer_joint_quantization.sh`.
- Add focused tests under `tests/` and update `README.md`.

### Task 1: Strict Integer Primitives

**Files:**
- Create: `spn_quant/integer_ops.py`
- Create: `tests/test_integer_ops.py`

- [ ] **Step 1: Write failing code-domain and requantization tests**

```python
def test_signed_codes_use_symmetric_a4_domain():
    codes, scale = quantize_signed(
        torch.tensor([-2.0, 0.0, 2.0]), bits=4, maximum=2.0)
    assert codes.dtype == torch.int8
    assert codes.tolist() == [-7, 0, 7]
    assert scale == pytest.approx(2.0 / 7.0)


def test_q31_requantization_is_signed_and_saturating():
    values = torch.tensor([-30, -3, 3, 30], dtype=torch.int32)
    output = requantize_int32(
        values, torch.tensor(0.5), torch.tensor(1.0), -7, 7)
    assert output.tolist() == [-7, -2, 2, 7]
```

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. pytest -q tests/test_integer_ops.py -k 'signed_codes or q31'`.
Expected: import failure because `spn_quant.integer_ops` is absent.

- [ ] **Step 3: Implement signed/unsigned codes and Q31 requantization**

Implement these exact public APIs:

```python
quantize_signed(tensor, bits, maximum) -> (int8_codes, scale)
quantize_unsigned(tensor, bits, maximum) -> (uint8_codes, scale)
requantize_int32(values, source_scale, target_scale, qmin, qmax,
                 fraction_bits=31) -> int32_codes
```

Multiplier creation may use float64; applying it uses INT64 product, signed
rounding, right shift, and explicit clamp. Validate finite positive scales,
broadcast shape, bit range, and product bounds. Do not add fallback scales.

- [ ] **Step 4: Write failing INT8 and shifted-U8 matmul tests**

```python
def test_shifted_u8_int8_mm_matches_unsigned_reference():
    probability = torch.tensor([[0, 128, 255]], dtype=torch.uint8)
    value = torch.tensor([[2], [-3], [4]], dtype=torch.int8)
    expected = probability.to(torch.int32) @ value.to(torch.int32)
    torch.testing.assert_close(
        uint8_int8_mm_int32(probability, value), expected)
```

Also test `int8_mm_int32` and batched variants against explicit integer sums.

- [ ] **Step 5: Implement strict `_int_mm` paths**

`int8_mm_int32` calls `torch._int_mm` directly after dtype/rank/shape checks.
No float matmul fallback is allowed. Full unsigned A8 uses:

```python
shifted = (left.to(torch.int16) - 128).to(torch.int8)
product = torch._int_mm(shifted, right)
correction = 128 * right.to(torch.int32).sum(dim=0, keepdim=True)
return product + correction
```

Implement batched/head execution as deterministic per-matrix `_int_mm` calls.

- [ ] **Step 6: Implement INT8 im2col**

Use `F.pad` plus tensor `.unfold()` views. Support the official concat Conv
contract only: groups=1 and dilation=1. Unsupported parameters raise.

- [ ] **Step 7: Verify and commit**

Run `PYTHONPATH=. pytest -q tests/test_integer_ops.py`, then commit:

```bash
git add spn_quant/integer_ops.py tests/test_integer_ops.py
git commit -m "feat: add strict integer quantization primitives"
```

### Task 2: Deterministic Block Scale Search

**Files:**
- Create: `spn_quant/scale_search.py`
- Create: `tests/test_scale_search.py`

- [ ] **Step 1: Write failing bounded-cache tests**

Test that `CalibrationCache(sample_limit, byte_limit)` stores detached
contiguous float32 CPU tensors, preserves order, rejects non-finite tensors,
and raises instead of silently evicting when limits are exceeded.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. pytest -q tests/test_scale_search.py -k cache`.

- [ ] **Step 3: Implement immutable calibration storage**

Expose only `append(tuple_of_tensors)` and `samples()`. Account exact storage
bytes and require every constructor argument explicitly.

- [ ] **Step 4: Write failing coordinate-search tests**

```python
def test_coordinate_search_is_deterministic():
    search = CoordinateScaleSearch(
        parameter_names=("q", "k"), factors=(1.0, 0.75, 0.5), rounds=2)
    result = search.run(
        {"q": 1.0, "k": 1.0},
        lambda x: (x["q"] - 0.75) ** 2 + (x["k"] - 0.5) ** 2,
        sample_count=4)
    assert result.values == {"q": 0.75, "k": 0.5}
```

- [ ] **Step 5: Implement search and audit rows**

Search declared parameters and factors in order, keep the current value on
ties, reject non-finite objectives, and record round, parameter, factor,
objective, selected, and sample count for every candidate.

- [ ] **Step 6: Verify and commit**

Run `PYTHONPATH=. pytest -q tests/test_scale_search.py`, then commit
`spn_quant/scale_search.py` and its test with message
`feat: add deterministic block scale search`.

### Task 3: Integer Attention Controller

**Files:**
- Create: `spn_quant/completionformer_attention.py`
- Create: `tests/test_completionformer_attention.py`

- [ ] **Step 1: Write failing K/V separation and per-head tests**

Construct Q/K/V with two heads and different K/V ranges. After observe/freeze,
assert `scales["q"]`, `scales["k"]`, and `scales["v"]` each have shape `(2,)`
and K/V scales differ.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. pytest -q tests/test_completionformer_attention.py -k scales`.

- [ ] **Step 3: Implement phases and integer data flow**

Expose explicit methods:

```python
observe(q, k, v, fp_context)
freeze()
quantize(q, k, v)
disable()
manifest()
statistics()
search_rows()
```

Per-head maxima reduce all axes except head. Quantized execution uses signed
A4 Q/K/V, strict INT32 QK, scaled FP16 softmax, unsigned A8 probability,
shifted-U8 INT32 AV, and dequantized context for the existing projection.

- [ ] **Step 4: Add failing accumulator and probability tests**

Verify exact QK/AV integer sums, probability code range `[0, 255]`, exact zero
and one, Q/K token-length differences, output shape, and non-finite rejection.

- [ ] **Step 5: Implement block-objective scale selection**

Use normalized context MSE against calibration-only FP context. Search one
clip factor per Q/K/V role while retaining per-head maxima. Probability maximum
is exactly one and is not tuned on evaluation data.

- [ ] **Step 6: Add metrics and commit**

Record Q/K/V SQNR, score SQNR, probability KL/zero/saturation, and context MSE.
Run the full test file and commit with message
`feat: add CompletionFormer integer attention`.

### Task 4: Split Concat Integer Convolution

**Files:**
- Create: `spn_quant/completionformer_concat.py`
- Create: `tests/test_completionformer_concat.py`

- [ ] **Step 1: Write failing branch-order and partial-Conv tests**

Test that the first `C` channels are the Transformer branch and the second `C`
channels are the CNN branch. Compare two INT32 partial convolutions against a
direct integer convolution on a hand-computable tensor.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. pytest -q tests/test_completionformer_concat.py -k partial`.

- [ ] **Step 3: Implement strict split convolution**

Validate the official `concat_conv` shape and parameters. Split W4 codes on
the input-channel axis, build INT8 patches, and call `_int_mm` once per branch.
Keep both outputs INT32 until explicit accumulator alignment.

- [ ] **Step 4: Write failing scale and bias tests**

Assert branch accumulator scales may differ, both are requantized to the
declared target accumulator scale, bias uses that same per-output scale, and
output codes remain in A4/A8 range.

- [ ] **Step 5: Implement calibration and joint scale search**

Observe branch tensors and official float block output. Keep W4 per-output
weight scales fixed. Search Transformer branch, CNN branch, common accumulator,
and output clipping factors against normalized block-output MSE.

- [ ] **Step 6: Add metrics and commit**

Record branch SQNR/zero/saturation, partial requantization error, output SQNR,
and block MSE. Run the full test file and commit with message
`feat: add split concat integer convolution`.

### Task 5: CompletionFormer Adapter And Boundary Ownership

**Files:**
- Create: `spn_quant/adapters/completionformer_joint.py`
- Modify: `spn_quant/adapters/__init__.py`
- Modify: `scripts/hardware_aligned_quantization.py`
- Create: `tests/test_completionformer_joint_adapter.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [ ] **Step 1: Write a failing externally-owned-input test**

Assert an owned module input receives no generic input QDQ while its W4 weight
and unowned output remain active. The metadata must list the owned boundary.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. pytest -q tests/test_hardware_aligned_quantization.py -k externally_owned_input`.

- [ ] **Step 3: Add strict input ownership**

Add required `externally_owned_inputs` constructor state, validate every name,
skip its generic input observer/QDQ, and expose it in metadata. New code reads
required config using attributes or `[]`; it does not use `getattr`, `.get`,
`try/except`, or fallback quantizers.

- [ ] **Step 4: Write failing adapter discovery and restoration tests**

Use an official-structure toy model. Assert Attention/concat discovery,
Q/K/V and concat ownership, disabled-forward identity, exact call counts, and
restoration of every patched forward on `close()`.

- [ ] **Step 5: Implement joint adapter**

Require CompletionFormer and validate 16 Attention plus 16 concat modules in
the real model. Reproduce official q/kv/sr/norm/proj order without modifying
external source. Expose observe, freeze, configure, disable, manifests,
metrics, search rows, and close. Every configure argument is mandatory.

- [ ] **Step 6: Verify and commit**

Run both adapter and hardware-instrumentor test files. Commit all five files
with message `feat: install CompletionFormer joint integer adapter`.

### Task 6: NYU Runner Ablations And Output Tables

**Files:**
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`

- [ ] **Step 1: Write failing configuration-contract tests**

Require this exact order:

```text
FP32
JIQ_RTN_W4A4
JIQ_Attention_W4A4
JIQ_Concat_W4A4
JIQ_Joint_W4A4
JIQ_W4A8
```

Every non-FP config explicitly contains `attention_enabled`, `concat_enabled`,
`qkv_bits`, `concat_bits`, and `output_bits`. Reject non-CompletionFormer use.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. pytest -q tests/test_run_nyu_rtn_quantization.py -k completionformer_joint`.

- [ ] **Step 3: Add backend setup and one calibration pass**

Add `completionformer_joint` to backend choices and propagation-adapter use.
Observe ordinary hardware boundaries, propagation A8 boundaries, Attention,
and concat on the same deterministic calibration indices, then freeze each
controller explicitly.

- [ ] **Step 4: Implement isolated config execution**

Before each config, restore original parameters and disable all controllers.
`JIQ_RTN_W4A4` reproduces current uniform RTN behavior. Attention-only and
concat-only enable exactly one joint family. W4A8 uses eight-bit ordinary,
Q/K/V, concat, and output activations while retaining W4 weights.

- [ ] **Step 5: Persist strict tables**

Write `attention_integer_manifest.csv`, `concat_integer_manifest.csv`,
`joint_scale_search.csv`, `attention_metrics.csv`, and `concat_metrics.csv`.
Reject missing/duplicate site-config rows before metadata is written.

- [ ] **Step 6: Verify and commit**

Run the runner test file and commit with message
`feat: run CompletionFormer joint quantization ablations`.

### Task 7: Reproducible Launcher And Prediction Figures

**Files:**
- Create: `scripts/run_completionformer_joint_quantization.sh`
- Create: `scripts/plot_completionformer_joint_quantization.py`
- Create: `tests/test_plot_completionformer_joint_quantization.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing plot-contract tests**

Build temporary metrics and NPZ predictions. Assert missing samples fail,
columns are GT/FP32/current W4A4/attention/concat/joint/W4A8, and aggregate and
prediction figures are written.

- [ ] **Step 2: Verify RED and implement plotting**

Run the test, implement Arial output with non-rotated labels, grid below
artists, shared depth/error ranges, then rerun. Produce:

```text
aggregate_rmse.png
local_attention_concat_metrics.png
prediction_comparison_64.png
```

- [ ] **Step 3: Add exact fixed-64 launcher**

The shell script runs `scripts/run_nyu_rtn_quantization.py` with:

```text
backend: completionformer_joint
run dir: /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/completionformer_iter18
sample metrics: /workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation/primary/rtn/completionformer/sample_metrics.csv
data root: /workspace/CSPN/cspn_pytorch
output: /workspace/SPN_Quantization/profile_logs/nyu_completionformer_joint_integer_64
calibration/evaluation: 64/64
```

Export predictions for all six configs. Use `PYTHONPATH="$ROOT"`; do not write
to `/tmp`.

- [ ] **Step 4: Document execution semantics**

README states that A4 codes are unpacked to INT8 lanes, QK/AV and concat use
INT32 accumulation, softmax is FP16, unsigned A8 uses shifted INT8 plus
correction, and this reference path makes no native speedup claim.

- [ ] **Step 5: Verify and commit**

Run the plot tests and `bash -n` on the launcher. Commit with message
`feat: report CompletionFormer joint quantization`.

### Task 8: Full Verification And Fixed-64 Evaluation

**Files:**
- Generate only: `/workspace/SPN_Quantization/profile_logs/nyu_completionformer_joint_integer_64/`

- [ ] **Step 1: Run complete tests**

Run `PYTHONPATH=. pytest -q`. Expected: all tests pass with no collection
errors or warnings.

- [ ] **Step 2: Run CUDA integer checks**

Run marked CUDA tests for official-shape QK, AV, and concat matrices. Assert
INT32 outputs and compare bounded slices with CPU integer references.

- [ ] **Step 3: Run the fixed-64 experiment**

Run `bash scripts/run_completionformer_joint_quantization.sh`. Do not retrain.

- [ ] **Step 4: Audit outputs**

Verify every config has 64 unique sample rows and predictions, all metrics are
finite, enabled manifests contain 16 Attention and 16 concat modules,
calibration/evaluation indices match metadata, and official source/checkpoint
hashes match the strict reference.

- [ ] **Step 5: Generate figures**

```bash
PYTHONPATH=. python scripts/plot_completionformer_joint_quantization.py \
  --root /workspace/SPN_Quantization/profile_logs/nyu_completionformer_joint_integer_64/completionformer \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_completionformer_joint_integer_64/analysis
```

- [ ] **Step 6: Report measured results**

Report mean/median/p95 RMSE and FP32 delta, paired joint-vs-current W4A4
differences, probability KL, concat block MSE, saturation/zero rates, and
representative prediction errors. State plainly if joint optimization fails.

- [ ] **Step 7: Final regression and commit**

Run `PYTHONPATH=. pytest -q`, inspect `git status --short`, and commit verified
source/test/documentation changes. Do not add generated `profile_logs`.
