# CSPN W8A8 Im2Col Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict PA-W8A8 CSPN diagnostic that measures activation, weight and local Conv output error across spatial-token and `channel@kernel-offset` axes, then exports 3D line visualizations.

**Architecture:** A focused `spn_quant.im2col_diagnostics` module owns layout validation, chunked exact counters and persisted metric schemas. A runner reuses the existing official CSPN loader, stratified metadata and hardware-aligned PA-W8A8 instrumentor; an activation-recorder callback observes exact pre-QDQ and post-QDQ Conv inputs without changing inference. A separate plotting script consumes only persisted CSV/NPZ artifacts.

**Tech Stack:** Python, PyTorch `torch.nn.functional.unfold`, NumPy, Matplotlib, CSV/JSON, pytest.

---

## File Structure

- Create `spn_quant/im2col_diagnostics.py`: Im2Col layout, streaming accumulators, recorder and table reductions.
- Create `scripts/run_nyu_cspn_w8a8_im2col.py`: official CSPN PA-W8A8 calibration/evaluation orchestration and artifact writing.
- Create `scripts/plot_nyu_cspn_w8a8_im2col.py`: deterministic 3D K-axis and spatial-waterfall rendering.
- Create `tests/test_im2col_diagnostics.py`: numerical layout, Conv equivalence, chunk invariance and validation tests.
- Create `tests/test_run_nyu_cspn_w8a8_im2col.py`: strict metadata/configuration/manifest contract tests.
- Create `tests/test_plot_nyu_cspn_w8a8_im2col.py`: plotting-from-artifacts test.
- Create `docs/2026-08-14-cspn-w8a8-im2col-results.md`: measured production results and interpretation.

### Task 1: Im2Col Layout and Exact Local Error

**Files:**
- Create: `spn_quant/im2col_diagnostics.py`
- Create: `tests/test_im2col_diagnostics.py`

- [ ] **Step 1: Write failing layout and Conv-equivalence tests**

Create tests using an asymmetric `Conv2d(2, 3, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0), bias=True)`. Assert that:

```python
layout = ConvIm2ColLayout.from_module("conv", module)
patches = layout.unfold(inputs)
weights = layout.flatten_weight(module.weight)
matrix_output = torch.einsum("bkm,ok->bom", patches, weights)
matrix_output = matrix_output.reshape(
    inputs.shape[0], module.out_channels,
    layout.output_shape(inputs)[0], layout.output_shape(inputs)[1])
matrix_output += module.bias.reshape(1, -1, 1, 1)
torch.testing.assert_close(matrix_output, module(inputs))
assert layout.decode_k(0) == (0, 0, 0)
assert layout.decode_k(5) == (0, 1, 2)
assert layout.decode_k(6) == (1, 0, 0)
```

Add rejection tests for `groups=2` and `ConvTranspose2d`.

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
python -m pytest -q tests/test_im2col_diagnostics.py
```

Expected: collection fails because `spn_quant.im2col_diagnostics` does not exist.

- [ ] **Step 3: Implement the strict layout API**

Implement:

```python
@dataclass(frozen=True)
class ConvIm2ColLayout:
    module: str
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]

    @classmethod
    def from_module(cls, name: str, module: nn.Conv2d): ...
    def output_shape(self, inputs: torch.Tensor) -> Tuple[int, int]: ...
    def unfold(self, inputs: torch.Tensor) -> torch.Tensor: ...
    def flatten_weight(self, weight: torch.Tensor) -> torch.Tensor: ...
    def decode_k(self, index: int) -> Tuple[int, int, int]: ...
```

Use `F.unfold` with the module geometry. Validate rank, channel count, weight shape, output dimensions and finite tensors. Do not accept grouped or transposed convolution.

Add `local_output_error(fp_patches, quantized_patches, fp_weight, quantized_weight)` returning `[B, M]` error energy from the exact FP and quantized pre-output-QDQ matrix products.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python -m pytest -q tests/test_im2col_diagnostics.py
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add spn_quant/im2col_diagnostics.py tests/test_im2col_diagnostics.py
git commit -m "feat: add strict Conv Im2Col diagnostics"
```

### Task 2: Streaming Channel-Offset and Spatial Statistics

**Files:**
- Modify: `spn_quant/im2col_diagnostics.py`
- Modify: `tests/test_im2col_diagnostics.py`

- [ ] **Step 1: Add failing exact-statistics and chunk-invariance tests**

Use hand-constructed FP/quantized activations with known new zeros and a
`1 x 2` kernel. Feed the same tensors with token chunk sizes `1` and `7` and
assert equality of exact counters, energies and layer reductions. Verify the
`channel, kh, kw` rows and exact top-token ordering.

```python
left = ConvIm2ColAccumulator(layout, percentile_capacity=64, token_topk=4)
right = ConvIm2ColAccumulator(layout, percentile_capacity=64, token_topk=4)
left.update(
    fp, qdq, codes, quantizer, fp_weight, qdq_weight,
    sample_index=9, token_chunk=1)
right.update(
    fp, qdq, codes, quantizer, fp_weight, qdq_weight,
    sample_index=9, token_chunk=7)
assert left.exact_state() == right.exact_state()
assert left.top_tokens() == right.top_tokens()
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
python -m pytest -q tests/test_im2col_diagnostics.py
```

Expected: failure because `ConvIm2ColAccumulator` is undefined.

- [ ] **Step 3: Implement bounded streaming accumulation**

Implement `ConvIm2ColAccumulator` with float64 exact sums for each
`Cin x Kh x Kw` cell and each layer. Track count, signal/error/zero-collapse/
rounding/clipping energy, reference/quantized/new zeros, saturation, min/max,
weight signal/error and exact per-token local output error. Use deterministic
bounded samples only for percentile estimates.

Implement these public methods:

```python
def update(self, reference, quantized, codes, quantizer,
           fp_weight, quantized_weight, sample_index, token_chunk): ...
def exact_state(self) -> Dict[str, object]: ...
def channel_offset_rows(self) -> List[Dict[str, object]]: ...
def channel_rows(self) -> List[Dict[str, object]]: ...
def offset_rows(self) -> List[Dict[str, object]]: ...
def layer_row(self) -> Dict[str, object]: ...
def spatial_arrays(self, sample_index: int) -> Dict[str, np.ndarray]: ...
def top_token_rows(self) -> List[Dict[str, object]]: ...
```

Classify quantization errors using the exact quantizer `scale_for`, `qmin` and
`qmax`, matching the existing activation-resolution definitions.

- [ ] **Step 4: Implement the strict activation-recorder adapter**

Add `CSPNW8A8Im2ColRecorder`. Its `record` method accepts the existing
instrumentor callback signature and records only `kind == "input"` for
supported ordinary Conv2d modules. Require call index zero, exact module
identity, W8 weight availability and one sample index set through
`begin_sample(index)` before every forward. `end_sample()` validates complete
coverage and clears per-sample spatial arrays after persistence.

- [ ] **Step 5: Run core tests and commit**

Run:

```bash
python -m pytest -q tests/test_im2col_diagnostics.py
```

Expected: all tests pass.

```bash
git add spn_quant/im2col_diagnostics.py tests/test_im2col_diagnostics.py
git commit -m "feat: stream CSPN Im2Col axis metrics"
```

### Task 3: Strict Official CSPN PA-W8A8 Runner

**Files:**
- Create: `scripts/run_nyu_cspn_w8a8_im2col.py`
- Create: `tests/test_run_nyu_cspn_w8a8_im2col.py`

- [ ] **Step 1: Write failing runner contract tests**

Test that the runner:

- accepts required CLI values without embedded path or sample defaults;
- reads `calibration_indices`, `evaluation_indices` and `seed` with direct
  dictionary indexing;
- requires exactly 128 unique train indices and 64 unique evaluation indices;
- selects only `FP32` and `PA_W8A8` from `build_propagation_configurations`;
- rejects any PA-W8A8 contract whose ordinary bits are not W8A8 or whose
  propagation fields are not A8/Q13;
- writes manifests without modifying source evaluation artifacts.

- [ ] **Step 2: Run the runner tests and verify failure**

Run:

```bash
python -m pytest -q tests/test_run_nyu_cspn_w8a8_im2col.py
```

Expected: collection fails because the runner does not exist.

- [ ] **Step 3: Implement required CLI and metadata validation**

Require these arguments:

```text
--device
--checkpoint
--stratified-metadata
--output-dir
--percentile-capacity
--token-topk
--token-chunk
--plot-layer-count
--plot-sample-count
```

Load the checkpoint and official model through existing CSPN helpers. Read all
configuration and metadata fields with direct attribute or dictionary access;
do not use `getattr`, `dict.get`, exception swallowing or fallback behavior.

- [ ] **Step 4: Reuse exact PA-W8A8 calibration and evaluation**

Build the existing hardware-aligned CSPN instrumentor and propagation adapter.
Run the 128 declared calibration samples, freeze observers, configure exact
`PA_W8A8`, attach `CSPNW8A8Im2ColRecorder`, and run the fixed 64 validation
samples. Persist each sample's Hout x Wout arrays immediately, then release
them before the next sample.

Capture FP32 and PA-W8A8 depth metrics before and during diagnostics and assert
that diagnostic attachment does not change predictions. Write all declared
CSV/JSON/NPZ artifacts atomically under the new output root.

- [ ] **Step 5: Run runner tests and commit**

Run:

```bash
python -m pytest -q tests/test_run_nyu_cspn_w8a8_im2col.py tests/test_im2col_diagnostics.py
```

Expected: all tests pass.

```bash
git add scripts/run_nyu_cspn_w8a8_im2col.py tests/test_run_nyu_cspn_w8a8_im2col.py
git commit -m "feat: run official CSPN W8A8 Im2Col analysis"
```

### Task 4: Persisted 3D-Line Plotting

**Files:**
- Create: `scripts/plot_nyu_cspn_w8a8_im2col.py`
- Create: `tests/test_plot_nyu_cspn_w8a8_im2col.py`

- [ ] **Step 1: Write failing plotting tests**

Create synthetic channel-offset CSV rows and spatial NPZ arrays. Invoke the
plot CLI in a subprocess and assert PNG/PDF pairs, a complete figure manifest,
unrotated readable labels, no figure title and nonblank PNG pixels.

- [ ] **Step 2: Run plotting tests and verify failure**

Run:

```bash
python -m pytest -q tests/test_plot_nyu_cspn_w8a8_im2col.py
```

Expected: failure because the plotting script does not exist.

- [ ] **Step 3: Implement K-axis and spatial waterfall figures**

Read only persisted artifacts. Generate one offset-colored 3D line per K-axis
metric and row-wise 3D waterfall lines for patch RMS, A8 error and local Conv
output error. Use Arial, no title, grid behind data, fixed view angles and
separate z axes for incompatible metrics. Record every selected module, sample,
metric, source path and output path in `figure_manifest.csv`.

- [ ] **Step 4: Run plotting tests and commit**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_w8a8_im2col.py
git add scripts/plot_nyu_cspn_w8a8_im2col.py tests/test_plot_nyu_cspn_w8a8_im2col.py
git commit -m "feat: plot CSPN Im2Col 3D distributions"
```

### Task 5: Production Run, Interpretation and Verification

**Files:**
- Create: `docs/2026-08-14-cspn-w8a8-im2col-results.md`

- [ ] **Step 1: Run the strict PA-W8A8 experiment**

Run the new runner in the CUDA-capable environment using the converged CSPN
checkpoint, current stratified metadata and a new output directory under
`/workspace/SPN_Quantization/profile_logs`. Do not write runtime artifacts to
`/tmp` or existing evaluation roots.

- [ ] **Step 2: Generate all declared figures**

Run the plotting script against the production output. Inspect representative
K-axis and spatial PNGs with the image viewer and verify that axes, offset lines
and spatial geometry are legible and nonblank.

- [ ] **Step 3: Validate artifact identities and reductions**

Run a strict validation script that checks module coverage, finite metrics,
128/64 metadata identities, channel/offset reduction equality, token array
shapes, direct Conv error agreement and unchanged FP32/PA-W8A8 depth metrics.

- [ ] **Step 4: Write measured findings**

Document the highest-risk layers, channels, offsets and spatial patches; state
whether W8A8 error follows activation magnitude, weight magnitude, borders or
specific semantic regions. Distinguish measured local Conv sensitivity from
the deferred endpoint-gradient attribution.

- [ ] **Step 5: Run complete verification and commit**

Run:

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: the full suite passes, diff check is empty and only the intended
result document remains before its commit.

```bash
git add docs/2026-08-14-cspn-w8a8-im2col-results.md
git commit -m "docs: report CSPN W8A8 Im2Col distributions"
```
