# CSPN Full Im2Col Matrix Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture and render complete, unsampled CSPN PA-W8A8 weight and activation Im2Col matrices as FP32/QDQ/error 3D line-curtain figures.

**Architecture:** A focused matrix module validates native Conv captures, reconstructs exact `W_col` and `X_col`, and converts every matrix element into one line-curtain coordinate. A strict capture CLI rebuilds the existing official CSPN PA-W8A8 experiment and stores one worst-sample native capture per selected module. A persisted-data plotting CLI renders one full panel at a time, releases its multi-million-point artists, then composes the three panels without sampling.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib `Line3DCollection`, Pillow, CSV/JSON, pytest.

---

## File Structure

- Create `spn_quant/im2col_matrix_visualization.py`: capture schema, strict recorder, full Im2Col reconstruction and no-sampling line coordinates.
- Create `tests/test_im2col_matrix_visualization.py`: numerical capture, reconstruction, orientation and coverage tests.
- Create `scripts/capture_nyu_cspn_w8a8_im2col_matrices.py`: official-model calibration and selected native tensor persistence.
- Create `tests/test_capture_nyu_cspn_w8a8_im2col_matrices.py`: manifest, ranking and CLI contract tests.
- Create `scripts/plot_nyu_cspn_w8a8_im2col_matrices.py`: persisted full-matrix line-curtain rendering and panel composition.
- Create `tests/test_plot_nyu_cspn_w8a8_im2col_matrices.py`: synthetic no-sampling figure and manifest tests.
- Modify `docs/2026-08-14-cspn-w8a8-im2col-results.md`: record full-matrix artifacts and measured interpretation.

### Task 1: Strict Native Capture and Full-Matrix Reconstruction

**Files:**
- Create: `spn_quant/im2col_matrix_visualization.py`
- Create: `tests/test_im2col_matrix_visualization.py`

- [ ] **Step 1: Write failing reconstruction tests**

Use an asymmetric `Conv2d(2, 3, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0))`. Test exact weight and activation matrix shape/order:

```python
capture = ConvMatrixCapture.from_tensors(
    module="conv", sample_index=7, layer=conv,
    reference_input=inputs, quantized_input=qdq_inputs,
    original_weight=conv.weight, quantized_weight=qdq_weight,
    activation_bits=8, activation_unsigned=False,
    activation_scale=torch.tensor(0.125))
matrices = capture.matrices()
assert matrices.reference_weight.shape == (3, 12)
assert matrices.reference_activation.shape[1] == 12
torch.testing.assert_close(
    matrices.reference_activation.T.unsqueeze(0),
    F.unfold(inputs, (2, 3), padding=(1, 0), stride=(2, 1)))
```

Add tests that absolute error matrices are exact and that non-finite tensors,
changed geometry and non-FP32 persisted arrays fail loudly.

- [ ] **Step 2: Run tests and verify the missing-module failure**

```bash
python -m pytest -q tests/test_im2col_matrix_visualization.py
```

Expected: collection fails because the matrix module does not exist.

- [ ] **Step 3: Implement capture schema and reconstruction**

Implement immutable `ConvMatrixGeometry`, `ConvMatrixCapture` and
`ConvUnfoldedMatrices`. Use `ConvIm2ColLayout.unfold` and `flatten_weight`,
transpose activation from `[1,K,M]` to `[M,K]`, and save only native FP32
tensors plus strict metadata in NPZ.

- [ ] **Step 4: Add failing recorder identity tests**

Configure `{7: {"first"}, 9: {"second"}}`; assert capture only for declared
module/sample pairs, one input call, channel dimension one and complete sample
coverage.

- [ ] **Step 5: Implement `FullMatrixCaptureRecorder`**

Consume the existing hardware-aligned callback. Copy selected reference/QDQ
feature maps and exact FP/W8 weights to CPU FP32. Read quantizer fields through
declared direct attributes `bits`, `unsigned` and `scale_for(reference)`;
unsupported formats raise.

- [ ] **Step 6: Verify and commit Task 1**

```bash
python -m pytest -q tests/test_im2col_matrix_visualization.py
git add spn_quant/im2col_matrix_visualization.py tests/test_im2col_matrix_visualization.py
git commit -m "feat: reconstruct full CSPN Im2Col matrices"
```

### Task 2: Exact No-Sampling Line Curtains

**Files:**
- Modify: `spn_quant/im2col_matrix_visualization.py`
- Modify: `tests/test_im2col_matrix_visualization.py`

- [ ] **Step 1: Write failing line-coverage tests**

```python
wide = np.arange(15, dtype=np.float32).reshape(3, 5)
tall = np.arange(15, dtype=np.float32).reshape(5, 3)
for matrix in (wide, tall):
    curtain = full_line_curtain(matrix)
    assert curtain.rendered_elements == matrix.size
    np.testing.assert_array_equal(curtain.reconstruct(), matrix)
```

Assert a wide matrix yields one line per row and a tall matrix one line per
column while preserving x=column and y=row coordinates.

- [ ] **Step 2: Run tests and verify failure**

```bash
python -m pytest -q tests/test_im2col_matrix_visualization.py
```

Expected: `full_line_curtain` is undefined.

- [ ] **Step 3: Implement exact line coordinates**

Implement `FullLineCurtain` and `full_line_curtain(matrix)`. Each line is
float32 `[N,3]` containing `(x=column, y=row, z=value)`. Reject empty,
non-rank-2, non-FP32 and non-finite matrices. Do not stride or select indices.

- [ ] **Step 4: Verify and commit Task 2**

```bash
python -m pytest -q tests/test_im2col_matrix_visualization.py
git add spn_quant/im2col_matrix_visualization.py tests/test_im2col_matrix_visualization.py
git commit -m "feat: preserve full matrices in 3D line curtains"
```

### Task 3: Official CSPN PA-W8A8 Capture CLI

**Files:**
- Create: `scripts/capture_nyu_cspn_w8a8_im2col_matrices.py`
- Create: `tests/test_capture_nyu_cspn_w8a8_im2col_matrices.py`

- [ ] **Step 1: Write failing protocol and ranking tests**

Construct synthetic source manifests and test:

```python
selection = select_worst_samples(experiment)
assert selection == {"conv1": 7, "decoder.conv": 11}
```

Require exactly the selected modules, highest local output error per module,
sample-index tie breaking, required CLI arguments and a nonexistent output
subdirectory.

- [ ] **Step 2: Run tests and verify the missing-script failure**

```bash
python -m pytest -q tests/test_capture_nyu_cspn_w8a8_im2col_matrices.py
```

- [ ] **Step 3: Implement strict source loading**

Require `--experiment-dir`, `--device` and `--fold-max-error`. Read all source
fields with direct dictionary indexing. Validate checkpoint, metadata,
architecture and exact W8A8/Q13 identities. Reuse the official runner's model,
calibration and hardware setup helpers.

- [ ] **Step 4: Implement calibrated selected capture**

Repeat 128-sample calibration, configure PA-W8A8, attach the recorder and run
each unique selected sample once. Save one capture per module and verify the
recorder does not alter prediction. Write a strict capture manifest containing
module, sample, native shapes, K size, token count and activation quantizer
metadata.

- [ ] **Step 5: Verify and commit Task 3**

```bash
python -m pytest -q tests/test_capture_nyu_cspn_w8a8_im2col_matrices.py tests/test_im2col_matrix_visualization.py
git add scripts/capture_nyu_cspn_w8a8_im2col_matrices.py tests/test_capture_nyu_cspn_w8a8_im2col_matrices.py
git commit -m "feat: capture official CSPN W8A8 Conv matrices"
```

### Task 4: Persisted Full-Matrix Plotter

**Files:**
- Create: `scripts/plot_nyu_cspn_w8a8_im2col_matrices.py`
- Create: `tests/test_plot_nyu_cspn_w8a8_im2col_matrices.py`

- [ ] **Step 1: Write failing synthetic plotting tests**

Create one distinct synthetic capture. Invoke the CLI and assert weight and
activation PNG/PDF triptychs, nonblank pixels, Arial-first style, shared
FP/QDQ z maxima, independent error maximum and:

```python
assert int(row["rendered_elements"]) == int(row["matrix_elements"])
assert row["sampling"] == "none"
```

- [ ] **Step 2: Run tests and verify the missing-script failure**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_w8a8_im2col_matrices.py
```

- [ ] **Step 3: Implement sequential full-panel rendering**

Render one panel with `Line3DCollection`, explicit bounds, Arial-first style,
grid below data and fixed camera. Render FP, QDQ and error panels sequentially
to in-memory PNG buffers and close each figure before constructing the next
full matrix. Compose the three raster panels with Pillow and save PNG/PDF,
without temporary files.

- [ ] **Step 4: Implement persisted CLI and figure manifest**

Require `--capture-dir`, `--output-dir`, `--font-size`, `--dpi`,
`--line-width`, `--elevation`, `--azimuth`. Generate both triptychs for every
capture. Record matrix shape/elements, rendered elements, line orientation,
`sampling=none`, paths and z maxima.

- [ ] **Step 5: Verify and commit Task 4**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_w8a8_im2col_matrices.py tests/test_im2col_matrix_visualization.py
git add scripts/plot_nyu_cspn_w8a8_im2col_matrices.py tests/test_plot_nyu_cspn_w8a8_im2col_matrices.py
git commit -m "feat: plot complete CSPN Im2Col matrices"
```

### Task 5: Production Capture, Figures and Verification

**Files:**
- Modify: `docs/2026-08-14-cspn-w8a8-im2col-results.md`

- [ ] **Step 1: Run official capture**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/capture_nyu_cspn_w8a8_im2col_matrices.py \
  --experiment-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_w8a8_im2col_64 \
  --device cuda:0 --fold-max-error 0.05
```

- [ ] **Step 2: Generate complete no-sampling figures**

```bash
python scripts/plot_nyu_cspn_w8a8_im2col_matrices.py \
  --capture-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_w8a8_im2col_64/full_matrix_visualization \
  --output-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_w8a8_im2col_64/full_matrix_visualization/figures \
  --font-size 13 --dpi 160 --line-width 0.25 --elevation 25 --azimuth -58
```

Expected: six captures and 12 triptych PNG/PDF pairs.

- [ ] **Step 3: Validate and inspect artifacts**

Reconstruct every capture and assert exact identity, FP32 finite values and
element counts. Assert no `.pending` files and unchanged source artifacts.
Inspect stem, deepest encoder and final decoder-fusion figures.

- [ ] **Step 4: Document measured findings**

Add matrix dimensions, channel@offset ridges, patch spikes and error
concentration to the result document. Distinguish local A8 error from
accumulated network error.

- [ ] **Step 5: Run complete verification and commit**

```bash
python -m pytest -q
git diff --check
git status --short
git add docs/2026-08-14-cspn-w8a8-im2col-results.md
git commit -m "docs: report full CSPN Im2Col matrix distributions"
```

Expected: full suite passes and worktree is clean.
