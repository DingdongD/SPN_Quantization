# CSPN Five-Frame Sequence Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a reproducible CSPN inference tool that predicts frames `0001` through `0005`, exports frame-level diagnostics, and measures unregistered adjacent-frame consistency.

**Architecture:** A single focused command-line module owns synthetic RGB/EXR loading, NYU-compatible preprocessing, deterministic shared sparse sampling, strict checkpoint loading, inference, metrics, and artifact rendering. Unit tests exercise pure data and metric functions with small arrays and toy models; the final smoke test uses the real five-frame dataset and A100 GPU.

**Tech Stack:** Python 3, PyTorch, torchvision, OpenCV OpenEXR, NumPy, Pillow, Matplotlib, unittest/pytest.

---

## File map

- Create `scripts/export_cspn_sequence_predictions.py`: reusable functions and CLI for loading, predicting, measuring, and rendering the five-frame pilot.
- Create `tests/test_export_cspn_sequence_predictions.py`: unit tests for masks, metrics, checkpoint normalisation, and output rendering.
- Generate files only under `/workspace/VoxelNet/cspn_predictions/`; generated data is not committed.

### Task 1: Depth validity, shared sparse sampling, and metrics

**Files:**
- Create: `tests/test_export_cspn_sequence_predictions.py`
- Create: `scripts/export_cspn_sequence_predictions.py`

- [ ] **Step 1: Write failing unit tests for validity and deterministic sampling**

Add imports and tests that define the exact public interfaces:

```python
import unittest

import numpy as np

from scripts import export_cspn_sequence_predictions as sequence


class DepthInputTest(unittest.TestCase):
    def test_sanitize_depth_rejects_nonfinite_nonpositive_and_over_range(self):
        depth = np.array([[1.0, np.inf, 0.0, -1.0, 10.1]], dtype=np.float32)
        clean, valid = sequence.sanitize_depth(depth, max_depth=10.0)
        np.testing.assert_array_equal(valid, [[True, False, False, False, False]])
        np.testing.assert_array_equal(clean, [[1.0, 0.0, 0.0, 0.0, 0.0]])

    def test_shared_sparse_depth_uses_same_500_coordinates(self):
        depths = np.stack([
            np.full((24, 32), 1.0, dtype=np.float32),
            np.full((24, 32), 2.0, dtype=np.float32),
        ])
        sparse, mask = sequence.build_shared_sparse_depths(
            depths, depths > 0.0, count=500, seed=2026)
        self.assertEqual(int(mask.sum()), 500)
        self.assertEqual(int(np.count_nonzero(sparse[0])), 500)
        self.assertEqual(int(np.count_nonzero(sparse[1])), 500)
        np.testing.assert_array_equal(sparse[0] > 0.0, sparse[1] > 0.0)

    def test_shared_sparse_depth_rejects_too_few_common_pixels(self):
        depths = np.ones((2, 4, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "common valid pixels"):
            sequence.build_shared_sparse_depths(
                depths, depths > 0.0, count=500, seed=2026)
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
python -m pytest -q tests/test_export_cspn_sequence_predictions.py
```

Expected: collection fails because `scripts.export_cspn_sequence_predictions` does not exist.

- [ ] **Step 3: Implement the validity and sparse-sampling functions**

Create the script with these functions and constants:

```python
#!/usr/bin/env python3
"""Export CSPN predictions and unregistered temporal diagnostics."""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np

MAX_DEPTH = 10.0
OUTPUT_HEIGHT = 228
OUTPUT_WIDTH = 304


def sanitize_depth(depth, max_depth=MAX_DEPTH):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth))
    return np.where(valid, depth, 0.0).astype(np.float32), valid


def build_shared_sparse_depths(depths, valid_masks, count=500, seed=2026):
    depths = np.asarray(depths, dtype=np.float32)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    if depths.shape != valid_masks.shape or depths.ndim != 3:
        raise ValueError("depths and valid_masks must have shape [frames, height, width]")
    common = np.all(valid_masks, axis=0)
    candidates = np.flatnonzero(common)
    if candidates.size < int(count):
        raise ValueError("common valid pixels %d are fewer than %d" %
                         (candidates.size, count))
    chosen = np.random.default_rng(seed).choice(candidates, size=count, replace=False)
    mask = np.zeros(common.size, dtype=bool)
    mask[chosen] = True
    mask = mask.reshape(common.shape)
    return (depths * mask[None]).astype(np.float32), mask
```

- [ ] **Step 4: Add failing tests for frame and temporal metrics**

```python
class MetricTest(unittest.TestCase):
    def test_frame_metrics_use_only_valid_ground_truth(self):
        gt = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
        pred = np.array([[2.0, 2.0, 9.0]], dtype=np.float32)
        result = sequence.frame_metrics(gt, pred, gt > 0.0)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5))
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["abs_rel"], 0.5)

    def test_temporal_metrics_measure_change_residual(self):
        gt = np.array([[[1.0, 2.0]], [[2.0, 4.0]]], dtype=np.float32)
        pred = np.array([[[1.0, 2.0]], [[3.0, 5.0]]], dtype=np.float32)
        valid = np.ones_like(gt, dtype=bool)
        rows, maps = sequence.temporal_metrics(gt, pred, valid, [1, 2])
        self.assertEqual(rows[0]["pair"], "0001->0002")
        self.assertAlmostEqual(rows[0]["rmse"], 1.0)
        np.testing.assert_array_equal(maps[0]["residual"], [[1.0, 1.0]])
```

- [ ] **Step 5: Implement metrics and verify the task**

Implement `frame_metrics(gt, pred, valid)` using float64 reductions for RMSE,
MAE, AbsRel, coverage, and valid-pixel count. Implement
`temporal_metrics(depths, predictions, valid_masks, frame_ids)` by taking
adjacent differences and returning one row and one map dictionary per pair.
Raise `ValueError` on empty masks or non-finite metric results.

Run:

```bash
python -m pytest -q tests/test_export_cspn_sequence_predictions.py
```

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add scripts/export_cspn_sequence_predictions.py tests/test_export_cspn_sequence_predictions.py
git commit -m "feat: add CSPN sequence data primitives"
```

### Task 2: Real RGB/EXR preprocessing and strict CSPN checkpoint loading

**Files:**
- Modify: `scripts/export_cspn_sequence_predictions.py`
- Modify: `tests/test_export_cspn_sequence_predictions.py`

- [ ] **Step 1: Write failing preprocessing and checkpoint tests**

```python
from pathlib import Path
import tempfile

import torch


class PreprocessingTest(unittest.TestCase):
    def test_preprocess_pair_has_cspn_geometry(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        rgb_out, depth_out, valid = sequence.preprocess_pair(rgb, depth)
        self.assertEqual(rgb_out.shape, (3, 228, 304))
        self.assertEqual(depth_out.shape, (228, 304))
        self.assertEqual(valid.shape, (228, 304))
        self.assertTrue(valid.all())


class CheckpointTest(unittest.TestCase):
    def test_normalize_state_accepts_only_known_legacy_difference(self):
        model = torch.nn.Linear(2, 1)
        state = {"module.weight": model.weight.detach().clone(),
                 "module.bias": model.bias.detach().clone(),
                 "module.post_process_layer.sum_conv.weight": torch.ones(1, 8, 1, 1, 1)}
        report = sequence.load_compatible_state(model, state, allowed_missing=())
        self.assertEqual(report["ignored"], ["post_process_layer.sum_conv.weight"])

    def test_normalize_state_rejects_unknown_key(self):
        model = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        state["unknown"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected checkpoint keys"):
            sequence.load_compatible_state(model, state, allowed_missing=())
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest -q tests/test_export_cspn_sequence_predictions.py -k 'Preprocessing or Checkpoint'
```

Expected: failures report missing `preprocess_pair` and
`load_compatible_state`.

- [ ] **Step 3: Implement image loading and preprocessing**

Add `read_exr_depth(path)` using `cv2.imread(..., cv2.IMREAD_UNCHANGED)`.
Require three equal channels within `np.allclose(..., equal_nan=True)`, then
select channel zero. Add `load_rgb(path)` with Pillow. Add
`preprocess_pair(rgb, depth)` using torchvision functional resize to short
side 240 and centre crop to `228 x 304`; RGB uses bilinear and depth uses
nearest interpolation. Return RGB as `float32 CHW` in `[0, 1]`, sanitized
depth, and its validity mask.

- [ ] **Step 4: Implement strict checkpoint loading and model construction**

Add `load_compatible_state(model, state, allowed_missing)` that strips one
leading `module.`, validates and removes an all-ones
`post_process_layer.sum_conv.weight` of shape `(1, 8, 1, 1, 1)`, calls
`load_state_dict(strict=False)`, and raises for every missing key outside
`allowed_missing` or every unexpected key. Add `build_cspn(checkpoint,
device)` that constructs `resnet50` with 24 iterations and permits only
`*_up_pool.weights` derived buffers to be missing.

- [ ] **Step 5: Run tests and commit Task 2**

```bash
python -m pytest -q tests/test_export_cspn_sequence_predictions.py
git add scripts/export_cspn_sequence_predictions.py tests/test_export_cspn_sequence_predictions.py
git commit -m "feat: load synthetic frames for CSPN inference"
```

Expected: all sequence prediction tests pass.

### Task 3: Artifact export, plots, metadata, and CLI

**Files:**
- Modify: `scripts/export_cspn_sequence_predictions.py`
- Modify: `tests/test_export_cspn_sequence_predictions.py`

- [ ] **Step 1: Write a failing artifact-export test**

```python
class ArtifactTest(unittest.TestCase):
    def test_write_artifacts_creates_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_ids = [1, 2]
            rgb = np.zeros((2, 3, 8, 10), dtype=np.float32)
            depth = np.ones((2, 8, 10), dtype=np.float32)
            sparse = depth.copy()
            pred = depth.copy()
            valid = np.ones_like(depth, dtype=bool)
            manifest = sequence.write_artifacts(
                Path(tmp), frame_ids, rgb, sparse, depth, pred, valid,
                checkpoint_sha256="abc", model_config={"iteration": 24})
            self.assertEqual(len(manifest["frame_npz"]), 2)
            self.assertEqual(len(manifest["frame_panels"]), 2)
            for name in ("sequence_overview", "temporal_overview",
                         "frame_metrics", "temporal_metrics", "metadata"):
                self.assertTrue(Path(manifest[name]).is_file())
```

- [ ] **Step 2: Run the artifact test and verify failure**

Run:

```bash
python -m pytest -q tests/test_export_cspn_sequence_predictions.py::ArtifactTest -v
```

Expected: failure reports missing `write_artifacts`.

- [ ] **Step 3: Implement artifacts and visualisations**

Use Matplotlib's `Agg` backend. Implement:

- `write_frame_panel`: five columns for RGB, sparse depth, ground truth,
  prediction, and absolute error, with depth range 0–10 m and error range
  0–3 m;
- `write_sequence_overview`: five rows, one per frame, with RGB, ground truth,
  prediction, and error;
- `write_temporal_overview`: four rows, one per adjacent pair, with truth
  change, prediction change, and residual on a symmetric colour scale;
- `write_csv`: stable explicit columns for frame and temporal rows;
- `write_artifacts`: five NPZ files, five frame panels, two overview images,
  two CSV files, and JSON metadata containing the literal warning
  `temporal_alignment: unregistered`.

Every NPZ must contain `frame_id`, `rgb`, `sparse`, `gt`, `pred_raw`,
`pred_clamped`, `valid`, and `abs_err`. Close every Matplotlib figure after
saving.

- [ ] **Step 4: Implement inference orchestration and CLI**

Add `predict_sequence(model, rgb, sparse, device)` that applies the existing
`legacy_cspn_rgb`, concatenates RGB and sparse depth into shape
`1 x 4 x 228 x 304`, runs `torch.inference_mode()`, and rejects non-finite
outputs. Add CLI arguments with these defaults:

```text
--data-root /workspace/VoxelNet/train
--scene BeachApartmentInterior_My_ir
--frames 1 2 3 4 5
--checkpoint /workspace/VoxelNet/cspn_models/best_model.pth
--sparse-count 500
--seed 2026
--device cuda:0
--out-dir /workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005
```

The `main()` function resolves all frame pairs, preprocesses all five before
sampling, verifies the model input shapes, runs inference, computes SHA-256,
and calls `write_artifacts`.

- [ ] **Step 5: Run all unit tests and commit Task 3**

```bash
python -m pytest -q tests/test_export_cspn_sequence_predictions.py
git add scripts/export_cspn_sequence_predictions.py tests/test_export_cspn_sequence_predictions.py
git commit -m "feat: export CSPN sequence predictions"
```

Expected: all sequence prediction tests pass.

### Task 4: Real five-frame GPU validation

**Files:**
- Execute: `scripts/export_cspn_sequence_predictions.py`
- Verify generated data under `/workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005/`

- [ ] **Step 1: Run the five-frame pilot**

```bash
python scripts/export_cspn_sequence_predictions.py \
  --data-root /workspace/VoxelNet/train \
  --scene BeachApartmentInterior_My_ir \
  --frames 1 2 3 4 5 \
  --checkpoint /workspace/VoxelNet/cspn_models/best_model.pth \
  --sparse-count 500 \
  --seed 2026 \
  --device cuda:0 \
  --out-dir /workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005
```

Expected: exit code 0 and a summary naming five predictions and four temporal
pairs.

- [ ] **Step 2: Verify the generated artifact contract**

Run this read-only verifier:

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

import numpy as np

root = Path('/workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005')
npz_paths = sorted(root.glob('frame_*.npz'))
panel_paths = sorted(root.glob('frame_*_panel.png'))
assert len(npz_paths) == 5, len(npz_paths)
assert len(panel_paths) == 5, len(panel_paths)
for path in npz_paths:
    with np.load(path, allow_pickle=False) as item:
        assert item['pred_raw'].shape == (228, 304)
        assert np.isfinite(item['pred_raw']).all()
        assert np.count_nonzero(item['sparse']) == 500
for csv_name, expected_rows in [('frame_metrics.csv', 5),
                                ('temporal_metrics.csv', 4)]:
    with (root / csv_name).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected_rows
    for row in rows:
        for key in ('rmse', 'mae'):
            assert np.isfinite(float(row[key]))
metadata = json.loads((root / 'run_metadata.json').read_text(encoding='utf-8'))
assert metadata['temporal_alignment'] == 'unregistered'
for name in ('sequence_overview.png', 'temporal_overview_unregistered.png'):
    assert (root / name).is_file()
print('verified 5 frames, 4 temporal pairs, and all artifact groups')
PY
```

Expected: `verified 5 frames, 4 temporal pairs, and all artifact groups`.

- [ ] **Step 3: Inspect visual output**

Open `sequence_overview.png` and `temporal_overview_unregistered.png`. Confirm
RGB, truth, prediction, and error maps are oriented consistently; colour bars
are readable; invalid depth is not rendered as a valid long range; and no
subplot is blank.

- [ ] **Step 4: Run regression verification**

```bash
python -m pytest -q tests/test_export_cspn_sequence_predictions.py
git status --short
```

Expected: all new tests pass. Git status shows only pre-existing unrelated
changes plus no uncommitted changes from this feature.

- [ ] **Step 5: Report results**

Report the five per-frame RMSE/MAE/AbsRel values, the four unregistered
temporal residual RMSE values, checkpoint/model compatibility, output path,
and direct links to both overview images. Explicitly state that the temporal
numbers are not motion-compensated.
