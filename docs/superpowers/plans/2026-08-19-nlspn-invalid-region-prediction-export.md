# NLSPN Invalid-Region Prediction Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export complete specialized NLSPN predictions, 16-bit metric depth, GT-plus-prediction fills, and invalid masks for the 150 retained formal-test frames without modifying the validated training result.

**Architecture:** Add one standalone exporter whose pure pixel-conversion functions are independently testable. The exporter validates the exact 30-window source contract, writes a temporary sibling tree, validates every generated PNG and manifest row against the source arrays, rechecks source NPZ hashes, and atomically publishes a previously absent destination.

**Tech Stack:** Python 3.11, NumPy, Pillow, Matplotlib `viridis`, CSV, SHA-256, pytest.

---

## File structure

- Create `scripts/export_nlspn_invalid_region_predictions.py`: pixel encoders, source validation, manifest writing, export validation, atomic publication, and CLI.
- Create `tests/test_export_nlspn_invalid_region_predictions.py`: unit and integration coverage for every output contract.
- Create through the exporter `/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1_prediction_exports`: the real 150-frame output; this stays outside Git.

### Task 1: Implement exact pixel products

**Files:**
- Create: `scripts/export_nlspn_invalid_region_predictions.py`
- Create: `tests/test_export_nlspn_invalid_region_predictions.py`

- [ ] **Step 1: Write failing pixel-contract tests**

Create the test module with these tests:

```python
import numpy as np
import pytest

from scripts import export_nlspn_invalid_region_predictions as exporter


def test_depth_to_millimetres_rounds_and_clips():
    depth = np.array([[-1.0, 0.0, 1.234, 1.2346, 10.0, 11.0]], dtype=np.float32)
    actual = exporter.depth_to_millimetres(depth)
    assert actual.dtype == np.uint16
    assert actual.tolist() == [[0, 0, 1234, 1235, 10000, 10000]]


def test_compose_fill_uses_gt_only_where_valid():
    gt = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    prediction = np.array([[8.0, 8.1], [8.2, 8.3]], dtype=np.float32)
    actual = exporter.compose_gt_with_prediction(gt, valid, prediction)
    np.testing.assert_array_equal(actual, [[1.0, 8.1], [3.0, 8.3]])


def test_invalid_mask_is_white_only_for_invalid_gt():
    valid = np.array([[True, False], [False, True]])
    actual = exporter.invalid_mask(valid)
    assert actual.dtype == np.uint8
    assert actual.tolist() == [[0, 255], [255, 0]]


def test_colorize_depth_has_fixed_zero_to_ten_metre_scale():
    depth = np.array([[0.0, 5.0, 10.0, -1.0, 11.0]], dtype=np.float32)
    actual = exporter.colorize_depth(depth)
    assert actual.shape == (1, 5, 3)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[0, 0], actual[0, 3])
    np.testing.assert_array_equal(actual[0, 2], actual[0, 4])
    assert not np.array_equal(actual[0, 0], actual[0, 1])
    assert not np.array_equal(actual[0, 1], actual[0, 2])


@pytest.mark.parametrize("function_name", [
    "depth_to_millimetres", "colorize_depth"])
def test_prediction_products_reject_nonfinite_values(function_name):
    function = getattr(exporter, function_name)
    with pytest.raises(ValueError, match="finite"):
        function(np.array([[np.nan]], dtype=np.float32))
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
python -m pytest -q tests/test_export_nlspn_invalid_region_predictions.py -x
```

Expected: collection fails because `export_nlspn_invalid_region_predictions` does not exist.

- [ ] **Step 3: Add the pure conversion functions**

Create `scripts/export_nlspn_invalid_region_predictions.py` with imports,
constants, and these functions:

```python
#!/usr/bin/env python3
"""Export full NLSPN predictions and explicit invalid-region fills."""

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
import numpy as np
from PIL import Image


SPATIAL_SHAPE = (228, 304)
WINDOW_COUNT = 30
FRAMES_PER_WINDOW = 5
PNG_NAMES = (
    "specialized_full_color.png",
    "specialized_depth_mm.png",
    "gt_with_prediction_fill.png",
    "invalid_mask.png",
)
MANIFEST_FIELDS = (
    "window", "scene", "frame_id", "invalid_pixel_count",
    "invalid_fraction", "prediction_min_m", "prediction_max_m",
    "prediction_mean_m", "specialized_full_color",
    "specialized_depth_mm", "gt_with_prediction_fill", "invalid_mask",
)


def _finite_depth(depth):
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("depth must be two-dimensional")
    if not np.isfinite(value).all():
        raise ValueError("depth values must be finite")
    return value


def depth_to_millimetres(depth):
    value = _finite_depth(depth)
    return np.rint(np.clip(value, 0.0, 10.0) * 1000.0).astype(np.uint16)


def compose_gt_with_prediction(gt, valid, prediction):
    gt = _finite_depth(gt)
    prediction = _finite_depth(prediction)
    valid = np.asarray(valid, dtype=bool)
    if gt.shape != prediction.shape or gt.shape != valid.shape:
        raise ValueError("GT, validity, and prediction shapes differ")
    return np.where(valid, gt, prediction).astype(np.float32, copy=False)


def invalid_mask(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("validity mask must be two-dimensional")
    return np.where(valid, 0, 255).astype(np.uint8)


def colorize_depth(depth):
    value = _finite_depth(depth)
    normalized = np.clip(value, 0.0, 10.0) / 10.0
    return cm.get_cmap("viridis")(normalized, bytes=True)[..., :3].astype(np.uint8)
```

- [ ] **Step 4: Run the pixel tests**

Run:

```bash
python -m pytest -q tests/test_export_nlspn_invalid_region_predictions.py
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit the pixel products**

```bash
git add scripts/export_nlspn_invalid_region_predictions.py \
  tests/test_export_nlspn_invalid_region_predictions.py
git commit -m "feat: encode NLSPN invalid-region depth products"
```

### Task 2: Validate the 30-window source contract

**Files:**
- Modify: `scripts/export_nlspn_invalid_region_predictions.py`
- Modify: `tests/test_export_nlspn_invalid_region_predictions.py`

- [ ] **Step 1: Add a synthetic formal-tree fixture and failing source tests**

Append:

```python
from pathlib import Path


def make_source(root):
    windows = root / "windows"
    windows.mkdir(parents=True)
    identities = []
    for index in range(30):
        scene = "room3" if index < 15 else "room7"
        start = index * 10 + 1
        name = "{:02d}_{}_{:04d}_{:04d}".format(
            index + 1, scene, start, start + 4)
        directory = windows / name
        directory.mkdir()
        frame_ids = np.arange(start, start + 5, dtype=np.int32)
        gt = np.full((5, 228, 304), 2.0, dtype=np.float32)
        valid = np.ones((5, 228, 304), dtype=bool)
        valid[:, 70:90, 90:120] = False
        gt[~valid] = 0.0
        specialized = np.full((5, 228, 304), 8.25, dtype=np.float32)
        np.savez_compressed(
            directory / "predictions.npz",
            scenes=np.asarray([scene] * 5), frame_ids=frame_ids,
            gt=gt, valid=valid, specialized=specialized)
        identities.extend((scene, int(frame_id)) for frame_id in frame_ids)
    return windows, identities


def test_load_source_frames_accepts_exact_formal_geometry(tmp_path):
    windows, identities = make_source(tmp_path)
    frames, fingerprints = exporter.load_source_frames(windows)
    assert [(row["scene"], row["frame_id"]) for row in frames] == identities
    assert len(frames) == 150
    assert len(fingerprints) == 30
    assert all(len(value) == 64 for value in fingerprints.values())


def test_load_source_frames_rejects_duplicate_identity(tmp_path):
    windows, _ = make_source(tmp_path)
    path = sorted(windows.iterdir())[1] / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    first = sorted(windows.iterdir())[0] / "predictions.npz"
    with np.load(first, allow_pickle=False) as archive:
        payload["scenes"][0] = archive["scenes"][0]
        payload["frame_ids"][0] = archive["frame_ids"][0]
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="duplicate"):
        exporter.load_source_frames(windows)


def test_load_source_frames_rejects_nonfinite_prediction(tmp_path):
    windows, _ = make_source(tmp_path)
    path = sorted(windows.iterdir())[0] / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["specialized"][0, 0, 0] = np.nan
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="finite"):
        exporter.load_source_frames(windows)


def test_load_source_frames_rejects_nonboolean_validity(tmp_path):
    windows, _ = make_source(tmp_path)
    path = sorted(windows.iterdir())[0] / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["valid"] = payload["valid"].astype(np.uint8)
    payload["valid"][0, 0, 0] = 2
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="boolean-compatible"):
        exporter.load_source_frames(windows)
```

- [ ] **Step 2: Run the new tests and verify the missing API failure**

Run:

```bash
python -m pytest -q tests/test_export_nlspn_invalid_region_predictions.py -x
```

Expected: FAIL because `load_source_frames` is undefined.

- [ ] **Step 3: Implement source hashing and aligned-frame loading**

Add:

```python
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_frames(windows_root):
    windows_root = Path(windows_root)
    directories = sorted(path for path in windows_root.iterdir() if path.is_dir())
    if len(directories) != WINDOW_COUNT:
        raise ValueError("source must contain exactly 30 window directories")
    frames = []
    fingerprints = {}
    seen = set()
    for directory in directories:
        path = directory / "predictions.npz"
        if not path.is_file():
            raise ValueError("window is missing predictions.npz: {}".format(directory.name))
        fingerprints[directory.name] = file_sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            required = ("scenes", "frame_ids", "gt", "valid", "specialized")
            if any(name not in archive.files for name in required):
                raise ValueError("prediction archive is missing required arrays")
            values = {name: archive[name] for name in required}
        if values["scenes"].shape != (5,) or values["frame_ids"].shape != (5,):
            raise ValueError("window identity arrays must contain five frames")
        for name in ("gt", "valid", "specialized"):
            if values[name].shape != (5,) + SPATIAL_SHAPE:
                raise ValueError("{} has the wrong shape".format(name))
        scenes = [str(value) for value in values["scenes"]]
        frame_ids = [int(value) for value in values["frame_ids"]]
        if len(set(scenes)) != 1 or any(
                right != left + 1 for left, right in zip(frame_ids, frame_ids[1:])):
            raise ValueError("window frames must be one scene and consecutive")
        for offset, (scene, frame_id) in enumerate(zip(scenes, frame_ids)):
            identity = (scene, frame_id)
            if identity in seen:
                raise ValueError("source contains duplicate scene/frame identity")
            seen.add(identity)
            prediction = _finite_depth(values["specialized"][offset])
            gt = _finite_depth(values["gt"][offset])
            raw_valid = np.asarray(values["valid"][offset])
            if raw_valid.dtype != np.bool_ and not np.isin(raw_valid, (0, 1)).all():
                raise ValueError("validity values must be boolean-compatible")
            valid = raw_valid.astype(bool, copy=False)
            frames.append({
                "window": directory.name, "scene": scene,
                "frame_id": frame_id, "gt": gt, "valid": valid,
                "specialized": prediction,
            })
    if len(frames) != WINDOW_COUNT * FRAMES_PER_WINDOW:
        raise ValueError("source must contain exactly 150 frames")
    return frames, fingerprints
```

- [ ] **Step 4: Run all exporter tests**

Run:

```bash
python -m pytest -q tests/test_export_nlspn_invalid_region_predictions.py
```

Expected: 10 tests pass.

- [ ] **Step 5: Commit source validation**

```bash
git add scripts/export_nlspn_invalid_region_predictions.py \
  tests/test_export_nlspn_invalid_region_predictions.py
git commit -m "feat: validate retained NLSPN prediction windows"
```

### Task 3: Write, validate, and atomically publish an export

**Files:**
- Modify: `scripts/export_nlspn_invalid_region_predictions.py`
- Modify: `tests/test_export_nlspn_invalid_region_predictions.py`

- [ ] **Step 1: Add failing end-to-end export tests**

Append:

```python
import csv
from PIL import Image


def test_export_predictions_writes_exact_products_and_manifest(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    result = exporter.export_predictions(windows, target)
    assert result == {"window_count": 30, "frame_count": 150, "png_count": 600}
    with (target / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 150
    first = rows[0]
    frame = target / "windows" / first["window"] / "frame_0001"
    assert {path.name for path in frame.iterdir()} == set(exporter.PNG_NAMES)
    with Image.open(frame / "specialized_depth_mm.png") as image:
        assert image.size == (304, 228)
        depth = np.asarray(image)
    assert depth.dtype in (np.dtype("uint16"), np.dtype("int32"))
    assert np.all(depth == 8250)
    with Image.open(frame / "invalid_mask.png") as image:
        mask = np.asarray(image)
    assert set(np.unique(mask)) == {0, 255}
    exporter.validate_export(target, windows)


def test_export_predictions_refuses_to_overwrite_completed_target(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        exporter.export_predictions(windows, target)
    assert marker.read_text() == "keep"


def test_validate_export_detects_changed_hybrid_pixel(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    exporter.export_predictions(windows, target)
    path = target / "windows/01_room3_0001_0005/frame_0001/gt_with_prediction_fill.png"
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    array[0, 0] = 0
    Image.fromarray(array, mode="RGB").save(path)
    with pytest.raises(ValueError, match="hybrid"):
        exporter.validate_export(target, windows)


def test_validate_export_rejects_extra_frame_file(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    exporter.export_predictions(windows, target)
    path = target / "windows/01_room3_0001_0005/frame_0001/extra.txt"
    path.write_text("unexpected")
    with pytest.raises(ValueError, match="exact contract"):
        exporter.validate_export(target, windows)
```

- [ ] **Step 2: Run the tests and verify export API failure**

Run:

```bash
python -m pytest -q tests/test_export_nlspn_invalid_region_predictions.py -x
```

Expected: FAIL because `export_predictions` is undefined.

- [ ] **Step 3: Implement PNG writers, manifest rows, and validation**

Add the following responsibilities to the exporter:

```python
def _save_png(path, array, mode):
    Image.fromarray(array, mode=mode).save(path)


def _frame_paths(window, frame_id):
    prefix = Path("windows") / window / "frame_{:04d}".format(frame_id)
    return {name[:-4]: prefix / name for name in PNG_NAMES}


def write_export(root, frames):
    root = Path(root)
    rows = []
    for frame in frames:
        paths = _frame_paths(frame["window"], frame["frame_id"])
        directory = root / next(iter(paths.values())).parent
        directory.mkdir(parents=True, exist_ok=False)
        prediction = frame["specialized"]
        hybrid = compose_gt_with_prediction(
            frame["gt"], frame["valid"], prediction)
        mask = invalid_mask(frame["valid"])
        products = {
            "specialized_full_color": (colorize_depth(prediction), "RGB"),
            "specialized_depth_mm": (depth_to_millimetres(prediction), "I;16"),
            "gt_with_prediction_fill": (colorize_depth(hybrid), "RGB"),
            "invalid_mask": (mask, "L"),
        }
        for name, (array, mode) in products.items():
            _save_png(root / paths[name], array, mode)
        invalid_count = int((~frame["valid"]).sum())
        rows.append({
            "window": frame["window"], "scene": frame["scene"],
            "frame_id": frame["frame_id"],
            "invalid_pixel_count": invalid_count,
            "invalid_fraction": invalid_count / float(np.prod(SPATIAL_SHAPE)),
            "prediction_min_m": float(prediction.min()),
            "prediction_max_m": float(prediction.max()),
            "prediction_mean_m": float(prediction.mean()),
            **{name: str(path) for name, path in paths.items()},
        })
    with (root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_export(root, source_windows):
    root = Path(root).resolve()
    frames, _ = load_source_frames(source_windows)
    expected = {(row["scene"], row["frame_id"]): row for row in frames}
    if {path.name for path in root.iterdir()} != {"manifest.csv", "windows"}:
        raise ValueError("export root differs from the exact contract")
    expected_windows = {row["window"] for row in frames}
    actual_windows = {
        path.name for path in (root / "windows").iterdir() if path.is_dir()}
    if actual_windows != expected_windows:
        raise ValueError("export window directories differ from source")
    with (root / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 150:
        raise ValueError("manifest must contain exactly 150 rows")
    png_count = 0
    seen = set()
    for row in rows:
        key = (row["scene"], int(row["frame_id"]))
        if key not in expected or key in seen:
            raise ValueError("manifest identity differs from source")
        seen.add(key)
        source = expected[key]
        if row["window"] != source["window"]:
            raise ValueError("manifest window differs from source")
        invalid_count = int((~source["valid"]).sum())
        expected_scalars = {
            "invalid_pixel_count": invalid_count,
            "invalid_fraction": invalid_count / float(np.prod(SPATIAL_SHAPE)),
            "prediction_min_m": float(source["specialized"].min()),
            "prediction_max_m": float(source["specialized"].max()),
            "prediction_mean_m": float(source["specialized"].mean()),
        }
        if int(row["invalid_pixel_count"]) != expected_scalars["invalid_pixel_count"]:
            raise ValueError("manifest invalid pixel count differs from source")
        for name in (
                "invalid_fraction", "prediction_min_m", "prediction_max_m",
                "prediction_mean_m"):
            if not math.isclose(float(row[name]), expected_scalars[name],
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("manifest {} differs from source".format(name))
        expected_arrays = {
            "specialized_full_color": colorize_depth(source["specialized"]),
            "specialized_depth_mm": depth_to_millimetres(source["specialized"]),
            "gt_with_prediction_fill": colorize_depth(compose_gt_with_prediction(
                source["gt"], source["valid"], source["specialized"])),
            "invalid_mask": invalid_mask(source["valid"]),
        }
        expected_paths = _frame_paths(source["window"], source["frame_id"])
        frame_directory = root / next(iter(expected_paths.values())).parent
        if {path.name for path in frame_directory.iterdir()} != set(PNG_NAMES):
            raise ValueError("frame directory differs from the exact contract")
        for name, expected_array in expected_arrays.items():
            if row[name] != str(expected_paths[name]):
                raise ValueError("manifest PNG path differs from exact contract")
            relative = Path(row[name])
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError("manifest PNG path escapes or is missing")
            with Image.open(path) as image:
                actual = np.asarray(image)
            if actual.shape != expected_array.shape:
                raise ValueError("PNG has the wrong shape")
            if not np.array_equal(actual, expected_array):
                label = "hybrid" if name == "gt_with_prediction_fill" else name
                raise ValueError("{} PNG differs from source formula".format(label))
            png_count += 1
    if set(expected) != seen or png_count != 600:
        raise ValueError("export must contain exactly 600 PNGs")
    return {"window_count": 30, "frame_count": 150, "png_count": 600}
```

- [ ] **Step 4: Implement atomic publication and CLI**

Add:

```python
def export_predictions(source_windows, target):
    source_windows = Path(source_windows).resolve()
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    frames, before = load_source_frames(source_windows)
    temporary = Path(tempfile.mkdtemp(
        prefix="." + target.name + "-", dir=str(target.parent)))
    try:
        write_export(temporary, frames)
        result = validate_export(temporary, source_windows)
        _, after = load_source_frames(source_windows)
        if before != after:
            raise RuntimeError("source predictions changed during export")
        os.replace(str(temporary), str(target))
        return validate_export(target, source_windows)
    except Exception:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise


def make_parser():
    parser = argparse.ArgumentParser(
        description="Export full NLSPN invalid-region predictions")
    parser.add_argument(
        "--source-windows",
        default="/workspace/VoxelNet/nlspn_finetune/"
                "full_rmse_scene_disjoint_v1/windows")
    parser.add_argument(
        "--target",
        default="/workspace/VoxelNet/nlspn_finetune/"
                "full_rmse_scene_disjoint_v1_prediction_exports")
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    result = export_predictions(cli.source_windows, cli.target)
    print(result)
    return result


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run focused tests and syntax checks**

Run:

```bash
python -m pytest -q tests/test_export_nlspn_invalid_region_predictions.py
python -m py_compile scripts/export_nlspn_invalid_region_predictions.py
git diff --check
```

Expected: 14 tests pass; compilation and diff checks exit 0.

- [ ] **Step 6: Commit the atomic exporter**

```bash
git add scripts/export_nlspn_invalid_region_predictions.py \
  tests/test_export_nlspn_invalid_region_predictions.py
git commit -m "feat: export complete NLSPN prediction depth maps"
```

### Task 4: Generate and verify the real 150-frame export

**Files:**
- Create through CLI: `/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1_prediction_exports`
- Verify: all generated PNGs and `manifest.csv`

- [ ] **Step 1: Confirm immutable source and absent destination**

Run:

```bash
test -d /workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1/windows
test ! -e /workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1_prediction_exports
find /workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1/windows \
  -mindepth 1 -maxdepth 1 -type d | wc -l
```

Expected: both `test` commands exit 0 and the count is `30`.

- [ ] **Step 2: Run the exporter once**

Run:

```bash
python scripts/export_nlspn_invalid_region_predictions.py
```

Expected output:

```text
{'window_count': 30, 'frame_count': 150, 'png_count': 600}
```

- [ ] **Step 3: Independently validate counts, modes, dimensions, and fill values**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from scripts import export_nlspn_invalid_region_predictions as exporter

source = Path('/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1/windows')
root = Path('/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1_prediction_exports')
print(exporter.validate_export(root, source))
with (root / 'manifest.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 150
assert sum(1 for _ in root.rglob('*.png')) == 600
for row in rows:
    with Image.open(root / row['specialized_depth_mm']) as image:
        assert image.size == (304, 228)
        values = np.asarray(image)
    assert values.min() >= 0 and values.max() <= 10000
print({'rows': len(rows), 'pngs': 600, 'depth_range_mm': '0..10000'})
PY
```

Expected: both dictionaries print and all assertions pass.

- [ ] **Step 4: Inspect six representative filled outputs**

Open `specialized_full_color.png`, `gt_with_prediction_fill.png`, and
`invalid_mask.png` for window indices 01, 06, 11, 16, 21, and 26. These are
one low-, medium-, and high-motion sample for room3 and room7. Require native
304 by 228 dimensions, fixed colors, nonblank output, correct invalid mask,
and prediction values visible inside every invalid GT region.

- [ ] **Step 5: Run full verification**

Run:

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: the complete project test suite passes, diff check exits 0, and the
Git worktree is clean.

- [ ] **Step 6: Finish the branch**

Invoke `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch`. Present merge, push/PR, keep, and
discard options and take no branch action without the user's choice.
