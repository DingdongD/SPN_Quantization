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
from matplotlib import colormaps
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
    return colormaps["viridis"](normalized, bytes=True)[..., :3].astype(
        np.uint8)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_frames(windows_root):
    windows_root = Path(windows_root)
    directories = sorted(path for path in windows_root.iterdir()
                         if path.is_dir())
    if len(directories) != WINDOW_COUNT:
        raise ValueError("source must contain exactly 30 window directories")
    frames = []
    fingerprints = {}
    seen = set()
    for directory in directories:
        path = directory / "predictions.npz"
        if not path.is_file():
            raise ValueError(
                "window is missing predictions.npz: {}".format(
                    directory.name))
        fingerprints[directory.name] = file_sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            required = ("scenes", "frame_ids", "gt", "valid", "specialized")
            if any(name not in archive.files for name in required):
                raise ValueError("prediction archive is missing required arrays")
            values = {name: archive[name] for name in required}
        if (values["scenes"].shape != (FRAMES_PER_WINDOW,) or
                values["frame_ids"].shape != (FRAMES_PER_WINDOW,)):
            raise ValueError("window identity arrays must contain five frames")
        for name in ("gt", "valid", "specialized"):
            expected_shape = (FRAMES_PER_WINDOW,) + SPATIAL_SHAPE
            if values[name].shape != expected_shape:
                raise ValueError("{} has the wrong shape".format(name))
        scenes = [str(value) for value in values["scenes"]]
        frame_ids = [int(value) for value in values["frame_ids"]]
        if len(set(scenes)) != 1 or any(
                right != left + 1
                for left, right in zip(frame_ids, frame_ids[1:])):
            raise ValueError("window frames must be one scene and consecutive")
        for offset, (scene, frame_id) in enumerate(zip(scenes, frame_ids)):
            identity = (scene, frame_id)
            if identity in seen:
                raise ValueError("source contains duplicate scene/frame identity")
            seen.add(identity)
            prediction = _finite_depth(values["specialized"][offset])
            gt = _finite_depth(values["gt"][offset])
            raw_valid = np.asarray(values["valid"][offset])
            if (raw_valid.dtype != np.bool_ and
                    not np.isin(raw_valid, (0, 1)).all()):
                raise ValueError(
                    "validity values must be boolean-compatible")
            valid = raw_valid.astype(bool, copy=False)
            frames.append({
                "window": directory.name,
                "scene": scene,
                "frame_id": frame_id,
                "gt": gt,
                "valid": valid,
                "specialized": prediction,
            })
    if len(frames) != WINDOW_COUNT * FRAMES_PER_WINDOW:
        raise ValueError("source must contain exactly 150 frames")
    return frames, fingerprints


def _save_png(path, array, mode):
    image = Image.fromarray(array)
    if image.mode != mode:
        raise ValueError(
            "PNG array inferred mode {}, expected {}".format(image.mode, mode))
    image.save(path)


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
            "specialized_depth_mm": (
                depth_to_millimetres(prediction), "I;16"),
            "gt_with_prediction_fill": (colorize_depth(hybrid), "RGB"),
            "invalid_mask": (mask, "L"),
        }
        for name, (array, mode) in products.items():
            _save_png(root / paths[name], array, mode)
        invalid_count = int((~frame["valid"]).sum())
        rows.append({
            "window": frame["window"],
            "scene": frame["scene"],
            "frame_id": frame["frame_id"],
            "invalid_pixel_count": invalid_count,
            "invalid_fraction": (
                invalid_count / float(np.prod(SPATIAL_SHAPE))),
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
    windows_root = root / "windows"
    if {path.name for path in windows_root.iterdir()} != expected_windows:
        raise ValueError("export window directories differ from source")
    if not all((windows_root / name).is_dir() for name in expected_windows):
        raise ValueError("export window entry is not a directory")
    with (root / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != WINDOW_COUNT * FRAMES_PER_WINDOW:
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
            "invalid_fraction": (
                invalid_count / float(np.prod(SPATIAL_SHAPE))),
            "prediction_min_m": float(source["specialized"].min()),
            "prediction_max_m": float(source["specialized"].max()),
            "prediction_mean_m": float(source["specialized"].mean()),
        }
        if int(row["invalid_pixel_count"]) != invalid_count:
            raise ValueError(
                "manifest invalid pixel count differs from source")
        for name in ("invalid_fraction", "prediction_min_m",
                     "prediction_max_m", "prediction_mean_m"):
            if not math.isclose(float(row[name]), expected_scalars[name],
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "manifest {} differs from source".format(name))
        expected_arrays = {
            "specialized_full_color": colorize_depth(source["specialized"]),
            "specialized_depth_mm": depth_to_millimetres(
                source["specialized"]),
            "gt_with_prediction_fill": colorize_depth(
                compose_gt_with_prediction(
                    source["gt"], source["valid"],
                    source["specialized"])),
            "invalid_mask": invalid_mask(source["valid"]),
        }
        expected_paths = _frame_paths(source["window"], source["frame_id"])
        frame_directory = root / next(iter(expected_paths.values())).parent
        if ({path.name for path in frame_directory.iterdir()} !=
                set(PNG_NAMES)):
            raise ValueError("frame directory differs from the exact contract")
        for name, expected_array in expected_arrays.items():
            if row[name] != str(expected_paths[name]):
                raise ValueError(
                    "manifest PNG path differs from exact contract")
            path = (root / Path(row[name])).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError("manifest PNG path escapes or is missing")
            with Image.open(path) as image:
                actual = np.asarray(image)
            if actual.shape != expected_array.shape:
                raise ValueError("PNG has the wrong shape")
            if not np.array_equal(actual, expected_array):
                label = ("hybrid" if name == "gt_with_prediction_fill"
                         else name)
                raise ValueError(
                    "{} PNG differs from source formula".format(label))
            png_count += 1
    if set(expected) != seen or png_count != len(frames) * len(PNG_NAMES):
        raise ValueError("export must contain exactly 600 PNGs")
    return {"window_count": WINDOW_COUNT,
            "frame_count": WINDOW_COUNT * FRAMES_PER_WINDOW,
            "png_count": png_count}


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
        validate_export(temporary, source_windows)
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
