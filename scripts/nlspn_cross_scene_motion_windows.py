"""Deterministic motion-window selection for cross-scene NLSPN evaluation."""

from pathlib import Path
import csv
import json
import os
import re

import numpy as np
from PIL import Image


SCENES = (
    "bedroom_ir",
    "livingroom_ir",
    "room3",
    "room4",
    "room6",
    "room7",
)

THUMBNAIL_SIZE = (76, 57)
_RGB_PATTERN = re.compile(r"^(\d{4})\.jpg$")
_DEPTH_PATTERN = re.compile(r"^Image(\d{4})\.exr$")
FIXED_THRESHOLD = 2.0 / 255.0
FIXED_DILATION_RADIUS = 8
BASE_METHOD_ORDER = ("full", "zero_flow", "rgb_diff", "global_diff")
RAFT_METHOD_ORDER = BASE_METHOD_ORDER + ("raft_gop2",)
METHOD_ORDER = BASE_METHOD_ORDER
ROOT_ARTIFACTS = (
    "selected_windows.csv",
    "cross_scene_summary.csv",
    "report.md",
    "run_metadata.json",
)


def require_fixed_configs(configs):
    """Validate and serialize the approved conservative cache configs."""
    if set(configs) != {"rgb_diff", "global_diff"}:
        raise ValueError("fixed configs require rgb_diff and global_diff")
    result = {}
    for variant in ("rgb_diff", "global_diff"):
        config = configs[variant]
        if getattr(config, "variant", None) != variant:
            raise ValueError("fixed config variant mismatch for {}".format(variant))
        threshold = float(getattr(config, "threshold", float("nan")))
        radius = int(getattr(config, "dilation_radius", -1))
        if (not np.isfinite(threshold) or
                abs(threshold - FIXED_THRESHOLD) > 1e-15):
            raise ValueError("fixed config threshold must equal 2/255")
        if radius != FIXED_DILATION_RADIUS:
            raise ValueError("fixed config dilation radius must equal 8")
        result[variant] = {
            "threshold": threshold,
            "dilation_radius": radius,
        }
    return result


def _validated_ids(frame_ids):
    ids = tuple(frame_ids)
    if not ids or any(not isinstance(item, (int, np.integer)) for item in ids):
        raise ValueError("frame IDs must be positive integers")
    ids = tuple(int(item) for item in ids)
    if any(item <= 0 for item in ids):
        raise ValueError("frame IDs must be positive")
    if tuple(sorted(ids)) != ids:
        raise ValueError("frame IDs must be sorted")
    if len(set(ids)) != len(ids):
        raise ValueError("frame IDs must be unique")
    return ids


def _validated_thumbnails(ids, thumbnails):
    arrays = {}
    shape = None
    for frame_id in ids:
        if frame_id not in thumbnails:
            raise ValueError("missing thumbnail for frame {}".format(frame_id))
        array = np.asarray(thumbnails[frame_id])
        if array.dtype != np.float32:
            raise ValueError("thumbnails must use float32")
        if array.ndim != 2:
            raise ValueError("thumbnails must be grayscale 2-D arrays")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError("thumbnails must have the same shape")
        if not np.all(np.isfinite(array)):
            raise ValueError("thumbnails must contain finite values")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("thumbnails must lie in [0,1]")
        arrays[frame_id] = array
    return arrays


def enumerate_motion_windows(frame_ids, thumbnails):
    """Return all legal consecutive five-frame windows in frame order."""
    ids = _validated_ids(frame_ids)
    arrays = _validated_thumbnails(ids, thumbnails)

    candidates = []
    for start_index in range(max(0, len(ids) - 4)):
        window = ids[start_index:start_index + 5]
        if any(right != left + 1 for left, right in zip(window, window[1:])):
            continue
        pair_scores = [
            float(np.mean(np.abs(arrays[right] - arrays[left]), dtype=np.float64))
            for left, right in zip(window, window[1:])
        ]
        candidates.append({
            "frame_ids": list(window),
            "pair_scores": pair_scores,
            "motion_score": float(np.mean(pair_scores, dtype=np.float64)),
        })

    if not candidates:
        raise ValueError("no five consecutive complete frames are available")
    return candidates


def select_motion_window(frame_ids, thumbnails):
    """Choose the highest-mean-MAD consecutive five-frame window."""
    candidates = enumerate_motion_windows(frame_ids, thumbnails)
    return min(
        candidates,
        key=lambda item: (-item["motion_score"], item["frame_ids"][0]),
    )


def _indexed_files(directory, pattern, label):
    if not directory.is_dir():
        raise ValueError("missing {} directory: {}".format(label, directory))
    indexed = {}
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match is None:
            continue
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("empty {} file: {}".format(label, path))
        frame_id = int(match.group(1))
        if frame_id in indexed:
            raise ValueError("duplicate {} frame {}".format(label, frame_id))
        indexed[frame_id] = path
    if not indexed:
        raise ValueError("no {} frames found in {}".format(label, directory))
    return indexed


def _load_thumbnail(path):
    try:
        with Image.open(str(path)) as image:
            image = image.convert("L")
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize(THUMBNAIL_SIZE, resample=resampling)
            return np.asarray(image, dtype=np.float32) / np.float32(255.0)
    except Exception as error:
        raise ValueError("unreadable RGB JPEG: {}".format(path)) from error


def scan_scene_candidates(scene_root):
    """Scan one scene and return all legal five-frame motion windows."""
    root = Path(scene_root)
    if not root.is_dir():
        raise ValueError("missing scene directory: {}".format(root))
    rgb = _indexed_files(root / "rgb", _RGB_PATTERN, "RGB")
    depth = _indexed_files(root / "depth", _DEPTH_PATTERN, "depth")
    ids = tuple(sorted(set(rgb).intersection(depth)))
    thumbnails = {frame_id: _load_thumbnail(rgb[frame_id]) for frame_id in ids}
    results = enumerate_motion_windows(ids, thumbnails)
    for result in results:
        result.update({
            "scene": root.name,
            "start_frame": result["frame_ids"][0],
            "end_frame": result["frame_ids"][-1],
        })
    return results


def scan_scene(scene_root):
    """Scan one ``rgb``/``depth`` scene and select its motion-rich window."""
    candidates = scan_scene_candidates(scene_root)
    return min(candidates, key=lambda item: (
        -item["motion_score"], item["frame_ids"][0]))


def _finite_float(row, name):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        raise ValueError("metric {} is missing or invalid".format(name))
    if not np.isfinite(value):
        raise ValueError("metric {} must be finite".format(name))
    return value


def resolve_summary_method_order(rows):
    methods = {str(row.get("method")) for row in rows}
    if methods == set(BASE_METHOD_ORDER):
        return BASE_METHOD_ORDER
    if methods == set(RAFT_METHOD_ORDER):
        return RAFT_METHOD_ORDER
    raise ValueError("summary rows do not match an approved method schema")


def build_scene_summary(scene, frame_metrics):
    """Pool five frame rows per method using valid-pixel weighting."""
    if scene not in SCENES:
        raise ValueError("unapproved scene: {}".format(scene))
    rows = list(frame_metrics)
    methods = resolve_summary_method_order(rows)
    result = []
    for method in methods:
        selected = [row for row in rows if str(row.get("method")) == method]
        frame_ids = [int(row.get("frame_id", -1)) for row in selected]
        if len(selected) != 5 or len(set(frame_ids)) != 5:
            raise ValueError(
                "{} requires five unique frame rows".format(method))
        valid_total = 0
        squared_total = np.float64(0.0)
        absolute_total = np.float64(0.0)
        latency_total = np.float64(0.0)
        for row in selected:
            valid_value = _finite_float(row, "valid_pixels")
            if valid_value <= 0 or not valid_value.is_integer():
                raise ValueError("valid_pixels must be a positive integer")
            valid = int(valid_value)
            rmse = _finite_float(row, "rmse")
            mae = _finite_float(row, "mae")
            latency = _finite_float(row, "latency_ms")
            if rmse < 0 or mae < 0 or latency < 0:
                raise ValueError("metrics cannot be negative")
            valid_total += valid
            squared_total += np.float64(rmse) ** 2 * valid
            absolute_total += np.float64(mae) * valid
            latency_total += np.float64(latency)
        result.append({
            "scene": scene,
            "method": method,
            "rmse": float(np.sqrt(squared_total / valid_total)),
            "mae": float(absolute_total / valid_total),
            "valid_pixels": valid_total,
            "latency_ms": float(latency_total),
        })
    full_rmse = result[0]["rmse"]
    if full_rmse <= 0:
        raise ValueError("full RMSE must be positive")
    for row in result:
        ratio = row["rmse"] / full_rmse
        row["rmse_ratio"] = float(ratio)
        row["passes_1pct"] = bool(ratio <= 1.01)
    return result


def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _validated_windows(windows):
    by_scene = {str(row.get("scene")): dict(row) for row in windows}
    if set(by_scene) != set(SCENES) or len(list(windows)) != len(SCENES):
        raise ValueError("windows must contain the six approved scenes")
    result = []
    for scene in SCENES:
        row = by_scene[scene]
        ids = _validated_ids(row.get("frame_ids", ()))
        if len(ids) != 5 or any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("window must contain five consecutive frame IDs")
        scores = [float(item) for item in row.get("pair_scores", ())]
        score = float(row.get("motion_score", float("nan")))
        if (len(scores) != 4 or not np.isfinite(scores).all() or
                not np.isfinite(score)):
            raise ValueError("window motion scores are invalid")
        if int(row.get("start_frame", -1)) != ids[0] or \
                int(row.get("end_frame", -1)) != ids[-1]:
            raise ValueError("window bounds do not match frame IDs")
        result.append({
            "scene": scene,
            "start_frame": ids[0],
            "end_frame": ids[-1],
            "frame_ids": json.dumps(list(ids), separators=(",", ":")),
            "pair_scores": json.dumps(scores, separators=(",", ":")),
            "motion_score": score,
        })
    return result


def _validated_summaries(summary_rows):
    rows = list(summary_rows)
    methods = resolve_summary_method_order(rows)
    expected = {(scene, method) for scene in SCENES for method in methods}
    keyed = {(str(row.get("scene")), str(row.get("method"))): dict(row)
             for row in rows}
    expected_count = len(SCENES) * len(methods)
    if len(rows) != expected_count or set(keyed) != expected:
        raise ValueError(
            "summary must contain {} unique scene-method rows".format(
                expected_count))
    result = []
    for scene in SCENES:
        for method in methods:
            row = keyed[(scene, method)]
            valid = _finite_float(row, "valid_pixels")
            values = {name: _finite_float(row, name) for name in (
                "rmse", "mae", "latency_ms", "rmse_ratio")}
            if valid <= 0 or any(value < 0 for value in values.values()):
                raise ValueError("summary metrics are invalid")
            result.append({
                "scene": scene,
                "method": method,
                "rmse": values["rmse"],
                "mae": values["mae"],
                "valid_pixels": int(valid),
                "latency_ms": values["latency_ms"],
                "rmse_ratio": values["rmse_ratio"],
                "passes_1pct": bool(row.get("passes_1pct")),
            })
    return result


def _all_scene_summary(rows):
    methods = resolve_summary_method_order(rows)
    pooled = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        valid = sum(row["valid_pixels"] for row in selected)
        rmse = float(np.sqrt(sum(
            np.float64(row["rmse"]) ** 2 * row["valid_pixels"]
            for row in selected) / valid))
        mae = float(sum(
            np.float64(row["mae"]) * row["valid_pixels"]
            for row in selected) / valid)
        pooled.append({
            "method": method,
            "rmse": rmse,
            "mae": mae,
            "valid_pixels": valid,
            "latency_ms": float(sum(row["latency_ms"] for row in selected)),
        })
    full = pooled[0]["rmse"]
    for row in pooled:
        row["rmse_ratio"] = row["rmse"] / full
        row["passes_1pct"] = row["rmse_ratio"] <= 1.01
    return pooled


def _render_report(windows, rows, pooled):
    lines = [
        "# NLSPN cross-scene motion-window evaluation",
        "",
        "## Selected windows",
        "",
        "| Scene | Frames | Motion score |",
        "|---|---:|---:|",
    ]
    for row in windows:
        lines.append("| {} | {}-{} | {:.9f} |".format(
            row["scene"], row["start_frame"], row["end_frame"],
            row["motion_score"]))
    lines.extend([
        "",
        "## Per-scene pooled metrics",
        "",
        "| Scene | Method | RMSE (m) | RMSE/full | <=1% | Latency (ms) |",
        "|---|---|---:|---:|:---:|---:|",
    ])
    for row in rows:
        lines.append("| {} | {} | {:.9f} | {:.6f} | {} | {:.3f} |".format(
            row["scene"], row["method"], row["rmse"], row["rmse_ratio"],
            "yes" if row["passes_1pct"] else "no", row["latency_ms"]))
    lines.extend([
        "",
        "## All-scene pooled metrics",
        "",
        "| Method | RMSE (m) | MAE (m) | RMSE/full | <=1% | Latency (ms) |",
        "|---|---:|---:|---:|:---:|---:|",
    ])
    for row in pooled:
        lines.append("| {} | {:.9f} | {:.9f} | {:.6f} | {} | {:.3f} |".format(
            row["method"], row["rmse"], row["mae"], row["rmse_ratio"],
            "yes" if row["passes_1pct"] else "no", row["latency_ms"]))
    return "\n".join(lines) + "\n"


def write_root_artifacts(output_root, windows, summary_rows, metadata):
    """Write the exact four deterministic cross-scene root artifacts."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    unexpected_files = [path.name for path in output_root.iterdir()
                        if path.is_file() and path.name not in ROOT_ARTIFACTS]
    unexpected_dirs = [path.name for path in output_root.iterdir()
                       if path.is_dir() and path.name not in SCENES]
    if unexpected_files or unexpected_dirs:
        raise RuntimeError("cross-scene output contains unapproved artifacts")
    window_rows = _validated_windows(list(windows))
    rows = _validated_summaries(summary_rows)
    methods = resolve_summary_method_order(rows)
    pooled = _all_scene_summary(rows)
    incomplete = dict(metadata)
    declared_methods = incomplete.get("method_order")
    if declared_methods is not None and tuple(declared_methods) != methods:
        raise ValueError("metadata method order differs from summary rows")
    incomplete.update({
        "complete": False,
        "scenes": list(SCENES),
        "method_order": list(methods),
    })
    _atomic_json(output_root / "run_metadata.json", incomplete)
    _atomic_csv(output_root / "selected_windows.csv", window_rows, (
        "scene", "start_frame", "end_frame", "frame_ids", "pair_scores",
        "motion_score"))
    _atomic_csv(output_root / "cross_scene_summary.csv", rows, (
        "scene", "method", "rmse", "mae", "valid_pixels", "latency_ms",
        "rmse_ratio", "passes_1pct"))
    _atomic_text(output_root / "report.md", _render_report(
        window_rows, rows, pooled))
    completed = dict(incomplete)
    completed.update({
        "complete": True,
        "artifact_count": len(ROOT_ARTIFACTS),
        "artifacts": list(ROOT_ARTIFACTS),
        "selected_window_count": len(window_rows),
        "summary_row_count": len(rows),
        "all_scene_summary": pooled,
    })
    _atomic_json(output_root / "run_metadata.json", completed)
    if any(not (output_root / name).is_file() or
           (output_root / name).stat().st_size <= 0 for name in ROOT_ARTIFACTS):
        raise RuntimeError("cross-scene root artifacts are incomplete")
    return completed
