"""Reproducible five-frame visualization helpers for causal NLSPN."""

from __future__ import division

import csv
import copy
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_in_memory_gop2 as online
from scripts import nlspn_temporal_residual as residual
from scripts import spn_sequence_io


BASE_METHOD_ORDER = ("full", "zero_flow", "rgb_diff", "global_diff")
RAFT_METHOD_ORDER = BASE_METHOD_ORDER + ("raft_gop2",)
METHOD_ORDER = BASE_METHOD_ORDER
METHOD_NAMES = {
    "full": "Full NLSPN",
    "zero_flow": "Zero-flow",
    "rgb_diff": "RGB-diff",
    "global_diff": "Global-diff",
    "raft_gop2": "RAFT-GOP2",
}
FRAME_IDS = tuple(range(1, 6))
FINAL_ARTIFACTS = (
    "nlspn_frame_difference_depth_comparison.png",
    "nlspn_frame_difference_error_comparison.png",
    "predictions.npz",
    "frame_metrics.csv",
    "run_metadata.json",
    "worker.log",
)


def _selected(value):
    return str(value).strip().lower() in ("true", "1")


def load_selected_configs(path):
    with open(str(path), "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if _selected(row.get("selected", ""))]
    configs = {}
    for variant in ("rgb_diff", "global_diff"):
        matches = [row for row in selected if row.get("variant") == variant]
        if len(matches) != 1:
            raise ValueError("%s requires exactly one selected row" % variant)
        row = matches[0]
        configs[variant] = cache.CacheConfig(
            variant=variant,
            threshold=float(row["threshold"]),
            dilation_radius=int(row["dilation_radius"]))
    if len(selected) != 2:
        raise ValueError("sweep contains an unexpected selected variant")
    return configs, residual.file_sha256(path)


def payload_frame_ids(payload):
    """Return five strictly consecutive integer frame IDs."""
    raw = np.asarray(payload.get("frame_ids"))
    if raw.shape != (5,):
        raise ValueError("visualization payload frame_ids geometry is invalid")
    frame_ids = raw.astype(np.int64)
    if not np.array_equal(raw, frame_ids):
        raise ValueError("visualization frame IDs must be integers")
    if np.any(frame_ids <= 0) or not np.all(np.diff(frame_ids) == 1):
        raise ValueError(
            "visualization requires five positive consecutive frame IDs")
    return tuple(int(item) for item in frame_ids)


def _validate_payload(payload):
    expected = {
        "frame_ids": (5,),
        "rgb": (5, 3, residual.HEIGHT, residual.WIDTH),
        "sparse": (5, residual.HEIGHT, residual.WIDTH),
        "gt": (5, residual.HEIGHT, residual.WIDTH),
        "valid": (5, residual.HEIGHT, residual.WIDTH),
    }
    for name, shape in expected.items():
        if name not in payload or tuple(np.asarray(payload[name]).shape) != shape:
            raise ValueError("visualization payload %s geometry is invalid" % name)
    payload_frame_ids(payload)
    sparse = np.asarray(payload["sparse"])
    if not np.all(np.count_nonzero(sparse > 0.0, axis=(1, 2)) == 500):
        raise ValueError("each frame must contain exactly 500 sparse points")
    valid = np.asarray(payload["valid"], dtype=bool)
    if not np.isfinite(np.asarray(payload["gt"])[valid]).all():
        raise ValueError("valid ground truth contains non-finite values")


def prediction_method_order(predictions):
    keys = set(predictions)
    if keys == set(BASE_METHOD_ORDER):
        return BASE_METHOD_ORDER
    if keys == set(RAFT_METHOD_ORDER):
        return RAFT_METHOD_ORDER
    raise ValueError("predictions do not match an approved method schema")


def _validate_predictions(predictions):
    methods = prediction_method_order(predictions)
    expected = (5, residual.HEIGHT, residual.WIDTH)
    for method in methods:
        value = np.asarray(predictions[method])
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError("%s prediction is invalid" % method)
    return methods


def collect_frame_metrics(payload, predictions, latency_rows):
    _validate_payload(payload)
    methods = _validate_predictions(predictions)
    frame_ids = payload_frame_ids(payload)
    latency = {}
    for row in latency_rows:
        key = (str(row["method"]), int(row["frame_id"]))
        if key in latency:
            raise ValueError("duplicate visualization latency row")
        latency[key] = float(row["latency_ms"])
    expected_keys = set(
        (method, frame_id) for method in methods
        for frame_id in frame_ids)
    if set(latency) != expected_keys:
        raise ValueError("visualization latency rows are incomplete")
    rows = []
    for method in methods:
        for index, frame_id in enumerate(frame_ids):
            metrics = spn_sequence_io.frame_metrics(
                payload["gt"][index], predictions[method][index],
                payload["valid"][index])
            metrics.update({
                "method": method,
                "frame_id": frame_id,
                "frame_kind": (
                    "FULL" if method == "full" else
                    online.frame_kind(index)),
                "latency_ms": latency[(method, frame_id)],
                "sparse_count": int(np.count_nonzero(
                    payload["sparse"][index] > 0.0)),
            })
            rows.append(metrics)
    return rows


def _masked(value, valid):
    return np.ma.masked_where(~np.asarray(valid, dtype=bool), value)


def _colormap(name):
    cmap = copy.copy(plt.get_cmap(name))
    cmap.set_bad(color="white")
    return cmap


def _draw(axis, value, title, cmap, vmin, vmax):
    image = axis.imshow(
        value, cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", aspect="auto")
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def _external_colorbar(figure, mappable, label):
    figure.subplots_adjust(right=0.88)
    axis = figure.add_axes([0.91, 0.18, 0.016, 0.64])
    return figure.colorbar(mappable, cax=axis, label=label)


def _save_png_atomic(figure, path):
    path = str(path)
    temporary = path + ".tmp"
    figure.savefig(temporary, format="png", dpi=150, facecolor="white")
    import os
    os.replace(temporary, path)


def common_error_max(payload, predictions):
    _validate_payload(payload)
    methods = _validate_predictions(predictions)
    valid = np.asarray(payload["valid"], dtype=bool)
    gt = np.asarray(payload["gt"], dtype=np.float32)
    values = np.concatenate([
        np.abs(np.asarray(predictions[method]) - gt)[valid].astype(np.float64)
        for method in methods])
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("visualization errors are empty or non-finite")
    return max(float(np.percentile(values, 99.0)), 1e-3)


def render_depth_comparison(path, payload, predictions):
    _validate_payload(payload)
    methods = _validate_predictions(predictions)
    columns = ("gt",) + methods
    figure, axes = plt.subplots(
        5, len(columns), figsize=(3.1 * len(columns), 12.25), squeeze=False)
    last_image = None
    cmap = _colormap("viridis")
    for row, frame_id in enumerate(payload_frame_ids(payload)):
        for column, method in enumerate(columns):
            if method == "gt":
                value = _masked(payload["gt"][row], payload["valid"][row])
                title = "Ground truth"
            else:
                value = predictions[method][row]
                title = METHOD_NAMES[method]
            last_image = _draw(
                axes[row, column], value, title if row == 0 else "",
                cmap, 0.0, residual.MAX_DEPTH)
            if column == 0:
                axes[row, column].set_ylabel("Frame %04d" % frame_id)
    figure.suptitle(
        "Five-frame causal NLSPN depth completion: identical RGB + "
        "500 sparse points")
    figure.subplots_adjust(
        top=0.94, right=0.88, wspace=0.04, hspace=0.12)
    _external_colorbar(figure, last_image, "Depth (m)")
    _save_png_atomic(figure, path)
    return figure


def render_error_comparison(path, payload, predictions):
    _validate_payload(payload)
    methods = _validate_predictions(predictions)
    error_max = common_error_max(payload, predictions)
    gt = np.asarray(payload["gt"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=bool)
    errors = dict(
        (method, np.abs(np.asarray(predictions[method]) - gt))
        for method in methods)
    figure, axes = plt.subplots(
        5, len(methods), figsize=(3.1 * len(methods), 12.25), squeeze=False)
    last_image = None
    cmap = _colormap("magma")
    for row, frame_id in enumerate(payload_frame_ids(payload)):
        for column, method in enumerate(methods):
            last_image = _draw(
                axes[row, column], _masked(errors[method][row], valid[row]),
                METHOD_NAMES[method] if row == 0 else "", cmap,
                0.0, error_max)
            if column == 0:
                axes[row, column].set_ylabel("Frame %04d" % frame_id)
    figure.suptitle(
        "Five-frame absolute error (common valid-pixel 99th percentile)")
    figure.subplots_adjust(
        top=0.94, right=0.88, wspace=0.04, hspace=0.12)
    _external_colorbar(figure, last_image, "Absolute error (m)")
    _save_png_atomic(figure, path)
    return figure


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _write_csv_atomic(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError("visualization metrics cannot be empty")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _write_npz_atomic(path, payload, predictions):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    arrays = {
        "frame_ids": np.asarray(payload["frame_ids"], dtype=np.int32),
        "rgb": np.asarray(payload["rgb"], dtype=np.float32),
        "sparse": np.asarray(payload["sparse"], dtype=np.float32),
        "gt": np.asarray(payload["gt"], dtype=np.float32),
        "valid": np.asarray(payload["valid"], dtype=bool),
    }
    methods = prediction_method_order(predictions)
    arrays.update(dict(
        (method, np.asarray(predictions[method], dtype=np.float32))
        for method in methods))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(str(temporary), str(path))


def write_artifacts(output_dir, payload, predictions, metrics, metadata):
    _validate_payload(payload)
    methods = _validate_predictions(predictions)
    metrics = list(metrics)
    expected_metric_count = 5 * len(methods)
    if len(metrics) != expected_metric_count:
        raise ValueError(
            "visualization requires exactly %d metric rows" %
            expected_metric_count)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [
        path.name for path in output_dir.iterdir()
        if path.is_file() and path.name not in FINAL_ARTIFACTS]
    if unexpected:
        raise RuntimeError("visualization directory has unapproved artifacts")
    incomplete = dict(metadata)
    incomplete["complete"] = False
    incomplete["method_order"] = list(methods)
    _write_json_atomic(output_dir / "run_metadata.json", incomplete)
    _write_npz_atomic(output_dir / "predictions.npz", payload, predictions)
    _write_csv_atomic(output_dir / "frame_metrics.csv", metrics)
    depth_figure = render_depth_comparison(
        output_dir / "nlspn_frame_difference_depth_comparison.png",
        payload, predictions)
    plt.close(depth_figure)
    error_figure = render_error_comparison(
        output_dir / "nlspn_frame_difference_error_comparison.png",
        payload, predictions)
    plt.close(error_figure)
    temporary_log = output_dir / "worker.log.tmp"
    temporary_log.write_text(
        "Worker completed; launcher will replace this log.\n",
        encoding="utf-8")
    os.replace(str(temporary_log), str(output_dir / "worker.log"))
    completed = dict(incomplete)
    completed.update({
        "complete": True,
        "artifact_count": len(FINAL_ARTIFACTS),
        "artifacts": list(FINAL_ARTIFACTS),
        "depth_vmin_m": 0.0,
        "depth_vmax_m": residual.MAX_DEPTH,
        "error_vmin_m": 0.0,
        "error_vmax_m": common_error_max(payload, predictions),
    })
    _write_json_atomic(output_dir / "run_metadata.json", completed)
    missing = [
        name for name in FINAL_ARTIFACTS
        if not (output_dir / name).is_file() or
        (output_dir / name).stat().st_size == 0]
    if missing:
        raise RuntimeError("visualization artifacts are incomplete")
    return completed
