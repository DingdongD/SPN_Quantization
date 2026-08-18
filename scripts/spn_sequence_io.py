#!/usr/bin/env python3
"""Shared data contracts for the four-model sequence comparison."""

from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path

import numpy as np


HEIGHT = 228
WIDTH = 304
SPARSE_COUNT = 500
MAX_DEPTH = 10.0
FRAME_IDS = tuple(range(1, 6))


def _update_array_digest(digest, name, value):
    value = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))


def canonical_input_digest(frame_ids, rgb, sparse, gt, valid):
    digest = hashlib.sha256()
    for name, value in (
            ("frame_ids", np.asarray(frame_ids, dtype=np.int64)),
            ("rgb", rgb),
            ("sparse", sparse),
            ("gt", gt),
            ("valid", np.asarray(valid, dtype=np.uint8))):
        _update_array_digest(digest, name, value)
    return digest.hexdigest()


def load_canonical_frames(root, frame_ids=FRAME_IDS):
    frame_ids = tuple(int(frame_id) for frame_id in frame_ids)
    rows = []
    for frame_id in frame_ids:
        path = Path(root) / ("frame_%04d.npz" % frame_id)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        with np.load(str(path), allow_pickle=False) as item:
            required = {
                "frame_id", "rgb", "sparse", "gt", "valid",
                "pred_raw", "pred_clamped",
            }
            missing = sorted(required - set(item.files))
            if missing:
                raise ValueError("canonical NPZ missing keys: %s" % missing)
            row = {key: np.asarray(item[key]).copy() for key in required}
        if int(row["frame_id"]) != frame_id:
            raise ValueError("canonical frame ID mismatch")
        rows.append(row)

    data = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "rgb": np.stack([row["rgb"] for row in rows]).astype(np.float32),
        "sparse": np.stack([row["sparse"] for row in rows]).astype(np.float32),
        "gt": np.stack([row["gt"] for row in rows]).astype(np.float32),
        "valid": np.stack([row["valid"] for row in rows]).astype(bool),
        "cspn_pred_raw": np.stack(
            [row["pred_raw"] for row in rows]).astype(np.float32),
        "cspn_pred_clamped": np.stack(
            [row["pred_clamped"] for row in rows]).astype(np.float32),
    }
    expected_frames = len(frame_ids)
    if data["rgb"].shape != (expected_frames, 3, HEIGHT, WIDTH):
        raise ValueError(
            "canonical RGB shape must be %dx3x228x304" % expected_frames)
    for key in (
            "sparse", "gt", "valid", "cspn_pred_raw", "cspn_pred_clamped"):
        if data[key].shape != (expected_frames, HEIGHT, WIDTH):
            raise ValueError(
                "canonical %s shape must be %dx228x304" %
                (key, expected_frames))

    masks = data["sparse"] > 0.0
    if any(int(mask.sum()) != SPARSE_COUNT for mask in masks):
        raise ValueError("each frame must contain exactly 500 sparse values")
    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise ValueError("frames do not use shared sparse coordinates")
    if not np.isfinite(data["rgb"]).all():
        raise ValueError("canonical RGB contains non-finite values")
    if not np.isfinite(data["sparse"]).all():
        raise ValueError("canonical sparse depth contains non-finite values")
    if not np.isfinite(data["gt"][data["valid"]]).all():
        raise ValueError(
            "canonical valid ground truth contains non-finite values")
    if not np.isfinite(data["cspn_pred_raw"]).all():
        raise ValueError("canonical CSPN prediction contains non-finite values")

    data["input_digest"] = canonical_input_digest(
        data["frame_ids"], data["rgb"], data["sparse"], data["gt"],
        data["valid"])
    return data


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_worker_result(path, model, frame_ids, pred_raw, input_digest,
                        checkpoint_digest, metadata, runtime_seconds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_raw = np.asarray(pred_raw, dtype=np.float32)
    expected_shape = (len(frame_ids), HEIGHT, WIDTH)
    if pred_raw.shape != expected_shape:
        raise ValueError(
            "worker prediction shape must be %s" % (expected_shape,))
    if not np.isfinite(pred_raw).all():
        raise ValueError("worker prediction contains non-finite values")
    pred_clamped = np.clip(pred_raw, 1e-6, MAX_DEPTH).astype(np.float32)
    record = dict(metadata)
    record.update({
        "model": str(model),
        "input_digest": str(input_digest),
        "checkpoint_digest": str(checkpoint_digest),
        "runtime_seconds": float(runtime_seconds),
    })
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            model=np.asarray(str(model)),
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
            pred_raw=pred_raw,
            pred_clamped=pred_clamped,
            input_digest=np.asarray(str(input_digest)),
            checkpoint_digest=np.asarray(str(checkpoint_digest)),
            metadata=np.asarray(json.dumps(record, sort_keys=True)),
        )
    os.replace(str(temporary), str(path))


def load_worker_result(path, expected_model, expected_frame_ids,
                       expected_input_digest, expected_checkpoint_digest):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(str(path), allow_pickle=False) as item:
        required = {
            "model", "frame_ids", "pred_raw", "pred_clamped",
            "input_digest", "checkpoint_digest", "metadata",
        }
        missing = sorted(required - set(item.files))
        if missing:
            raise ValueError("worker result missing keys: %s" % missing)
        result = {key: np.asarray(item[key]).copy() for key in required}

    model = str(result["model"])
    frame_ids = result["frame_ids"].astype(np.int64)
    input_digest = str(result["input_digest"])
    checkpoint_digest = str(result["checkpoint_digest"])
    if model != str(expected_model):
        raise ValueError("worker result model mismatch")
    if not np.array_equal(
            frame_ids, np.asarray(expected_frame_ids, dtype=np.int64)):
        raise ValueError("worker result frame IDs mismatch")
    if input_digest != str(expected_input_digest):
        raise ValueError("worker result input digest mismatch")
    if checkpoint_digest != str(expected_checkpoint_digest):
        raise ValueError("worker result checkpoint digest mismatch")
    expected_shape = (len(frame_ids), HEIGHT, WIDTH)
    for key in ("pred_raw", "pred_clamped"):
        value = result[key].astype(np.float32)
        if value.shape != expected_shape:
            raise ValueError(
                "worker result %s shape must be %s" % (key, expected_shape))
        if not np.isfinite(value).all():
            raise ValueError("worker result %s contains non-finite values" % key)
        result[key] = value
    result["model"] = model
    result["frame_ids"] = frame_ids
    result["input_digest"] = input_digest
    result["checkpoint_digest"] = checkpoint_digest
    result["metadata"] = json.loads(str(result["metadata"]))
    return result


def _validated_metric_arrays(gt, pred, valid):
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if gt.shape != pred.shape or gt.shape != valid.shape:
        raise ValueError("gt, pred, and valid must have identical shapes")
    if not np.any(valid):
        raise ValueError("metric mask contains no valid pixels")
    if np.any(gt[valid] <= 0.0):
        raise ValueError("valid ground truth must be positive")
    if not np.isfinite(gt[valid]).all() or not np.isfinite(pred[valid]).all():
        raise ValueError("metric inputs contain non-finite valid values")
    return gt, pred, valid


def frame_metrics(gt, pred, valid):
    gt, pred, valid = _validated_metric_arrays(gt, pred, valid)
    gt_values = gt[valid].astype(np.float64)
    differences = pred[valid].astype(np.float64) - gt_values
    result = {
        "rmse": float(np.sqrt(np.mean(differences ** 2))),
        "mae": float(np.mean(np.abs(differences))),
        "abs_rel": float(np.mean(np.abs(differences) / gt_values)),
        "valid_pixels": int(valid.sum()),
        "valid_coverage": float(valid.mean()),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("frame metrics contain non-finite values")
    return result


def temporal_metrics(depths, predictions, valid_masks, frame_ids):
    depths = np.asarray(depths, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    frame_ids = tuple(int(frame_id) for frame_id in frame_ids)
    if depths.shape != predictions.shape or depths.shape != valid_masks.shape:
        raise ValueError(
            "depths, predictions, and valid masks must have identical shapes")
    if depths.ndim != 3 or depths.shape[0] != len(frame_ids):
        raise ValueError("sequence arrays and frame IDs disagree")

    rows = []
    maps = []
    for index in range(len(frame_ids) - 1):
        valid = valid_masks[index] & valid_masks[index + 1]
        gt_change = depths[index + 1] - depths[index]
        pred_change = predictions[index + 1] - predictions[index]
        residual = pred_change - gt_change
        if not np.any(valid):
            raise ValueError("temporal metric mask contains no valid pixels")
        if not np.isfinite(residual[valid]).all():
            raise ValueError("temporal residual contains non-finite values")
        values = residual[valid].astype(np.float64)
        rows.append({
            "pair": "%04d->%04d" % (frame_ids[index], frame_ids[index + 1]),
            "from_frame": frame_ids[index],
            "to_frame": frame_ids[index + 1],
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "mae": float(np.mean(np.abs(values))),
            "valid_pixels": int(valid.sum()),
            "valid_coverage": float(valid.mean()),
        })
        maps.append({
            "pair": rows[-1]["pair"],
            "gt_change": gt_change,
            "pred_change": pred_change,
            "residual": residual,
            "valid": valid,
        })
    return rows, maps
