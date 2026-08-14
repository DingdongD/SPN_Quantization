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
        raise ValueError(
            "depths and valid_masks must have shape [frames, height, width]")
    common = np.all(valid_masks, axis=0)
    candidates = np.flatnonzero(common)
    if candidates.size < int(count):
        raise ValueError(
            "common valid pixels %d are fewer than %d" %
            (candidates.size, count))
    chosen = np.random.default_rng(seed).choice(
        candidates, size=int(count), replace=False)
    mask = np.zeros(common.size, dtype=bool)
    mask[chosen] = True
    mask = mask.reshape(common.shape)
    return (depths * mask[None]).astype(np.float32), mask


def _validated_metric_arrays(gt, pred, valid):
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if gt.shape != pred.shape or gt.shape != valid.shape:
        raise ValueError("gt, pred, and valid must have identical shapes")
    if not np.any(valid):
        raise ValueError("metric mask contains no valid pixels")
    if not np.isfinite(gt[valid]).all() or not np.isfinite(pred[valid]).all():
        raise ValueError("metric inputs contain non-finite valid values")
    return gt, pred, valid


def frame_metrics(gt, pred, valid):
    gt, pred, valid = _validated_metric_arrays(gt, pred, valid)
    gt_valid = gt[valid].astype(np.float64)
    diff = pred[valid].astype(np.float64) - gt_valid
    result = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "abs_rel": float(np.mean(np.abs(diff) / gt_valid)),
        "valid_pixels": int(valid.sum()),
        "valid_coverage": float(valid.mean()),
    }
    if not all(np.isfinite(value) for key, value in result.items()
               if key != "valid_pixels"):
        raise ValueError("frame metrics contain non-finite values")
    return result


def temporal_metrics(depths, predictions, valid_masks, frame_ids):
    depths = np.asarray(depths, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    if depths.shape != predictions.shape or depths.shape != valid_masks.shape:
        raise ValueError(
            "depths, predictions, and valid_masks must have identical shapes")
    if depths.ndim != 3 or depths.shape[0] != len(frame_ids):
        raise ValueError("sequence arrays and frame_ids must describe the same frames")

    rows = []
    maps = []
    for index in range(len(frame_ids) - 1):
        valid = valid_masks[index] & valid_masks[index + 1]
        gt_change = depths[index + 1] - depths[index]
        pred_change = predictions[index + 1] - predictions[index]
        residual = pred_change - gt_change
        _, _, valid = _validated_metric_arrays(gt_change, pred_change, valid)
        values = residual[valid].astype(np.float64)
        row = {
            "pair": "%04d->%04d" % (frame_ids[index], frame_ids[index + 1]),
            "from_frame": int(frame_ids[index]),
            "to_frame": int(frame_ids[index + 1]),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "mae": float(np.mean(np.abs(values))),
            "valid_pixels": int(valid.sum()),
            "valid_coverage": float(valid.mean()),
        }
        if not np.isfinite(row["rmse"]) or not np.isfinite(row["mae"]):
            raise ValueError("temporal metrics contain non-finite values")
        rows.append(row)
        maps.append({
            "pair": row["pair"],
            "gt_change": gt_change,
            "pred_change": pred_change,
            "residual": residual,
            "valid": valid,
        })
    return rows, maps
