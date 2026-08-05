#!/usr/bin/env python3
"""Validation and sample selection for quantized depth predictions."""

from __future__ import division

import math
from pathlib import Path

import numpy as np


def prediction_metrics(gt, pred):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError("gt and pred must have identical shapes")
    valid_gt = np.isfinite(gt) & (gt > 1e-4)
    finite = valid_gt & np.isfinite(pred)
    diff = pred[finite] - gt[finite]
    if diff.size:
        rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
        mae = float(np.mean(np.abs(diff.astype(np.float64))))
    else:
        rmse = float("nan")
        mae = float("nan")
    nonfinite = int(np.count_nonzero(valid_gt & ~np.isfinite(pred)))
    valid_gt_pixels = int(np.count_nonzero(valid_gt))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "nonfinite_pixels": nonfinite,
        "num_pixels": int(np.count_nonzero(finite)),
        "valid_gt_pixels": valid_gt_pixels,
        "nonfinite_rate": (nonfinite / float(valid_gt_pixels)
                           if valid_gt_pixels else 0.0),
    }


def validate_sample_sets(prediction_root, model_configs, expected_indices):
    prediction_root = Path(prediction_root)
    expected = set(int(index) for index in expected_indices)
    output = {}
    for config in model_configs:
        paths = sorted((prediction_root / config).glob("sample_*.npz"))
        observed = []
        for path in paths:
            with np.load(str(path), allow_pickle=False) as payload:
                observed.append(int(payload["sample_index"]))
        if len(observed) != len(set(observed)):
            raise ValueError("duplicate prediction sample for %s" % config)
        if set(observed) != expected or len(paths) != len(expected):
            raise ValueError("prediction sample mismatch for %s" % config)
        output[config] = paths
    return output


def _finite_indices(base):
    return [index for index in sorted(base)
            if math.isfinite(float(base[index]["RMSE"]))]


def _nearest_unselected(indices, rows, target, selected):
    choices = [index for index in indices if index not in selected]
    if not choices:
        raise ValueError("not enough unique finite samples for selection")
    return min(choices, key=lambda index: (
        abs(float(rows[index]["RMSE"]) - target), index))


def select_representative_samples(rows, model):
    by_role = {}
    for row in rows:
        by_role.setdefault(row["role"], {})[int(row["sample_index"])] = row
    if "w4a4" not in by_role or "sparse_a8" not in by_role:
        raise ValueError("selection requires w4a4 and sparse_a8 rows")
    base = by_role["w4a4"]
    sparse = by_role["sparse_a8"]
    if set(base) != set(sparse):
        raise ValueError("w4a4 and sparse_a8 sample sets differ")
    finite = _finite_indices(base)
    if len(finite) < 3 or len(base) < 4:
        raise ValueError("selection requires at least four samples")

    selected = []
    reasons = []
    values = [float(base[index]["RMSE"]) for index in finite]
    for reason, quantile in (("median_w4a4", 0.5), ("p90_w4a4", 0.9)):
        target = float(np.quantile(values, quantile))
        selected.append(_nearest_unselected(
            finite, base, target, selected))
        reasons.append(reason)

    maximum = max((index for index in finite if index not in selected),
                  key=lambda index: (float(base[index]["RMSE"]), -index))
    selected.append(maximum)
    reasons.append("maximum_w4a4")

    remaining = [index for index in sorted(base) if index not in selected]
    if model == "cspn":
        fourth = max(remaining, key=lambda index: (
            float(base[index].get("nonfinite_rate", 0.0)), -index))
        fourth_reason = "maximum_nonfinite"
    else:
        def recovery(index):
            base_rmse = float(base[index]["RMSE"])
            sparse_rmse = float(sparse[index]["RMSE"])
            if not math.isfinite(base_rmse) or not math.isfinite(sparse_rmse):
                return float("-inf")
            return base_rmse - sparse_rmse
        fourth = max(remaining, key=lambda index: (recovery(index), -index))
        fourth_reason = "maximum_sparse_recovery"
    selected.append(fourth)
    reasons.append(fourth_reason)

    return [{
        "model": model,
        "sample_index": int(index),
        "reason": reason,
    } for index, reason in zip(selected, reasons)]


def cross_validate_metrics(computed_rows, formal_rows, tolerance=1e-5):
    formal = dict((
        (row["model"], row["config"], int(row["sample_index"])), row)
        for row in formal_rows)
    for row in computed_rows:
        key = (row["model"], row["config"], int(row["sample_index"]))
        if key not in formal:
            raise ValueError("missing formal metric for %s" % (key,))
        for field in ("RMSE", "MAE"):
            if not np.isclose(float(row[field]), float(formal[key][field]),
                              rtol=tolerance, atol=tolerance,
                              equal_nan=True):
                raise ValueError("metric mismatch for %s %s" % (key, field))
        if int(row.get("nonfinite_pixels", 0)) != int(float(
                formal[key].get("nonfinite_pixels", 0))):
            raise ValueError("metric mismatch for %s nonfinite_pixels" %
                             (key,))
