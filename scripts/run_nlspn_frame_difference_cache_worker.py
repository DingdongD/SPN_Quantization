#!/usr/bin/env python3
"""Legacy-environment worker for NLSPN frame-difference cache pilots."""

from __future__ import print_function

import numpy as np
import torch

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_in_memory_gop2 as online


MASK_FIELDS = (
    "stable_fraction",
    "changed_fraction",
    "photometric_changed_fraction",
    "sparse_changed_fraction",
    "out_of_bounds_fraction",
)


def _as_prediction(value, shape):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.shape != tuple(shape) or not np.isfinite(value).all():
        raise ValueError("prediction geometry or values are invalid")
    return value


def run_calibration_sweep(engine, calibration_payloads, full_predictions):
    payloads = tuple(calibration_payloads)
    if not payloads:
        raise ValueError("calibration requires at least one clip")
    frame_count = int(sum(payload["frame_ids"].size for payload in payloads))
    full_predictions = list(full_predictions)
    if len(full_predictions) != frame_count:
        raise ValueError("full reference prediction count is invalid")
    frame_id_min = min(int(payload["frame_ids"][0]) for payload in payloads)
    frame_id_max = max(int(payload["frame_ids"][-1]) for payload in payloads)
    rows = []
    for variant in ("rgb_diff", "global_diff"):
        for config in cache.candidate_configs(variant):
            prediction_index = 0
            full_sse = 0.0
            variant_sse = 0.0
            valid_pixels = 0
            metric_values = dict((field, []) for field in MASK_FIELDS)
            for payload in payloads:
                engine.reset()
                for local_index in range(payload["frame_ids"].size):
                    if online.frame_kind(local_index) == "I":
                        result = engine.infer_i(
                            payload["rgb"][local_index],
                            payload["sparse"][local_index], local_index)
                    else:
                        result = engine.infer_p(
                            payload["rgb"][local_index],
                            payload["sparse"][local_index], local_index,
                            config)
                        for field in MASK_FIELDS:
                            metric_values[field].append(
                                float(result.mask_metrics[field]))
                    gt = np.asarray(payload["gt"][local_index],
                                    dtype=np.float64)
                    valid = np.asarray(payload["valid"][local_index],
                                       dtype=bool)
                    prediction = _as_prediction(result.prediction, gt.shape)
                    full = _as_prediction(
                        full_predictions[prediction_index], gt.shape)
                    full_error = full.astype(np.float64)[valid] - gt[valid]
                    variant_error = prediction.astype(
                        np.float64)[valid] - gt[valid]
                    full_sse += float(np.sum(full_error ** 2))
                    variant_sse += float(np.sum(variant_error ** 2))
                    valid_pixels += int(np.count_nonzero(valid))
                    prediction_index += 1
            if prediction_index != frame_count or valid_pixels <= 0:
                raise RuntimeError("calibration did not consume exact frames")
            row = {
                "variant": variant,
                "threshold": float(config.threshold),
                "dilation_radius": int(config.dilation_radius),
                "sse": variant_sse,
                "full_sse": full_sse,
                "valid_pixels": valid_pixels,
                "rmse": float(np.sqrt(variant_sse / valid_pixels)),
                "full_rmse": float(np.sqrt(full_sse / valid_pixels)),
                "frame_count": frame_count,
                "frame_id_min": frame_id_min,
                "frame_id_max": frame_id_max,
                "selected": False,
            }
            for field, values in metric_values.items():
                row[field] = float(np.mean(values)) if values else 0.0
            rows.append(row)
    return rows
