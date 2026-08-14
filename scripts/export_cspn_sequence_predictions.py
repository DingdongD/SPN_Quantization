#!/usr/bin/env python3
"""Export CSPN predictions and unregistered temporal diagnostics."""

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MAX_DEPTH = 10.0
OUTPUT_HEIGHT = 228
OUTPUT_WIDTH = 304


def load_rgb(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("rb") as stream:
        return np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8).copy()


def read_exr_depth(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("failed to decode EXR depth: %s" % path)
    if depth.ndim != 3 or depth.shape[2] != 3:
        raise ValueError("EXR depth must contain three channels: %s" % path)
    if not (np.allclose(depth[..., 0], depth[..., 1], equal_nan=True) and
            np.allclose(depth[..., 0], depth[..., 2], equal_nan=True)):
        raise ValueError("EXR depth channels differ: %s" % path)
    return depth[..., 0].astype(np.float32, copy=True)


def sanitize_depth(depth, max_depth=MAX_DEPTH):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth))
    return np.where(valid, depth, 0.0).astype(np.float32), valid


def preprocess_pair(rgb, depth, max_depth=MAX_DEPTH):
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape [height, width, 3]")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("rgb and depth must have identical image geometry")

    clean_depth, valid = sanitize_depth(depth, max_depth=max_depth)
    rgb_image = Image.fromarray(rgb, mode="RGB")
    depth_image = Image.fromarray(clean_depth, mode="F")
    valid_image = Image.fromarray(valid.astype(np.uint8) * 255, mode="L")

    rgb_image = TF.resize(
        rgb_image, 240, interpolation=InterpolationMode.BILINEAR)
    depth_image = TF.resize(
        depth_image, 240, interpolation=InterpolationMode.NEAREST)
    valid_image = TF.resize(
        valid_image, 240, interpolation=InterpolationMode.NEAREST)
    rgb_image = TF.center_crop(rgb_image, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    depth_image = TF.center_crop(depth_image, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    valid_image = TF.center_crop(valid_image, (OUTPUT_HEIGHT, OUTPUT_WIDTH))

    rgb_out = TF.to_tensor(rgb_image).numpy().astype(np.float32)
    depth_out = np.asarray(depth_image, dtype=np.float32).copy()
    valid_out = np.asarray(valid_image, dtype=np.uint8) > 0
    depth_out, depth_valid = sanitize_depth(depth_out, max_depth=max_depth)
    valid_out &= depth_valid
    depth_out[~valid_out] = 0.0
    return rgb_out, depth_out, valid_out


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


def load_compatible_state(model, state, allowed_missing=()):
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    normalized = {}
    for key, value in state.items():
        normalized[key.removeprefix("module.")] = value

    ignored = []
    legacy_key = "post_process_layer.sum_conv.weight"
    if legacy_key in normalized:
        value = normalized.pop(legacy_key)
        if (tuple(value.shape) != (1, 8, 1, 1, 1) or
                not bool(torch.all(value == 1).item())):
            raise RuntimeError("invalid CSPN fixed sum kernel in checkpoint")
        ignored.append(legacy_key)

    incompatible = model.load_state_dict(normalized, strict=False)
    allowed_missing = set(allowed_missing)
    missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing:
        raise RuntimeError("missing checkpoint keys: %s" % missing)
    if unexpected:
        raise RuntimeError("unexpected checkpoint keys: %s" % unexpected)
    return {
        "ignored": ignored,
        "missing": missing,
        "allowed_missing": sorted(
            set(incompatible.missing_keys) & allowed_missing),
        "unexpected": unexpected,
    }


def build_cspn(checkpoint, device):
    repo_root = Path(__file__).resolve().parents[1]
    models_root = repo_root / "models"
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))
    import torch_resnet_cspn_nyu as cspn_model

    model = cspn_model.resnet50(
        pretrained=False,
        cspn_config={"step": 24, "kernel": 3, "norm_type": "8sum"})
    try:
        state = torch.load(
            str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    allowed_missing = {
        key for key in model.state_dict() if key.endswith("._up_pool.weights")}
    report = load_compatible_state(
        model, state, allowed_missing=allowed_missing)
    return model.to(device).eval(), report
