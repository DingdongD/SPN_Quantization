"""Shared contracts and math for NLSPN temporal-residual validation."""

import hashlib
from pathlib import Path

import numpy as np


HEIGHT = 228
WIDTH = 304
SPARSE_COUNT = 500
SPARSE_SEED = 2026
MAX_DEPTH = 10.0
PILOT_CLIPS = (
    (1, 32),
    (282, 313),
    (563, 594),
    (844, 875),
    (1126, 1157),
    (1407, 1438),
    (1688, 1719),
    (1969, 2000),
)


def clip_frame_ids(clips):
    return tuple(
        frame_id
        for start, end in clips
        for frame_id in range(int(start), int(end) + 1)
    )


def clip_pairs(clips):
    return tuple(
        (frame_id, frame_id + 1)
        for start, end in clips
        for frame_id in range(int(start), int(end))
    )


def validate_clip_payload(payload):
    frame_ids = np.asarray(payload["frame_ids"])
    frame_count = frame_ids.size
    expected = {
        "rgb": (frame_count, 3, HEIGHT, WIDTH),
        "sparse": (frame_count, HEIGHT, WIDTH),
        "gt": (frame_count, HEIGHT, WIDTH),
        "valid": (frame_count, HEIGHT, WIDTH),
    }
    for key, shape in expected.items():
        value = np.asarray(payload[key])
        if value.shape != shape:
            raise ValueError("%s must have shape %r" % (key, shape))
    sparse_count = np.count_nonzero(payload["sparse"], axis=(1, 2))
    if not np.all(sparse_count == SPARSE_COUNT):
        raise ValueError("every frame must contain exactly 500 sparse points")
    if not np.isfinite(payload["rgb"]).all():
        raise ValueError("RGB contains non-finite values")
    for key in ("sparse", "gt"):
        if not np.isfinite(np.asarray(payload[key])).all():
            raise ValueError("%s contains non-finite values" % key)
    return payload


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backward_warp(source, flow):
    import torch
    import torch.nn.functional as torch_f

    if source.ndim != 4 or flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("source and flow must be BCHW tensors")
    batch, _, height, width = source.shape
    if flow.shape != (batch, 2, height, width):
        raise ValueError("flow shape does not match source")
    y_axis = torch.arange(
        height, device=source.device, dtype=source.dtype)
    x_axis = torch.arange(
        width, device=source.device, dtype=source.dtype)
    try:
        ys, xs = torch.meshgrid(y_axis, x_axis, indexing="ij")
    except TypeError as error:
        if "indexing" not in str(error):
            raise
        ys, xs = torch.meshgrid(y_axis, x_axis)
    sample_x = xs[None] + flow[:, 0]
    sample_y = ys[None] + flow[:, 1]
    in_bounds = (
        (sample_x >= 0) & (sample_x <= width - 1) &
        (sample_y >= 0) & (sample_y <= height - 1)
    )
    grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = torch_f.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped, in_bounds[:, None]


def sparse_residual_seed(sparse, base):
    sparse = np.asarray(sparse, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    if sparse.shape != base.shape:
        raise ValueError("sparse and base shapes differ")
    mask = sparse > 0.0
    seed = np.zeros_like(sparse)
    seed[mask] = sparse[mask] - base[mask]
    return seed, mask


def pooled_quality(full, reconstructed, gt, valid):
    full = np.asarray(full, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (full.shape == reconstructed.shape == gt.shape == valid.shape):
        raise ValueError("quality arrays must have identical shapes")
    if not np.any(valid):
        raise ValueError("quality mask is empty")
    full_error = full[valid] - gt[valid]
    reconstructed_error = reconstructed[valid] - gt[valid]
    rmse_full = float(np.sqrt(np.mean(full_error ** 2)))
    if rmse_full <= 0.0:
        raise ValueError("full baseline RMSE must be positive")
    rmse_reconstructed = float(
        np.sqrt(np.mean(reconstructed_error ** 2)))
    ratio = rmse_reconstructed / rmse_full
    return {
        "rmse_full": rmse_full,
        "rmse_reconstructed": rmse_reconstructed,
        "quality_ratio": ratio,
        "passes": bool(ratio <= 1.01),
    }


def residual_statistics(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("residual statistics require finite values")
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mae": float(np.mean(absolute)),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "residual_energy": float(np.sum(values ** 2)),
        "fraction_below_1cm": float(np.mean(absolute <= 0.01)),
        "fraction_below_2cm": float(np.mean(absolute <= 0.02)),
        "fraction_below_5cm": float(np.mean(absolute <= 0.05)),
        "fraction_below_10cm": float(np.mean(absolute <= 0.10)),
    }
