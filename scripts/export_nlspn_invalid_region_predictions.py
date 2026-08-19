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
