#!/usr/bin/env python3
"""Orchestrate the frozen-NLSPN temporal residual pilot."""

import os
from pathlib import Path
import tempfile

import numpy as np

from scripts import export_cspn_sequence_predictions as sequence
from scripts import nlspn_temporal_residual as residual


def load_preprocessed_frame(data_root, scene, frame_id):
    rgb_path = Path(data_root) / scene / "rgb" / ("%04d.jpg" % frame_id)
    depth_path = (
        Path(data_root) / scene / "depth" / ("Image%04d.exr" % frame_id))
    return sequence.preprocess_pair(
        sequence.load_rgb(rgb_path), sequence.read_exr_depth(depth_path))


def write_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_clip(data_root, scene, frame_ids, output, seed):
    frames = [
        load_preprocessed_frame(data_root, scene, frame_id)
        for frame_id in frame_ids
    ]
    rgb, gt, valid = (np.stack(values) for values in zip(*frames))
    sparse, sparse_mask = sequence.build_shared_sparse_depths(
        gt, valid, count=residual.SPARSE_COUNT, seed=seed)
    payload = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "rgb": rgb.astype(np.float32),
        "sparse": sparse.astype(np.float32),
        "gt": gt.astype(np.float32),
        "valid": valid.astype(bool),
        "sparse_mask": sparse_mask.astype(bool),
    }
    residual.validate_clip_payload(payload)
    write_npz_atomic(output, **payload)
    return payload
