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


def build_raft(device):
    from torchvision.models.optical_flow import (
        Raft_Small_Weights, raft_small)

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    return model, weights.transforms(), weights


def predict_backward_flow(model, transform, rgb, device, batch_size=4):
    import time

    import torch
    import torch.nn.functional as torch_f

    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[1:] != (3, 228, 304):
        raise ValueError("RAFT RGB must have shape [frames, 3, 228, 304]")
    if rgb.shape[0] < 2:
        raise ValueError("RAFT requires at least two frames")
    if int(batch_size) <= 0:
        raise ValueError("RAFT batch size must be positive")
    rgb_tensor = torch.from_numpy(rgb)
    flows = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for first in range(0, rgb_tensor.shape[0] - 1, int(batch_size)):
            last = min(first + int(batch_size), rgb_tensor.shape[0] - 1)
            current = rgb_tensor[first + 1:last + 1].to(device)
            previous = rgb_tensor[first:last].to(device)
            current = torch_f.pad(current, (0, 0, 2, 2), mode="replicate")
            previous = torch_f.pad(previous, (0, 0, 2, 2), mode="replicate")
            current, previous = transform(current, previous)
            prediction = model(current, previous)[-1]
            flows.append(prediction[:, :, 2:-2].cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    flow = torch.cat(flows).numpy().astype(np.float32)
    expected = (rgb_tensor.shape[0] - 1, 2, 228, 304)
    if flow.shape != expected or not np.isfinite(flow).all():
        raise ValueError("RAFT flow has invalid shape or values")
    return flow, seconds
