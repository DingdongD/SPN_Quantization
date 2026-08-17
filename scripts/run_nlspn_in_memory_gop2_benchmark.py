#!/usr/bin/env python3
"""Orchestrate pure-memory NLSPN GOP2 benchmarking."""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKER_PATH = Path(__file__).with_name(
    "run_nlspn_in_memory_gop2_worker.py")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
DEFAULT_CHECKPOINT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS_JSON = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/args.json")
DEFAULT_RAFT_WEIGHTS = Path(
    "/root/.cache/torch/hub/checkpoints/"
    "raft_small_C_T_V2-01064c6d.pth")
DEFAULT_OUTPUT = Path(
    "/workspace/VoxelNet/nlspn_in_memory_gop2/"
    "BeachApartmentInterior_My_ir/pilot_256")
MAX_ABS_TOLERANCE = 1e-3
RMSE_TOLERANCE = 1e-4


def encode_array(value):
    value = np.asarray(value)
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode_array(value):
    if not isinstance(value, str):
        raise TypeError("encoded array must be text")
    try:
        binary = base64.b64decode(value.encode("ascii"), validate=True)
        with io.BytesIO(binary) as stream:
            result = np.load(stream, allow_pickle=False)
    except (ValueError, TypeError, OSError) as error:
        raise ValueError("invalid encoded array payload") from error
    return np.asarray(result)


def compare_raft_flows(reference, compatible):
    reference = np.asarray(reference, dtype=np.float32)
    compatible = np.asarray(compatible, dtype=np.float32)
    if reference.shape != compatible.shape:
        raise ValueError("RAFT parity flow shapes differ")
    if reference.shape != (1, 2, 228, 304):
        raise ValueError("RAFT parity flow shape must be [1, 2, 228, 304]")
    if not np.isfinite(reference).all() or not np.isfinite(compatible).all():
        raise ValueError("RAFT parity flow contains non-finite values")
    difference = compatible.astype(np.float64) - reference.astype(np.float64)
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    return {
        "shape": list(reference.shape),
        "max_abs": max_abs,
        "rmse": rmse,
        "max_abs_tolerance": MAX_ABS_TOLERANCE,
        "rmse_tolerance": RMSE_TOLERANCE,
        "passes": bool(
            max_abs <= MAX_ABS_TOLERANCE and rmse <= RMSE_TOLERANCE),
    }


def require_raft_parity(reference, compatible):
    result = compare_raft_flows(reference, compatible)
    if not result["passes"]:
        raise RuntimeError(
            "RAFT parity failed: max_abs=%g rmse=%g" %
            (result["max_abs"], result["rmse"]))
    return result


def parse_worker_pipe_payload(stdout):
    if not isinstance(stdout, str):
        raise TypeError("worker output must be text")
    payload = None
    for line in reversed(stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict) and "array" in candidate:
            payload = candidate
            break
    if payload is None:
        raise ValueError("worker output has no array payload")
    array = decode_array(payload["array"])
    expected_shape = tuple(int(value) for value in payload.get("shape", ()))
    if tuple(array.shape) != expected_shape:
        raise ValueError("worker payload shape metadata differs")
    if str(array.dtype) != str(payload.get("dtype")):
        raise ValueError("worker payload dtype metadata differs")
    metadata = dict(payload)
    del metadata["array"]
    return array, metadata


def build_worker_command(stage, data_root, scene, checkpoint, args_json,
                         raft_weights, device, output_dir=None, clips=None,
                         warmup_repeats=None, timed_repeats=None):
    command = [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH),
        "--stage", str(stage),
        "--data-root", str(data_root),
        "--scene", str(scene),
        "--checkpoint", str(checkpoint),
        "--args-json", str(args_json),
        "--raft-weights", str(raft_weights),
        "--device", str(device),
    ]
    if output_dir is not None:
        command.extend(("--output-dir", str(output_dir)))
    if clips is not None:
        command.extend(("--clips", str(clips)))
    if warmup_repeats is not None:
        command.extend(("--warmup-repeats", str(int(warmup_repeats))))
    if timed_repeats is not None:
        command.extend(("--timed-repeats", str(int(timed_repeats))))
    source = NLSPN_ROOT / "src"
    paths = (REPO_ROOT, source, source / "model" / "deformconv")
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in paths] + ([existing] if existing else []))
    return command, environment

