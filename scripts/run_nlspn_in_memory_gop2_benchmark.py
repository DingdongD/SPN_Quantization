#!/usr/bin/env python3
"""Orchestrate pure-memory NLSPN GOP2 benchmarking."""

import argparse
import base64
import csv
import io
import json
import os
from pathlib import Path
import subprocess
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
                         warmup_repeats=None, timed_repeats=None, seed=None,
                         parity=None):
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
    if seed is not None:
        command.extend(("--seed", str(int(seed))))
    if parity is not None:
        command.extend(("--parity-json", json.dumps(
            parity, sort_keys=True, separators=(",", ":"))))
    source = NLSPN_ROOT / "src"
    paths = (REPO_ROOT, source, source / "model" / "deformconv")
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in paths] + ([existing] if existing else []))
    return command, environment


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def validate_final_artifacts(output_dir, expected_frame_count,
                             timed_repeats, checkpoint_digest,
                             raft_digest):
    output_dir = Path(output_dir)
    expected_files = {
        "run_metadata.json", "summary.json", "frame_metrics.csv",
        "clip_summary.csv", "report.md",
    }
    actual_files = {
        path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("benchmark output artifact scope is invalid")
    if any((output_dir / name).stat().st_size == 0
           for name in expected_files):
        raise RuntimeError("benchmark output contains an empty artifact")
    metadata = _read_json(output_dir / "run_metadata.json")
    summary = _read_json(output_dir / "summary.json")
    expected_timed = int(expected_frame_count) * int(timed_repeats)
    expected_metadata = {
        "complete": True,
        "artifact_count": 5,
        "frame_count": int(expected_frame_count),
        "timed_repeats": int(timed_repeats),
        "timed_frames_per_path": expected_timed,
        "checkpoint_sha256": str(checkpoint_digest),
        "raft_weight_sha256": str(raft_digest),
        "intermediate_tensor_cache": False,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError("benchmark metadata mismatch for %s" % key)
    if (summary.get("quality_frame_count") != int(expected_frame_count) or
            summary.get("timed_repeats") != int(timed_repeats)):
        raise RuntimeError("benchmark summary frame counts differ")
    quality = summary.get("quality", {})
    ratio = float(quality.get("quality_ratio", float("nan")))
    if not quality.get("passes") or not np.isfinite(ratio) or ratio > 1.01:
        raise RuntimeError("benchmark pooled RMSE quality gate failed")
    with (output_dir / "frame_metrics.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.DictReader(stream))
    for path in ("full", "gop2"):
        rows = [row for row in frame_rows if row.get("path") == path]
        if len(rows) != expected_timed:
            raise RuntimeError("benchmark timed row count differs for %s" % path)
        latencies = np.asarray(
            [float(row["latency_ms"]) for row in rows], dtype=np.float64)
        if not np.isfinite(latencies).all() or np.any(latencies < 0.0):
            raise RuntimeError("benchmark latency values are invalid")
    return {
        "metadata": metadata,
        "summary": summary,
        "frame_rows": frame_rows,
    }


def _run_worker(command, environment):
    completed = subprocess.run(
        command, cwd=str(REPO_ROOT), env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "pure-memory worker failed with code %d:\n%s" %
            (completed.returncode, completed.stdout))
    return completed.stdout


def run_reference_flow(data_root, scene, device):
    from scripts import run_nlspn_temporal_residual_validation as validation

    rgb = np.stack([
        validation.load_preprocessed_frame(
            data_root, scene, frame_id)[0]
        for frame_id in (1, 2)
    ]).astype(np.float32)
    model, transform, weights = validation.build_raft(
        __import__("torch").device(device))
    flow, seconds = validation.predict_backward_flow(
        model, transform, rgb, __import__("torch").device(device),
        batch_size=1)
    weight_path = validation.resolve_raft_weight_path(weights)
    return flow, {
        "runtime_seconds": float(seconds),
        "weight_path": str(weight_path.resolve()),
        "weight_sha256": validation.residual.file_sha256(weight_path),
    }


def run_parity(cli):
    reference, reference_metadata = run_reference_flow(
        cli.data_root, cli.scene, cli.device)
    command, environment = build_worker_command(
        "raft-parity", cli.data_root, cli.scene, cli.checkpoint,
        cli.args_json, cli.raft_weights, cli.device)
    compatible, compatible_metadata = parse_worker_pipe_payload(
        _run_worker(command, environment))
    expected_digest = reference_metadata["weight_sha256"]
    if (compatible_metadata.get("weight_sha256") != expected_digest or
            expected_digest != _file_sha256(cli.raft_weights)):
        raise RuntimeError("RAFT parity weight identity differs")
    parity = require_raft_parity(reference, compatible)
    parity.update({
        "weight_sha256": expected_digest,
        "reference": reference_metadata,
        "compatible": compatible_metadata,
    })
    return parity


def _file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_clip_text(value):
    if value is None:
        from scripts import nlspn_temporal_residual as residual
        return residual.PILOT_CLIPS
    clips = []
    for item in str(value).split(","):
        start, end = (int(field) for field in item.split(":"))
        if start <= 0 or end < start:
            raise ValueError("invalid clip bounds")
        clips.append((start, end))
    return tuple(clips)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark pure-memory causal NLSPN GOP2")
    parser.add_argument("--stage", choices=("raft-parity", "all"),
                        default="all")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--args-json", default=str(DEFAULT_ARGS_JSON))
    parser.add_argument("--raft-weights", default=str(DEFAULT_RAFT_WEIGHTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clips")
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timed-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    parity = run_parity(cli)
    if cli.stage == "raft-parity":
        print(json.dumps(parity, indent=2, sort_keys=True))
        return parity
    clips = _parse_clip_text(cli.clips)
    frame_count = sum(end - start + 1 for start, end in clips)
    command, environment = build_worker_command(
        "benchmark", cli.data_root, cli.scene, cli.checkpoint,
        cli.args_json, cli.raft_weights, cli.device,
        output_dir=cli.output_dir, clips=cli.clips,
        warmup_repeats=cli.warmup_repeats,
        timed_repeats=cli.timed_repeats, seed=cli.seed, parity=parity)
    _run_worker(command, environment)
    result = validate_final_artifacts(
        cli.output_dir, frame_count, cli.timed_repeats,
        _file_sha256(cli.checkpoint), _file_sha256(cli.raft_weights))
    response = {
        "output_dir": str(Path(cli.output_dir).resolve()),
        "quality_ratio": result["summary"]["quality"]["quality_ratio"],
        "speedup": result["summary"]["speedup"],
        "parity": parity,
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return response


if __name__ == "__main__":
    main()
