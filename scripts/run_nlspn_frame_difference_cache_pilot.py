#!/usr/bin/env python3
"""Launch and validate the causal NLSPN frame-difference cache pilot."""

import argparse
import csv
import hashlib
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
    "run_nlspn_frame_difference_cache_worker.py")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
DEFAULT_CHECKPOINT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS_JSON = DEFAULT_CHECKPOINT.with_name("args.json")
DEFAULT_OUTPUT = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "BeachApartmentInterior_My_ir/pilot_256")
FINAL_ARTIFACTS = {
    "run_metadata.json", "summary.json", "threshold_sweep.csv",
    "frame_metrics.csv", "clip_summary.csv", "report.md",
}


def build_worker_command(data_root, scene, checkpoint, args_json, output_dir,
                         device, clips=None, calibration_clip_count=4,
                         warmup_repeats=1, timed_repeats=5, seed=2026):
    command = [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH),
        "--data-root", str(data_root),
        "--scene", str(scene),
        "--checkpoint", str(checkpoint),
        "--args-json", str(args_json),
        "--output-dir", str(output_dir),
        "--device", str(device),
        "--calibration-clip-count", str(int(calibration_clip_count)),
        "--warmup-repeats", str(int(warmup_repeats)),
        "--timed-repeats", str(int(timed_repeats)),
        "--seed", str(int(seed)),
    ]
    if clips is not None:
        command.extend(("--clips", str(clips)))
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


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_final_artifacts(output_dir, expected_frame_count,
                             timed_repeats, checkpoint_digest):
    output_dir = Path(output_dir)
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != FINAL_ARTIFACTS:
        raise RuntimeError("pilot output artifact scope is invalid")
    if any((output_dir / name).stat().st_size == 0
           for name in FINAL_ARTIFACTS):
        raise RuntimeError("pilot output contains an empty artifact")
    metadata = _read_json(output_dir / "run_metadata.json")
    summary = _read_json(output_dir / "summary.json")
    sweep_rows = _read_csv(output_dir / "threshold_sweep.csv")
    frame_rows = _read_csv(output_dir / "frame_metrics.csv")
    clip_rows = _read_csv(output_dir / "clip_summary.csv")
    expected_metadata = {
        "complete": True,
        "artifact_count": 6,
        "frame_count": int(expected_frame_count),
        "timed_repeats": int(timed_repeats),
        "checkpoint_sha256": str(checkpoint_digest),
        "intermediate_tensor_cache": False,
        "raft_constructed": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError("pilot metadata mismatch for %s" % key)
    if len(sweep_rows) != 24:
        raise RuntimeError("pilot threshold sweep must contain 24 rows")
    if sum(row.get("selected", "").lower() == "true"
           for row in sweep_rows) != 2:
        raise RuntimeError("pilot threshold sweep selection is invalid")
    variants = ("full", "zero_flow", "rgb_diff", "global_diff")
    expected_rows = len(variants) * int(expected_frame_count) * int(
        timed_repeats)
    if len(frame_rows) != expected_rows:
        raise RuntimeError("pilot timed frame row count is invalid")
    paths = summary.get("paths", {})
    if set(paths) != set(variants):
        raise RuntimeError("pilot summary paths are incomplete")
    for variant in variants:
        rows = [row for row in frame_rows if row.get("variant") == variant]
        if len(rows) != int(expected_frame_count) * int(timed_repeats):
            raise RuntimeError("pilot path frame count is invalid")
        latencies = np.asarray(
            [float(row["latency_ms"]) for row in rows], dtype=np.float64)
        if not np.isfinite(latencies).all() or np.any(latencies < 0.0):
            raise RuntimeError("pilot latency values are invalid")
        quality_rows = [row for row in rows if row.get("repeat") == "0"]
        if (len(quality_rows) != int(expected_frame_count) or
                any(not row.get("valid_pixels") for row in quality_rows)):
            raise RuntimeError("pilot repeat-zero quality rows are invalid")
        quality = paths[variant].get("quality", {})
        if set(quality) != {"calibration", "heldout", "all"}:
            raise RuntimeError("pilot quality splits are incomplete")
        values = [
            float(quality[split]["quality_ratio"])
            for split in ("calibration", "heldout", "all")]
        if not np.isfinite(values).all():
            raise RuntimeError("pilot quality values are non-finite")
    return {
        "metadata": metadata,
        "summary": summary,
        "sweep_rows": sweep_rows,
        "frame_rows": frame_rows,
        "clip_rows": clip_rows,
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_worker(command, environment):
    completed = subprocess.run(
        command, cwd=str(REPO_ROOT), env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    if completed.returncode != 0:
        raise RuntimeError("frame-difference worker failed with code %d:\n%s" %
                           (completed.returncode, completed.stdout))
    return completed.stdout


def _parse_clips(value):
    from scripts import nlspn_temporal_residual as residual
    if value is None:
        return residual.PILOT_CLIPS
    result = []
    for item in str(value).split(","):
        start, end = (int(field) for field in item.split(":"))
        if start <= 0 or end < start:
            raise ValueError("invalid clip bounds")
        result.append((start, end))
    return tuple(result)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run NLSPN frame-difference cache pilot")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--args-json", default=str(DEFAULT_ARGS_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clips")
    parser.add_argument("--calibration-clip-count", type=int, default=4)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timed-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    clips = _parse_clips(cli.clips)
    frame_count = sum(end - start + 1 for start, end in clips)
    command, environment = build_worker_command(
        cli.data_root, cli.scene, cli.checkpoint, cli.args_json,
        cli.output_dir, cli.device, clips=cli.clips,
        calibration_clip_count=cli.calibration_clip_count,
        warmup_repeats=cli.warmup_repeats,
        timed_repeats=cli.timed_repeats, seed=cli.seed)
    _run_worker(command, environment)
    result = validate_final_artifacts(
        cli.output_dir, frame_count, cli.timed_repeats,
        _file_sha256(cli.checkpoint))
    response = {
        "output_dir": str(Path(cli.output_dir).resolve()),
        "paths": dict(
            (variant, {
                "quality_ratio": values["quality"]["all"]["quality_ratio"],
                "speedup": values["speedup"],
            }) for variant, values in result["summary"]["paths"].items()),
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return response


if __name__ == "__main__":
    main()
