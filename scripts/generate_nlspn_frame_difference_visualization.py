#!/usr/bin/env python3
"""Generate and validate the five-frame causal NLSPN comparison."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKER_PATH = Path(__file__).with_name(
    "run_nlspn_frame_difference_visualization_worker.py")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
DEFAULT_CHECKPOINT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS_JSON = DEFAULT_CHECKPOINT.with_name("args.json")
DEFAULT_FORMAL_DIR = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "BeachApartmentInterior_My_ir/pilot_256")
DEFAULT_OUTPUT_DIR = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "BeachApartmentInterior_My_ir/frames_0001_0005_visualization")

from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_temporal_residual as residual


def build_worker_command(data_root, scene, checkpoint, args_json, formal_dir,
                         output_dir, device, seed):
    command = [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH),
        "--data-root", str(data_root),
        "--scene", str(scene),
        "--checkpoint", str(checkpoint),
        "--args-json", str(args_json),
        "--formal-dir", str(formal_dir),
        "--output-dir", str(output_dir),
        "--device", str(device),
        "--seed", str(int(seed)),
    ]
    source = NLSPN_ROOT / "src"
    paths = (REPO_ROOT, source, source / "model" / "deformconv")
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in paths] + ([existing] if existing else []))
    return command, environment


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("visualization metadata must be an object")
    return result


def validate_final_artifacts(output_dir, checkpoint_digest, sweep_digest):
    output_dir = Path(output_dir)
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != set(visual.FINAL_ARTIFACTS):
        raise RuntimeError("visualization artifact scope is invalid")
    if any((output_dir / name).stat().st_size == 0
           for name in visual.FINAL_ARTIFACTS):
        raise RuntimeError("visualization contains an empty artifact")
    metadata = _read_json(output_dir / "run_metadata.json")
    frame_ids = metadata.get("frame_ids")
    if (not isinstance(frame_ids, list) or len(frame_ids) != 5 or
            any(not isinstance(item, int) or item <= 0
                for item in frame_ids) or
            any(right != left + 1
                for left, right in zip(frame_ids, frame_ids[1:]))):
        raise RuntimeError("visualization metadata frame IDs are invalid")
    expected = {
        "complete": True,
        "artifact_count": 6,
        "checkpoint_sha256": str(checkpoint_digest),
        "formal_sweep_sha256": str(sweep_digest),
        "frame_ids": frame_ids,
        "depth_vmin_m": 0.0,
        "depth_vmax_m": 10.0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError("visualization metadata mismatch for %s" % key)
    configs = metadata.get("selected_configs", {})
    if set(configs) != {"rgb_diff", "global_diff"}:
        raise RuntimeError("visualization selected configs are incomplete")
    for variant in configs:
        if (not np.isfinite(float(configs[variant]["threshold"])) or
                int(configs[variant]["dilation_radius"]) <= 0):
            raise RuntimeError("visualization selected config is invalid")
    with (output_dir / "frame_metrics.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    keys = {(row.get("method"), int(row.get("frame_id", -1)))
            for row in metrics}
    expected_keys = {(method, frame_id) for method in visual.METHOD_ORDER
                     for frame_id in frame_ids}
    if len(metrics) != 20 or keys != expected_keys:
        raise RuntimeError("visualization metric rows are invalid")
    for row in metrics:
        values = [float(row[name]) for name in (
            "rmse", "mae", "valid_pixels", "latency_ms")]
        if not np.isfinite(values).all() or values[2] <= 0 or values[3] < 0:
            raise RuntimeError("visualization metrics are non-finite")
    archive = {}
    with np.load(output_dir / "predictions.npz", allow_pickle=False) as item:
        required = {
            "frame_ids", "rgb", "sparse", "gt", "valid",
            "full", "zero_flow", "rgb_diff", "global_diff"}
        if set(item.files) != required:
            raise RuntimeError("visualization archive keys are invalid")
        archive = dict((name, np.asarray(item[name]).copy())
                       for name in required)
    if not np.array_equal(
            archive["frame_ids"], np.asarray(frame_ids, dtype=np.int32)):
        raise RuntimeError("visualization archive frame IDs are invalid")
    if archive["rgb"].shape != (5, 3, 228, 304):
        raise RuntimeError("visualization archive RGB shape is invalid")
    for name in ("sparse", "gt", "valid") + visual.METHOD_ORDER:
        if archive[name].shape != (5, 228, 304):
            raise RuntimeError("visualization archive %s shape is invalid" % name)
    for method in visual.METHOD_ORDER:
        if not np.isfinite(archive[method]).all():
            raise RuntimeError("visualization prediction is non-finite")
    depth_size = Image.open(
        output_dir / "nlspn_frame_difference_depth_comparison.png").size
    error_size = Image.open(
        output_dir / "nlspn_frame_difference_error_comparison.png").size
    if min(depth_size + error_size) <= 0:
        raise RuntimeError("visualization PNG geometry is invalid")
    return {
        "metadata": metadata,
        "metrics": metrics,
        "archive": archive,
        "depth_image_size": depth_size,
        "error_image_size": error_size,
    }


def _directory_digests(path):
    return dict(
        (item.name, residual.file_sha256(item))
        for item in sorted(Path(path).iterdir()) if item.is_file())


def _run_worker(command, environment, output_dir):
    completed = subprocess.run(
        command, cwd=str(REPO_ROOT), env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "worker.log.tmp"
    temporary.write_text(
        "Command: %s\n\n%s" % (" ".join(command), completed.stdout),
        encoding="utf-8")
    os.replace(str(temporary), str(output_dir / "worker.log"))
    if completed.returncode != 0:
        raise RuntimeError(
            "visualization worker failed with code %d; see %s" %
            (completed.returncode, output_dir / "worker.log"))
    return completed.stdout


def _pooled_metrics(rows):
    result = {}
    for method in visual.METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        valid = sum(int(row["valid_pixels"]) for row in selected)
        squared = sum(float(row["rmse"]) ** 2 * int(row["valid_pixels"])
                      for row in selected)
        absolute = sum(float(row["mae"]) * int(row["valid_pixels"])
                       for row in selected)
        result[method] = {
            "rmse": float(np.sqrt(squared / valid)),
            "mae": float(absolute / valid),
        }
    return result


def make_parser():
    parser = argparse.ArgumentParser(
        description="Generate five-frame causal NLSPN visualizations")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--args-json", default=str(DEFAULT_ARGS_JSON))
    parser.add_argument("--formal-dir", default=str(DEFAULT_FORMAL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    formal_before = _directory_digests(cli.formal_dir)
    _, sweep_digest = visual.load_selected_configs(
        Path(cli.formal_dir) / "threshold_sweep.csv")
    command, environment = build_worker_command(
        cli.data_root, cli.scene, cli.checkpoint, cli.args_json,
        cli.formal_dir, cli.output_dir, cli.device, cli.seed)
    _run_worker(command, environment, cli.output_dir)
    result = validate_final_artifacts(
        cli.output_dir, residual.file_sha256(cli.checkpoint), sweep_digest)
    formal_after = _directory_digests(cli.formal_dir)
    if formal_before != formal_after:
        raise RuntimeError("formal pilot directory was modified")
    response = {
        "output_dir": str(Path(cli.output_dir).resolve()),
        "depth_figure": str((Path(cli.output_dir) /
            "nlspn_frame_difference_depth_comparison.png").resolve()),
        "error_figure": str((Path(cli.output_dir) /
            "nlspn_frame_difference_error_comparison.png").resolve()),
        "pooled_metrics": _pooled_metrics(result["metrics"]),
        "error_vmax_m": result["metadata"]["error_vmax_m"],
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return response


if __name__ == "__main__":
    main()
