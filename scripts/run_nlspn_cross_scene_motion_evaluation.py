#!/usr/bin/env python3
"""Select, run, and independently validate six NLSPN motion windows."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKER_PATH = Path(__file__).with_name(
    "run_nlspn_cross_scene_motion_worker.py")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
DEFAULT_CHECKPOINT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS_JSON = DEFAULT_CHECKPOINT.with_name("args.json")
DEFAULT_FORMAL_DIR = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "BeachApartmentInterior_My_ir/pilot_256")
DEFAULT_OUTPUT_ROOT = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "cross_scene_motion_windows")

from scripts import generate_nlspn_frame_difference_visualization as visual_launcher
from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_temporal_residual as residual


def build_worker_command(data_root, manifest, checkpoint, args_json,
                         formal_dir, output_root, device, seed):
    command = [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH),
        "--data-root", str(data_root),
        "--manifest", str(manifest),
        "--checkpoint", str(checkpoint),
        "--args-json", str(args_json),
        "--formal-dir", str(formal_dir),
        "--output-root", str(output_root),
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


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def prepare_windows(data_root, manifest_path):
    windows = [motion.scan_scene(Path(data_root) / scene)
               for scene in motion.SCENES]
    _atomic_json(manifest_path, {"windows": windows})
    return windows


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError("expected a JSON object: {}".format(path))
    return value


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _bool_value(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def _validate_fixed_config_dict(configs):
    if set(configs) != {"rgb_diff", "global_diff"}:
        raise RuntimeError("selected configs are incomplete")
    for variant in ("rgb_diff", "global_diff"):
        row = configs[variant]
        if (abs(float(row["threshold"]) - 2.0 / 255.0) > 1e-15 or
                int(row["dilation_radius"]) != 8):
            raise RuntimeError("selected configs differ from fixed values")


def validate_final_tree(output_root, windows, checkpoint_digest, sweep_digest,
                        scene_validator=None):
    scene_validator = scene_validator or visual_launcher.validate_final_artifacts
    output_root = Path(output_root)
    expected_children = set(motion.ROOT_ARTIFACTS).union(motion.SCENES)
    actual_children = {path.name for path in output_root.iterdir()}
    if actual_children != expected_children:
        raise RuntimeError("cross-scene output tree scope is invalid")
    for name in motion.ROOT_ARTIFACTS:
        path = output_root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("cross-scene root artifact is missing or empty")
    for scene in motion.SCENES:
        scene_dir = output_root / scene
        actual = {path.name for path in scene_dir.iterdir() if path.is_file()}
        if actual != set(visual.FINAL_ARTIFACTS):
            raise RuntimeError("{} artifact scope is invalid".format(scene))
        if any((scene_dir / name).stat().st_size <= 0
               for name in visual.FINAL_ARTIFACTS):
            raise RuntimeError("{} contains an empty artifact".format(scene))

    root_metadata = _read_json(output_root / "run_metadata.json")
    expected_metadata = {
        "complete": True,
        "checkpoint_sha256": str(checkpoint_digest),
        "formal_sweep_sha256": str(sweep_digest),
        "model_load_count": 1,
        "selected_window_count": 6,
        "summary_row_count": 24,
    }
    for key, value in expected_metadata.items():
        if root_metadata.get(key) != value:
            raise RuntimeError("root metadata mismatch for {}".format(key))
    _validate_fixed_config_dict(root_metadata.get("selected_configs", {}))

    selected_rows = _read_csv(output_root / "selected_windows.csv")
    if len(selected_rows) != 6:
        raise RuntimeError("selected window row count is invalid")
    expected_by_scene = {row["scene"]: row for row in windows}
    if [row.get("scene") for row in selected_rows] != list(motion.SCENES):
        raise RuntimeError("selected window order is invalid")
    for row in selected_rows:
        expected = expected_by_scene[row["scene"]]
        if (json.loads(row["frame_ids"]) != expected["frame_ids"] or
                int(row["start_frame"]) != expected["start_frame"] or
                int(row["end_frame"]) != expected["end_frame"] or
                not np.isclose(float(row["motion_score"]),
                               float(expected["motion_score"]), rtol=0, atol=1e-15)):
            raise RuntimeError("selected window contents are invalid")

    recomputed = []
    for scene in motion.SCENES:
        expected = expected_by_scene[scene]
        result = scene_validator(
            output_root / scene, checkpoint_digest, sweep_digest)
        metadata = result["metadata"]
        if (metadata.get("scene") != scene or
                metadata.get("frame_ids") != expected["frame_ids"] or
                not np.isclose(float(metadata.get("motion_score")),
                               float(expected["motion_score"]),
                               rtol=0, atol=1e-15)):
            raise RuntimeError("{} metadata differs from selection".format(scene))
        recomputed.extend(motion.build_scene_summary(scene, result["metrics"]))

    summary_rows = _read_csv(output_root / "cross_scene_summary.csv")
    actual_by_key = {(row.get("scene"), row.get("method")): row
                     for row in summary_rows}
    if len(summary_rows) != 24 or len(actual_by_key) != 24:
        raise RuntimeError("cross-scene summary keys are invalid")
    for expected in recomputed:
        key = (expected["scene"], expected["method"])
        if key not in actual_by_key:
            raise RuntimeError("cross-scene summary row is missing")
        actual = actual_by_key[key]
        for name in ("rmse", "mae", "latency_ms", "rmse_ratio"):
            if not np.isclose(float(actual[name]), float(expected[name]),
                              rtol=1e-12, atol=1e-12):
                raise RuntimeError("cross-scene summary {} differs".format(name))
        if (int(actual["valid_pixels"]) != expected["valid_pixels"] or
                _bool_value(actual["passes_1pct"]) != expected["passes_1pct"]):
            raise RuntimeError("cross-scene summary gate differs")
    return {
        "scene_count": len(motion.SCENES),
        "summary_row_count": len(recomputed),
        "model_load_count": root_metadata["model_load_count"],
        "metadata": root_metadata,
    }


def _directory_digests(path):
    return dict(
        (item.name, residual.file_sha256(item))
        for item in sorted(Path(path).iterdir()) if item.is_file())


def _run_worker(command, environment):
    completed = subprocess.run(
        command, cwd=str(REPO_ROOT), env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "cross-scene worker failed with code {}:\n{}".format(
                completed.returncode, completed.stdout))
    return completed.stdout


def _replace_scene_logs(output_root, command, worker_output):
    content = "Command: {}\n\n{}".format(" ".join(command), worker_output)
    for scene in motion.SCENES:
        path = Path(output_root) / scene / "worker.log"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(str(temporary), str(path))


def make_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed causal NLSPN caches on six scenes")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--args-json", default=str(DEFAULT_ARGS_JSON))
    parser.add_argument("--formal-dir", default=str(DEFAULT_FORMAL_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    formal_before = _directory_digests(cli.formal_dir)
    _, sweep_digest = visual.load_selected_configs(
        Path(cli.formal_dir) / "threshold_sweep.csv")
    checkpoint_digest = residual.file_sha256(cli.checkpoint)
    with tempfile.TemporaryDirectory(prefix="nlspn-cross-scene-") as temporary:
        manifest = Path(temporary) / "manifest.json"
        windows = prepare_windows(cli.data_root, manifest)
        command, environment = build_worker_command(
            cli.data_root, manifest, cli.checkpoint, cli.args_json,
            cli.formal_dir, cli.output_root, cli.device, cli.seed)
        worker_output = _run_worker(command, environment)
    _replace_scene_logs(cli.output_root, command, worker_output)
    result = validate_final_tree(
        cli.output_root, windows, checkpoint_digest, sweep_digest)
    if formal_before != _directory_digests(cli.formal_dir):
        raise RuntimeError("formal pilot directory was modified")
    response = {
        "complete": True,
        "output_root": str(Path(cli.output_root).resolve()),
        "selected_windows": windows,
        "all_scene_summary": result["metadata"]["all_scene_summary"],
        "model_load_count": result["model_load_count"],
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return response


if __name__ == "__main__":
    main()
