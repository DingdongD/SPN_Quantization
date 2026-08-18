#!/usr/bin/env python3
"""Run, validate, and recoverably promote six-scene RAFT-GOP2 results."""

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
    "run_nlspn_cross_scene_raft_visualization_worker.py")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
DEFAULT_CHECKPOINT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS_JSON = DEFAULT_CHECKPOINT.with_name("args.json")
DEFAULT_FORMAL_DIR = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "BeachApartmentInterior_My_ir/pilot_256")
DEFAULT_TARGET = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "cross_scene_motion_windows")
DEFAULT_STAGING = DEFAULT_TARGET.with_name(
    "cross_scene_motion_windows_raft_staging")
DEFAULT_BACKUP = DEFAULT_TARGET.with_name(
    "cross_scene_motion_windows_pre_raft_backup")

from scripts import generate_nlspn_frame_difference_visualization as scene_launcher
from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_temporal_residual as residual
from scripts import nlspn_validated_output_promotion as promotion
from scripts import raft_small_compat


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError("expected a JSON object: {}".format(path))
    return value


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def load_recorded_windows(output_root):
    rows = _read_csv(Path(output_root) / "selected_windows.csv")
    if len(rows) != len(motion.SCENES) or \
            [row.get("scene") for row in rows] != list(motion.SCENES):
        raise ValueError("recorded windows do not contain approved scene order")
    windows = []
    for row in rows:
        try:
            frame_ids = json.loads(row["frame_ids"])
            pair_scores = json.loads(row["pair_scores"])
            window = {
                "scene": row["scene"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "frame_ids": frame_ids,
                "pair_scores": pair_scores,
                "motion_score": float(row["motion_score"]),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("recorded window row is invalid")
        if (not isinstance(frame_ids, list) or len(frame_ids) != 5 or
                any(not isinstance(item, int) or item <= 0
                    for item in frame_ids) or
                any(right != left + 1
                    for left, right in zip(frame_ids, frame_ids[1:])) or
                window["start_frame"] != frame_ids[0] or
                window["end_frame"] != frame_ids[-1] or
                not isinstance(pair_scores, list) or len(pair_scores) != 4 or
                not np.isfinite(np.asarray(pair_scores, dtype=np.float64)).all() or
                not np.isfinite(window["motion_score"])):
            raise ValueError("recorded window contents are invalid")
        windows.append(window)
    return windows


def build_worker_command(data_root, manifest, checkpoint, args_json,
                         formal_dir, output_root, raft_weights, device, seed):
    command = [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH),
        "--data-root", str(data_root),
        "--manifest", str(manifest),
        "--checkpoint", str(checkpoint),
        "--args-json", str(args_json),
        "--formal-dir", str(formal_dir),
        "--output-root", str(output_root),
        "--raft-weights", str(raft_weights),
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


def _bool_value(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def _validate_fixed_configs(configs):
    if set(configs) != {"rgb_diff", "global_diff"}:
        raise RuntimeError("selected configs are incomplete")
    for variant in ("rgb_diff", "global_diff"):
        row = configs[variant]
        if (abs(float(row["threshold"]) - 2.0 / 255.0) > 1e-15 or
                int(row["dilation_radius"]) != 8):
            raise RuntimeError("selected configs differ from fixed values")


def _validate_raft_metadata(metadata, raft_digest):
    expected = {
        "method_order": list(visual.RAFT_METHOD_ORDER),
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
        "raft_weight_sha256": str(raft_digest),
        "raft_flow_updates": 12,
        "raft_flow_direction": "current_to_previous",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("RAFT metadata is invalid")


def validate_raft_final_tree(output_root, windows, checkpoint_digest,
                             sweep_digest, raft_digest,
                             selected_windows_digest,
                             scene_validator=None):
    scene_validator = scene_validator or scene_launcher.validate_final_artifacts
    output_root = Path(output_root)
    expected_children = set(motion.ROOT_ARTIFACTS).union(motion.SCENES)
    if {path.name for path in output_root.iterdir()} != expected_children:
        raise RuntimeError("RAFT output tree scope is invalid")
    for name in motion.ROOT_ARTIFACTS:
        path = output_root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("RAFT root artifact is missing or empty")
    if residual.file_sha256(
            output_root / "selected_windows.csv") != selected_windows_digest:
        raise RuntimeError("recorded selected windows changed")
    for scene in motion.SCENES:
        scene_dir = output_root / scene
        actual = {path.name for path in scene_dir.iterdir() if path.is_file()}
        if actual != set(visual.FINAL_ARTIFACTS):
            raise RuntimeError("{} artifact scope is invalid".format(scene))
        if any((scene_dir / name).stat().st_size <= 0
               for name in visual.FINAL_ARTIFACTS):
            raise RuntimeError("{} contains an empty artifact".format(scene))

    root_metadata = _read_json(output_root / "run_metadata.json")
    expected_root = {
        "complete": True,
        "checkpoint_sha256": str(checkpoint_digest),
        "formal_sweep_sha256": str(sweep_digest),
        "selected_window_count": 6,
        "summary_row_count": 30,
    }
    if any(root_metadata.get(key) != value
           for key, value in expected_root.items()):
        raise RuntimeError("RAFT root metadata is invalid")
    _validate_raft_metadata(root_metadata, raft_digest)
    _validate_fixed_configs(root_metadata.get("selected_configs", {}))

    expected_by_scene = {window["scene"]: window for window in windows}
    if tuple(expected_by_scene) != motion.SCENES:
        raise RuntimeError("RAFT expected window order is invalid")
    recomputed = []
    for scene in motion.SCENES:
        expected = expected_by_scene[scene]
        result = scene_validator(
            output_root / scene, checkpoint_digest, sweep_digest)
        if tuple(result.get("method_order", ())) != visual.RAFT_METHOD_ORDER:
            raise RuntimeError("scene method order is invalid")
        metadata = result["metadata"]
        if (metadata.get("scene") != scene or
                metadata.get("frame_ids") != expected["frame_ids"] or
                not np.isclose(float(metadata.get("motion_score")),
                               float(expected["motion_score"]),
                               rtol=0, atol=1e-15)):
            raise RuntimeError("{} metadata differs from selection".format(scene))
        _validate_raft_metadata(metadata, raft_digest)
        recomputed.extend(motion.build_scene_summary(scene, result["metrics"]))

    summary_rows = _read_csv(output_root / "cross_scene_summary.csv")
    actual_by_key = {(row.get("scene"), row.get("method")): row
                     for row in summary_rows}
    if len(summary_rows) != 30 or len(actual_by_key) != 30:
        raise RuntimeError("RAFT summary keys are invalid")
    for expected in recomputed:
        key = (expected["scene"], expected["method"])
        if key not in actual_by_key:
            raise RuntimeError("RAFT summary row is missing")
        actual = actual_by_key[key]
        for name in ("rmse", "mae", "latency_ms", "rmse_ratio"):
            if not np.isclose(float(actual[name]), float(expected[name]),
                              rtol=1e-12, atol=1e-12):
                raise RuntimeError("RAFT summary {} differs".format(name))
        if (int(actual["valid_pixels"]) != expected["valid_pixels"] or
                _bool_value(actual["passes_1pct"]) != expected["passes_1pct"]):
            raise RuntimeError("RAFT summary gate differs")
    return {
        "scene_count": len(motion.SCENES),
        "summary_row_count": len(recomputed),
        "nlspn_model_load_count": root_metadata["nlspn_model_load_count"],
        "raft_model_load_count": root_metadata["raft_model_load_count"],
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
            "RAFT worker failed with code {}:\n{}".format(
                completed.returncode, completed.stdout))
    return completed.stdout


def _replace_scene_logs(output_root, command, worker_output):
    content = "Command: {}\n\n{}".format(" ".join(command), worker_output)
    for scene in motion.SCENES:
        path = Path(output_root) / scene / "worker.log"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(str(temporary), str(path))


def run_evaluation(cli):
    target = Path(cli.target_root)
    staging = Path(cli.staging_root)
    backup = Path(cli.backup_root)
    if not target.is_dir():
        raise ValueError("completed target directory does not exist")
    if staging.exists():
        raise FileExistsError("staging path already exists: {}".format(staging))
    if backup.exists():
        raise FileExistsError("backup path already exists: {}".format(backup))
    windows = load_recorded_windows(target)
    selected_digest = residual.file_sha256(target / "selected_windows.csv")
    raft_digest = residual.file_sha256(cli.raft_weights)
    if raft_digest != raft_small_compat.EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("official RAFT-Small weight digest mismatch")
    checkpoint_digest = residual.file_sha256(cli.checkpoint)
    _, sweep_digest = visual.load_selected_configs(
        Path(cli.formal_dir) / "threshold_sweep.csv")
    formal_before = _directory_digests(cli.formal_dir)

    with tempfile.TemporaryDirectory(prefix="nlspn-raft-visual-") as temporary:
        manifest = Path(temporary) / "manifest.json"
        _atomic_json(manifest, {"windows": windows})
        command, environment = build_worker_command(
            cli.data_root, manifest, cli.checkpoint, cli.args_json,
            cli.formal_dir, staging, cli.raft_weights, cli.device, cli.seed)
        worker_output = _run_worker(command, environment)
    _replace_scene_logs(staging, command, worker_output)

    def validator(path):
        return validate_raft_final_tree(
            path, windows, checkpoint_digest, sweep_digest, raft_digest,
            selected_digest)

    staged = validator(staging)
    if formal_before != _directory_digests(cli.formal_dir):
        raise RuntimeError("formal pilot directory was modified")
    promoted = promotion.promote_validated_output(
        staging, target, backup, validator)
    if formal_before != _directory_digests(cli.formal_dir):
        raise RuntimeError("formal pilot directory was modified")
    response = {
        "complete": True,
        "target_root": promoted["target"],
        "backup_root": promoted["backup"],
        "scene_count": staged["scene_count"],
        "summary_row_count": staged["summary_row_count"],
        "nlspn_model_load_count": staged["nlspn_model_load_count"],
        "raft_model_load_count": staged["raft_model_load_count"],
        "selected_windows": windows,
        "all_scene_summary": staged["metadata"]["all_scene_summary"],
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Generate and promote six-scene RAFT-GOP2 visualizations")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--args-json", default=str(DEFAULT_ARGS_JSON))
    parser.add_argument("--formal-dir", default=str(DEFAULT_FORMAL_DIR))
    parser.add_argument("--raft-weights", default=str(
        raft_small_compat.DEFAULT_WEIGHT_PATH))
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET))
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING))
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    return run_evaluation(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
