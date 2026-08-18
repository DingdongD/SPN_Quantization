#!/usr/bin/env python3
"""Run, validate, and publish a stratified 90-window NLSPN evaluation."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKER_PATH = Path(__file__).with_name(
    "run_nlspn_stratified_motion_worker.py")
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
    "cross_scene_motion_stratified_5x3")
DEFAULT_STAGING = DEFAULT_TARGET.with_name(
    "cross_scene_motion_stratified_5x3_staging")
DEFAULT_RAFT_WEIGHTS = Path(
    "/root/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth")

from scripts import generate_nlspn_frame_difference_visualization as visual_launcher
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_stratified_motion_sampling as sampling
from scripts import nlspn_stratified_motion_statistics as statistics
from scripts import nlspn_temporal_residual as residual
from scripts import nlspn_validated_output_promotion as promotion
from scripts import raft_small_compat


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
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError("invalid boolean value")


def compare_rows(actual_rows, expected_rows, key_fields, label):
    """Compare serialized rows with independently recomputed typed rows."""
    actual_rows = list(actual_rows)
    expected_rows = list(expected_rows)
    actual = {tuple(str(row[name]) for name in key_fields): row
              for row in actual_rows}
    expected = {tuple(str(row[name]) for name in key_fields): row
                for row in expected_rows}
    if len(actual) != len(actual_rows) or set(actual) != set(expected):
        raise RuntimeError("{} row keys differ".format(label))
    for key, expected_row in expected.items():
        actual_row = actual[key]
        if set(actual_row) != set(expected_row):
            raise RuntimeError("{} row fields differ".format(label))
        for name, expected_value in expected_row.items():
            value = actual_row[name]
            if isinstance(expected_value, bool):
                equal = _bool_value(value) == expected_value
            elif isinstance(expected_value, (int, np.integer)):
                try:
                    equal = int(value) == int(expected_value)
                except (TypeError, ValueError):
                    equal = False
            elif isinstance(expected_value, (float, np.floating)):
                try:
                    equal = np.isclose(float(value), float(expected_value),
                                       rtol=1e-12, atol=1e-12)
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = str(value) == str(expected_value)
            if not equal:
                raise RuntimeError(
                    "{} row {} differs for {}".format(label, name, key))


def _directory_digests(path):
    return {item.name: residual.file_sha256(item)
            for item in sorted(Path(path).iterdir()) if item.is_file()}


def preflight(cli):
    target = Path(cli.target_root).resolve()
    staging_value = getattr(cli, "staging_root", None)
    staging = (Path(staging_value).resolve() if staging_value else
               target.with_name(target.name + "_staging"))
    if target.exists():
        raise FileExistsError("target path already exists: {}".format(target))
    if staging.exists():
        raise FileExistsError("staging path already exists: {}".format(staging))
    if target == staging or target.parent != staging.parent:
        raise ValueError("target and staging must be distinct siblings")
    required = (
        (Path(cli.data_root), "data root", True),
        (Path(cli.checkpoint), "checkpoint", False),
        (Path(cli.args_json), "args JSON", False),
        (Path(cli.formal_dir), "formal directory", True),
        (Path(cli.formal_dir) / "threshold_sweep.csv", "formal sweep", False),
        (Path(cli.raft_weights), "RAFT weights", False),
    )
    for path, label, directory in required:
        valid = path.is_dir() if directory else path.is_file()
        if not valid:
            raise FileNotFoundError("{} is missing: {}".format(label, path))
    raft_digest = residual.file_sha256(cli.raft_weights)
    if raft_digest != raft_small_compat.EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("official RAFT-Small weight digest mismatch")
    return target, staging


def _run_worker(command, environment):
    completed = subprocess.run(
        command, cwd=str(REPO_ROOT), env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("stratified worker failed with code {}:\n{}".format(
            completed.returncode, completed.stdout))
    return completed.stdout


def _replace_window_logs(output_root, windows, command, worker_output):
    content = "Command: {}\n\n{}".format(" ".join(command), worker_output)
    for window in windows:
        path = Path(output_root) / window["window_id"] / "worker.log"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(str(temporary), str(path))


def _load_all_frame_metrics(output_root, windows):
    rows = []
    for window in windows:
        source = _read_csv(
            Path(output_root) / window["window_id"] / "frame_metrics.csv")
        if len(source) != 25:
            raise RuntimeError("each window must contain 25 frame metric rows")
        for raw in source:
            row = dict(raw)
            for name, expected in (("window_id", window["window_id"]),
                                   ("scene", window["scene"]),
                                   ("stratum", window["stratum"])):
                if name in row and row[name] != expected:
                    raise RuntimeError("frame metric {} differs".format(name))
                row[name] = expected
            rows.append(row)
    if len(rows) != 2250:
        raise RuntimeError("frame metric table must contain 2250 rows")
    return rows


def _verify_png(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("PNG is missing or empty: {}".format(path))
    with Image.open(str(path)) as image:
        image.verify()


def validate_final_tree(output_root, windows, checkpoint_digest, sweep_digest,
                        raft_digest, manifest_digest, args_digest=None,
                        formal_digests=None, bootstrap_replicates=2000,
                        artifact_validator=None):
    artifact_validator = (artifact_validator or
                          visual_launcher.validate_final_artifacts)
    root = Path(output_root)
    windows = sampling.validate_selected_windows(
        windows, sampling.SELECTION_SEED)
    stored_windows = sampling.load_selected_windows_csv(
        root / "selected_windows.csv", sampling.SELECTION_SEED)
    stored_digest = sampling.canonical_manifest_sha256(stored_windows)
    if stored_digest != manifest_digest or stored_digest != \
            sampling.canonical_manifest_sha256(windows):
        raise RuntimeError("selected manifest digest differs")

    expected_root = set(statistics.ROOT_ARTIFACTS).union(sampling.motion.SCENES)
    if {path.name for path in root.iterdir()} != expected_root:
        raise RuntimeError("stratified output root scope is invalid")
    for name in statistics.ROOT_ARTIFACTS:
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("root artifact is missing or empty: {}".format(name))

    by_cell = {(scene, stratum): [] for scene in sampling.motion.SCENES
               for stratum in sampling.STRATA}
    for window in windows:
        by_cell[(window["scene"], window["stratum"])].append(window)
    all_metrics = []
    for scene in sampling.motion.SCENES:
        scene_dir = root / scene
        if {path.name for path in scene_dir.iterdir()} != set(sampling.STRATA):
            raise RuntimeError("scene stratum scope is invalid: {}".format(scene))
        for stratum in sampling.STRATA:
            stratum_dir = scene_dir / stratum
            selected = by_cell[(scene, stratum)]
            expected_names = {Path(row["window_id"]).name for row in selected}
            if {path.name for path in stratum_dir.iterdir()} != expected_names:
                raise RuntimeError("window scope is invalid: {}/{}".format(
                    scene, stratum))
            for window in selected:
                window_dir = root / window["window_id"]
                if {path.name for path in window_dir.iterdir()} != \
                        set(visual.FINAL_ARTIFACTS):
                    raise RuntimeError("window artifact scope is invalid")
                result = artifact_validator(
                    window_dir, checkpoint_digest, sweep_digest)
                metadata = result["metadata"]
                required = {
                    "complete": True,
                    "scene": scene,
                    "stratum": stratum,
                    "window_id": window["window_id"],
                    "frame_ids": window["frame_ids"],
                    "raft_weight_sha256": str(raft_digest),
                    "raft_flow_direction": "current_to_previous",
                    "raft_flow_updates": 12,
                    "nlspn_model_load_count": 1,
                    "raft_model_load_count": 1,
                }
                if args_digest is not None:
                    required["args_sha256"] = str(args_digest)
                for name, expected in required.items():
                    if metadata.get(name) != expected:
                        raise RuntimeError(
                            "window metadata mismatch for {}".format(name))
                if formal_digests is not None and \
                        metadata.get("formal_directory_digests") != formal_digests:
                    raise RuntimeError("window formal digest snapshot differs")
                if tuple(result.get("method_order", ())) != \
                        visual.RAFT_METHOD_ORDER:
                    raise RuntimeError("window method order differs")
                archive = result["archive"]
                for method in visual.RAFT_METHOD_ORDER:
                    if not np.isfinite(archive[method]).all():
                        raise RuntimeError("prediction is non-finite")
                for method in visual.RAFT_METHOD_ORDER[1:]:
                    if not np.array_equal(archive[method][[0, 2, 4]],
                                          archive["full"][[0, 2, 4]]):
                        raise RuntimeError(
                            "I-frame prediction differs from Full NLSPN")
                for row in result["metrics"]:
                    value = dict(row)
                    value.update({"scene": scene, "stratum": stratum,
                                  "window_id": window["window_id"]})
                    all_metrics.append(value)
                _verify_png(window_dir /
                            "nlspn_frame_difference_depth_comparison.png")
                _verify_png(window_dir /
                            "nlspn_frame_difference_error_comparison.png")

    p_rows = statistics.derive_p_frame_metrics(windows, all_metrics)
    summaries = statistics.build_stratified_summary(
        p_rows, bootstrap_replicates, sampling.SELECTION_SEED)
    correlations = statistics.build_correlation_summary(p_rows)
    compare_rows(_read_csv(root / "p_frame_metrics.csv"), p_rows,
                 ("window_id", "method", "frame_id"), "P-frame")
    compare_rows(_read_csv(root / "stratified_summary.csv"), summaries,
                 ("stratum", "method"), "summary")
    compare_rows(_read_csv(root / "correlation_summary.csv"), correlations,
                 ("method", "target"), "correlation")
    expected_report = statistics._render_report(
        windows, summaries, correlations)
    if (root / "report.md").read_text(encoding="utf-8") != expected_report:
        raise RuntimeError("report differs from recomputed statistics")
    _verify_png(root / "motion_error_scatter.png")
    _verify_png(root / "stratified_error_boxplot.png")

    metadata = _read_json(root / "run_metadata.json")
    required_root = {
        "complete": True,
        "selected_window_count": 90,
        "frame_count": 450,
        "p_frame_count": 180,
        "p_frame_row_count": 900,
        "summary_row_count": 15,
        "correlation_row_count": 13,
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
        "checkpoint_sha256": str(checkpoint_digest),
        "formal_sweep_sha256": str(sweep_digest),
        "raft_weight_sha256": str(raft_digest),
        "manifest_sha256": str(manifest_digest),
        "bootstrap_replicates": int(bootstrap_replicates),
        "method_order": list(visual.RAFT_METHOD_ORDER),
        "schedule": "I,P,I,P,I",
    }
    if args_digest is not None:
        required_root["args_sha256"] = str(args_digest)
    if formal_digests is not None:
        required_root["formal_directory_digests"] = formal_digests
    for name, expected in required_root.items():
        if metadata.get(name) != expected:
            raise RuntimeError("root metadata mismatch for {}".format(name))
    compare_rows(metadata.get("stratified_summary", []), summaries,
                 ("stratum", "method"), "embedded summary")
    compare_rows(metadata.get("correlation_summary", []), correlations,
                 ("method", "target"), "embedded correlation")
    return {
        "window_count": 90, "frame_count": 450,
        "p_frame_count": 180, "p_frame_row_count": 900,
        "nlspn_model_load_count": 1, "raft_model_load_count": 1,
        "metadata": metadata, "summaries": summaries,
        "correlations": correlations,
    }


def run_evaluation(cli, discover_fn=None):
    target, staging = preflight(cli)
    discover_fn = discover_fn or sampling.discover_selected_windows
    windows = discover_fn(cli.data_root, cli.seed)
    windows = sampling.validate_selected_windows(windows, cli.seed)

    formal_before = _directory_digests(cli.formal_dir)
    _, sweep_digest = visual.load_selected_configs(
        Path(cli.formal_dir) / "threshold_sweep.csv")
    checkpoint_digest = residual.file_sha256(cli.checkpoint)
    args_digest = residual.file_sha256(cli.args_json)
    raft_digest = residual.file_sha256(cli.raft_weights)
    manifest_digest = sampling.canonical_manifest_sha256(windows, cli.seed)
    staging.mkdir(parents=False)
    sampling.write_selected_windows_csv(
        staging / "selected_windows.csv", windows, cli.seed)

    with tempfile.TemporaryDirectory(
            prefix="nlspn-stratified-motion-") as temporary:
        manifest = Path(temporary) / "manifest.json"
        sampling.write_manifest_json(manifest, windows, cli.seed)
        command, environment = build_worker_command(
            cli.data_root, manifest, cli.checkpoint, cli.args_json,
            cli.formal_dir, staging, cli.raft_weights, cli.device, cli.seed)
        worker_output = _run_worker(command, environment)
    _replace_window_logs(staging, windows, command, worker_output)

    frame_metrics = _load_all_frame_metrics(staging, windows)
    p_rows = statistics.derive_p_frame_metrics(windows, frame_metrics)
    summaries = statistics.build_stratified_summary(
        p_rows, cli.bootstrap_replicates, cli.seed)
    correlations = statistics.build_correlation_summary(p_rows)
    root_metadata = {
        "schema_version": 1,
        "selection_seed": int(cli.seed),
        "bootstrap_replicates": int(cli.bootstrap_replicates),
        "bootstrap_seed": int(cli.seed),
        "checkpoint_sha256": checkpoint_digest,
        "args_sha256": args_digest,
        "formal_sweep_sha256": sweep_digest,
        "formal_directory_digests": formal_before,
        "raft_weight_sha256": raft_digest,
        "manifest_sha256": manifest_digest,
        "method_order": list(visual.RAFT_METHOD_ORDER),
        "schedule": "I,P,I,P,I",
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
        "device": str(cli.device),
        "numpy_version": str(np.__version__),
    }
    statistics.write_root_artifacts(
        staging, windows, p_rows, summaries, correlations, root_metadata)
    validate_args = (
        windows, checkpoint_digest, sweep_digest, raft_digest,
        manifest_digest, args_digest, formal_before,
        cli.bootstrap_replicates)
    validate_final_tree(staging, *validate_args)
    if _directory_digests(cli.formal_dir) != formal_before:
        raise RuntimeError("formal pilot directory was modified")
    promotion.promote_new_validated_output(
        staging, target,
        lambda root: validate_final_tree(root, *validate_args))
    result = validate_final_tree(target, *validate_args)
    if _directory_digests(cli.formal_dir) != formal_before:
        raise RuntimeError("formal pilot directory was modified")

    ranges = {name: {
        "minimum": min(row["motion_score"] for row in windows
                       if row["stratum"] == name),
        "maximum": max(row["motion_score"] for row in windows
                       if row["stratum"] == name),
    } for name in sampling.STRATA}
    response = {
        "complete": True,
        "target_root": str(target),
        "window_count": result["window_count"],
        "frame_count": result["frame_count"],
        "p_frame_count": result["p_frame_count"],
        "p_frame_row_count": result["p_frame_row_count"],
        "nlspn_model_load_count": result["nlspn_model_load_count"],
        "raft_model_load_count": result["raft_model_load_count"],
        "stratum_motion_ranges": ranges,
        "stratified_summary": result["summaries"],
        "correlation_summary": result["correlations"],
        "report": str(target / "report.md"),
        "plots": [str(target / "motion_error_scatter.png"),
                  str(target / "stratified_error_boxplot.png")],
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run a stratified 90-window causal NLSPN evaluation")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--args-json", default=str(DEFAULT_ARGS_JSON))
    parser.add_argument("--formal-dir", default=str(DEFAULT_FORMAL_DIR))
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET))
    parser.add_argument("--staging-root", default=None)
    parser.add_argument("--raft-weights", default=str(DEFAULT_RAFT_WEIGHTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=sampling.SELECTION_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    if cli.staging_root is None:
        target = Path(cli.target_root)
        cli.staging_root = str(target.with_name(target.name + "_staging"))
    if cli.seed != sampling.SELECTION_SEED:
        raise ValueError("formal evaluation seed must be 2026")
    if cli.bootstrap_replicates != 2000:
        raise ValueError("formal evaluation requires 2000 bootstrap replicates")
    return run_evaluation(cli)


if __name__ == "__main__":
    main()
