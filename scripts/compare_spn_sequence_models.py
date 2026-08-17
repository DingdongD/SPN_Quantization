#!/usr/bin/env python3
"""Compare four depth-completion models on five canonical sequence frames."""

import argparse
import os
from pathlib import Path
import subprocess

import numpy as np

from scripts import spn_sequence_io as sequence_io


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
EXTERNAL_MODELS = MODEL_ORDER[1:]
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}

DEFAULT_CANONICAL_DIR = (
    "/workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/"
    "frames_0001_0005")
DEFAULT_OUTPUT_DIR = (
    "/workspace/VoxelNet/spn_model_comparison/BeachApartmentInterior_My_ir/"
    "frames_0001_0005")


def default_worker_specs(worker_path):
    root = Path(
        "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines")
    return {
        "dyspn": {
            "worker": str(worker_path),
            "environment": "pointkan",
            "pythonpath": "/workspace/external_depth_completion_models/DySPN",
            "checkpoint": str(root / "dyspn_iter6" / "best.pt"),
            "args_json": str(root / "dyspn_iter6" / "args.json"),
        },
        "nlspn": {
            "worker": str(worker_path),
            "environment": "completionformer-py37",
            "pythonpath": (
                "/workspace/external_depth_completion_models/"
                "NLSPN_ECCV20/src:"
                "/workspace/external_depth_completion_models/"
                "NLSPN_ECCV20/src/model/deformconv"),
            "checkpoint": str(root / "nlspn_iter18" / "best.pt"),
            "args_json": str(root / "nlspn_iter18" / "args.json"),
        },
        "completionformer": {
            "worker": str(worker_path),
            "environment": "completionformer-py37",
            "pythonpath": (
                "/workspace/CompletionFormer/src:"
                "/workspace/CompletionFormer/src/model/deformconv"),
            "checkpoint": str(
                root / "completionformer_iter18" / "best.pt"),
            "args_json": str(
                root / "completionformer_iter18" / "args.json"),
        },
    }


def build_worker_command(model, spec, canonical_dir, output, device):
    command = [
        "conda", "run", "-n", spec["environment"], "python",
        spec["worker"],
        "--model", model,
        "--canonical-dir", str(canonical_dir),
        "--checkpoint", spec["checkpoint"],
        "--args-json", spec["args_json"],
        "--output", str(output),
        "--device", str(device),
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = spec["pythonpath"] + (
        os.pathsep + existing if existing else "")
    return command, env


def _write_text_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_or_reuse_worker(model, spec, canonical_dir, model_dir, device,
                        frame_ids, input_digest, force=False,
                        runner=subprocess.run):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_dir / "predictions.npz"
    log_path = model_dir / "worker.log"
    checkpoint_digest = sequence_io.file_sha256(spec["checkpoint"])

    if result_path.is_file() and not force:
        try:
            result = sequence_io.load_worker_result(
                result_path,
                model,
                frame_ids,
                input_digest,
                checkpoint_digest,
            )
        except (FileNotFoundError, ValueError):
            pass
        else:
            if not log_path.is_file():
                _write_text_atomic(
                    log_path,
                    "Reused digest-matching cached worker result.\n")
            return result, True

    command, env = build_worker_command(
        model, spec, canonical_dir, result_path, device)
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    log = (
        "Command: %s\n\n[stdout]\n%s\n[stderr]\n%s\n" %
        (" ".join(command), completed.stdout, completed.stderr)
    )
    _write_text_atomic(log_path, log)
    if completed.returncode != 0:
        raise RuntimeError(
            "%s worker exited with exit code %d; see %s" %
            (model, completed.returncode, log_path))
    result = sequence_io.load_worker_result(
        result_path,
        model,
        frame_ids,
        input_digest,
        checkpoint_digest,
    )
    return result, False


def expected_artifacts(output_dir, frame_ids):
    output_dir = Path(output_dir)
    paths = []
    for model in MODEL_ORDER:
        model_dir = output_dir / model
        for frame_id in frame_ids:
            paths.extend([
                model_dir / ("frame_%04d.npz" % int(frame_id)),
                model_dir / ("frame_%04d.png" % int(frame_id)),
            ])
        if model in EXTERNAL_MODELS:
            paths.extend([
                model_dir / "predictions.npz",
                model_dir / "worker.log",
            ])
    paths.extend([
        output_dir / "four_model_depth_comparison.png",
        output_dir / "four_model_error_comparison.png",
        output_dir / "four_model_temporal_comparison_unregistered.png",
        output_dir / "four_model_frame_metrics.csv",
        output_dir / "four_model_temporal_metrics.csv",
        output_dir / "run_metadata.json",
    ])
    return paths


def collect_metrics(frame_ids, gt, valid, predictions):
    frame_ids = tuple(int(frame_id) for frame_id in frame_ids)
    frame_rows = []
    temporal_rows = []
    temporal_maps = {}
    for model in MODEL_ORDER:
        prediction = predictions[model]
        for index, frame_id in enumerate(frame_ids):
            row = sequence_io.frame_metrics(
                gt[index], prediction[index], valid[index])
            row.update({
                "model": model,
                "frame_id": frame_id,
                "sparse_count": sequence_io.SPARSE_COUNT,
            })
            frame_rows.append(row)
        rows, maps = sequence_io.temporal_metrics(
            gt, prediction, valid, frame_ids)
        for row in rows:
            row.update({"model": model, "alignment": "unregistered"})
            temporal_rows.append(row)
        temporal_maps[model] = maps
    return frame_rows, temporal_rows, temporal_maps


def make_parser():
    parser = argparse.ArgumentParser(
        description="Compare CSPN, DySPN, NLSPN, and CompletionFormer")
    parser.add_argument("--canonical-dir", default=DEFAULT_CANONICAL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser


def load_model_predictions(canonical, canonical_dir, output_dir, device,
                           specs, force=False,
                           worker_executor=run_or_reuse_worker):
    predictions = {
        "cspn": np.asarray(
            canonical["cspn_pred_clamped"], dtype=np.float32).copy()
    }
    worker_info = {}
    for model in EXTERNAL_MODELS:
        result, reused = worker_executor(
            model,
            specs[model],
            canonical_dir,
            Path(output_dir) / model,
            device,
            canonical["frame_ids"],
            canonical["input_digest"],
            force=force,
        )
        predictions[model] = result["pred_clamped"]
        worker_info[model] = {
            "reused": bool(reused),
            "checkpoint_digest": result["checkpoint_digest"],
            "metadata": result["metadata"],
        }
    return predictions, worker_info
