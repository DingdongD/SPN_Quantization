#!/usr/bin/env python3
"""Compare four depth-completion models on five canonical sequence frames."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
DEFAULT_CSPN_CHECKPOINT = "/workspace/VoxelNet/cspn_models/best_model.pth"


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
            "pred_raw": result.get("pred_raw", result["pred_clamped"]),
        }
    return predictions, worker_info


def _save_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(str(temporary), str(path))


def _write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(str(temporary), str(path))


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _masked(value, valid):
    return np.ma.masked_where(~np.asarray(valid, dtype=bool), value)


def _draw_image(ax, value, title, cmap=None, vmin=None, vmax=None):
    image = ax.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def _add_right_colorbar(fig, mappable, label, bottom=0.18, top=0.82):
    fig.subplots_adjust(right=0.88)
    colorbar_axis = fig.add_axes([0.91, bottom, 0.016, top - bottom])
    return fig.colorbar(mappable, cax=colorbar_axis, label=label)


def _render_frame_panel(path, frame_id, rgb, sparse, gt, valid, prediction,
                        metrics):
    absolute_error = np.abs(prediction - gt)
    valid_errors = absolute_error[valid]
    error_max = max(
        float(np.percentile(valid_errors, 99.0)) if valid_errors.size else 0.0,
        1e-3,
    )
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.25))
    _draw_image(axes[0], np.moveaxis(rgb, 0, -1), "RGB %04d" % frame_id)
    _draw_image(
        axes[1], np.ma.masked_where(sparse <= 0.0, sparse), "Sparse (500)",
        cmap="viridis", vmin=0.0, vmax=sequence_io.MAX_DEPTH)
    _draw_image(
        axes[2], _masked(gt, valid), "Ground truth", cmap="viridis",
        vmin=0.0, vmax=sequence_io.MAX_DEPTH)
    _draw_image(
        axes[3], prediction,
        "Prediction\nRMSE %.3f m" % metrics["rmse"], cmap="viridis",
        vmin=0.0, vmax=sequence_io.MAX_DEPTH)
    _draw_image(
        axes[4], _masked(absolute_error, valid),
        "Absolute error\nMAE %.3f m" % metrics["mae"], cmap="magma",
        vmin=0.0, vmax=error_max)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _render_depth_comparison(path, frame_ids, gt, valid, predictions):
    columns = ("gt",) + MODEL_ORDER
    fig, axes = plt.subplots(
        len(frame_ids), len(columns),
        figsize=(3.1 * len(columns), 2.45 * len(frame_ids)),
        squeeze=False,
    )
    last_image = None
    for row, frame_id in enumerate(frame_ids):
        for column, name in enumerate(columns):
            if name == "gt":
                value = _masked(gt[row], valid[row])
                title = "Ground truth"
            else:
                value = predictions[name][row]
                title = MODEL_NAMES[name]
            last_image = _draw_image(
                axes[row, column], value,
                title if row == 0 else "",
                cmap="viridis", vmin=0.0, vmax=sequence_io.MAX_DEPTH)
            if column == 0:
                axes[row, column].set_ylabel("Frame %04d" % int(frame_id))
    fig.suptitle("Five-frame depth completion: identical RGB + 500 sparse points")
    fig.subplots_adjust(top=0.94, right=0.88, wspace=0.04, hspace=0.12)
    _add_right_colorbar(fig, last_image, "Depth (m)")
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _render_error_comparison(path, frame_ids, gt, valid, predictions):
    errors = {
        model: np.abs(predictions[model] - gt) for model in MODEL_ORDER
    }
    all_valid = np.concatenate([
        errors[model][valid].astype(np.float64) for model in MODEL_ORDER])
    error_max = max(float(np.percentile(all_valid, 99.0)), 1e-3)
    fig, axes = plt.subplots(
        len(frame_ids), len(MODEL_ORDER),
        figsize=(3.1 * len(MODEL_ORDER), 2.45 * len(frame_ids)),
        squeeze=False,
    )
    last_image = None
    for row, frame_id in enumerate(frame_ids):
        for column, model in enumerate(MODEL_ORDER):
            last_image = _draw_image(
                axes[row, column], _masked(errors[model][row], valid[row]),
                MODEL_NAMES[model] if row == 0 else "", cmap="magma",
                vmin=0.0, vmax=error_max)
            if column == 0:
                axes[row, column].set_ylabel("Frame %04d" % int(frame_id))
    fig.suptitle("Four-model absolute error comparison (common 99th-percentile scale)")
    fig.subplots_adjust(top=0.94, right=0.88, wspace=0.04, hspace=0.12)
    _add_right_colorbar(fig, last_image, "Absolute error (m)")
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _render_temporal_comparison(path, temporal_maps):
    valid_values = []
    for model in MODEL_ORDER:
        for item in temporal_maps[model]:
            valid_values.append(np.abs(item["residual"][item["valid"]]))
    all_values = np.concatenate(valid_values).astype(np.float64)
    residual_max = max(float(np.percentile(all_values, 99.0)), 1e-3)
    pair_count = len(temporal_maps[MODEL_ORDER[0]])
    fig, axes = plt.subplots(
        pair_count, len(MODEL_ORDER),
        figsize=(3.1 * len(MODEL_ORDER), 2.45 * pair_count),
        squeeze=False,
    )
    last_image = None
    for row in range(pair_count):
        for column, model in enumerate(MODEL_ORDER):
            item = temporal_maps[model][row]
            last_image = _draw_image(
                axes[row, column],
                _masked(item["residual"], item["valid"]),
                MODEL_NAMES[model] if row == 0 else "", cmap="coolwarm",
                vmin=-residual_max, vmax=residual_max)
            if column == 0:
                axes[row, column].set_ylabel(item["pair"])
    fig.suptitle("UNREGISTERED IMAGE-SPACE TEMPORAL RESIDUAL")
    fig.text(
        0.5, 0.012,
        "No camera-pose or optical-flow compensation; models infer each frame independently.",
        ha="center", fontsize=9)
    fig.subplots_adjust(top=0.92, bottom=0.06, right=0.88, wspace=0.04, hspace=0.15)
    _add_right_colorbar(
        fig, last_image, "Temporal residual (m)", bottom=0.18, top=0.78)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def write_all_artifacts(output_dir, frame_ids, rgb, sparse, gt, valid,
                        predictions, metadata, raw_predictions=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = tuple(int(frame_id) for frame_id in frame_ids)
    raw_predictions = predictions if raw_predictions is None else raw_predictions
    frame_rows, temporal_rows, temporal_maps = collect_metrics(
        frame_ids, gt, valid, predictions)
    rows_by_key = {
        (row["model"], row["frame_id"]): row for row in frame_rows
    }

    for model in MODEL_ORDER:
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for index, frame_id in enumerate(frame_ids):
            prediction = np.asarray(predictions[model][index], dtype=np.float32)
            raw_prediction = np.asarray(
                raw_predictions[model][index], dtype=np.float32)
            absolute_error = np.where(
                valid[index], np.abs(prediction - gt[index]), 0.0).astype(
                    np.float32)
            _save_npz_atomic(
                model_dir / ("frame_%04d.npz" % frame_id),
                frame_id=np.asarray(frame_id, dtype=np.int64),
                rgb=np.asarray(rgb[index], dtype=np.float32),
                sparse=np.asarray(sparse[index], dtype=np.float32),
                gt=np.asarray(gt[index], dtype=np.float32),
                valid=np.asarray(valid[index], dtype=bool),
                pred_raw=raw_prediction,
                pred_clamped=prediction,
                abs_err=absolute_error,
            )
            _render_frame_panel(
                model_dir / ("frame_%04d.png" % frame_id),
                frame_id,
                rgb[index],
                sparse[index],
                gt[index],
                valid[index],
                prediction,
                rows_by_key[(model, frame_id)],
            )

    frame_fields = (
        "model", "frame_id", "rmse", "mae", "abs_rel", "valid_pixels",
        "valid_coverage", "sparse_count")
    temporal_fields = (
        "model", "pair", "from_frame", "to_frame", "rmse", "mae",
        "valid_pixels", "valid_coverage", "alignment")
    _write_csv_atomic(
        output_dir / "four_model_frame_metrics.csv", frame_fields, frame_rows)
    _write_csv_atomic(
        output_dir / "four_model_temporal_metrics.csv",
        temporal_fields,
        temporal_rows,
    )
    _render_depth_comparison(
        output_dir / "four_model_depth_comparison.png",
        frame_ids, gt, valid, predictions)
    _render_error_comparison(
        output_dir / "four_model_error_comparison.png",
        frame_ids, gt, valid, predictions)
    _render_temporal_comparison(
        output_dir / "four_model_temporal_comparison_unregistered.png",
        temporal_maps)

    metadata_path = output_dir / "run_metadata.json"
    required_without_metadata = [
        path for path in expected_artifacts(output_dir, frame_ids)
        if path != metadata_path]
    missing = [
        str(path) for path in required_without_metadata
        if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("comparison artifacts are incomplete: %s" % missing)
    metadata["complete"] = True
    metadata["frame_metric_rows"] = len(frame_rows)
    metadata["temporal_metric_rows"] = len(temporal_rows)
    metadata["artifact_count"] = len(expected_artifacts(output_dir, frame_ids))
    _write_json_atomic(metadata_path, metadata)
    return frame_rows, temporal_rows


def build_run_metadata(canonical_dir, output_dir, device, canonical, specs,
                       worker_info, cspn_checkpoint=DEFAULT_CSPN_CHECKPOINT):
    cspn_checkpoint = Path(cspn_checkpoint)
    models = {
        "cspn": {
            "architecture": "CSPN ResNet-50",
            "iteration": 24,
            "norm_type": "8sum_abs",
            "checkpoint": str(cspn_checkpoint),
            "checkpoint_digest": sequence_io.file_sha256(cspn_checkpoint),
            "reused_existing_prediction": True,
            "input_semantics": "concatenated RGB+sparse depth",
        }
    }
    for model in EXTERNAL_MODELS:
        info = worker_info[model]
        models[model] = {
            "environment": specs[model]["environment"],
            "checkpoint": specs[model]["checkpoint"],
            "args_json": specs[model]["args_json"],
            "checkpoint_digest": info["checkpoint_digest"],
            "reused_worker_result": bool(info["reused"]),
            "worker_log": str(Path(output_dir) / model / "worker.log"),
            "input_semantics": "separate RGB and sparse-depth tensors",
            "configuration": dict(info["metadata"]),
        }
    return {
        "complete": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dir": str(Path(canonical_dir)),
        "output_dir": str(Path(output_dir)),
        "device": str(device),
        "input_digest": canonical["input_digest"],
        "source_geometry": [480, 640],
        "network_geometry": [sequence_io.HEIGHT, sequence_io.WIDTH],
        "input_geometry": [sequence_io.HEIGHT, sequence_io.WIDTH],
        "rgb_tensor_shape": list(canonical["rgb"].shape),
        "sparse_tensor_shape": list(canonical["sparse"].shape),
        "sparse_count": sequence_io.SPARSE_COUNT,
        "temporal_alignment": "unregistered",
        "temporal_warning": (
            "Image-space adjacent-frame diagnostics without camera-pose or "
            "optical-flow compensation; all models infer frames independently."),
        "models": models,
    }


def run_comparison(cli, canonical_loader=sequence_io.load_canonical_frames,
                   worker_specs_builder=default_worker_specs,
                   prediction_loader=load_model_predictions,
                   metadata_builder=build_run_metadata,
                   artifact_writer=write_all_artifacts):
    canonical_dir = Path(cli.canonical_dir)
    output_dir = Path(cli.output_dir)
    worker_path = Path(__file__).with_name("run_spn_sequence_worker.py")
    canonical = canonical_loader(canonical_dir)
    specs = worker_specs_builder(worker_path)
    predictions, worker_info = prediction_loader(
        canonical,
        canonical_dir,
        output_dir,
        cli.device,
        specs,
        force=cli.force,
    )
    raw_predictions = {"cspn": canonical["cspn_pred_raw"]}
    for model in EXTERNAL_MODELS:
        raw_predictions[model] = worker_info[model]["pred_raw"]
    metadata = metadata_builder(
        canonical_dir,
        output_dir,
        cli.device,
        canonical,
        specs,
        worker_info,
    )
    artifact_writer(
        output_dir,
        canonical["frame_ids"],
        canonical["rgb"],
        canonical["sparse"],
        canonical["gt"],
        canonical["valid"],
        predictions,
        metadata,
        raw_predictions=raw_predictions,
    )
    return {
        "complete": bool(metadata["complete"]),
        "output_dir": str(output_dir.resolve()),
        "input_digest": canonical["input_digest"],
        "models": list(MODEL_ORDER),
    }


def main(argv=None):
    cli = make_parser().parse_args(argv)
    summary = run_comparison(cli)
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
