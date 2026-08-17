#!/usr/bin/env python3
"""Orchestrate the frozen-NLSPN temporal residual pilot."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlparse

import numpy as np

from scripts import export_cspn_sequence_predictions as sequence
from scripts import nlspn_temporal_residual as residual


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = Path(__file__).with_name(
    "run_nlspn_temporal_residual_worker.py")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
DEFAULT_CHECKPOINT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS_JSON = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/args.json")
DEFAULT_OUTPUT = Path(
    "/workspace/VoxelNet/nlspn_temporal_residual_validation/"
    "BeachApartmentInterior_My_ir/pilot_256")


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


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finalize_metadata(path, metadata, required_artifacts):
    missing = [
        str(artifact) for artifact in required_artifacts
        if not Path(artifact).is_file() or Path(artifact).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError("pilot artifacts are incomplete: %s" % missing)
    completed = dict(metadata)
    completed["complete"] = True
    completed["artifact_count"] = len(tuple(required_artifacts))
    write_json_atomic(path, completed)
    metadata.clear()
    metadata.update(completed)
    return completed


def write_csv_atomic(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".csv", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_npz(path):
    with np.load(str(path), allow_pickle=False) as data:
        return dict((key, data[key]) for key in data.files)


def _scalar(payload, key):
    if key not in payload:
        raise KeyError(key)
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError("%s must be scalar" % key)
    return value.item()


def cache_matches(path, expected_scalars, frame_ids, expected_shapes):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = load_npz(path)
        if not np.array_equal(
                payload["frame_ids"], np.asarray(frame_ids, dtype=np.int32)):
            return False
        for key, value in expected_scalars.items():
            if str(_scalar(payload, key)) != str(value):
                return False
        for key, shape in expected_shapes.items():
            array = np.asarray(payload[key])
            if array.shape != tuple(shape):
                return False
            if array.dtype != bool and not np.isfinite(array).all():
                return False
        return True
    except (KeyError, OSError, ValueError, TypeError):
        return False


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


def resolve_raft_weight_path(weights):
    import torch

    filename = Path(urlparse(weights.url).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("official RAFT weight was not cached: %s" % path)
    return path


def run_worker(command, env, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command, env=env, cwd=str(REPO_ROOT), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "NLSPN worker exited with code %d; see %s" %
            (completed.returncode, log_path))
    return completed.stdout


def _depth_metrics(gt, prediction, valid):
    gt = np.asarray(gt, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (gt.shape == prediction.shape == valid.shape) or not np.any(valid):
        raise ValueError("invalid depth metric arrays")
    error = prediction[valid] - gt[valid]
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "abs_rel": float(np.mean(np.abs(error) / gt[valid])),
        "valid_pixels": int(valid.sum()),
    }


def _correlations(left, right):
    from scipy.stats import spearmanr

    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size == 0:
        raise ValueError("correlation arrays must be non-empty and aligned")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("correlation arrays contain non-finite values")
    constant = bool(np.std(left) == 0.0 or np.std(right) == 0.0)
    if constant:
        return {"pearson": 0.0, "spearman": 0.0,
                "constant_input": True}
    pearson = float(np.corrcoef(left, right)[0, 1])
    spearman = float(spearmanr(left, right).statistic)
    if not np.isfinite(pearson) or not np.isfinite(spearman):
        raise ValueError("correlation result is non-finite")
    return {"pearson": pearson, "spearman": spearman,
            "constant_input": False}


def _warp_rgb(rgb, flow):
    import torch

    source = torch.from_numpy(
        np.asarray(rgb[:-1], dtype=np.float32))
    flow_tensor = torch.from_numpy(np.asarray(flow, dtype=np.float32))
    return residual.backward_warp(source, flow_tensor)[0].numpy()


def analyze_clip(record):
    payload = record["input"]
    baseline = record["baseline"]
    flow = record["flow"]
    propagation = record["propagation"]
    frame_ids = np.asarray(payload["frame_ids"], dtype=np.int32)
    warped_rgb = _warp_rgb(payload["rgb"], flow["backward_flow"])
    frame_rows = []
    pair_rows = []
    methods = {
        "full": baseline["pred"][1:],
        "oracle": propagation["oracle_prediction"],
        "causal": propagation["causal_prediction"],
    }
    for local_index, frame_id in enumerate(frame_ids[1:]):
        for method, prediction in methods.items():
            row = {
                "clip": record["clip"],
                "frame_id": int(frame_id),
                "method": method,
            }
            row.update(_depth_metrics(
                payload["gt"][local_index + 1],
                prediction[local_index],
                payload["valid"][local_index + 1]))
            frame_rows.append(row)

    for index, (from_frame, to_frame) in enumerate(
            zip(frame_ids[:-1], frame_ids[1:])):
        raw = baseline["pred"][index + 1] - baseline["pred"][index]
        aligned = baseline["pred"][index + 1] - propagation["base"][index]
        relationship_valid = (
            payload["valid"][index] & payload["valid"][index + 1] &
            propagation["in_bounds"][index])
        photometric = np.mean(np.abs(
            payload["rgb"][index + 1] - warped_rgb[index]), axis=0)
        flow_magnitude = np.linalg.norm(
            flow["backward_flow"][index], axis=0)
        sparse_mask = payload["sparse"][index + 1] > 0.0
        sparse_values = (
            payload["sparse"][index + 1] - propagation["base"][index])
        row = {
            "clip": record["clip"],
            "pair": "%04d->%04d" % (from_frame, to_frame),
            "from_frame": int(from_frame),
            "to_frame": int(to_frame),
            "in_bounds_coverage": float(
                propagation["in_bounds"][index].mean()),
        }
        for prefix, values in (("raw", raw), ("aligned", aligned)):
            stats = residual.residual_statistics(values[relationship_valid])
            row.update(dict((prefix + "_" + key, value)
                            for key, value in stats.items()))
        for name, diagnostic in (
                ("photometric", photometric),
                ("flow_magnitude", flow_magnitude)):
            correlation = _correlations(
                np.abs(aligned[relationship_valid]),
                diagnostic[relationship_valid])
            row.update(dict((name + "_" + key, value)
                            for key, value in correlation.items()))
        sparse_correlation = _correlations(
            aligned[sparse_mask], sparse_values[sparse_mask])
        row.update(dict(("sparse_" + key, value)
                        for key, value in sparse_correlation.items()))
        pair_rows.append(row)

    valid = payload["valid"][1:]
    gt = payload["gt"][1:]
    oracle_quality = residual.pooled_quality(
        methods["full"], methods["oracle"], gt, valid)
    causal_quality = residual.pooled_quality(
        methods["full"], methods["causal"], gt, valid)
    clip_summary = {
        "clip": record["clip"],
        "start_frame": int(frame_ids[0]),
        "end_frame": int(frame_ids[-1]),
        "frame_count": int(frame_ids.size),
        "pair_count": int(frame_ids.size - 1),
        "rmse_full": causal_quality["rmse_full"],
        "rmse_oracle": oracle_quality["rmse_reconstructed"],
        "oracle_quality_ratio": oracle_quality["quality_ratio"],
        "oracle_passes": oracle_quality["passes"],
        "rmse_causal": causal_quality["rmse_reconstructed"],
        "causal_quality_ratio": causal_quality["quality_ratio"],
        "causal_passes": causal_quality["passes"],
    }
    return frame_rows, pair_rows, clip_summary


def _render_quality_ratios(path, frame_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = dict(
        ((row["clip"], row["frame_id"]), row["rmse"])
        for row in frame_rows if row["method"] == "full")
    causal = [row for row in frame_rows if row["method"] == "causal"]
    ratios = [
        row["rmse"] / full[(row["clip"], row["frame_id"])]
        for row in causal
    ]
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(np.arange(len(ratios)), ratios, linewidth=1.2,
              color="#2f4b7c", label="Causal / full frame RMSE")
    axis.axhline(1.01, color="#d62728", linestyle="--", linewidth=1.2,
                 label="1% gate")
    cursor = 0
    for clip in dict.fromkeys(row["clip"] for row in causal):
        count = sum(row["clip"] == clip for row in causal)
        if cursor:
            axis.axvline(cursor - 0.5, color="#999999", linewidth=0.6)
        cursor += count
    axis.set_xlabel("Causal frame index across pilot clips")
    axis.set_ylabel("Per-frame RMSE ratio")
    axis.set_title("NLSPN warped-history reconstruction quality")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_residual_overview(path, records, frame_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    causal_rmse = dict(
        ((row["clip"], row["frame_id"]), row["rmse"])
        for row in frame_rows if row["method"] == "causal")
    selections = []
    for record in records:
        frame_ids = record["input"]["frame_ids"]
        candidates = list(range(frame_ids.size - 1))
        largest = max(
            candidates,
            key=lambda index: causal_rmse[
                (record["clip"], int(frame_ids[index + 1]))])
        for index in dict.fromkeys((0, largest)):
            selections.append((record, index))

    aligned_values = []
    causal_errors = []
    for record, index in selections:
        baseline = record["baseline"]
        propagation = record["propagation"]
        aligned_values.append(
            baseline["pred"][index + 1] - propagation["base"][index])
        causal_errors.append(
            propagation["causal_prediction"][index] -
            record["input"]["gt"][index + 1])
    residual_limit = max(0.05, float(np.percentile(
        np.abs(np.concatenate([value.ravel()
                               for value in aligned_values])), 99)))
    error_limit = max(0.05, float(np.percentile(
        np.abs(np.concatenate([value.ravel()
                               for value in causal_errors])), 99)))

    figure, axes = plt.subplots(
        len(selections), 5, squeeze=False,
        figsize=(15, max(2.4 * len(selections), 3.0)))
    for row, (record, index) in enumerate(selections):
        payload = record["input"]
        baseline = record["baseline"]
        propagation = record["propagation"]
        frame_id = int(payload["frame_ids"][index + 1])
        raw = baseline["pred"][index + 1] - baseline["pred"][index]
        aligned = baseline["pred"][index + 1] - propagation["base"][index]
        sparse_mask = payload["sparse"][index + 1] > 0
        sparse_residual = np.full_like(aligned, np.nan)
        sparse_residual[sparse_mask] = (
            payload["sparse"][index + 1][sparse_mask] -
            propagation["base"][index][sparse_mask])
        causal_error = (
            propagation["causal_prediction"][index] -
            payload["gt"][index + 1])
        axes[row, 0].imshow(np.moveaxis(payload["rgb"][index + 1], 0, -1))
        axes[row, 1].imshow(
            raw, cmap="RdBu_r", vmin=-residual_limit,
            vmax=residual_limit)
        axes[row, 2].imshow(
            aligned, cmap="RdBu_r", vmin=-residual_limit,
            vmax=residual_limit)
        axes[row, 3].imshow(
            sparse_residual, cmap="RdBu_r", vmin=-residual_limit,
            vmax=residual_limit)
        axes[row, 4].imshow(
            causal_error, cmap="RdBu_r", vmin=-error_limit,
            vmax=error_limit)
        axes[row, 0].set_ylabel("%s\n%04d" % (record["clip"], frame_id))
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(
            axes[0], ("Current RGB", "Raw output residual",
                      "Flow-aligned residual", "Sparse residual seeds",
                      "Causal GT error")):
        axis.set_title(title)
    figure.suptitle(
        "Frozen NLSPN adjacent-frame residual mapping", y=0.998)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_analysis_artifacts(output_dir, records):
    output_dir = Path(output_dir)
    frame_rows = []
    pair_rows = []
    clip_rows = []
    for record in records:
        clip_frame, clip_pair, clip_summary = analyze_clip(record)
        frame_rows.extend(clip_frame)
        pair_rows.extend(clip_pair)
        clip_rows.append(clip_summary)

    full = np.concatenate([
        record["baseline"]["pred"][1:] for record in records])
    oracle = np.concatenate([
        record["propagation"]["oracle_prediction"] for record in records])
    causal = np.concatenate([
        record["propagation"]["causal_prediction"] for record in records])
    gt = np.concatenate([record["input"]["gt"][1:] for record in records])
    valid = np.concatenate([
        record["input"]["valid"][1:] for record in records])
    oracle_quality = residual.pooled_quality(full, oracle, gt, valid)
    causal_quality = residual.pooled_quality(full, causal, gt, valid)

    raw_values = []
    aligned_values = []
    for record in records:
        baseline = record["baseline"]["pred"]
        propagation = record["propagation"]
        payload = record["input"]
        for index in range(baseline.shape[0] - 1):
            mask = (payload["valid"][index] &
                    payload["valid"][index + 1] &
                    propagation["in_bounds"][index])
            raw_values.append((baseline[index + 1] - baseline[index])[mask])
            aligned_values.append(
                (baseline[index + 1] - propagation["base"][index])[mask])
    raw_stats = residual.residual_statistics(np.concatenate(raw_values))
    aligned_stats = residual.residual_statistics(
        np.concatenate(aligned_values))

    if causal_quality["passes"]:
        interpretation = "causal_pass"
    elif oracle_quality["passes"]:
        interpretation = "oracle_only"
    else:
        interpretation = "oracle_fail"
    summary = {
        "frame_count": int(sum(record["input"]["frame_ids"].size
                               for record in records)),
        "pair_count": int(sum(record["input"]["frame_ids"].size - 1
                              for record in records)),
        "oracle": oracle_quality,
        "causal": causal_quality,
        "raw_residual": raw_stats,
        "aligned_residual": aligned_stats,
        "alignment_rmse_ratio": aligned_stats["rmse"] / raw_stats["rmse"],
        "interpretation": interpretation,
    }
    paths = {
        "frame_metrics": output_dir / "frame_metrics.csv",
        "pair_metrics": output_dir / "pair_metrics.csv",
        "clip_summary": output_dir / "clip_summary.csv",
        "summary": output_dir / "summary.json",
        "residual_overview": output_dir / "residual_mapping_overview.png",
        "quality_plot": output_dir / "quality_ratio_by_frame.png",
    }
    write_csv_atomic(paths["frame_metrics"], frame_rows)
    write_csv_atomic(paths["pair_metrics"], pair_rows)
    write_csv_atomic(paths["clip_summary"], clip_rows)
    write_json_atomic(paths["summary"], summary)
    _render_residual_overview(paths["residual_overview"], records, frame_rows)
    _render_quality_ratios(paths["quality_plot"], frame_rows)
    return summary, tuple(paths.values())


def build_worker_command(stage, input_path, output_path, checkpoint,
                         args_json, device, baseline=None, flow=None):
    command = [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH),
        "--stage", str(stage),
        "--input", str(input_path),
        "--checkpoint", str(checkpoint),
        "--args-json", str(args_json),
        "--output", str(output_path),
        "--device", str(device),
    ]
    if baseline is not None:
        command.extend(("--baseline", str(baseline)))
    if flow is not None:
        command.extend(("--flow", str(flow)))
    source = NLSPN_ROOT / "src"
    paths = (REPO_ROOT, source, source / "model" / "deformconv")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in paths] + ([existing] if existing else []))
    return command, env


def make_parser():
    parser = argparse.ArgumentParser(
        description="Validate causal NLSPN temporal residual mapping")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--args-json", default=str(DEFAULT_ARGS_JSON))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raft-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke-frames", type=int, nargs="+", default=None,
        help=argparse.SUPPRESS)
    return parser


def resolve_clips(args):
    if args.smoke_frames is None:
        return residual.PILOT_CLIPS
    frames = tuple(int(value) for value in args.smoke_frames)
    if (len(frames) != 2 or frames[0] >= frames[1] or
            frames[0] < 1 or frames[1] > 2000):
        raise ValueError("smoke mode requires exactly two increasing frame IDs")
    return ((frames[0], frames[1]),)


def run_pilot(args, metadata):
    import torch
    import torchvision

    checkpoint = Path(args.checkpoint)
    args_json = Path(args.args_json)
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    if not args_json.is_file():
        raise FileNotFoundError(str(args_json))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for %s" % device)
    if args.raft_batch_size <= 0:
        raise ValueError("RAFT batch size must be positive")

    output_dir = Path(args.output_dir)
    clips = resolve_clips(args)
    checkpoint_digest = residual.file_sha256(checkpoint)
    args_digest = residual.file_sha256(args_json)
    raft_model, raft_transform, raft_weights = build_raft(device)
    raft_weight_path = resolve_raft_weight_path(raft_weights)
    raft_weight_digest = residual.file_sha256(raft_weight_path)
    metadata.update({
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_digest": checkpoint_digest,
        "args_json": str(args_json.resolve()),
        "args_digest": args_digest,
        "raft_weight": str(raft_weight_path.resolve()),
        "raft_weight_digest": raft_weight_digest,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "network_geometry": [228, 304],
        "source_geometry": [480, 640],
        "sparse_count": residual.SPARSE_COUNT,
        "seed": int(args.seed),
        "clip_runs": [],
    })
    write_json_atomic(output_dir / "run_metadata.json", metadata)

    records = []
    required = []
    for start, end in clips:
        frame_ids = tuple(range(int(start), int(end) + 1))
        clip_name = "%04d_%04d" % (start, end)
        clip_dir = output_dir / ("clip_" + clip_name)
        clip_dir.mkdir(parents=True, exist_ok=True)
        input_path = clip_dir / "input.npz"
        baseline_path = clip_dir / "baseline.npz"
        flow_path = clip_dir / "flow.npz"
        propagation_path = clip_dir / "propagation.npz"
        baseline_log = clip_dir / "baseline_worker.log"
        propagation_log = clip_dir / "propagation_worker.log"

        if args.force or not input_path.is_file():
            payload = prepare_clip(
                Path(args.data_root), args.scene, frame_ids,
                input_path, seed=args.seed)
        else:
            payload = load_npz(input_path)
            residual.validate_clip_payload(payload)
            if not np.array_equal(
                    payload["frame_ids"], np.asarray(frame_ids, np.int32)):
                payload = prepare_clip(
                    Path(args.data_root), args.scene, frame_ids,
                    input_path, seed=args.seed)
        input_digest = residual.file_sha256(input_path)
        frame_count = len(frame_ids)

        baseline_shapes = {
            "pred": (frame_count, 228, 304),
            "pred_init": (frame_count, 228, 304),
            "guidance": (frame_count, 8, 228, 304),
            "confidence": (frame_count, 1, 228, 304),
            "offset": (frame_count, 16, 228, 304),
            "aff": (frame_count, 9, 228, 304),
        }
        baseline_valid = cache_matches(
            baseline_path,
            {"stage": "baseline", "input_digest": input_digest,
             "checkpoint_digest": checkpoint_digest},
            frame_ids, baseline_shapes)
        baseline_valid = (
            baseline_valid and baseline_log.is_file() and
            baseline_log.stat().st_size > 0)
        if args.force or not baseline_valid:
            command, env = build_worker_command(
                "baseline", input_path, baseline_path,
                checkpoint, args_json, args.device)
            run_worker(command, env, baseline_log)
            if not cache_matches(
                    baseline_path,
                    {"stage": "baseline", "input_digest": input_digest,
                     "checkpoint_digest": checkpoint_digest},
                    frame_ids, baseline_shapes):
                raise RuntimeError("baseline worker result failed validation")

        flow_shapes = {
            "backward_flow": (frame_count - 1, 2, 228, 304),
        }
        flow_valid = cache_matches(
            flow_path,
            {"stage": "flow", "input_digest": input_digest,
             "weight_digest": raft_weight_digest},
            frame_ids, flow_shapes)
        if args.force or not flow_valid:
            backward_flow, flow_seconds = predict_backward_flow(
                raft_model, raft_transform, payload["rgb"], device,
                batch_size=args.raft_batch_size)
            write_npz_atomic(
                flow_path,
                frame_ids=np.asarray(frame_ids, dtype=np.int32),
                stage=np.asarray("flow"),
                input_digest=np.asarray(input_digest),
                weight_digest=np.asarray(raft_weight_digest),
                weight_path=np.asarray(str(raft_weight_path.resolve())),
                runtime_seconds=np.asarray(flow_seconds, dtype=np.float64),
                backward_flow=backward_flow)
            if not cache_matches(
                    flow_path,
                    {"stage": "flow", "input_digest": input_digest,
                     "weight_digest": raft_weight_digest},
                    frame_ids, flow_shapes):
                raise RuntimeError("RAFT result failed validation")

        propagation_shapes = {
            key: (frame_count - 1, 228, 304)
            for key in ("base", "in_bounds", "oracle_residual",
                        "oracle_prediction", "causal_residual",
                        "causal_prediction")
        }
        propagation_scalars = {
            "stage": "propagate",
            "input_digest": input_digest,
            "checkpoint_digest": checkpoint_digest,
            "baseline_digest": residual.file_sha256(baseline_path),
            "flow_digest": residual.file_sha256(flow_path),
        }
        propagation_valid = cache_matches(
            propagation_path, propagation_scalars,
            frame_ids, propagation_shapes)
        propagation_valid = (
            propagation_valid and propagation_log.is_file() and
            propagation_log.stat().st_size > 0)
        if args.force or not propagation_valid:
            command, env = build_worker_command(
                "propagate", input_path, propagation_path,
                checkpoint, args_json, args.device,
                baseline=baseline_path, flow=flow_path)
            run_worker(command, env, propagation_log)
            if not cache_matches(
                    propagation_path, propagation_scalars,
                    frame_ids, propagation_shapes):
                raise RuntimeError(
                    "propagation worker result failed validation")

        baseline = load_npz(baseline_path)
        flow = load_npz(flow_path)
        propagation = load_npz(propagation_path)
        record = {
            "clip": clip_name,
            "input": payload,
            "baseline": baseline,
            "flow": flow,
            "propagation": propagation,
        }
        records.append(record)
        metadata["clip_runs"].append({
            "clip": clip_name,
            "input_digest": input_digest,
            "baseline_runtime_seconds": float(
                _scalar(baseline, "runtime_seconds")),
            "flow_runtime_seconds": float(
                _scalar(flow, "runtime_seconds")),
            "oracle_runtime_seconds": float(
                _scalar(propagation, "oracle_runtime_seconds")),
            "causal_runtime_seconds": float(
                _scalar(propagation, "causal_runtime_seconds")),
        })
        required.extend((
            input_path, baseline_path, flow_path, propagation_path,
            baseline_log, propagation_log))
        write_json_atomic(output_dir / "run_metadata.json", metadata)
        print("completed clip %s" % clip_name, flush=True)

    del raft_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary, analysis_artifacts = write_analysis_artifacts(
        output_dir, records)
    required.extend(analysis_artifacts)
    metadata["summary"] = summary
    finalize_metadata(
        output_dir / "run_metadata.json", metadata, tuple(required))
    completion = {
        "complete": True,
        "output_dir": str(output_dir.resolve()),
        "frame_count": metadata["frame_count"],
        "pair_count": metadata["pair_count"],
        "interpretation": summary["interpretation"],
        "causal_quality_ratio": summary["causal"]["quality_ratio"],
    }
    print(json.dumps(completion, sort_keys=True), flush=True)
    return completion


def main(argv=None):
    args = make_parser().parse_args(argv)
    clips = resolve_clips(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        "complete": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "clips": [[int(start), int(end)] for start, end in clips],
        "frame_count": len(residual.clip_frame_ids(clips)),
        "pair_count": len(residual.clip_pairs(clips)),
        "device": args.device,
    }
    write_json_atomic(metadata_path, metadata)
    try:
        result = run_pilot(args, metadata)
    except Exception as error:
        metadata["error"] = str(error)
        write_json_atomic(metadata_path, metadata)
        raise
    return result


if __name__ == "__main__":
    main()
