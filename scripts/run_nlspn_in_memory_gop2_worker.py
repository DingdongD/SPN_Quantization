#!/usr/bin/env python3
"""Run the pure-memory NLSPN GOP2 pipeline in its legacy environment."""

from __future__ import print_function

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import export_cspn_sequence_predictions as sequence
from scripts import nlspn_temporal_residual as residual
from scripts import nlspn_in_memory_gop2 as online
from scripts import raft_small_compat
from scripts import run_spn_sequence_worker as spn_worker


FINAL_ARTIFACTS = (
    "run_metadata.json",
    "summary.json",
    "frame_metrics.csv",
    "clip_summary.csv",
    "report.md",
)


def parse_clips(value):
    if value is None or str(value).strip() == "":
        return residual.PILOT_CLIPS
    clips = []
    for item in str(value).split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError("clips must use start:end syntax")
        start, end = (int(field) for field in fields)
        if start <= 0 or end < start:
            raise ValueError("clip bounds are invalid")
        clips.append((start, end))
    if not clips:
        raise ValueError("at least one clip is required")
    return tuple(clips)


def load_preprocessed_frame(data_root, scene, frame_id):
    root = Path(data_root) / scene
    rgb = sequence.load_rgb(root / "rgb" / ("%04d.jpg" % int(frame_id)))
    depth = sequence.read_exr_depth(
        root / "depth" / ("Image%04d.exr" % int(frame_id)))
    return sequence.preprocess_pair(rgb, depth)


def load_in_memory_clips(data_root, scene, clips, seed=2026,
                         frame_loader=load_preprocessed_frame):
    payloads = []
    for start, end in clips:
        frame_ids = np.arange(int(start), int(end) + 1, dtype=np.int32)
        if frame_ids.size < 2:
            raise ValueError("every temporal clip requires at least two frames")
        frames = [
            frame_loader(data_root, scene, int(frame_id))
            for frame_id in frame_ids
        ]
        rgb, gt, valid = (
            np.stack(values) for values in zip(*frames))
        sparse, sparse_mask = sequence.build_shared_sparse_depths(
            gt, valid, count=residual.SPARSE_COUNT, seed=int(seed))
        payload = {
            "frame_ids": frame_ids,
            "rgb": rgb.astype(np.float32),
            "sparse": sparse.astype(np.float32),
            "gt": gt.astype(np.float32),
            "valid": valid.astype(bool),
            "sparse_mask": sparse_mask.astype(bool),
        }
        residual.validate_clip_payload(payload)
        payloads.append(payload)
    return tuple(payloads)


def _clip_name(frame_ids):
    return "%04d-%04d" % (int(frame_ids[0]), int(frame_ids[-1]))


def run_warmup(engine, payloads, path, repeats=1):
    if path not in ("full", "gop2"):
        raise ValueError("unknown benchmark path")
    if int(repeats) < 0:
        raise ValueError("warm-up repeats cannot be negative")
    started = time.perf_counter()
    for _ in range(int(repeats)):
        for payload in payloads:
            engine.reset()
            for local_index in range(payload["frame_ids"].size):
                rgb = payload["rgb"][local_index]
                sparse = payload["sparse"][local_index]
                if path == "full":
                    engine.infer_full(rgb, sparse)
                elif online.frame_kind(local_index) == "I":
                    engine.infer_i(rgb, sparse, local_index)
                else:
                    engine.infer_p(rgb, sparse, local_index)
    return float(time.perf_counter() - started)


def run_timed_path(engine, payloads, path, repeats):
    if path not in ("full", "gop2"):
        raise ValueError("unknown benchmark path")
    if int(repeats) <= 0:
        raise ValueError("timed repeats must be positive")
    records = []
    first_predictions = []
    for repeat in range(int(repeats)):
        for payload in payloads:
            engine.reset()
            clip = _clip_name(payload["frame_ids"])
            for local_index, frame_id in enumerate(payload["frame_ids"]):
                rgb = payload["rgb"][local_index]
                sparse = payload["sparse"][local_index]
                if path == "full":
                    result = engine.infer_full(rgb, sparse)
                elif online.frame_kind(local_index) == "I":
                    result = engine.infer_i(rgb, sparse, local_index)
                else:
                    result = engine.infer_p(rgb, sparse, local_index)
                records.append({
                    "path": path,
                    "repeat": repeat,
                    "clip": clip,
                    "frame_id": int(frame_id),
                    "local_index": int(local_index),
                    "kind": result.kind,
                    "latency_ms": float(result.latency_ms),
                })
                if repeat == 0:
                    prediction = result.prediction
                    if isinstance(prediction, torch.Tensor):
                        prediction = prediction.detach().cpu().numpy()
                    prediction = np.asarray(prediction, dtype=np.float32)
                    if prediction.shape != (residual.HEIGHT, residual.WIDTH):
                        raise ValueError("prediction geometry is invalid")
                    if not np.isfinite(prediction).all():
                        raise ValueError("prediction contains non-finite values")
                    first_predictions.append(prediction.copy())
    return records, first_predictions


def _reset_peak_memory(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device):
    device = torch.device(device)
    if device.type != "cuda":
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def execute_benchmark(engine, payloads, warmup_repeats=1,
                      timed_repeats=5):
    payloads = tuple(payloads)
    if not payloads:
        raise ValueError("benchmark requires at least one clip")
    warmup = {
        "full_seconds": run_warmup(
            engine, payloads, "full", warmup_repeats),
        "gop2_seconds": run_warmup(
            engine, payloads, "gop2", warmup_repeats),
    }

    device = getattr(engine, "device", torch.device("cpu"))
    _reset_peak_memory(device)
    full_records, full_predictions = run_timed_path(
        engine, payloads, "full", timed_repeats)
    full_memory = _peak_memory(device)

    _reset_peak_memory(device)
    gop2_records, gop2_predictions = run_timed_path(
        engine, payloads, "gop2", timed_repeats)
    gop2_memory = _peak_memory(device)

    frame_ids = np.concatenate([
        payload["frame_ids"] for payload in payloads])
    clip_names = np.concatenate([
        np.repeat(_clip_name(payload["frame_ids"]),
                  payload["frame_ids"].size)
        for payload in payloads])
    gt = np.concatenate([payload["gt"] for payload in payloads])
    valid = np.concatenate([payload["valid"] for payload in payloads])
    full_predictions = np.stack(full_predictions).astype(np.float32)
    gop2_predictions = np.stack(gop2_predictions).astype(np.float32)
    quality_rows = online.frame_quality_rows(
        frame_ids, clip_names, full_predictions, gop2_predictions,
        gt, valid)
    quality = online.pooled_quality_summary(
        full_predictions, gop2_predictions, gt, valid)
    clip_rows = online.clip_quality_rows(quality_rows)

    summary = online.benchmark_summary(
        [row["latency_ms"] for row in full_records],
        [row["latency_ms"] for row in gop2_records],
        [row["kind"] for row in gop2_records],
        quality,
    )
    summary["warmup"] = warmup
    summary["memory"] = {
        "full": full_memory,
        "gop2": gop2_memory,
    }
    summary["quality_frame_count"] = int(frame_ids.size)
    summary["timed_repeats"] = int(timed_repeats)

    quality_by_frame = dict(
        ((row["clip"], row["frame_id"]), row) for row in quality_rows)
    frame_rows = full_records + gop2_records
    for row in frame_rows:
        if row["repeat"] != 0:
            continue
        quality_row = quality_by_frame[(row["clip"], row["frame_id"])]
        for key in (
                "rmse_full", "rmse_gop2", "quality_ratio",
                "passes_1pct", "full_squared_error",
                "gop2_squared_error", "valid_pixels"):
            row[key] = quality_row[key]
    return {
        "summary": summary,
        "frame_rows": frame_rows,
        "clip_rows": clip_rows,
    }


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv_atomic(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty CSV")
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
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_text_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".md", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(str(value))
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _render_report(summary):
    quality = summary.get("quality", {})
    return (
        "# NLSPN Pure-In-Memory GOP2 Benchmark\n\n"
        "- Measured end-to-end speedup: %.6fx\n"
        "- GOP2 pooled quality gate: %s\n\n"
        "The earlier 1.57x value was a component projection, not this "
        "end-to-end measurement.\n" % (
            float(summary.get("speedup", float("nan"))),
            "pass" if quality.get("passes") else "fail",
        )
    )


def write_final_artifacts(output_dir, metadata, summary, frame_rows, clip_rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = [
        path for path in output_dir.iterdir()
        if path.is_file() and (
            path.suffix.lower() in (".npz", ".npy") or
            any(token in path.name.lower() for token in (
                "prediction", "flow", "guidance", "confidence", "cache")))
    ]
    if forbidden:
        raise RuntimeError("output directory contains tensor cache artifacts")
    incomplete = dict(metadata)
    incomplete["complete"] = False
    _write_json_atomic(output_dir / "run_metadata.json", incomplete)
    _write_json_atomic(output_dir / "summary.json", summary)
    _write_csv_atomic(output_dir / "frame_metrics.csv", frame_rows)
    _write_csv_atomic(output_dir / "clip_summary.csv", clip_rows)
    _write_text_atomic(output_dir / "report.md", _render_report(summary))
    missing = [
        name for name in FINAL_ARTIFACTS
        if not (output_dir / name).is_file() or
        (output_dir / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError("final benchmark artifacts are incomplete")
    completed = dict(incomplete)
    completed["complete"] = True
    completed["artifact_count"] = len(FINAL_ARTIFACTS)
    _write_json_atomic(output_dir / "run_metadata.json", completed)
    return completed


def _read_args(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        args = json.load(stream)
    expected = {
        "model": "nlspn",
        "iteration": 18,
        "nlspn_network": "resnet34",
    }
    if not isinstance(args, dict) or any(
            args.get(key) != value for key, value in expected.items()):
        raise ValueError("unexpected frozen NLSPN configuration")
    return args


def build_models(checkpoint, args_json, raft_weights, device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for requested device")
    args = _read_args(args_json)
    started = time.perf_counter()
    nlspn, model_metadata = spn_worker.build_model(
        "nlspn", args, device)
    spn_worker.load_checkpoint_strict(nlspn, Path(checkpoint))
    nlspn.eval()
    if bool(nlspn.args.preserve_input):
        raise ValueError("NLSPN preserve_input must remain false")
    if nlspn.args.affinity != "TGASS" or nlspn.prop_layer.prop_time != 18:
        raise ValueError("NLSPN propagation configuration changed")
    raft = raft_small_compat.raft_small()
    raft_small_compat.load_official_weights(raft, raft_weights)
    raft.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = float(time.perf_counter() - started)
    metadata = dict(model_metadata)
    metadata.update({
        "model_load_seconds": load_seconds,
        "torch_version": str(torch.__version__),
        "device": str(device),
    })
    return nlspn, raft, metadata


def run_raft_parity(cli):
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for requested device")
    previous = load_preprocessed_frame(
        cli.data_root, cli.scene, 1)[0]
    current = load_preprocessed_frame(
        cli.data_root, cli.scene, 2)[0]
    model = raft_small_compat.raft_small()
    raft_small_compat.load_official_weights(model, cli.raft_weights)
    model.to(device).eval()
    with torch.no_grad():
        current_tensor = torch.from_numpy(current[None]).to(device)
        previous_tensor = torch.from_numpy(previous[None]).to(device)
        flow = raft_small_compat.predict_backward_flow(
            model, current_tensor, previous_tensor).cpu().numpy().astype(
                np.float32)
    from scripts import run_nlspn_in_memory_gop2_benchmark as orchestrator
    payload = {
        "array": orchestrator.encode_array(flow),
        "shape": list(flow.shape),
        "dtype": str(flow.dtype),
        "weight_sha256": residual.file_sha256(cli.raft_weights),
        "torch_version": str(torch.__version__),
        "device": str(device),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def run_benchmark(cli):
    clips = parse_clips(cli.clips)
    input_started = time.perf_counter()
    payloads = load_in_memory_clips(
        cli.data_root, cli.scene, clips, seed=cli.seed)
    input_seconds = float(time.perf_counter() - input_started)
    nlspn, raft, model_metadata = build_models(
        cli.checkpoint, cli.args_json, cli.raft_weights, cli.device)
    engine = online.InMemoryGOP2Engine(nlspn, raft, cli.device)
    result = execute_benchmark(
        engine, payloads, warmup_repeats=cli.warmup_repeats,
        timed_repeats=cli.timed_repeats)
    summary = result["summary"]
    summary["startup"] = {
        "input_load_seconds": input_seconds,
        "model_load_seconds": model_metadata["model_load_seconds"],
    }
    frame_count = int(sum(
        payload["frame_ids"].size for payload in payloads))
    parity = json.loads(cli.parity_json) if cli.parity_json else None
    metadata = {
        "complete": False,
        "scene": str(cli.scene),
        "clips": [list(clip) for clip in clips],
        "frame_count": frame_count,
        "timed_repeats": int(cli.timed_repeats),
        "warmup_repeats": int(cli.warmup_repeats),
        "timed_frames_per_path": frame_count * int(cli.timed_repeats),
        "seed": int(cli.seed),
        "sparse_points": residual.SPARSE_COUNT,
        "height": residual.HEIGHT,
        "width": residual.WIDTH,
        "schedule": "I,P",
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_sha256": residual.file_sha256(cli.checkpoint),
        "raft_weights": str(Path(cli.raft_weights).resolve()),
        "raft_weight_sha256": residual.file_sha256(cli.raft_weights),
        "raft_parity": parity,
        "model": model_metadata,
        "numpy_version": str(np.__version__),
        "gpu": (
            torch.cuda.get_device_name(torch.device(cli.device))
            if torch.device(cli.device).type == "cuda" else "cpu"),
        "intermediate_tensor_cache": False,
    }
    completed = write_final_artifacts(
        cli.output_dir, metadata, summary,
        result["frame_rows"], result["clip_rows"])
    response = {
        "output_dir": str(Path(cli.output_dir).resolve()),
        "complete": completed["complete"],
        "quality_ratio": summary["quality"]["quality_ratio"],
        "quality_passes": summary["quality"]["passes"],
        "speedup": summary["speedup"],
    }
    print(json.dumps(response, sort_keys=True), flush=True)
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run pure-memory NLSPN GOP2 stages")
    parser.add_argument(
        "--stage", required=True, choices=("raft-parity", "benchmark"))
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--raft-weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir")
    parser.add_argument("--clips")
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timed-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-json")
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    torch.set_num_threads(1)
    if cli.stage == "raft-parity":
        return run_raft_parity(cli)
    if not cli.output_dir:
        raise ValueError("benchmark stage requires --output-dir")
    if cli.warmup_repeats < 0 or cli.timed_repeats <= 0:
        raise ValueError("benchmark repeat counts are invalid")
    return run_benchmark(cli)


if __name__ == "__main__":
    main()
