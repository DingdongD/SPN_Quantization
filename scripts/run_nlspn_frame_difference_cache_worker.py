#!/usr/bin/env python3
"""Legacy-environment worker for NLSPN frame-difference cache pilots."""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_in_memory_gop2 as online
from scripts import nlspn_temporal_residual as residual
from scripts import run_nlspn_in_memory_gop2_worker as legacy_worker
from scripts import run_spn_sequence_worker as spn_worker


MASK_FIELDS = (
    "stable_fraction",
    "changed_fraction",
    "photometric_changed_fraction",
    "sparse_changed_fraction",
    "out_of_bounds_fraction",
)

FINAL_ARTIFACTS = (
    "run_metadata.json",
    "summary.json",
    "threshold_sweep.csv",
    "frame_metrics.csv",
    "clip_summary.csv",
    "report.md",
)


def _clip_name(frame_ids):
    return "%04d-%04d" % (int(frame_ids[0]), int(frame_ids[-1]))


def _as_prediction(value, shape):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.shape != tuple(shape) or not np.isfinite(value).all():
        raise ValueError("prediction geometry or values are invalid")
    return value


def run_calibration_sweep(engine, calibration_payloads, full_predictions):
    payloads = tuple(calibration_payloads)
    if not payloads:
        raise ValueError("calibration requires at least one clip")
    frame_count = int(sum(payload["frame_ids"].size for payload in payloads))
    full_predictions = list(full_predictions)
    if len(full_predictions) != frame_count:
        raise ValueError("full reference prediction count is invalid")
    frame_id_min = min(int(payload["frame_ids"][0]) for payload in payloads)
    frame_id_max = max(int(payload["frame_ids"][-1]) for payload in payloads)
    rows = []
    for variant in ("rgb_diff", "global_diff"):
        for config in cache.candidate_configs(variant):
            prediction_index = 0
            full_sse = 0.0
            variant_sse = 0.0
            valid_pixels = 0
            metric_values = dict((field, []) for field in MASK_FIELDS)
            for payload in payloads:
                engine.reset()
                for local_index in range(payload["frame_ids"].size):
                    if online.frame_kind(local_index) == "I":
                        result = engine.infer_i(
                            payload["rgb"][local_index],
                            payload["sparse"][local_index], local_index)
                    else:
                        result = engine.infer_p(
                            payload["rgb"][local_index],
                            payload["sparse"][local_index], local_index,
                            config)
                        for field in MASK_FIELDS:
                            metric_values[field].append(
                                float(result.mask_metrics[field]))
                    gt = np.asarray(payload["gt"][local_index],
                                    dtype=np.float64)
                    valid = np.asarray(payload["valid"][local_index],
                                       dtype=bool)
                    prediction = _as_prediction(result.prediction, gt.shape)
                    full = _as_prediction(
                        full_predictions[prediction_index], gt.shape)
                    full_error = full.astype(np.float64)[valid] - gt[valid]
                    variant_error = prediction.astype(
                        np.float64)[valid] - gt[valid]
                    full_sse += float(np.sum(full_error ** 2))
                    variant_sse += float(np.sum(variant_error ** 2))
                    valid_pixels += int(np.count_nonzero(valid))
                    prediction_index += 1
            if prediction_index != frame_count or valid_pixels <= 0:
                raise RuntimeError("calibration did not consume exact frames")
            row = {
                "variant": variant,
                "threshold": float(config.threshold),
                "dilation_radius": int(config.dilation_radius),
                "sse": variant_sse,
                "full_sse": full_sse,
                "valid_pixels": valid_pixels,
                "rmse": float(np.sqrt(variant_sse / valid_pixels)),
                "full_rmse": float(np.sqrt(full_sse / valid_pixels)),
                "frame_count": frame_count,
                "frame_id_min": frame_id_min,
                "frame_id_max": frame_id_max,
                "selected": False,
            }
            for field, values in metric_values.items():
                row[field] = float(np.mean(values)) if values else 0.0
            rows.append(row)
    return rows


def _record(result, variant, repeat, payload, local_index, config=None):
    metrics = getattr(result, "mask_metrics", {}) or {}
    row = {
        "variant": variant,
        "repeat": int(repeat),
        "clip": _clip_name(payload["frame_ids"]),
        "frame_id": int(payload["frame_ids"][local_index]),
        "local_index": int(local_index),
        "kind": str(result.kind),
        "latency_ms": float(result.latency_ms),
        "threshold": ("" if config is None or config.threshold is None
                      else float(config.threshold)),
        "dilation_radius": (
            "" if config is None or config.dilation_radius is None
            else int(config.dilation_radius)),
        "dx": float(metrics.get("dx", 0.0)),
        "dy": float(metrics.get("dy", 0.0)),
    }
    for field in MASK_FIELDS:
        row[field] = float(metrics.get(field, 0.0))
    return row


def run_timed_variant(engine, payloads, config, repeats):
    if not isinstance(config, cache.CacheConfig):
        raise TypeError("timed variant requires CacheConfig")
    if int(repeats) <= 0:
        raise ValueError("timed repeats must be positive")
    records = []
    predictions = []
    for repeat in range(int(repeats)):
        for payload in payloads:
            engine.reset()
            for local_index in range(payload["frame_ids"].size):
                if online.frame_kind(local_index) == "I":
                    result = engine.infer_i(
                        payload["rgb"][local_index],
                        payload["sparse"][local_index], local_index)
                else:
                    result = engine.infer_p(
                        payload["rgb"][local_index],
                        payload["sparse"][local_index], local_index, config)
                records.append(_record(
                    result, config.variant, repeat, payload,
                    local_index, config))
                if repeat == 0:
                    predictions.append(_as_prediction(
                        result.prediction,
                        payload["gt"][local_index].shape).copy())
    return records, predictions


def run_timed_full(engine, payloads, repeats):
    if int(repeats) <= 0:
        raise ValueError("timed repeats must be positive")
    records = []
    predictions = []
    for repeat in range(int(repeats)):
        for payload in payloads:
            engine.reset()
            for local_index in range(payload["frame_ids"].size):
                result = engine.infer_full(
                    payload["rgb"][local_index],
                    payload["sparse"][local_index])
                records.append(_record(
                    result, "full", repeat, payload, local_index))
                if repeat == 0:
                    predictions.append(_as_prediction(
                        result.prediction,
                        payload["gt"][local_index].shape).copy())
    return records, predictions


def _warmup(engine, payloads, config, repeats):
    for _ in range(int(repeats)):
        for payload in payloads:
            engine.reset()
            for local_index in range(payload["frame_ids"].size):
                if config is None:
                    engine.infer_full(
                        payload["rgb"][local_index],
                        payload["sparse"][local_index])
                elif online.frame_kind(local_index) == "I":
                    engine.infer_i(
                        payload["rgb"][local_index],
                        payload["sparse"][local_index], local_index)
                else:
                    engine.infer_p(
                        payload["rgb"][local_index],
                        payload["sparse"][local_index], local_index, config)


def _quality_terms(full, variant, gt, valid):
    full = np.asarray(full, dtype=np.float64)
    variant = np.asarray(variant, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (full.shape == variant.shape == gt.shape == valid.shape):
        raise ValueError("quality arrays must have identical shapes")
    if not np.any(valid):
        raise ValueError("quality mask is empty")
    full_error = full[valid] - gt[valid]
    variant_error = variant[valid] - gt[valid]
    return (float(np.sum(full_error ** 2)),
            float(np.sum(variant_error ** 2)),
            int(np.count_nonzero(valid)))


def _quality_summary(full, variant, gt, valid, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("quality split cannot be empty")
    full_sse, variant_sse, count = _quality_terms(
        np.asarray(full)[indices], np.asarray(variant)[indices],
        np.asarray(gt)[indices], np.asarray(valid)[indices])
    rmse_full = float(np.sqrt(full_sse / count))
    if rmse_full <= 0.0:
        raise ValueError("full reference RMSE must be positive")
    rmse_variant = float(np.sqrt(variant_sse / count))
    ratio = float(rmse_variant / rmse_full)
    return {
        "rmse_full": rmse_full,
        "rmse_variant": rmse_variant,
        "quality_ratio": ratio,
        "passes_1pct": bool(ratio <= 1.01),
        "full_squared_error": full_sse,
        "variant_squared_error": variant_sse,
        "valid_pixels": count,
        "frame_count": int(indices.size),
    }


def _path_latency(records, include_kinds):
    latency = {
        "overall": online.latency_summary(
            [row["latency_ms"] for row in records])
    }
    if include_kinds:
        for key, kind in (("i", "I"), ("p", "P")):
            latency[key] = online.latency_summary([
                row["latency_ms"] for row in records
                if row["kind"] == kind])
    return latency


def execute_pilot(engine, payloads, calibration_clip_count=4,
                  warmup_repeats=1, timed_repeats=5):
    payloads = tuple(payloads)
    if not payloads:
        raise ValueError("pilot requires input clips")
    calibration_clip_count = int(calibration_clip_count)
    if not 0 < calibration_clip_count < len(payloads):
        raise ValueError("pilot requires nonempty calibration and held-out clips")
    if int(warmup_repeats) < 0 or int(timed_repeats) <= 0:
        raise ValueError("pilot repeat counts are invalid")

    _warmup(engine, payloads, None, warmup_repeats)
    full_records, full_predictions = run_timed_full(
        engine, payloads, timed_repeats)
    calibration_payloads = payloads[:calibration_clip_count]
    calibration_frames = int(sum(
        payload["frame_ids"].size for payload in calibration_payloads))
    sweep_rows = run_calibration_sweep(
        engine, calibration_payloads,
        full_predictions[:calibration_frames])
    selected = {
        variant: cache.select_calibration_config(
            [row for row in sweep_rows if row["variant"] == variant],
            variant)
        for variant in ("rgb_diff", "global_diff")
    }
    for row in sweep_rows:
        config = selected[row["variant"]]
        row["selected"] = bool(
            float(row["threshold"]) == float(config.threshold) and
            int(row["dilation_radius"]) == int(config.dilation_radius))

    final_configs = (
        cache.candidate_configs("zero_flow")[0],
        selected["rgb_diff"],
        selected["global_diff"],
    )
    records_by_variant = {"full": full_records}
    predictions_by_variant = {
        "full": np.stack(full_predictions).astype(np.float32)}
    for config in final_configs:
        _warmup(engine, payloads, config, warmup_repeats)
        records, predictions = run_timed_variant(
            engine, payloads, config, timed_repeats)
        records_by_variant[config.variant] = records
        predictions_by_variant[config.variant] = np.stack(
            predictions).astype(np.float32)

    gt = np.concatenate([payload["gt"] for payload in payloads])
    valid = np.concatenate([payload["valid"] for payload in payloads])
    frame_ids = np.concatenate([payload["frame_ids"] for payload in payloads])
    clip_names = np.concatenate([
        np.repeat(_clip_name(payload["frame_ids"]),
                  payload["frame_ids"].size)
        for payload in payloads])
    total_frames = int(frame_ids.size)
    split_indices = {
        "calibration": np.arange(calibration_frames),
        "heldout": np.arange(calibration_frames, total_frames),
        "all": np.arange(total_frames),
    }
    summary = {
        "paths": {},
        "external_raft_reference": {
            "quality_ratio": 1.0069332411236265,
            "speedup": 0.5267047643822169,
        },
        "timed_repeats": int(timed_repeats),
        "warmup_repeats": int(warmup_repeats),
        "calibration_frames": calibration_frames,
        "heldout_frames": total_frames - calibration_frames,
    }
    full_total = _path_latency(full_records, False)["overall"]["total_ms"]
    frame_rows = []
    clip_rows = []
    for variant in ("full", "zero_flow", "rgb_diff", "global_diff"):
        records = records_by_variant[variant]
        latency = _path_latency(records, variant != "full")
        quality = dict(
            (split, _quality_summary(
                predictions_by_variant["full"],
                predictions_by_variant[variant], gt, valid, indices))
            for split, indices in split_indices.items())
        path_total = latency["overall"]["total_ms"]
        summary["paths"][variant] = {
            "latency": latency,
            "speedup": float(full_total / path_total),
            "quality": quality,
        }
        frame_quality = []
        for index in range(total_frames):
            terms = _quality_terms(
                predictions_by_variant["full"][index],
                predictions_by_variant[variant][index],
                gt[index], valid[index])
            full_sse, variant_sse, count = terms
            rmse_full = float(np.sqrt(full_sse / count))
            rmse_variant = float(np.sqrt(variant_sse / count))
            frame_quality.append({
                "full_squared_error": full_sse,
                "variant_squared_error": variant_sse,
                "valid_pixels": count,
                "rmse_full": rmse_full,
                "rmse_variant": rmse_variant,
                "quality_ratio": float(rmse_variant / rmse_full),
            })
        for row in records:
            if row["repeat"] == 0:
                index = sum(1 for prior in frame_rows
                            if prior["variant"] == variant and
                            prior["repeat"] == 0)
                row.update(frame_quality[index])
                row["split"] = (
                    "calibration" if index < calibration_frames
                    else "heldout")
            frame_rows.append(row)
        for clip in dict.fromkeys(clip_names.tolist()):
            indices = np.flatnonzero(clip_names == clip)
            split = (
                "calibration" if int(indices[0]) < calibration_frames
                else "heldout")
            row = _quality_summary(
                predictions_by_variant["full"],
                predictions_by_variant[variant], gt, valid, indices)
            row.update({"variant": variant, "split": split, "clip": clip})
            clip_rows.append(row)
    return {
        "summary": summary,
        "selected_configs": selected,
        "sweep_rows": sweep_rows,
        "frame_rows": frame_rows,
        "clip_rows": clip_rows,
    }


def _render_report(result):
    summary = result["summary"]
    lines = [
        "# NLSPN Frame-Difference Cache Pilot",
        "",
        "Calibration selects RGB/global mask parameters before held-out "
        "evaluation. No RAFT, fallback, fine-tuning, or tensor cache is used.",
        "",
        "| Variant | All RMSE ratio | <=1% | Speedup |",
        "|---|---:|:---:|---:|",
    ]
    for variant in ("zero_flow", "rgb_diff", "global_diff"):
        path = summary["paths"][variant]
        quality = path["quality"]["all"]
        lines.append("| %s | %.9f | %s | %.6fx |" % (
            variant, quality["quality_ratio"],
            "pass" if quality["passes_1pct"] else "fail",
            path["speedup"]))
    lines.extend(["", "A speedup above 1.0x is faster than full NLSPN.", ""])
    return "\n".join(lines)


def write_final_artifacts(output_dir, metadata, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [
        path for path in output_dir.iterdir()
        if path.is_file() and path.name not in FINAL_ARTIFACTS]
    if unexpected:
        raise RuntimeError("output directory contains tensor cache or "
                           "unapproved artifacts")
    incomplete = dict(metadata)
    incomplete["complete"] = False
    legacy_worker._write_json_atomic(
        output_dir / "run_metadata.json", incomplete)
    legacy_worker._write_json_atomic(
        output_dir / "summary.json", result["summary"])
    legacy_worker._write_csv_atomic(
        output_dir / "threshold_sweep.csv", result["sweep_rows"])
    legacy_worker._write_csv_atomic(
        output_dir / "frame_metrics.csv", result["frame_rows"])
    legacy_worker._write_csv_atomic(
        output_dir / "clip_summary.csv", result["clip_rows"])
    legacy_worker._write_text_atomic(
        output_dir / "report.md", _render_report(result))
    missing = [
        name for name in FINAL_ARTIFACTS
        if not (output_dir / name).is_file() or
        (output_dir / name).stat().st_size == 0]
    if missing:
        raise RuntimeError("final pilot artifacts are incomplete")
    completed = dict(incomplete)
    completed["complete"] = True
    completed["artifact_count"] = len(FINAL_ARTIFACTS)
    legacy_worker._write_json_atomic(
        output_dir / "run_metadata.json", completed)
    return completed


def build_nlspn(checkpoint, args_json, device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for requested device")
    args = legacy_worker._read_args(args_json)
    started = time.perf_counter()
    model, metadata = spn_worker.build_model("nlspn", args, device)
    spn_worker.load_checkpoint_strict(model, Path(checkpoint))
    model.eval()
    if bool(model.args.preserve_input):
        raise ValueError("NLSPN preserve_input must remain false")
    if model.args.affinity != "TGASS" or model.prop_layer.prop_time != 18:
        raise ValueError("NLSPN propagation configuration changed")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metadata = dict(metadata)
    metadata.update({
        "model_load_seconds": float(time.perf_counter() - started),
        "torch_version": str(torch.__version__),
        "device": str(device),
    })
    return model, metadata


def run_benchmark(cli):
    clips = legacy_worker.parse_clips(cli.clips)
    input_started = time.perf_counter()
    payloads = legacy_worker.load_in_memory_clips(
        cli.data_root, cli.scene, clips, seed=cli.seed)
    input_seconds = float(time.perf_counter() - input_started)
    model, model_metadata = build_nlspn(
        cli.checkpoint, cli.args_json, cli.device)
    engine = cache.FrameDifferenceGOP2Engine(model, cli.device)
    if torch.device(cli.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(cli.device))
    result = execute_pilot(
        engine, payloads,
        calibration_clip_count=cli.calibration_clip_count,
        warmup_repeats=cli.warmup_repeats,
        timed_repeats=cli.timed_repeats)
    result["summary"]["startup"] = {
        "input_load_seconds": input_seconds,
        "model_load_seconds": model_metadata["model_load_seconds"],
    }
    if torch.device(cli.device).type == "cuda":
        result["summary"]["peak_cuda_memory"] = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(
                torch.device(cli.device))),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(
                torch.device(cli.device))),
        }
    else:
        result["summary"]["peak_cuda_memory"] = {
            "allocated_bytes": 0, "reserved_bytes": 0}
    frame_count = int(sum(
        payload["frame_ids"].size for payload in payloads))
    metadata = {
        "complete": False,
        "scene": str(cli.scene),
        "clips": [list(clip) for clip in clips],
        "calibration_clip_count": int(cli.calibration_clip_count),
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
        "args_json": str(Path(cli.args_json).resolve()),
        "args_sha256": residual.file_sha256(cli.args_json),
        "model": model_metadata,
        "numpy_version": str(np.__version__),
        "gpu": (torch.cuda.get_device_name(torch.device(cli.device))
                if torch.device(cli.device).type == "cuda" else "cpu"),
        "intermediate_tensor_cache": False,
        "raft_constructed": False,
        "causal": True,
        "fallback": False,
    }
    completed = write_final_artifacts(cli.output_dir, metadata, result)
    response = {
        "output_dir": str(Path(cli.output_dir).resolve()),
        "complete": bool(completed["complete"]),
        "selected_configs": dict(
            (variant, {
                "threshold": config.threshold,
                "dilation_radius": config.dilation_radius,
            }) for variant, config in result["selected_configs"].items()),
        "paths": dict(
            (variant, {
                "quality_ratio": values["quality"]["all"]["quality_ratio"],
                "speedup": values["speedup"],
            }) for variant, values in result["summary"]["paths"].items()),
    }
    print(json.dumps(response, sort_keys=True), flush=True)
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run legacy NLSPN frame-difference cache worker")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clips")
    parser.add_argument("--calibration-clip-count", type=int, default=4)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timed-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    torch.set_num_threads(1)
    return run_benchmark(cli)


if __name__ == "__main__":
    main()
