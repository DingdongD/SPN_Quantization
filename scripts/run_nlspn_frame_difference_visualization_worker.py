#!/usr/bin/env python3
"""Legacy-environment worker for five-frame causal NLSPN visualization."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_in_memory_gop2 as online
from scripts import nlspn_temporal_residual as residual
from scripts import run_nlspn_frame_difference_cache_worker as cache_worker
from scripts import run_nlspn_in_memory_gop2_worker as legacy_worker
from scripts import spn_sequence_io


def _prediction_array(result):
    value = result.prediction
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    expected = (residual.HEIGHT, residual.WIDTH)
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError("visualization inference prediction is invalid")
    return value.copy()


def run_inference(engine, payload, selected_configs):
    visual._validate_payload(payload)
    if set(selected_configs) != {"rgb_diff", "global_diff"}:
        raise ValueError("visualization selected configurations are incomplete")
    configs = (
        cache.candidate_configs("zero_flow")[0],
        selected_configs["rgb_diff"],
        selected_configs["global_diff"],
    )
    predictions = {}
    latency_rows = []
    engine.reset()
    full = []
    for local_index, frame_id in enumerate(payload["frame_ids"]):
        result = engine.infer_full(
            payload["rgb"][local_index], payload["sparse"][local_index])
        full.append(_prediction_array(result))
        latency_rows.append({
            "method": "full", "frame_id": int(frame_id),
            "latency_ms": float(result.latency_ms)})
    predictions["full"] = np.stack(full).astype(np.float32)
    for config in configs:
        engine.reset()
        values = []
        for local_index, frame_id in enumerate(payload["frame_ids"]):
            if online.frame_kind(local_index) == "I":
                result = engine.infer_i(
                    payload["rgb"][local_index],
                    payload["sparse"][local_index], local_index)
            else:
                result = engine.infer_p(
                    payload["rgb"][local_index],
                    payload["sparse"][local_index], local_index, config)
            values.append(_prediction_array(result))
            latency_rows.append({
                "method": config.variant, "frame_id": int(frame_id),
                "latency_ms": float(result.latency_ms)})
        predictions[config.variant] = np.stack(values).astype(np.float32)
    visual._validate_predictions(predictions)
    return {"predictions": predictions, "latency_rows": latency_rows}


def run_raft_inference(engine, payload):
    """Run one strict causal RAFT-GOP2 pass over a five-frame payload."""
    visual._validate_payload(payload)
    engine.reset()
    predictions = []
    latency_rows = []
    for local_index, frame_id in enumerate(payload["frame_ids"]):
        if online.frame_kind(local_index) == "I":
            result = engine.infer_i(
                payload["rgb"][local_index],
                payload["sparse"][local_index], local_index)
        else:
            result = engine.infer_p(
                payload["rgb"][local_index],
                payload["sparse"][local_index], local_index)
        predictions.append(_prediction_array(result))
        latency_rows.append({
            "method": "raft_gop2",
            "frame_id": int(frame_id),
            "latency_ms": float(result.latency_ms),
        })
    return {
        "prediction": np.stack(predictions).astype(np.float32),
        "latency_rows": latency_rows,
    }


def _directory_digests(path):
    path = Path(path)
    return dict(
        (item.name, residual.file_sha256(item))
        for item in sorted(path.iterdir()) if item.is_file())


def run_worker(cli):
    formal_dir = Path(cli.formal_dir)
    formal_before = _directory_digests(formal_dir)
    configs, sweep_digest = visual.load_selected_configs(
        formal_dir / "threshold_sweep.csv")
    payloads = legacy_worker.load_in_memory_clips(
        cli.data_root, cli.scene, ((1, 5),), seed=cli.seed)
    if len(payloads) != 1:
        raise RuntimeError("visualization loader returned unexpected clips")
    payload = payloads[0]
    model, model_metadata = cache_worker.build_nlspn(
        cli.checkpoint, cli.args_json, cli.device)
    engine = cache.FrameDifferenceGOP2Engine(model, cli.device)
    result = run_inference(engine, payload, configs)
    metrics = visual.collect_frame_metrics(
        payload, result["predictions"], result["latency_rows"])
    formal_after = _directory_digests(formal_dir)
    if formal_before != formal_after:
        raise RuntimeError("formal pilot directory changed during visualization")
    input_digest = spn_sequence_io.canonical_input_digest(
        payload["frame_ids"], payload["rgb"], payload["sparse"],
        payload["gt"], payload["valid"])
    metadata = {
        "complete": False,
        "scene": str(cli.scene),
        "frame_ids": [int(value) for value in payload["frame_ids"]],
        "height": residual.HEIGHT,
        "width": residual.WIDTH,
        "sparse_points": residual.SPARSE_COUNT,
        "seed": int(cli.seed),
        "device": str(cli.device),
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_sha256": residual.file_sha256(cli.checkpoint),
        "args_json": str(Path(cli.args_json).resolve()),
        "args_sha256": residual.file_sha256(cli.args_json),
        "formal_dir": str(formal_dir.resolve()),
        "formal_sweep_sha256": sweep_digest,
        "formal_directory_digests": formal_before,
        "formal_directory_modified": False,
        "input_digest": input_digest,
        "selected_configs": dict(
            (variant, {
                "threshold": float(config.threshold),
                "dilation_radius": int(config.dilation_radius),
            }) for variant, config in configs.items()),
        "model": model_metadata,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "schedule": "I,P,I,P,I",
    }
    completed = visual.write_artifacts(
        cli.output_dir, payload, result["predictions"], metrics, metadata)
    response = {
        "complete": bool(completed["complete"]),
        "output_dir": str(Path(cli.output_dir).resolve()),
        "error_vmax_m": float(completed["error_vmax_m"]),
    }
    print(json.dumps(response, sort_keys=True), flush=True)
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run five-frame NLSPN cache visualization worker")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    torch.set_num_threads(1)
    return run_worker(cli)


if __name__ == "__main__":
    main()
