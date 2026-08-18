#!/usr/bin/env python3
"""One-load legacy worker for six-scene strict RAFT-GOP2 visualization."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_in_memory_gop2 as online
from scripts import nlspn_temporal_residual as residual
from scripts import raft_small_compat
from scripts import run_nlspn_cross_scene_motion_worker as base_worker
from scripts import run_nlspn_frame_difference_visualization_worker as visual_worker
from scripts import run_nlspn_in_memory_gop2_worker as legacy_worker
from scripts import spn_sequence_io


def _directory_digests(path):
    path = Path(path)
    return dict(
        (item.name, residual.file_sha256(item))
        for item in sorted(path.iterdir()) if item.is_file())


def activate_cuda_device(device):
    """Align the current CUDA context with legacy custom extension tensors."""
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for requested device")
        torch.cuda.set_device(device)


def run_batch(cli, bundle_builder=None, payload_loader=None,
              cache_engine_factory=None, raft_engine_factory=None,
              cache_runner=None, raft_runner=None, artifact_writer=None,
              root_writer=None, config_loader=None,
              directory_digest_fn=None, file_digest_fn=None,
              input_digest_fn=None):
    bundle_builder = bundle_builder or legacy_worker.build_models
    payload_loader = payload_loader or legacy_worker.load_in_memory_clips
    cache_engine_factory = (
        cache_engine_factory or cache.FrameDifferenceGOP2Engine)
    raft_engine_factory = raft_engine_factory or online.InMemoryGOP2Engine
    cache_runner = cache_runner or visual_worker.run_inference
    raft_runner = raft_runner or visual_worker.run_raft_inference
    artifact_writer = artifact_writer or visual.write_artifacts
    root_writer = root_writer or motion.write_root_artifacts
    config_loader = config_loader or visual.load_selected_configs
    directory_digest_fn = directory_digest_fn or _directory_digests
    file_digest_fn = file_digest_fn or residual.file_sha256
    input_digest_fn = input_digest_fn or spn_sequence_io.canonical_input_digest

    windows = base_worker.load_manifest(cli.manifest)
    raft_digest = file_digest_fn(cli.raft_weights)
    if raft_digest != raft_small_compat.EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("official RAFT-Small weight digest mismatch")

    formal_dir = Path(cli.formal_dir)
    formal_before = directory_digest_fn(formal_dir)
    configs, sweep_digest = config_loader(formal_dir / "threshold_sweep.csv")
    serialized_configs = motion.require_fixed_configs(configs)
    checkpoint_digest = file_digest_fn(cli.checkpoint)
    args_digest = file_digest_fn(cli.args_json)

    nlspn, raft, model_metadata = bundle_builder(
        cli.checkpoint, cli.args_json, cli.raft_weights, cli.device)
    cache_engine = cache_engine_factory(nlspn, cli.device)
    raft_engine = raft_engine_factory(nlspn, raft, cli.device)
    summary_rows = []
    output_root = Path(cli.output_root)

    for window in windows:
        scene = window["scene"]
        payloads = payload_loader(
            cli.data_root, scene,
            ((window["start_frame"], window["end_frame"]),),
            seed=cli.seed)
        if len(payloads) != 1:
            raise RuntimeError("payload loader returned an unexpected clip count")
        payload = payloads[0]
        visual._validate_payload(payload)
        actual_ids = [int(value) for value in payload["frame_ids"]]
        if actual_ids != window["frame_ids"]:
            raise RuntimeError("payload frame IDs differ from manifest")

        four = cache_runner(cache_engine, payload, configs)
        raft_result = raft_runner(raft_engine, payload)
        predictions = dict(four["predictions"])
        predictions["raft_gop2"] = raft_result["prediction"]
        latency_rows = (
            list(four["latency_rows"]) + list(raft_result["latency_rows"]))
        if tuple(predictions) != visual.RAFT_METHOD_ORDER:
            raise RuntimeError("five-method prediction order is invalid")
        metrics = visual.collect_frame_metrics(
            payload, predictions, latency_rows)
        for row in metrics:
            row["scene"] = scene

        input_digest = input_digest_fn(
            payload["frame_ids"], payload["rgb"], payload["sparse"],
            payload["gt"], payload["valid"])
        metadata = {
            "complete": False,
            "scene": scene,
            "frame_ids": actual_ids,
            "start_frame": window["start_frame"],
            "end_frame": window["end_frame"],
            "pair_scores": window["pair_scores"],
            "motion_score": window["motion_score"],
            "height": residual.HEIGHT,
            "width": residual.WIDTH,
            "sparse_points": residual.SPARSE_COUNT,
            "seed": int(cli.seed),
            "device": str(cli.device),
            "checkpoint": str(Path(cli.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_digest,
            "args_json": str(Path(cli.args_json).resolve()),
            "args_sha256": args_digest,
            "formal_dir": str(formal_dir.resolve()),
            "formal_sweep_sha256": sweep_digest,
            "formal_directory_digests": formal_before,
            "formal_directory_modified": False,
            "input_digest": input_digest,
            "selected_configs": serialized_configs,
            "method_order": list(visual.RAFT_METHOD_ORDER),
            "model": model_metadata,
            "nlspn_model_load_count": 1,
            "raft_model_load_count": 1,
            "raft_weights": str(Path(cli.raft_weights).resolve()),
            "raft_weight_sha256": raft_digest,
            "raft_implementation": "raft_small_compat",
            "raft_flow_updates": 12,
            "raft_flow_direction": "current_to_previous",
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "schedule": "I,P,I,P,I",
        }
        completed = artifact_writer(
            output_root / scene, payload, predictions, metrics, metadata)
        if not completed.get("complete"):
            raise RuntimeError("scene artifact writer did not complete")
        summary_rows.extend(motion.build_scene_summary(scene, metrics))

    formal_after = directory_digest_fn(formal_dir)
    if formal_before != formal_after:
        raise RuntimeError("formal pilot directory changed during evaluation")
    root_metadata = {
        "complete": False,
        "seed": int(cli.seed),
        "device": str(cli.device),
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "args_json": str(Path(cli.args_json).resolve()),
        "args_sha256": args_digest,
        "formal_dir": str(formal_dir.resolve()),
        "formal_sweep_sha256": sweep_digest,
        "formal_directory_digests": formal_before,
        "formal_directory_modified": False,
        "selected_configs": serialized_configs,
        "method_order": list(visual.RAFT_METHOD_ORDER),
        "model": model_metadata,
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
        "raft_weights": str(Path(cli.raft_weights).resolve()),
        "raft_weight_sha256": raft_digest,
        "raft_implementation": "raft_small_compat",
        "raft_flow_updates": 12,
        "raft_flow_direction": "current_to_previous",
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "schedule": "I,P,I,P,I",
    }
    completed = root_writer(output_root, windows, summary_rows, root_metadata)
    response = {
        "complete": bool(completed.get("complete")),
        "output_root": str(output_root.resolve()),
        "scene_count": len(windows),
        "summary_row_count": len(summary_rows),
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
    }
    print(json.dumps(response, indent=2, sort_keys=True), flush=True)
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run one-load six-scene strict RAFT-GOP2 visualization")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--raft-weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    activate_cuda_device(cli.device)
    torch.set_num_threads(1)
    return run_batch(cli)


if __name__ == "__main__":
    main()
