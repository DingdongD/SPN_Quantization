#!/usr/bin/env python3
"""One-load legacy worker for 90-window stratified NLSPN evaluation."""

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
from scripts import nlspn_stratified_motion_sampling as sampling
from scripts import nlspn_temporal_residual as residual
from scripts import raft_small_compat
from scripts import run_nlspn_cross_scene_raft_visualization_worker as raft_worker
from scripts import run_nlspn_frame_difference_visualization_worker as visual_worker
from scripts import run_nlspn_in_memory_gop2_worker as legacy_worker
from scripts import spn_sequence_io


def load_manifest(path):
    return sampling.load_manifest_json(path)


def _directory_digests(path):
    return dict((item.name, residual.file_sha256(item))
                for item in sorted(Path(path).iterdir()) if item.is_file())


def build_window_metadata(window, payload, model_metadata,
                          checkpoint_digest, args_digest, sweep_digest,
                          raft_digest, formal_digests, selected_configs,
                          device, seed, input_digest):
    metadata = dict(window)
    metadata.update({
        "complete": False,
        "height": residual.HEIGHT,
        "width": residual.WIDTH,
        "sparse_points": residual.SPARSE_COUNT,
        "seed": int(seed),
        "device": str(device),
        "checkpoint_sha256": str(checkpoint_digest),
        "args_sha256": str(args_digest),
        "formal_sweep_sha256": str(sweep_digest),
        "formal_directory_digests": dict(formal_digests),
        "formal_directory_modified": False,
        "input_digest": str(input_digest),
        "selected_configs": selected_configs,
        "method_order": list(visual.RAFT_METHOD_ORDER),
        "model": model_metadata,
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
        "raft_weight_sha256": str(raft_digest),
        "raft_implementation": "raft_small_compat",
        "raft_flow_updates": 12,
        "raft_flow_direction": "current_to_previous",
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "schedule": "I,P,I,P,I",
        "frame_ids": [int(value) for value in payload["frame_ids"]],
    })
    return metadata


def run_batch(cli, bundle_builder=None, payload_loader=None,
              cache_engine_factory=None, raft_engine_factory=None,
              cache_runner=None, raft_runner_fn=None, artifact_writer=None,
              config_loader=None, directory_digest_fn=None,
              file_digest_fn=None, input_digest_fn=None, **compat):
    if raft_runner_fn is None and "raft_runner" in compat:
        raft_runner_fn = compat.pop("raft_runner")
    if compat:
        raise TypeError("unexpected worker dependency: {}".format(
            sorted(compat)[0]))
    bundle_builder = bundle_builder or legacy_worker.build_models
    payload_loader = payload_loader or legacy_worker.load_in_memory_clips
    cache_engine_factory = cache_engine_factory or cache.FrameDifferenceGOP2Engine
    raft_engine_factory = raft_engine_factory or online.InMemoryGOP2Engine
    cache_runner = cache_runner or visual_worker.run_inference
    raft_runner_fn = raft_runner_fn or visual_worker.run_raft_inference
    artifact_writer = artifact_writer or visual.write_artifacts
    config_loader = config_loader or visual.load_selected_configs
    directory_digest_fn = directory_digest_fn or _directory_digests
    file_digest_fn = file_digest_fn or residual.file_sha256
    input_digest_fn = input_digest_fn or spn_sequence_io.canonical_input_digest

    windows = load_manifest(cli.manifest)
    raft_digest = file_digest_fn(cli.raft_weights)
    if raft_digest != raft_small_compat.EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("official RAFT-Small weight digest mismatch")
    formal_before = directory_digest_fn(cli.formal_dir)
    configs, sweep_digest = config_loader(
        Path(cli.formal_dir) / "threshold_sweep.csv")
    serialized_configs = motion.require_fixed_configs(configs)
    checkpoint_digest = file_digest_fn(cli.checkpoint)
    args_digest = file_digest_fn(cli.args_json)

    raft_worker.activate_cuda_device(cli.device)
    nlspn, raft, model_metadata = bundle_builder(
        cli.checkpoint, cli.args_json, cli.raft_weights, cli.device)
    cache_engine = cache_engine_factory(nlspn, cli.device)
    raft_engine = raft_engine_factory(nlspn, raft, cli.device)
    metric_count = 0
    for window in windows:
        payloads = payload_loader(
            cli.data_root, window["scene"],
            ((window["start_frame"], window["end_frame"]),), seed=cli.seed)
        if len(payloads) != 1:
            raise RuntimeError("payload loader returned an unexpected clip count")
        payload = payloads[0]
        visual._validate_payload(payload)
        if [int(value) for value in payload["frame_ids"]] != window["frame_ids"]:
            raise RuntimeError("payload frame IDs differ from manifest")
        four = cache_runner(cache_engine, payload, configs)
        raft_result = raft_runner_fn(raft_engine, payload)
        predictions = dict(four["predictions"])
        predictions["raft_gop2"] = raft_result["prediction"]
        if tuple(predictions) != visual.RAFT_METHOD_ORDER:
            raise RuntimeError("five-method prediction order is invalid")
        latency_rows = list(four["latency_rows"]) + \
            list(raft_result["latency_rows"])
        metrics = visual.collect_frame_metrics(payload, predictions, latency_rows)
        if len(metrics) != 25:
            raise RuntimeError("each window requires 25 metric rows")
        for row in metrics:
            row.update({
                "scene": window["scene"],
                "stratum": window["stratum"],
                "window_id": window["window_id"],
            })
        input_digest = input_digest_fn(
            payload["frame_ids"], payload["rgb"], payload["sparse"],
            payload["gt"], payload["valid"])
        metadata = build_window_metadata(
            window, payload, model_metadata, checkpoint_digest, args_digest,
            sweep_digest, raft_digest, formal_before, serialized_configs,
            cli.device, cli.seed, input_digest)
        completed = artifact_writer(
            Path(cli.output_root) / window["window_id"], payload,
            predictions, metrics, metadata)
        if not completed.get("complete"):
            raise RuntimeError("window artifacts did not complete")
        metric_count += len(metrics)

    if directory_digest_fn(cli.formal_dir) != formal_before:
        raise RuntimeError("formal pilot directory changed during evaluation")
    response = {
        "complete": True,
        "window_count": len(windows),
        "frame_metric_row_count": metric_count,
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
    }
    print(json.dumps(response, indent=2, sort_keys=True), flush=True)
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run one-load stratified NLSPN motion evaluation")
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
    torch.set_num_threads(1)
    return run_batch(cli)


if __name__ == "__main__":
    main()
