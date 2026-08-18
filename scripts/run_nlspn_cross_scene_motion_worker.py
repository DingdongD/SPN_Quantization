#!/usr/bin/env python3
"""One-load legacy worker for six-scene causal NLSPN evaluation."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_temporal_residual as residual
from scripts import run_nlspn_frame_difference_cache_worker as cache_worker
from scripts import run_nlspn_frame_difference_visualization_worker as visual_worker
from scripts import run_nlspn_in_memory_gop2_worker as legacy_worker
from scripts import spn_sequence_io


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or set(value) != {"windows"}:
        raise ValueError("manifest must contain exactly a windows field")
    windows = value["windows"]
    if not isinstance(windows, list) or len(windows) != len(motion.SCENES):
        raise ValueError("manifest must contain exactly six windows")
    if [row.get("scene") for row in windows] != list(motion.SCENES):
        raise ValueError("manifest scenes are not in the approved order")
    validated = []
    for row in windows:
        if not isinstance(row, dict):
            raise ValueError("manifest windows must be objects")
        ids = row.get("frame_ids")
        if (not isinstance(ids, list) or len(ids) != 5 or
                any(not isinstance(item, int) or item <= 0 for item in ids) or
                any(right != left + 1 for left, right in zip(ids, ids[1:]))):
            raise ValueError("manifest window frame IDs must be consecutive")
        scores = row.get("pair_scores")
        motion_score = row.get("motion_score")
        if (not isinstance(scores, list) or len(scores) != 4 or
                not np.isfinite(np.asarray(scores, dtype=np.float64)).all() or
                not np.isfinite(float(motion_score))):
            raise ValueError("manifest motion scores are invalid")
        if (int(row.get("start_frame", -1)) != ids[0] or
                int(row.get("end_frame", -1)) != ids[-1]):
            raise ValueError("manifest window bounds are invalid")
        validated.append(dict(row))
    return validated


def _directory_digests(path):
    path = Path(path)
    return dict(
        (item.name, residual.file_sha256(item))
        for item in sorted(path.iterdir()) if item.is_file())


def run_batch(cli, model_builder=None, payload_loader=None,
              inference_runner=None, artifact_writer=None, root_writer=None,
              config_loader=None, engine_factory=None,
              directory_digest_fn=None, file_digest_fn=None,
              input_digest_fn=None):
    model_builder = model_builder or cache_worker.build_nlspn
    payload_loader = payload_loader or legacy_worker.load_in_memory_clips
    inference_runner = inference_runner or visual_worker.run_inference
    artifact_writer = artifact_writer or visual.write_artifacts
    root_writer = root_writer or motion.write_root_artifacts
    config_loader = config_loader or visual.load_selected_configs
    engine_factory = engine_factory or cache.FrameDifferenceGOP2Engine
    directory_digest_fn = directory_digest_fn or _directory_digests
    file_digest_fn = file_digest_fn or residual.file_sha256
    input_digest_fn = input_digest_fn or spn_sequence_io.canonical_input_digest

    windows = load_manifest(cli.manifest)
    formal_dir = Path(cli.formal_dir)
    formal_before = directory_digest_fn(formal_dir)
    configs, sweep_digest = config_loader(formal_dir / "threshold_sweep.csv")
    serialized_configs = motion.require_fixed_configs(configs)
    checkpoint_digest = file_digest_fn(cli.checkpoint)
    args_digest = file_digest_fn(cli.args_json)

    model, model_metadata = model_builder(
        cli.checkpoint, cli.args_json, cli.device)
    engine = engine_factory(model, cli.device)
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
        result = inference_runner(engine, payload, configs)
        metrics = visual.collect_frame_metrics(
            payload, result["predictions"], result["latency_rows"])
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
            "model": model_metadata,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "schedule": "I,P,I,P,I",
        }
        completed = artifact_writer(
            output_root / scene, payload, result["predictions"], metrics,
            metadata)
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
        "model": model_metadata,
        "model_load_count": 1,
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
    }
    print(json.dumps(response, indent=2, sort_keys=True), flush=True)
    return response


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run one-load six-scene causal NLSPN evaluation")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    torch.set_num_threads(1)
    return run_batch(cli)


if __name__ == "__main__":
    main()
