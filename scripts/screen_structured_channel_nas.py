#!/usr/bin/env python3
"""Screen interface-preserving channel-pruned SPN candidates on NYU."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import screen_four_model_depth_nas as depth_nas  # noqa: E402
from spn_quant.structured_channel_pruning import (  # noqa: E402
    prune_basic_block_hidden,
    prune_conv_bn_deconv_bridge,
    prune_mlp_hidden,
)


CANDIDATES = {
    "dyspn": (
        ("baseline", 1.0),
        ("bridge_87p5pct", 0.875),
        ("bridge_75pct", 0.75),
        ("bridge_62p5pct", 0.625),
        ("bridge62_stage5hidden_87p5pct", 0.875),
        ("bridge62_stage5hidden_75pct", 0.75),
        ("bridge62_stage5hidden_62p5pct", 0.625),
        ("bridge62_s5hidden62_s4hidden_75pct", 0.75),
        ("bridge62_s5hidden62_s4hidden_50pct", 0.5),
    ),
    "nlspn": (
        ("baseline", 1.0),
        ("bridge_87p5pct", 0.875),
        ("bridge_75pct", 0.75),
        ("bridge_62p5pct", 0.625),
        ("bridge62_stage5hidden_87p5pct", 0.875),
        ("bridge62_stage5hidden_75pct", 0.75),
        ("bridge62_stage5hidden_62p5pct", 0.625),
        ("bridge62_s5hidden62_s4hidden_75pct", 0.75),
        ("bridge62_s5hidden62_s4hidden_50pct", 0.5),
    ),
    "completionformer": (
        ("baseline", 1.0),
        ("mlp_87p5pct", 0.875),
        ("mlp_75pct", 0.75),
        ("mlp_62p5pct", 0.625),
        ("mlp62_stage4hidden_87p5pct", 0.875),
        ("mlp62_stage4hidden_75pct", 0.75),
        ("mlp62_stage4hidden_62p5pct", 0.625),
        ("mlp62_s4hidden62_s3hidden_80pct", 0.8),
        ("mlp62_s4hidden62_s3hidden_60pct", 0.6),
    ),
}


def candidate_ids(model_name, selected=None):
    available = tuple(
        candidate_id for candidate_id, _ in CANDIDATES[model_name])
    if selected is None:
        return available
    selected = tuple(selected)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError("unknown structured candidates: %s" %
                         sorted(unknown))
    return selected


def evaluation_indices(configured, dataset_size, sample_count):
    if sample_count == 0:
        return tuple(range(dataset_size))
    if sample_count < 0 or sample_count > len(configured):
        raise ValueError("sample count must be zero or within configured set")
    return tuple(int(value) for value in configured[:sample_count])


def _candidate_ratio(model_name, candidate_id):
    matches = [ratio for name, ratio in CANDIDATES[model_name]
               if name == candidate_id]
    if len(matches) != 1:
        raise ValueError("unknown structured candidate %s/%s" %
                         (model_name, candidate_id))
    return float(matches[0])


def _prune_basic_blocks(blocks, keep_ratio):
    reports = []
    for block_index, block in enumerate(blocks):
        original_width = int(block.conv1.out_channels)
        keep = int(round(original_width * keep_ratio))
        selected = prune_basic_block_hidden(block, keep)
        reports.append({
            "block_index": block_index,
            "original_width": original_width,
            "hidden_width": keep,
            "selected_channels": selected.tolist(),
        })
    return reports


def apply_structured_candidate(model, model_name, candidate_id,
                               bridge_width=None, mlp_ratio=None):
    ratio = _candidate_ratio(model_name, candidate_id)
    if candidate_id == "baseline":
        return {"candidate_id": candidate_id, "ratio": ratio}

    if model_name in ("dyspn", "nlspn"):
        owner = model.base if model_name == "dyspn" else model
        original_width = int(owner.conv6[0].out_channels)
        deep_compound = candidate_id.startswith(
            "bridge62_s5hidden62_s4hidden_")
        compound = deep_compound or candidate_id.startswith(
            "bridge62_stage5hidden_")
        bridge_ratio = 0.625 if compound else ratio
        keep = (int(bridge_width) if bridge_width is not None else
                int(round(original_width * bridge_ratio)))
        selected = prune_conv_bn_deconv_bridge(
            owner.conv6, owner.dec5, keep)
        report = {
            "candidate_id": candidate_id,
            "ratio": bridge_ratio,
            "bridge_width": keep,
            "selected_channels": selected.tolist(),
        }
        if compound:
            stage5_ratio = 0.625 if deep_compound else ratio
            blocks = _prune_basic_blocks(owner.conv5, stage5_ratio)
            report.update({
                "stage5_hidden_ratio": stage5_ratio,
                "pruned_block_count": len(blocks),
                "stage5_blocks": blocks,
            })
        if deep_compound:
            stage4_blocks = _prune_basic_blocks(owner.conv4, ratio)
            report.update({
                "stage4_hidden_ratio": ratio,
                "pruned_block_count": (
                    report["pruned_block_count"] + len(stage4_blocks)),
                "stage4_blocks": stage4_blocks,
            })
        return report

    if model_name == "completionformer":
        deep_compound = candidate_id.startswith(
            "mlp62_s4hidden62_s3hidden_")
        compound = deep_compound or candidate_id.startswith(
            "mlp62_stage4hidden_")
        keep_ratio = (0.625 if compound else ratio) \
            if mlp_ratio is None else float(mlp_ratio)
        former = model.backbone.former
        widths = []
        for stage_name in ("block1", "block2", "block3", "block4"):
            for block_index, block in enumerate(getattr(former, stage_name)):
                original_width = int(block.mlp.fc1.out_features)
                keep = int(round(original_width * keep_ratio))
                selected = prune_mlp_hidden(block.mlp, keep)
                widths.append({
                    "block": "%s.%d" % (stage_name, block_index),
                    "original_width": original_width,
                    "hidden_width": keep,
                    "selected_channels": selected.tolist(),
                })
        report = {
            "candidate_id": candidate_id,
            "ratio": keep_ratio,
            "pruned_mlp_count": len(widths),
            "mlp_widths": widths,
        }
        if compound:
            stage4_ratio = 0.625 if deep_compound else ratio
            blocks = _prune_basic_blocks(
                [block.resblock for block in former.block4], stage4_ratio)
            report.update({
                "stage4_hidden_ratio": stage4_ratio,
                "pruned_block_count": len(blocks),
                "stage4_blocks": blocks,
            })
        if deep_compound:
            stage3_blocks = _prune_basic_blocks(
                [block.resblock for block in former.block3], ratio)
            report.update({
                "stage3_hidden_ratio": ratio,
                "pruned_block_count": (
                    report["pruned_block_count"] + len(stage3_blocks)),
                "stage3_blocks": stage3_blocks,
            })
        return report
    raise ValueError("unsupported model: %s" % model_name)


def _prepare_model(runtime, model_name, depth_candidate, depth_checkpoint):
    model = runtime.build_model(runtime.device)
    paths = depth_nas.STAGE_PATHS[model_name]
    stages = depth_nas.stage_modules(model, paths)
    depths = dict(depth_nas.CANDIDATE_DEPTHS[model_name])[depth_candidate]
    depth_nas.apply_depths(model, paths, stages, depths)
    if depth_checkpoint is not None:
        checkpoint = torch.load(str(depth_checkpoint),
                                map_location=runtime.device)
        if checkpoint.get("model") != model_name or \
                checkpoint.get("candidate_id") != depth_candidate:
            raise ValueError("depth checkpoint identity mismatch")
        model.load_state_dict(checkpoint["net"], strict=True)
    return model, depths


def _cosine(left, right):
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    denominator = float(left.norm().item() * right.norm().item())
    if denominator == 0.0:
        return 1.0 if bool(torch.equal(left, right)) else 0.0
    return max(-1.0, min(1.0, float(torch.dot(left, right).item()) /
                         denominator))


def _evaluate_candidate(runtime, model, dataset, indices, references):
    squared_error_sum = 0.0
    valid_pixels = 0
    sample_rmse = []
    cosines = []
    latencies = []
    predictions = {}
    model.eval()
    with torch.no_grad():
        for position, index in enumerate(indices):
            sample = default_collate([dataset[int(index)]])
            model_args, target = runtime.model_input(sample, runtime.device)
            if position == 0:
                runtime.prediction(model(*model_args))
                torch.cuda.synchronize(runtime.device)
            start = time.perf_counter()
            prediction = runtime.prediction(model(*model_args))
            torch.cuda.synchronize(runtime.device)
            latencies.append(1000.0 * (time.perf_counter() - start))
            valid = target > 0.0001
            difference = prediction[valid].double() - target[valid].double()
            sample_squared_error = float(torch.sum(
                difference * difference).item())
            sample_pixels = int(valid.sum().item())
            squared_error_sum += sample_squared_error
            valid_pixels += sample_pixels
            sample_rmse.append(math.sqrt(
                sample_squared_error / float(sample_pixels)))
            prediction_cpu = prediction.detach().float().cpu()
            predictions[int(index)] = prediction_cpu
            if references is not None:
                cosines.append(_cosine(
                    prediction_cpu, references[int(index)]))
    return {
        "pooled_rmse_m": math.sqrt(
            squared_error_sum / float(valid_pixels)),
        "mean_sample_rmse_m": sum(sample_rmse) / len(sample_rmse),
        "mean_output_cosine": (sum(cosines) / len(cosines)
                               if cosines else 1.0),
        "median_latency_ms": statistics.median(latencies),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "sample_count": len(indices),
        "valid_pixels": valid_pixels,
    }, predictions


def run(config_path, model_name, depth_candidate, depth_checkpoint,
        device, sample_count, output, selected_candidates=None):
    if output.exists():
        raise FileExistsError("screen output already exists: %s" % output)
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    payload = source["models"][model_name]
    configured_indices = tuple(
        int(value) for value in payload["evaluation_indices"])
    runtime = NYUModelRuntime.from_args(quant._runtime_args(payload, device))
    quant.configure_runtime_execution(runtime)
    try:
        dataset = runtime.build_dataset("val")
        indices = evaluation_indices(
            configured_indices, len(dataset), sample_count)
        references = None
        baseline_rmse = None
        baseline_parameters = None
        rows = []
        reports = {}
        depths = None
        for candidate_id in candidate_ids(model_name, selected_candidates):
            model, depths = _prepare_model(
                runtime, model_name, depth_candidate, depth_checkpoint)
            report = apply_structured_candidate(
                model, model_name, candidate_id)
            metrics, predictions = _evaluate_candidate(
                runtime, model, dataset, indices, references)
            parameters = depth_nas.parameter_count(model)
            if candidate_id == "baseline":
                references = predictions
                baseline_rmse = metrics["mean_sample_rmse_m"]
                baseline_parameters = parameters
            row = {
                "model": model_name,
                "depth_candidate": depth_candidate,
                "structured_candidate": candidate_id,
                "sample_count": metrics["sample_count"],
                "mean_sample_rmse_m": metrics["mean_sample_rmse_m"],
                "pooled_rmse_m": metrics["pooled_rmse_m"],
                "relative_rmse_pct": 100.0 * (
                    metrics["mean_sample_rmse_m"] / baseline_rmse - 1.0),
                "mean_output_cosine": metrics["mean_output_cosine"],
                "parameters": parameters,
                "parameter_reduction_pct": 100.0 * (
                    1.0 - parameters / float(baseline_parameters)),
                "median_latency_ms": metrics["median_latency_ms"],
                "mean_latency_ms": metrics["mean_latency_ms"],
            }
            rows.append(row)
            reports[candidate_id] = report
            print(json.dumps(row, sort_keys=True), flush=True)
            del model
            torch.cuda.empty_cache()

        output.mkdir(parents=True)
        with (output / "structured_screen.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "format_version": 1,
            "model": model_name,
            "depth_candidate": depth_candidate,
            "depths": list(depths),
            "depth_checkpoint": ("" if depth_checkpoint is None else
                                 str(depth_checkpoint.resolve())),
            "device": device,
            "sample_indices": list(indices),
            "candidates": reports,
            "rows": rows,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return output / "manifest.json"
    finally:
        runtime.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--depth-candidate", required=True)
    parser.add_argument("--depth-checkpoint", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--sample-count", type=int, default=64,
        help="configured sample prefix, or zero for the full validation set")
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.depth_candidate,
              args.depth_checkpoint, args.device, args.sample_count,
              args.output, args.candidates))


if __name__ == "__main__":
    main()
