#!/usr/bin/env python3
"""Screen weight-inheriting stage-depth subnets on paired NYU samples."""

from __future__ import annotations

import argparse
from argparse import Namespace
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant_runner  # noqa: E402


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")

STAGE_PATHS = {
    "cspn": ("layer1", "layer2", "layer3", "layer4"),
    "dyspn": ("base.conv2", "base.conv3", "base.conv4", "base.conv5"),
    "nlspn": ("conv2", "conv3", "conv4", "conv5"),
    "completionformer": (
        "backbone.former.embed_layer1", "backbone.former.embed_layer2",
        "backbone.former.block1", "backbone.former.block2",
        "backbone.former.block3", "backbone.former.block4",
    ),
}

CANDIDATE_DEPTHS = {
    "cspn": (
        ("baseline", (2, 2, 2, 2)),
        ("drop_s1", (1, 2, 2, 2)),
        ("drop_s2", (2, 1, 2, 2)),
        ("drop_s3", (2, 2, 1, 2)),
        ("drop_s4", (2, 2, 2, 1)),
        ("minimal", (1, 1, 1, 1)),
    ),
    "dyspn": (
        ("baseline", (3, 4, 6, 3)),
        ("drop_s1", (2, 4, 6, 3)),
        ("drop_s2", (3, 2, 6, 3)),
        ("drop_s3", (3, 4, 3, 3)),
        ("drop_s4", (3, 4, 6, 2)),
        ("r18_depth", (2, 2, 2, 2)),
    ),
    "nlspn": (
        ("baseline", (3, 4, 6, 3)),
        ("drop_s1", (2, 4, 6, 3)),
        ("drop_s2", (3, 2, 6, 3)),
        ("drop_s3", (3, 4, 3, 3)),
        ("drop_s4", (3, 4, 6, 2)),
        ("r18_depth", (2, 2, 2, 2)),
    ),
    "completionformer": (
        ("baseline", (3, 4, 3, 4, 6, 3)),
        ("drop_cnn_s1", (2, 4, 3, 4, 6, 3)),
        ("drop_cnn_s2", (3, 2, 3, 4, 6, 3)),
        ("drop_pvt_s1", (3, 4, 2, 4, 6, 3)),
        ("drop_pvt_s2", (3, 4, 3, 2, 6, 3)),
        ("drop_pvt_s3", (3, 4, 3, 4, 3, 3)),
        ("drop_pvt_s4", (3, 4, 3, 4, 6, 2)),
        ("drop_pvt_s3_s4", (3, 4, 3, 4, 3, 2)),
        ("tiny_depth", (2, 2, 1, 2, 2, 1)),
    ),
}


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    current = model
    for component in path.split("."):
        current = getattr(current, component)
    return current


def replace_module(model: nn.Module, path: str, module: nn.Module) -> None:
    components = path.split(".")
    parent = model
    for component in components[:-1]:
        parent = getattr(parent, component)
    setattr(parent, components[-1], module)


def stage_modules(model: nn.Module, paths: Sequence[str]) -> tuple[tuple[nn.Module, ...], ...]:
    stages = []
    for path in paths:
        module = resolve_module(model, path)
        if not isinstance(module, (nn.Sequential, nn.ModuleList)):
            raise TypeError("NAS stage is not a sequence: %s" % path)
        stages.append(tuple(module.children()))
    return tuple(stages)


def apply_depths(model: nn.Module, paths: Sequence[str],
                 full_stages: Sequence[Sequence[nn.Module]],
                 depths: Sequence[int]) -> None:
    if not (len(paths) == len(full_stages) == len(depths)):
        raise ValueError("stage path and depth dimensions differ")
    for path, modules, depth in zip(paths, full_stages, depths):
        if int(depth) < 1 or int(depth) > len(modules):
            raise ValueError("invalid depth %d for %s with %d blocks" %
                             (depth, path, len(modules)))
        original = resolve_module(model, path)
        selected = list(modules[:int(depth)])
        replacement = nn.ModuleList(selected) if isinstance(
            original, nn.ModuleList) else nn.Sequential(*selected)
        replace_module(model, path, replacement)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    denominator = float(left.norm().item() * right.norm().item())
    if denominator == 0.0:
        return 1.0 if bool(torch.equal(left, right)) else 0.0
    value = float(torch.dot(left, right).item()) / denominator
    return max(-1.0, min(1.0, value))


def evaluate_candidate(runtime: NYUModelRuntime, model: nn.Module,
                       dataset, indices: Sequence[int],
                       references: Mapping[int, torch.Tensor] | None) -> tuple[dict, dict]:
    squared_error_sum = 0.0
    valid_pixels = 0
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
            squared_error_sum += float(torch.sum(difference * difference).item())
            valid_pixels += int(valid.sum().item())
            prediction_cpu = prediction.detach().float().cpu()
            predictions[int(index)] = prediction_cpu
            if references is not None:
                cosines.append(_cosine(prediction_cpu, references[int(index)]))
    metrics = {
        "pooled_rmse_m": math.sqrt(squared_error_sum / float(valid_pixels)),
        "mean_output_cosine": sum(cosines) / len(cosines) if cosines else 1.0,
        "median_latency_ms": statistics.median(latencies),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "sample_count": len(indices),
        "valid_pixels": valid_pixels,
    }
    return metrics, predictions


def run(config_path: Path, model_name: str, device: str,
        sample_count: int, output: Path) -> Path:
    if model_name not in MODEL_ORDER:
        raise ValueError("unsupported model: %s" % model_name)
    if output.exists():
        raise FileExistsError("screen output already exists: %s" % output)
    config = quant_runner._load_run_config(config_path)
    source = quant_runner._model_source(config_path, config)
    model_payload = source["models"][model_name]
    all_indices = tuple(int(value) for value in model_payload["evaluation_indices"])
    if sample_count < 1 or sample_count > len(all_indices):
        raise ValueError("sample count must be within configured evaluation set")
    indices = all_indices[:sample_count]
    runtime = NYUModelRuntime.from_args(
        quant_runner._runtime_args(model_payload, device))
    quant_runner.configure_runtime_execution(runtime)
    try:
        model = runtime.build_model(runtime.device)
        dataset = runtime.build_dataset("val")
        paths = STAGE_PATHS[model_name]
        full_stages = stage_modules(model, paths)
        expected = CANDIDATE_DEPTHS[model_name][0][1]
        observed = tuple(len(stage) for stage in full_stages)
        if observed != expected:
            raise RuntimeError("baseline depths differ: %s != %s" %
                               (observed, expected))
        rows = []
        references = None
        baseline_rmse = None
        baseline_parameters = None
        for candidate_id, depths in CANDIDATE_DEPTHS[model_name]:
            apply_depths(model, paths, full_stages, depths)
            metrics, predictions = evaluate_candidate(
                runtime, model, dataset, indices, references)
            parameters = parameter_count(model)
            if candidate_id == "baseline":
                references = predictions
                baseline_rmse = metrics["pooled_rmse_m"]
                baseline_parameters = parameters
            row = {
                "model": model_name,
                "candidate_id": candidate_id,
                "depths": ";".join(str(value) for value in depths),
                "sample_count": metrics["sample_count"],
                "pooled_rmse_m": metrics["pooled_rmse_m"],
                "relative_rmse_pct": 100.0 * (
                    metrics["pooled_rmse_m"] / baseline_rmse - 1.0),
                "mean_output_cosine": metrics["mean_output_cosine"],
                "parameters": parameters,
                "parameter_reduction_pct": 100.0 * (
                    1.0 - parameters / float(baseline_parameters)),
                "median_latency_ms": metrics["median_latency_ms"],
                "mean_latency_ms": metrics["mean_latency_ms"],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        output.mkdir(parents=True)
        with (output / "depth_screen.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "model": model_name,
            "device": device,
            "checkpoint": str(runtime.checkpoint),
            "evaluation_indices": list(indices),
            "stage_paths": list(paths),
            "screen_type": "inherited-weight depth ablation; no fine-tuning",
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output / "depth_screen.csv"
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.device, args.sample_count,
              args.output))


if __name__ == "__main__":
    main()
