#!/usr/bin/env python3
"""Audit full-validation and fixed-subset RMSE aggregation protocols."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")


@dataclass(frozen=True)
class SampleError:
    squared_error_sum: float
    valid_pixels: int

    @property
    def rmse(self) -> float:
        return math.sqrt(self.squared_error_sum / float(self.valid_pixels))


def aggregate_sample_errors(rows: Iterable[SampleError]) -> dict:
    rows = tuple(rows)
    if not rows:
        raise ValueError("at least one sample error is required")
    squared_error_sum = sum(row.squared_error_sum for row in rows)
    valid_pixels = sum(row.valid_pixels for row in rows)
    if valid_pixels <= 0:
        raise ValueError("evaluation contains no valid depth pixels")
    return {
        "sample_count": len(rows),
        "valid_pixels": valid_pixels,
        "mean_sample_rmse_m": sum(row.rmse for row in rows) / len(rows),
        "pooled_pixel_rmse_m": math.sqrt(
            squared_error_sum / float(valid_pixels)),
    }


def evaluate(runtime: NYUModelRuntime, model: torch.nn.Module, dataset,
             fixed_indices: Sequence[int]) -> dict:
    fixed = frozenset(int(index) for index in fixed_indices)
    if len(fixed) != len(fixed_indices):
        raise ValueError("fixed evaluation indices contain duplicates")
    if not fixed or min(fixed) < 0 or max(fixed) >= len(dataset):
        raise ValueError("fixed evaluation indices are outside the dataset")

    full_rows = []
    fixed_rows = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = default_collate([dataset[index]])
            model_args, target = runtime.model_input(sample, runtime.device)
            prediction = runtime.prediction(model(*model_args))
            valid = target > 0.0001
            pred = prediction[valid].double().clamp_min(1e-6)
            truth = target[valid].double()
            difference = pred - truth
            row = SampleError(
                squared_error_sum=float(difference.square().sum().item()),
                valid_pixels=int(valid.sum().item()),
            )
            full_rows.append(row)
            if index in fixed:
                fixed_rows.append(row)
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print("evaluated %d/%d" % (index + 1, len(dataset)), flush=True)

    if len(fixed_rows) != len(fixed):
        raise RuntimeError("fixed evaluation subset was not fully evaluated")
    return {
        "full_validation": aggregate_sample_errors(full_rows),
        "fixed_subset": aggregate_sample_errors(fixed_rows),
    }


def run(config_path: Path, model_name: str, device: str,
        output: Path) -> Path:
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    model_payload = source["models"][model_name]
    runtime = NYUModelRuntime.from_args(
        quant._runtime_args(model_payload, device))
    quant.configure_runtime_execution(runtime)
    try:
        model = runtime.build_model(runtime.device)
        dataset = runtime.build_dataset("val")
        metrics = evaluate(
            runtime, model, dataset, model_payload["evaluation_indices"])
        payload = {
            "format_version": 1,
            "model": model_name,
            "checkpoint": str(runtime.checkpoint),
            "dataset_samples": len(dataset),
            "fixed_evaluation_indices": [
                int(index) for index in model_payload["evaluation_indices"]],
            **metrics,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return output
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.device, args.output))


if __name__ == "__main__":
    main()
