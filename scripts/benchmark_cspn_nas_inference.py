#!/usr/bin/env python3
"""Benchmark CSPN NAS inference step counts and CUDA numerical modes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "models"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import evaluate_cspn_nas_official as evaluator
from spn_quant.nas.benchmark import (
    PRECISIONS,
    assert_gpu_idle,
    benchmark_model,
    gpu_environment,
    precision_context,
)


def parse_csv_choices(value: str, choices: Iterable[str], label: str) -> list[str]:
    allowed = set(choices)
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError("unsupported %s %r" % (label, item))
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise ValueError("at least one %s is required" % label)
    return parsed


def parse_positive_steps(value: str) -> list[int]:
    parsed = []
    for item in value.split(","):
        step = int(item.strip())
        if step <= 0:
            raise ValueError("CSPN steps must be positive")
        if step not in parsed:
            parsed.append(step)
    if not parsed:
        raise ValueError("at least one CSPN step count is required")
    return parsed


def prediction_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | bool]:
    difference = candidate.float() - reference.float()
    finite = bool(torch.isfinite(candidate).all().item())
    return {
        "finite": finite,
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
        "max_abs": float(torch.max(torch.abs(difference)).item()),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    with torch.inference_mode(), precision_context(
            precision, input_tensor.device.type):
        return model(input_tensor).detach()


def benchmark_configurations(
    run_dir: Path,
    *,
    checkpoint: str,
    device: torch.device,
    steps: Sequence[int],
    precisions: Sequence[str],
    warmup: int,
    iterations: int,
    repeats: int,
    allow_busy_gpu: bool,
) -> dict:
    if device.type != "cuda":
        raise ValueError("inference benchmark requires a CUDA device")
    environment = gpu_environment(device.index or 0)
    if not allow_busy_gpu:
        assert_gpu_idle(environment)
    torch.manual_seed(20260924)
    input_tensor = torch.randn(1, 4, 228, 304, device=device)
    rows = []
    for step in steps:
        model, _, checkpoint_path = evaluator.load_model(
            run_dir, device, checkpoint=checkpoint, cspn_steps=int(step))
        reference = _predict(model, input_tensor, "fp32")
        for precision in precisions:
            try:
                candidate = _predict(model, input_tensor, precision)
                error = prediction_error(reference, candidate)
                if not error["finite"]:
                    raise FloatingPointError("prediction contains non-finite values")
                timing = benchmark_model(
                    model, input_tensor, warmup=warmup,
                    iterations=iterations, repeats=repeats,
                    require_idle=False, environment=environment,
                    precision=precision)
                rows.append({
                    "cspn_steps": int(step),
                    "precision": precision,
                    "status": "ok",
                    "prediction_error": error,
                    **timing,
                })
            except (RuntimeError, ValueError, FloatingPointError) as error:
                rows.append({
                    "cspn_steps": int(step),
                    "precision": precision,
                    "status": "failed",
                    "reason": str(error),
                })
        del model
        torch.cuda.empty_cache()
    if not all(math.isfinite(float(row["median_ms"]))
               for row in rows if row["status"] == "ok"):
        raise RuntimeError("benchmark produced non-finite latency")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "environment": environment,
        "background_processes_accepted": bool(allow_busy_gpu),
        "protocol": {
            "batch_size": 1,
            "input_shape": list(input_tensor.shape),
            "warmup": int(warmup),
            "iterations": int(iterations),
            "repeats": int(repeats),
        },
        "results": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", default="24,18,12,8")
    parser.add_argument("--precisions", default=",".join(PRECISIONS))
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = benchmark_configurations(
        Path(args.run_dir), checkpoint=args.checkpoint,
        device=torch.device(args.device),
        steps=parse_positive_steps(args.steps),
        precisions=parse_csv_choices(
            args.precisions, PRECISIONS, "precision"),
        warmup=args.warmup, iterations=args.iterations,
        repeats=args.repeats, allow_busy_gpu=args.allow_busy_gpu)
    _atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
