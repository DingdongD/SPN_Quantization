#!/usr/bin/env python3
"""Prepare, rank, and benchmark the staged CSPN encoder NAS experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"
for path in (REPO_ROOT, MODELS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cspn_encoder_nas import build_cspn_nas
from spn_quant.nas.benchmark import (
    assert_gpu_idle,
    benchmark_model,
    gpu_environment,
)
from spn_quant.nas.search import make_search_split, select_successive_halving
from spn_quant.nas.spec import EncoderSpec, enumerate_encoder_specs
from spn_quant.nas.weights import transfer_prefix_state


FORMAT_VERSION = 1
ENCODER_MODULES = (
    "conv1_1", "bn1", "layer1", "layer2", "layer3", "layer4",
    "stem_skip_adapter", "stage1_skip_adapter", "stage2_skip_adapter",
    "bottleneck_adapter",
)
_FIXED_PARAMETER_COUNT: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def _stage_parameters(in_channels: int, out_channels: int, depth: int) -> int:
    if depth == 0:
        return in_channels * out_channels + 2 * out_channels
    total = in_channels * out_channels * 9 + 2 * out_channels
    total += out_channels * out_channels * 9 + 2 * out_channels
    if in_channels != out_channels:
        total += in_channels * out_channels + 2 * out_channels
    for _ in range(1, depth):
        total += 2 * out_channels * out_channels * 9 + 4 * out_channels
    return total


def _encoder_parameter_count(spec: EncoderSpec) -> int:
    total = 4 * spec.stem_width * 7 * 7 + 2 * spec.stem_width
    in_channels = spec.stem_width
    for out_channels, depth in zip(spec.widths, spec.depths):
        total += _stage_parameters(in_channels, out_channels, depth)
        in_channels = out_channels
    for in_channels, out_channels in zip(
        (spec.stem_width, spec.widths[0], spec.widths[1], spec.widths[3]),
        (64, 64, 128, 512),
    ):
        if in_channels != out_channels:
            total += in_channels * out_channels
    return total


def constructed_parameter_counts(spec: EncoderSpec) -> dict[str, int]:
    model = build_cspn_nas(spec, cspn_step=1)
    encoder = sum(
        parameter.numel()
        for name in ENCODER_MODULES
        for parameter in getattr(model, name).parameters()
    )
    full = sum(parameter.numel() for parameter in model.parameters())
    return {"encoder_parameters": encoder, "full_parameters": full}


def parameter_counts(spec: EncoderSpec) -> dict[str, int]:
    global _FIXED_PARAMETER_COUNT
    encoder = _encoder_parameter_count(spec)
    if _FIXED_PARAMETER_COUNT is None:
        control = constructed_parameter_counts(EncoderSpec.r18())
        _FIXED_PARAMETER_COUNT = (
            control["full_parameters"] - control["encoder_parameters"])
    return {
        "encoder_parameters": encoder,
        "full_parameters": encoder + _FIXED_PARAMETER_COUNT,
    }


def _csv_count(path: Path) -> int:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _identity(args: argparse.Namespace, total_count: int) -> dict[str, Any]:
    train_list = Path(args.train_list).resolve()
    checkpoint = Path(args.control_checkpoint).resolve()
    return {
        "format_version": FORMAT_VERSION,
        "seed": int(args.seed),
        "dev_count": int(args.dev_count),
        "total_count": int(total_count),
        "train_list": str(train_list),
        "train_list_sha256": _sha256(train_list),
        "control_checkpoint": str(checkpoint),
        "control_checkpoint_sha256": _sha256(checkpoint),
        "source_revision": _git_revision(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _ordered_specs(max_candidates: int) -> list[EncoderSpec]:
    control = EncoderSpec.r18()
    specs = [control]
    specs.extend(spec for spec in enumerate_encoder_specs() if spec != control)
    if max_candidates > 0:
        specs = specs[:max_candidates]
    return specs


def _training_command(
    args: argparse.Namespace, root: Path, spec_path: Path, spec: EncoderSpec,
) -> str:
    values = [
        args.python, str(REPO_ROOT / "scripts" / "train_nyu_iteration_sweep.py"),
        "--model", "cspn", "--iteration", "24", "--epochs", "5",
        "--device", "cuda:0", "--seed", str(args.seed),
        "--train-list", str(Path(args.train_list).resolve()),
        "--eval-list", str(Path(args.train_list).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--split-manifest", str((root / "split.json").resolve()),
        "--cspn-encoder-spec", str(spec_path.resolve()),
        "--cspn-control-checkpoint", str(Path(args.control_checkpoint).resolve()),
        "--save-root", str((root / "training").resolve()),
        "--run-name", "rung05-" + spec.slug,
    ]
    return " ".join(shlex.quote(str(value)) for value in values)


def _smoke_candidate(spec: EncoderSpec) -> dict[str, Any]:
    model = build_cspn_nas(spec, cspn_step=1).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 4, 228, 304))
    return {
        "candidate": spec.slug,
        "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
    }


def prepare_experiment(args: argparse.Namespace) -> dict[str, Any]:
    train_list = Path(args.train_list)
    checkpoint = Path(args.control_checkpoint)
    if not train_list.is_file():
        raise FileNotFoundError(train_list)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    total_count = _csv_count(train_list)
    identity = _identity(args, total_count)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "experiment.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError("experiment identity mismatch; use a new output root")
    else:
        _write_json(identity_path, identity)

    split = make_search_split(
        total_count, int(args.dev_count), seed=int(args.seed),
        source_list=train_list)
    split_path = root / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError("experiment identity mismatch: split changed")
    _write_json(split_path, split)

    rows = []
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    specs = _ordered_specs(int(args.max_candidates))
    for spec in specs:
        relative_spec = Path("candidates") / (spec.slug + ".json")
        spec_path = root / relative_spec
        _write_json(spec_path, spec.to_dict())
        counts = parameter_counts(spec)
        rows.append({
            "candidate": spec.slug,
            "spec_path": str(relative_spec),
            **counts,
            "status": "prepared",
        })
        commands.append(_training_command(args, root, spec_path, spec))
    _write_csv(
        root / "candidates.csv", rows,
        ["candidate", "spec_path", "encoder_parameters",
         "full_parameters", "status"])
    _atomic_text(root / "train_commands.sh", "\n".join(commands) + "\n")

    smoke_count = min(int(args.smoke_candidates), len(specs))
    smoke = [_smoke_candidate(spec) for spec in specs[:smoke_count]]
    if smoke:
        _write_json(root / "smoke.json", smoke)
    summary = {
        "output_root": str(root),
        "candidate_count": len(specs),
        "train_count": len(split["train_indices"]),
        "dev_count": len(split["dev_indices"]),
        "smoke_count": smoke_count,
    }
    _write_json(root / "prepare_summary.json", summary)
    return summary


def rank_results(results: Path, output: Path) -> list[dict[str, Any]]:
    with Path(results).open(newline="", encoding="utf-8") as stream:
        selected = select_successive_halving(csv.DictReader(stream))
    _write_json(Path(output), selected)
    return selected


def _load_checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "net" in payload:
        return payload["net"]
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint does not contain a state dictionary")
    return payload


def benchmark_candidates(
    root: Path,
    *,
    device: str = "cuda:0",
    limit: int = 0,
    warmup: int = 200,
    iterations: int = 1000,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    device_value = torch.device(device)
    if device_value.type != "cuda":
        raise ValueError("NAS latency benchmark requires a CUDA device")
    index = device_value.index or 0
    environment = gpu_environment(index)
    try:
        assert_gpu_idle(environment)
    except RuntimeError as error:
        _write_json(root / "benchmark_failure.json", {
            "status": "failed_preflight",
            "reason": str(error),
            "environment": environment,
        })
        raise

    identity = json.loads((root / "experiment.json").read_text())
    source_state = _load_checkpoint_state(Path(identity["control_checkpoint"]))
    with (root / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    if limit > 0:
        candidates = candidates[:limit]
    input_tensor = torch.randn(1, 4, 228, 304, device=device_value)
    results = []
    for row in candidates:
        spec = EncoderSpec.from_dict(json.loads(
            (root / row["spec_path"]).read_text(encoding="utf-8")))
        model = build_cspn_nas(spec, cspn_step=24)
        trained = root / "training" / ("rung05-" + spec.slug) / "best.pt"
        if trained.is_file():
            model.load_state_dict(_load_checkpoint_state(trained), strict=True)
            weight_source = str(trained)
        else:
            transfer_prefix_state(model, source_state)
            weight_source = identity["control_checkpoint"]
        model = model.to(device_value)
        metrics = benchmark_model(
            model, input_tensor, warmup=warmup, iterations=iterations,
            repeats=repeats, require_idle=False, environment=environment)
        results.append({
            "candidate": spec.slug,
            "median_ms": metrics["median_ms"],
            "p95_ms": metrics["p95_ms"],
            "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
            "weight_source": weight_source,
        })
        del model
        torch.cuda.empty_cache()
    _write_csv(
        root / "benchmark.csv", results,
        ["candidate", "median_ms", "p95_ms", "peak_cuda_memory_bytes",
         "weight_source"])
    _write_json(root / "benchmark_environment.json", environment)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train-list", required=True)
    prepare.add_argument("--control-checkpoint", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--seed", type=int, default=20260920)
    prepare.add_argument("--dev-count", type=int, default=670)
    prepare.add_argument("--max-candidates", type=int, default=0)
    prepare.add_argument("--smoke-candidates", type=int, default=3)
    prepare.add_argument("--python", default=sys.executable)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--output-root", required=True)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--limit", type=int, default=0)
    benchmark.add_argument("--warmup", type=int, default=200)
    benchmark.add_argument("--iterations", type=int, default=1000)
    benchmark.add_argument("--repeats", type=int, default=5)

    rank = subparsers.add_parser("rank")
    rank.add_argument("--results", required=True)
    rank.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_experiment(args)
    elif args.command == "benchmark":
        result = benchmark_candidates(
            Path(args.output_root), device=args.device, limit=args.limit,
            warmup=args.warmup, iterations=args.iterations,
            repeats=args.repeats)
    else:
        result = rank_results(Path(args.results), Path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
