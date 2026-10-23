#!/usr/bin/env python3
"""Run all four official vanilla SPN latency workers on one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spn_quant.gpu_latency import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
    validate_four_model_results,
    write_latency_reports,
)


def load_benchmark_config(path: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("unsupported GPU latency config format")
    if tuple(payload.get("models", {})) != MODEL_ORDER:
        raise ValueError("GPU latency config model order is incomplete")
    measurement = payload.get("measurement", {})
    required = {
        "precision": "fp32",
        "batch_size": 1,
        "input_residency": "gpu",
        "includes_propagation": True,
        "tf32": False,
        "python_gc": True,
        "cudnn_benchmark": True,
    }
    for key, expected in required.items():
        if measurement.get(key) != expected:
            raise ValueError("invalid measurement contract field: %s" % key)
    for key in ("warmup_iterations", "timed_iterations_per_round", "rounds"):
        if int(measurement.get(key, 0)) <= 0:
            raise ValueError("measurement count must be positive: %s" % key)
    if int(measurement["rounds"]) < 2:
        raise ValueError("at least two timed rounds are required")
    source = Path(payload["model_source_config"])
    if not source.is_absolute():
        source = Path(repo_root) / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError("model source config not found: %s" % source)
    payload["model_source_config"] = source
    for model_name in MODEL_ORDER:
        executable = Path(payload["models"][model_name]["python_executable"])
        if not executable.is_file():
            raise FileNotFoundError("model Python not found: %s" % executable)
    return payload


def build_worker_environment(
        base: Mapping[str, str], model_spec: Mapping[str, Any]) -> Dict[str, str]:
    environment = dict(base)
    for key, value in model_spec.get("environment", {}).items():
        value = str(value)
        if key == "LD_LIBRARY_PATH" and environment.get(key):
            value = value + os.pathsep + environment[key]
        environment[str(key)] = value
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def build_worker_command(
        repo_root: Path, benchmark_config_path: Path,
        config: Mapping[str, Any], model_name: str, device: str,
        output_path: Path, fixed_inputs: Path = None) -> Sequence[str]:
    del benchmark_config_path
    measurement = config["measurement"]
    model_spec = config["models"][model_name]
    command = [
        str(model_spec["python_executable"]),
        str(Path(repo_root) / "scripts" /
            "benchmark_nyu_vanilla_gpu_worker.py"),
        "--config", str(config["model_source_config"]),
        "--model", model_name,
        "--device", str(device),
        "--warmup-iterations", str(measurement["warmup_iterations"]),
        "--timed-iterations", str(
            measurement["timed_iterations_per_round"]),
        "--rounds", str(measurement["rounds"]),
        "--output", str(output_path),
    ]
    if measurement["python_gc"] is False:
        command.append("--disable-python-gc")
    if fixed_inputs is not None:
        command.extend(["--fixed-inputs", str(Path(fixed_inputs).resolve())])
    return command


def _gpu_state(index: int) -> Dict[str, Any]:
    command = [
        "nvidia-smi", "--id=%d" % index,
        "--query-gpu=index,name,uuid,utilization.gpu,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 6:
        raise RuntimeError("unexpected nvidia-smi GPU state")
    return {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "utilization_percent": int(fields[3]),
        "memory_used_mb": int(fields[4]), "temperature_c": int(fields[5]),
    }


def _write_plots(output_dir: Path, results: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = validate_four_model_results(results)
    labels = [MODEL_LABELS[item["model"]] for item in ordered]
    medians = [item["latency_ms"]["p50"] for item in ordered]
    p95 = [item["latency_ms"]["p95"] for item in ordered]
    lower = [medians[i] - item["latency_ms"]["min"]
             for i, item in enumerate(ordered)]
    upper = [p95[i] - medians[i] for i in range(len(ordered))]
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    bars = axis.bar(labels, medians, color="#2878b5")
    axis.errorbar(
        range(len(labels)), medians, yerr=[lower, upper], fmt="none",
        ecolor="#c82423", capsize=5, label="min to P95")
    axis.bar_label(bars, fmt="%.2f ms", padding=3)
    axis.set_ylabel("Full-forward latency (ms)")
    axis.set_title("Vanilla FP32 SPN latency on %s" % ordered[0]["gpu"]["name"])
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "latency_comparison.png", dpi=180)
    plt.close(figure)

    peaks = [item["memory_mb"]["peak_allocated"] for item in ordered]
    workspaces = [item["memory_mb"]["inference_workspace_peak"]
                  for item in ordered]
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    x_positions = list(range(len(labels)))
    width = 0.36
    axis.bar([value - width / 2 for value in x_positions], peaks,
             width=width, label="peak allocated", color="#2f9e44")
    axis.bar([value + width / 2 for value in x_positions], workspaces,
             width=width, label="forward workspace", color="#f08c00")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("CUDA memory (MiB)")
    axis.set_title("Vanilla FP32 SPN CUDA memory")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "memory_comparison.png", dpi=180)
    plt.close(figure)


def run_benchmark(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = load_benchmark_config(config_path, REPO_ROOT)
    output_dir = Path(args.output).resolve()
    if output_dir.exists():
        raise FileExistsError("benchmark output already exists: %s" % output_dir)
    initial_gpu = _gpu_state(args.gpu_index)
    if not args.allow_busy_gpu and initial_gpu["utilization_percent"] > 5:
        raise RuntimeError(
            "selected GPU is busy (%d%%): cuda:%d" %
            (initial_gpu["utilization_percent"], args.gpu_index))
    output_dir.mkdir(parents=True)
    per_model = output_dir / "per_model"
    per_model.mkdir()
    logs = output_dir / "logs"
    logs.mkdir()
    device = "cuda:%d" % args.gpu_index
    results = []
    for model_name in MODEL_ORDER:
        result_path = per_model / (model_name + ".json")
        command = build_worker_command(
            REPO_ROOT, config_path, config, model_name, device, result_path,
            fixed_inputs=getattr(args, "fixed_inputs", None))
        environment = build_worker_environment(
            os.environ, config["models"][model_name])
        completed = subprocess.run(
            command, cwd=str(REPO_ROOT), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        (logs / (model_name + ".log")).write_text(
            completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                "%s worker failed with exit code %d; see %s" % (
                    model_name, completed.returncode,
                    logs / (model_name + ".log")))
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
        print("%s: %.3f ms mean, %.3f ms P95" % (
            MODEL_LABELS[model_name], results[-1]["latency_ms"]["mean"],
            results[-1]["latency_ms"]["p95"]), flush=True)
    validate_four_model_results(results)
    write_latency_reports(output_dir, results)
    _write_plots(output_dir, results)
    final_gpu = _gpu_state(args.gpu_index)
    run_metadata = {
        "format_version": 1,
        "config": str(config_path),
        "model_source_config": str(config["model_source_config"]),
        "initial_gpu_state": initial_gpu,
        "final_gpu_state": final_gpu,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "four_spn_vanilla_gpu_latency.json")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--fixed-inputs", type=Path)
    return parser


def main(argv: Sequence[str] = None) -> int:
    output = run_benchmark(_parser().parse_args(argv))
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
