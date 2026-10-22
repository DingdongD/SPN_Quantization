"""Shared contracts and reports for SPN GPU latency measurements."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
MODEL_LABELS = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}
MEASUREMENT_SCOPE = "forward_only_gpu_resident_inputs"


def _linear_percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) +
                 sorted_values[upper] * fraction)


def summarize_latencies(samples_ms: Iterable[float]) -> Dict[str, float]:
    """Summarize positive CUDA Event measurements using linear percentiles."""
    values = [float(value) for value in samples_ms]
    if not values or any(
            not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("latency samples must be finite positive values")
    ordered = sorted(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": ordered[0],
        "p50": _linear_percentile(ordered, 0.50),
        "p90": _linear_percentile(ordered, 0.90),
        "p95": _linear_percentile(ordered, 0.95),
        "p99": _linear_percentile(ordered, 0.99),
        "max": ordered[-1],
        "fps_from_mean": 1000.0 / mean,
    }


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _require_fp32(tensor: torch.Tensor, owner: str) -> None:
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        raise RuntimeError("%s has dtype %s; strict FP32 is required" %
                           (owner, tensor.dtype))


def audit_fp32_execution(model: torch.nn.Module, model_args: Any) -> Dict[str, Any]:
    """Verify every floating model/input tensor is FP32 and report its size."""
    parameter_count = 0
    parameter_size_bytes = 0
    parameter_tensors = 0
    for name, parameter in model.named_parameters():
        _require_fp32(parameter, "parameter %s" % name)
        if parameter.is_floating_point():
            parameter_tensors += 1
            parameter_count += int(parameter.numel())
            parameter_size_bytes += int(parameter.numel() * parameter.element_size())
    floating_buffers = 0
    for name, buffer in model.named_buffers():
        _require_fp32(buffer, "buffer %s" % name)
        if buffer.is_floating_point():
            floating_buffers += 1
    input_tensors = tuple(_iter_tensors(model_args))
    if not input_tensors:
        raise RuntimeError("model execution has no tensor inputs")
    for index, tensor in enumerate(input_tensors):
        _require_fp32(tensor, "input %d" % index)
    return {
        "precision": "fp32",
        "parameter_tensor_count": parameter_tensors,
        "parameter_count": parameter_count,
        "parameter_size_bytes": parameter_size_bytes,
        "floating_buffer_count": floating_buffers,
        "input_tensor_count": len(input_tensors),
        "input_shapes": [list(tensor.shape) for tensor in input_tensors],
    }


def _same_settings(result: Mapping[str, Any]) -> Tuple[Any, ...]:
    settings = result["settings"]
    return tuple(settings[key] for key in (
        "warmup_iterations", "timed_iterations_per_round", "rounds",
        "tf32", "cudnn_benchmark", "python_gc_enabled_during_timing"))


def validate_four_model_results(
        results: Sequence[Mapping[str, Any]]) -> Tuple[Mapping[str, Any], ...]:
    """Validate comparability and return results in canonical model order."""
    if len(results) != len(MODEL_ORDER):
        raise ValueError("benchmark must contain exactly four model results")
    by_model = {str(result.get("model")): result for result in results}
    if set(by_model) != set(MODEL_ORDER) or len(by_model) != len(results):
        raise ValueError("benchmark results must cover each SPN model exactly once")
    ordered = tuple(by_model[name] for name in MODEL_ORDER)
    uuids = set()
    settings = set()
    for result in ordered:
        if result.get("precision") != "fp32":
            raise ValueError("all vanilla benchmark results must be strict FP32")
        if result.get("measurement_scope") != MEASUREMENT_SCOPE:
            raise ValueError("benchmark measurement scope is inconsistent")
        if result.get("batch_size") != 1:
            raise ValueError("benchmark batch size must equal one")
        if result.get("includes_propagation") is not True:
            raise ValueError("full-model benchmark must include propagation")
        uuid = str(result.get("gpu", {}).get("uuid", ""))
        if not uuid:
            raise ValueError("benchmark result is missing the physical GPU UUID")
        uuids.add(uuid)
        settings.add(_same_settings(result))
        latency = result.get("latency_ms", {})
        expected_count = int(result["settings"]["timed_iterations_per_round"]) * \
            int(result["settings"]["rounds"])
        if int(latency.get("count", -1)) != expected_count:
            raise ValueError("latency sample count differs from benchmark settings")
        for key in ("mean", "std", "min", "p50", "p90", "p95",
                    "p99", "max", "fps_from_mean"):
            value = float(latency.get(key, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("invalid latency statistic: %s" % key)
    if len(uuids) != 1:
        raise ValueError("all models must be measured on the same physical GPU")
    if len(settings) != 1:
        raise ValueError("all models must use identical benchmark settings")
    return ordered


def _summary_rows(results: Sequence[Mapping[str, Any]]) -> Sequence[Dict[str, Any]]:
    rows = []
    for result in validate_four_model_results(results):
        latency = result["latency_ms"]
        memory = result["memory_mb"]
        rows.append({
            "model": result["model"],
            "model_label": MODEL_LABELS[result["model"]],
            "architecture_class": result["architecture_class"],
            "propagation_iterations": result["propagation_iterations"],
            "parameter_count": result["parameter_count"],
            "parameter_size_mb": result["parameter_size_mb"],
            "mean_ms": latency["mean"],
            "std_ms": latency["std"],
            "min_ms": latency["min"],
            "p50_ms": latency["p50"],
            "p90_ms": latency["p90"],
            "p95_ms": latency["p95"],
            "p99_ms": latency["p99"],
            "max_ms": latency["max"],
            "fps": latency["fps_from_mean"],
            "baseline_allocated_mb": memory["baseline_allocated"],
            "peak_allocated_mb": memory["peak_allocated"],
            "peak_reserved_mb": memory["peak_reserved"],
            "inference_workspace_peak_mb": memory["inference_workspace_peak"],
        })
    return rows


def _atomic_text(path: Path, payload: str) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_latency_reports(
        output_dir: Path,
        results: Sequence[Mapping[str, Any]]) -> Dict[str, Path]:
    """Persist canonical CSV, Markdown, and full JSON benchmark artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = validate_four_model_results(results)
    rows = _summary_rows(ordered)
    csv_path = output_dir / "latency_summary.csv"
    temporary_csv = Path(str(csv_path) + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary_csv), str(csv_path))

    first = ordered[0]
    gc_state = "enabled" if first["settings"][
        "python_gc_enabled_during_timing"] else "disabled"
    lines = [
        "# Four-SPN vanilla FP32 GPU latency",
        "",
        "This report measures forward-only latency with GPU-resident inputs. ",
        "Dataset decoding, host-to-device transfer, checkpoint loading, and model ",
        "construction are excluded. Each full forward includes the model's native ",
        "propagation/refinement path.",
        "",
        "- GPU: `%s` (`%s`)" %
        (first["gpu"]["name"], first["gpu"]["uuid"]),
        "- Precision: strict FP32 (TF32 disabled)",
        "- Runtime: Python cyclic GC %s during timing" % gc_state,
        "- Batch/input: batch 1, NYU 228x304, validation identity %s" %
        first["sample_index"],
        "- Timing: CUDA Events; %s warmups, %sx%s measured forwards" % (
            first["settings"]["warmup_iterations"],
            first["settings"]["rounds"],
            first["settings"]["timed_iterations_per_round"]),
        "- Request mode: online batch-1, each request synchronized",
        "",
        "| Model | Prop. iters | Params (M) | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | FPS | Peak alloc. (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.1f |" % (
                row["model_label"], int(row["propagation_iterations"]),
                int(row["parameter_count"]) / 1.0e6, float(row["mean_ms"]),
                float(row["p50_ms"]), float(row["p95_ms"]),
                float(row["p99_ms"]), float(row["fps"]),
                float(row["peak_allocated_mb"])))
    lines.extend([
        "",
        "`P50` is the typical latency, while `P95/P99` expose run-to-run tail ",
        "variation. Framework and custom CUDA-extension versions are preserved ",
        "per model in `benchmark_manifest.json`.",
        "",
    ])
    markdown_path = output_dir / "latency_report.md"
    _atomic_text(markdown_path, "\n".join(lines))

    manifest_path = output_dir / "benchmark_manifest.json"
    manifest = {
        "format_version": 1,
        "measurement_scope": MEASUREMENT_SCOPE,
        "model_order": list(MODEL_ORDER),
        "results": list(ordered),
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "csv": csv_path,
        "markdown": markdown_path,
        "manifest": manifest_path,
    }


__all__ = [
    "MEASUREMENT_SCOPE", "MODEL_LABELS", "MODEL_ORDER",
    "audit_fp32_execution", "summarize_latencies",
    "validate_four_model_results", "write_latency_reports",
]
