"""Strict CPU-only timing and profiler helpers for vanilla SPN models."""

from __future__ import annotations

from collections import defaultdict
import gc
import math
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from spn_quant.gpu_latency import summarize_latencies


CATEGORIES = (
    "dense_compute",
    "attention",
    "sampling",
    "propagation",
    "elementwise_norm",
    "layout_memory",
    "other_unclassified",
)


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def audit_cpu_execution(
        model: torch.nn.Module, model_args: Sequence[Any]) -> Dict[str, Any]:
    """Require every parameter, buffer, and input tensor to reside on CPU."""
    tensor_count = 0
    for owner, tensors in (
            ("parameter", model.parameters()),
            ("buffer", model.buffers()),
            ("input", _iter_tensors(model_args))):
        for tensor in tensors:
            tensor_count += 1
            if tensor.device.type != "cpu":
                raise RuntimeError("%s tensor is not resident on CPU" % owner)
    if tensor_count == 0:
        raise RuntimeError("CPU execution audit found no tensors")
    return {
        "execution_device": "cpu",
        "audited_tensor_count": tensor_count,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def measure_cpu_forward(
        model: torch.nn.Module, model_args: Sequence[Any],
        warmup_iterations: int, timed_iterations: int, rounds: int,
        disable_python_gc: bool = False,
        perf_counter_ns=time.perf_counter_ns) -> Dict[str, Any]:
    """Measure serial complete CPU forwards with a monotonic wall clock."""
    if warmup_iterations <= 0 or timed_iterations <= 0 or rounds <= 0:
        raise ValueError("warmup, timed iterations, and rounds must be positive")
    model.eval()
    gc_was_enabled = gc.isenabled()
    if disable_python_gc and gc_was_enabled:
        gc.disable()
    try:
        with torch.inference_mode():
            for _ in range(int(warmup_iterations)):
                output = model(*model_args)
            del output
            samples = []
            round_summaries = []
            for round_index in range(int(rounds)):
                round_samples = []
                for _ in range(int(timed_iterations)):
                    start_ns = int(perf_counter_ns())
                    output = model(*model_args)
                    end_ns = int(perf_counter_ns())
                    elapsed_ms = (end_ns - start_ns) / 1.0e6
                    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                        raise RuntimeError("CPU wall clock returned invalid latency")
                    round_samples.append(elapsed_ms)
                    del output
                samples.extend(round_samples)
                summary = summarize_latencies(round_samples)
                round_summaries.append({
                    "round": round_index + 1,
                    "count": summary["count"],
                    "mean_ms": summary["mean"],
                    "p50_ms": summary["p50"],
                    "p95_ms": summary["p95"],
                    "min_ms": summary["min"],
                    "max_ms": summary["max"],
                })
        return {
            "latency_ms": summarize_latencies(samples),
            "raw_latency_ms": samples,
            "rounds": round_summaries,
        }
    finally:
        if disable_python_gc and gc_was_enabled:
            gc.enable()


def run_cpu_energy_workload(
        model: torch.nn.Module, model_args: Sequence[Any],
        active_seconds: float, monotonic=time.monotonic,
        monotonic_ns=time.monotonic_ns,
        perf_counter_ns=time.perf_counter_ns) -> Dict[str, Any]:
    """Run serial complete CPU forwards for a measured active interval."""
    if not math.isfinite(float(active_seconds)) or active_seconds <= 0.0:
        raise ValueError("active_seconds must be positive")
    active_start_ns = int(monotonic_ns())
    wall_start = float(monotonic())
    samples = []
    with torch.inference_mode():
        while float(monotonic()) - wall_start < float(active_seconds):
            start_ns = int(perf_counter_ns())
            output = model(*model_args)
            end_ns = int(perf_counter_ns())
            elapsed_ms = (end_ns - start_ns) / 1.0e6
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                raise RuntimeError("CPU wall clock returned invalid latency")
            samples.append(elapsed_ms)
            del output
    active_end_ns = int(monotonic_ns())
    return {
        "active_start_monotonic_ns": active_start_ns,
        "active_end_monotonic_ns": active_end_ns,
        "active_wall_seconds": (active_end_ns - active_start_ns) / 1.0e9,
        "completed_forwards": len(samples),
        "raw_latency_ms": samples,
        "latency_ms": summarize_latencies(samples),
    }


def classify_operator(name: str, ancestor_names: Sequence[str] = ()) -> str:
    """Map one CPU operator to one exclusive architecture category."""
    ancestors = tuple(str(value) for value in ancestor_names)
    if "SPN_PROPAGATION" in ancestors:
        return "propagation"
    if "SPN_ATTENTION" in ancestors:
        return "attention"
    value = str(name).lower()
    if any(token in value for token in (
            "softmax", "scaled_dot_product", "attention", "bmm")):
        return "attention"
    if any(token in value for token in (
            "grid_sampler", "grid_sample", "gather", "deform", "upsample",
            "interpolate", "index_select", "embedding", "max_pool")):
        return "sampling"
    if any(token in value for token in (
            "batch_norm", "batchnorm", "layer_norm", "normalization")):
        return "elementwise_norm"
    if any(token in value for token in (
            "conv", "gemm", "matmul", "linear", "addmm", "mm")):
        return "dense_compute"
    if any(token in value for token in (
            "cat", "copy", "contiguous", "permute", "transpose", "clone",
            "resize", "to_copy", "view", "reshape", "flatten", "pad",
            "slice", "select", "narrow", "as_strided", "unsqueeze",
            "squeeze", "empty", "zero", "zeros")):
        return "layout_memory"
    if any(token in value for token in (
            "relu", "sigmoid", "tanh", "mul", "add", "sub", "div",
            "norm", "normalize", "exp", "sqrt", "clamp", "where", "sum",
            "mean", "max", "min", "pow", "abs", "fill", "gelu")):
        return "elementwise_norm"
    return "other_unclassified"


def _ancestor_names(event: Any) -> tuple:
    names = []
    parent = getattr(event, "cpu_parent", None)
    while parent is not None:
        names.append(str(parent.name))
        parent = getattr(parent, "cpu_parent", None)
    return tuple(names)


def summarize_profile_events(
        events: Sequence[Any], profiled_iterations: int) -> Dict[str, Any]:
    """Aggregate exclusive operator self CPU time per profiled forward."""
    if int(profiled_iterations) <= 0:
        raise ValueError("profiled_iterations must be positive")
    sums = defaultdict(float)
    counts = defaultdict(int)
    operator_sums = defaultdict(float)
    classified = 0
    negative_values = []
    for event in events:
        name = str(event.name)
        if not (name.startswith("aten::") or
                name.startswith("torchvision::")):
            continue
        self_time = float(event.self_cpu_time_total)
        if not math.isfinite(self_time):
            raise ValueError(
                "profiler self CPU time must be finite and nonnegative: "
                "%s=%r" % (name, self_time))
        if self_time < 0.0:
            negative_values.append(self_time)
        category = classify_operator(name, _ancestor_names(event))
        sums[category] += self_time
        counts[category] += 1
        operator_sums[name] += self_time
        classified += 1
    means = {
        category: sums[category] / int(profiled_iterations)
        for category in CATEGORIES
    }
    negative_categories = {
        category: value for category, value in means.items() if value < 0.0
    }
    if negative_categories:
        raise ValueError(
            "profiler aggregated category time must be nonnegative: %r" %
            negative_categories)
    total = sum(means.values())
    return {
        "profiled_iterations": int(profiled_iterations),
        "classified_event_count": classified,
        "category_mean_self_cpu_time_us": means,
        "category_event_count": {
            category: counts[category] for category in CATEGORIES
        },
        "category_share_percent": {
            category: (100.0 * value / total if total else 0.0)
            for category, value in means.items()
        },
        "self_cpu_time_sum_mean_us": total,
        "negative_self_time_events": {
            "count": len(negative_values),
            "total_magnitude_us": -sum(negative_values),
            "minimum_us": min(negative_values) if negative_values else 0.0,
        },
        "top_operator_mean_self_cpu_time_us": [
            {"operator": name, "mean_self_cpu_time_us": value /
             int(profiled_iterations)}
            for name, value in sorted(
                operator_sums.items(), key=lambda item: item[1], reverse=True)[:50]
        ],
    }
