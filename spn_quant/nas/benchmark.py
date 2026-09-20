"""Repeatable CUDA-event benchmarking for CSPN NAS candidates."""

from __future__ import annotations

from contextlib import contextmanager
import os
import statistics
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class CudaEventTimer:
    def measure(self, callback: Callable[[], Any]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callback()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))


def _run_nvidia_smi(arguments: Sequence[str]) -> list[str]:
    output = subprocess.check_output(
        ["nvidia-smi", *arguments], stderr=subprocess.STDOUT,
    ).decode("utf-8")
    return [line.strip() for line in output.splitlines() if line.strip()]


def gpu_environment(device_index: int = 0) -> dict[str, Any]:
    rows = _run_nvidia_smi([
        "--query-gpu=index,uuid,name,driver_version,memory.total,"
        "memory.used,utilization.gpu,temperature.gpu,pstate,clocks.current.sm",
        "--format=csv,noheader,nounits",
    ])
    selected = None
    for row in rows:
        values = [value.strip() for value in row.split(",")]
        if int(values[0]) == int(device_index):
            selected = values
            break
    if selected is None:
        raise RuntimeError("GPU index %d was not reported by nvidia-smi" % device_index)
    environment = {
        "index": int(selected[0]),
        "uuid": selected[1],
        "name": selected[2],
        "driver_version": selected[3],
        "memory_total_mib": int(selected[4]),
        "memory_used_mib": int(selected[5]),
        "utilization_percent": int(selected[6]),
        "temperature_c": int(selected[7]),
        "pstate": selected[8],
        "sm_clock_mhz": int(selected[9]),
        "processes": [],
    }
    process_rows = _run_nvidia_smi([
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    for row in process_rows:
        values = [value.strip() for value in row.split(",")]
        if values[0] != environment["uuid"]:
            continue
        environment["processes"].append({
            "pid": int(values[1]),
            "process_name": values[2],
            "used_memory_mib": int(values[3]),
        })
    return environment


def assert_gpu_idle(
    environment: Mapping[str, Any],
    *,
    current_pid: Optional[int] = None,
) -> None:
    current_pid = os.getpid() if current_pid is None else int(current_pid)
    others = [
        process for process in environment.get("processes", [])
        if int(process["pid"]) != current_pid
    ]
    if others:
        raise RuntimeError(
            "GPU has other compute processes: %s" %
            ", ".join(str(process["pid"]) for process in others))


@contextmanager
def _strict_fp32():
    previous_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    previous_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)
    if previous_matmul is not None:
        torch.backends.cuda.matmul.allow_tf32 = False
    if previous_cudnn is not None:
        torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        if previous_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        if previous_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = previous_cudnn


def _prediction(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        output = output.get("pred")
    if not torch.is_tensor(output):
        raise TypeError("model output is not a prediction tensor")
    return output


def _validate_output(output: Any, expected_shape: Sequence[int]) -> torch.Tensor:
    prediction = _prediction(output)
    if tuple(prediction.shape) != tuple(expected_shape):
        raise RuntimeError(
            "model output shape %s does not match %s" %
            (tuple(prediction.shape), tuple(expected_shape)))
    if not bool(torch.isfinite(prediction).all().item()):
        raise FloatingPointError("model output contains non-finite values")
    return prediction


def benchmark_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    warmup: int = 200,
    iterations: int = 1000,
    repeats: int = 5,
    timer: Optional[Any] = None,
    require_idle: bool = True,
    expected_output_shape: Sequence[int] = (1, 1, 228, 304),
    environment: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if warmup < 0 or iterations <= 0 or repeats <= 0:
        raise ValueError("invalid benchmark iteration counts")
    device_index = input_tensor.device.index or 0
    if require_idle:
        environment = dict(environment or gpu_environment(device_index))
        assert_gpu_idle(environment)
    timer = timer or CudaEventTimer()
    was_training = model.training
    model.eval()
    measurements = []
    repetition_medians = []
    if input_tensor.is_cuda:
        torch.cuda.reset_peak_memory_stats(input_tensor.device)
    try:
        with _strict_fp32(), torch.inference_mode():
            for _ in range(warmup):
                _validate_output(model(input_tensor), expected_output_shape)
            for _ in range(repeats):
                current = []
                for _ in range(iterations):
                    output = []

                    def invoke() -> None:
                        output.append(model(input_tensor))

                    elapsed = timer.measure(invoke)
                    _validate_output(output[0], expected_output_shape)
                    current.append(float(elapsed))
                measurements.extend(current)
                repetition_medians.append(float(statistics.median(current)))
    finally:
        model.train(was_training)
    peak_memory = 0
    if input_tensor.is_cuda:
        peak_memory = int(torch.cuda.max_memory_allocated(input_tensor.device))
    return {
        "median_ms": float(statistics.median(repetition_medians)),
        "p95_ms": float(np.percentile(measurements, 95)),
        "repetition_medians_ms": repetition_medians,
        "warmup": int(warmup),
        "iterations": int(iterations),
        "repeats": int(repeats),
        "peak_cuda_memory_bytes": peak_memory,
        "environment": dict(environment or {}),
    }
