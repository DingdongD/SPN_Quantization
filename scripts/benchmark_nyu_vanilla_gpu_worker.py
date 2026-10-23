#!/usr/bin/env python3
"""Measure one official vanilla SPN full forward with CUDA Events."""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from spn_quant.gpu_latency import (  # noqa: E402
    MEASUREMENT_SCOPE,
    MODEL_ORDER,
    audit_fp32_execution,
    summarize_latencies,
)
from spn_quant.energy import (  # noqa: E402
    parse_power_trace,
    summarize_energy_trace,
    validate_energy_settings,
)


def _mb(value: int) -> float:
    return float(value) / float(1024 ** 2)


def measure_forward(
        model: torch.nn.Module, model_args: Sequence[Any],
        device: torch.device, warmup_iterations: int,
        timed_iterations: int, rounds: int,
        disable_python_gc: bool) -> Dict[str, Any]:
    """Measure synchronized full forwards; inputs must already reside on CUDA."""
    if warmup_iterations <= 0 or timed_iterations <= 0 or rounds <= 0:
        raise ValueError("warmup, timed iterations, and rounds must be positive")
    model.eval()
    gc_was_enabled = gc.isenabled()
    if disable_python_gc and gc_was_enabled:
        gc.disable()
    try:
        with torch.no_grad():
            for _ in range(int(warmup_iterations)):
                output = model(*model_args)
        del output
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

        samples = []
        round_summaries = []
        with torch.no_grad():
            for round_index in range(int(rounds)):
                round_samples = []
                for _ in range(int(timed_iterations)):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    output = model(*model_args)
                    end.record()
                    end.synchronize()
                    elapsed_ms = float(start.elapsed_time(end))
                    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                        raise RuntimeError("CUDA Event returned invalid latency")
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
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        return {
            "latency_ms": summarize_latencies(samples),
            "raw_latency_ms": samples,
            "rounds": round_summaries,
            "memory_mb": {
                "baseline_allocated": _mb(baseline_allocated),
                "baseline_reserved": _mb(baseline_reserved),
                "peak_allocated": _mb(peak_allocated),
                "peak_reserved": _mb(peak_reserved),
                "inference_workspace_peak": _mb(max(
                    0, peak_allocated - baseline_allocated)),
            },
        }
    finally:
        if disable_python_gc and gc_was_enabled:
            gc.enable()


def run_synchronized_energy_workload(
        model: torch.nn.Module, model_args: Sequence[Any],
        device: torch.device, active_seconds: float,
        monotonic=time.monotonic,
        monotonic_ns=time.monotonic_ns) -> Dict[str, Any]:
    """Run complete, individually synchronized forwards for a wall interval."""
    if not math.isfinite(float(active_seconds)) or active_seconds <= 0.0:
        raise ValueError("active_seconds must be positive")
    torch.cuda.synchronize(device)
    active_start_monotonic_ns = int(monotonic_ns())
    wall_start = float(monotonic())
    latency_ms = []
    with torch.no_grad():
        while float(monotonic()) - wall_start < float(active_seconds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(*model_args)
            end.record()
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                raise RuntimeError("CUDA Event returned invalid energy-run latency")
            latency_ms.append(elapsed_ms)
            del output
    torch.cuda.synchronize(device)
    active_end_monotonic_ns = int(monotonic_ns())
    return {
        "active_start_monotonic_ns": active_start_monotonic_ns,
        "active_end_monotonic_ns": active_end_monotonic_ns,
        "active_wall_seconds": (
            active_end_monotonic_ns - active_start_monotonic_ns) / 1.0e9,
        "completed_forwards": len(latency_ms),
        "raw_latency_ms": latency_ms,
        "latency_ms": summarize_latencies(latency_ms),
    }


def build_power_sampler_command(
        source: str, python_executable: Path, output: Path,
        interval_ms: float, gpu_index: int) -> list:
    """Build the explicit external-sampler command used by the worker."""
    if source not in ("nvml", "jtop"):
        raise ValueError("unsupported power source: %s" % source)
    return [
        str(python_executable),
        str(REPO_ROOT / "scripts" / "sample_hardware_power.py"),
        "--source", source,
        "--output", str(output),
        "--interval-ms", str(float(interval_ms)),
        "--gpu-index", str(int(gpu_index)),
    ]


def _has_power_sample_at_or_after(
        trace_path: Path, minimum_monotonic_ns: int) -> bool:
    if not trace_path.is_file():
        return False
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("kind") == "sample" and int(
                record.get("monotonic_ns", -1)) >= int(minimum_monotonic_ns):
            return True
    return False


def _wait_for_first_power_sample(
        process: subprocess.Popen, trace_path: Path,
        minimum_monotonic_ns: int,
        timeout_seconds: float = 15.0) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("power sampler exited before its first sample")
        if _has_power_sample_at_or_after(
                trace_path, minimum_monotonic_ns):
            return
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for the first power sample")


def measure_energy(
        model: torch.nn.Module, model_args: Sequence[Any],
        device: torch.device, args: argparse.Namespace) -> Dict[str, Any]:
    """Measure loaded-idle and complete-forward energy with raw evidence."""
    settings = {
        "warmup_iterations": int(args.warmup_iterations),
        "idle_seconds": float(args.energy_idle_seconds),
        "active_seconds": float(args.energy_active_seconds),
        "sample_interval_ms": float(args.power_sample_interval_ms),
        "synchronize_each_forward": True,
    }
    validate_energy_settings(settings)
    energy_output = Path(args.energy_output).resolve()
    trace_path = energy_output.with_suffix(".power.jsonl")
    sampler_log = energy_output.with_suffix(".sampler.log")
    sampler_python = Path(
        args.power_sampler_python or sys.executable).resolve()
    command = build_power_sampler_command(
        args.power_source, sampler_python, trace_path,
        float(args.power_sample_interval_ms), int(device.index))
    sampler_log.parent.mkdir(parents=True, exist_ok=True)
    with sampler_log.open("w", encoding="utf-8") as log_handle:
        sampler_start_monotonic_ns = time.monotonic_ns()
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), stdout=log_handle,
            stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            _wait_for_first_power_sample(
                process, trace_path, sampler_start_monotonic_ns)
            idle_start = time.monotonic_ns()
            time.sleep(float(args.energy_idle_seconds))
            idle_end = time.monotonic_ns()
            workload = run_synchronized_energy_workload(
                model, model_args, device,
                active_seconds=float(args.energy_active_seconds))
            time.sleep(max(0.1, 2.0 * float(
                args.power_sample_interval_ms) / 1000.0))
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                return_code = process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
                raise RuntimeError("power sampler did not stop cleanly")
    if return_code != 0:
        raise RuntimeError(
            "power sampler failed; inspect %s" % sampler_log)
    trace = parse_power_trace(trace_path)
    integrated = summarize_energy_trace(
        trace,
        idle_start_monotonic_ns=idle_start,
        idle_end_monotonic_ns=idle_end,
        active_start_monotonic_ns=workload[
            "active_start_monotonic_ns"],
        active_end_monotonic_ns=workload["active_end_monotonic_ns"],
        completed_forwards=workload["completed_forwards"],
    )
    result = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "measurement_scope": MEASUREMENT_SCOPE,
        "includes_propagation": True,
        "settings": settings,
        "phase_boundaries_monotonic_ns": {
            "idle_start": idle_start,
            "idle_end": idle_end,
            "active_start": workload["active_start_monotonic_ns"],
            "active_end": workload["active_end_monotonic_ns"],
        },
        "workload": workload,
        "power": integrated,
        "raw_power_trace": {
            "path": str(trace_path),
            "sha256": _sha256(trace_path),
            "sample_count": len(trace["samples"]),
        },
        "sampler_log": str(sampler_log),
    }
    _atomic_json(energy_output, result)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_input_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {
            key: _move_input_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_input_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_input_to_device(item, device) for item in value]
    raise ValueError(
        "fixed model inputs may contain only tensors, mappings, tuples, and "
        "lists; got %s" % type(value).__name__)


def load_fixed_model_args(
        path: Path, model_name: str, device: torch.device) -> tuple:
    """Load one captured, preprocessing-complete input onto the device."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("fixed input artifact not found: %s" % path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or model_name not in payload:
        raise ValueError(
            "fixed input artifact has no entry for model %s" % model_name)
    entry = payload[model_name]
    if not isinstance(entry, dict) or "args" not in entry:
        raise ValueError("fixed input entry has no args for model %s" % model_name)
    model_args = entry["args"]
    if not isinstance(model_args, tuple):
        raise ValueError("fixed model args must be stored as a tuple")
    moved = _move_input_to_device(model_args, device)
    if not tuple(_input_tensors(moved)):
        raise ValueError("fixed model args contain no tensors")
    return moved


def _runtime_args(spec: Mapping[str, Any], device: str) -> Namespace:
    return Namespace(
        model=str(spec["model"]), run_dir=Path(spec["run_dir"]),
        checkpoint=Path(spec["checkpoint"]),
        expected_architecture_class=str(spec["expected_architecture_class"]),
        required_cuda_extension=str(spec["required_cuda_extension"]),
        propagation_iterations=int(spec["propagation_iterations"]),
        data_root=Path(spec["data_root"]), device=str(device),
        checkpoint_architecture=str(spec["checkpoint_architecture"]),
        native_cuda_operator=spec.get("native_cuda_operator"),
    )


def configure_runtime_environment(
        data_root: Path, external_root: Path,
        completionformer_root: Path) -> None:
    """Set model roots before modules snapshot them at import time."""
    os.environ["SPN_DATA_ROOT"] = str(Path(data_root).resolve())
    os.environ["SPN_EXTERNAL_ROOT"] = str(Path(external_root).resolve())
    os.environ["COMPLETIONFORMER_ROOT"] = str(
        Path(completionformer_root).resolve())


def _configure_fp32_backend() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and \
            hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _gpu_telemetry(device: torch.device) -> Dict[str, Any]:
    command = [
        "nvidia-smi", "--id=%d" % int(device.index),
        "--query-gpu=index,name,uuid,driver_version,temperature.gpu,"
        "utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,pstate",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True)
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    keys = (
        "index", "name", "uuid", "driver_version", "temperature_c",
        "utilization_percent", "memory_used_mb", "sm_clock_mhz",
        "memory_clock_mhz", "power_w", "pstate",
    )
    if len(values) != len(keys):
        raise RuntimeError("unexpected nvidia-smi telemetry")
    return dict(zip(keys, values))


def _source_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if args.model not in MODEL_ORDER:
        raise ValueError("unsupported SPN model: %s" % args.model)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("benchmark requires an explicitly indexed CUDA device")
    _configure_fp32_backend()
    config_path = Path(args.config).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    spec = payload["models"][args.model]
    sample_indices = tuple(int(value) for value in spec["evaluation_indices"])
    if not sample_indices:
        raise ValueError("model source config has no evaluation identities")
    sample_index = sample_indices[0]

    if args.external_root is not None and \
            args.completionformer_root is not None:
        configure_runtime_environment(
            Path(spec["data_root"]), args.external_root,
            args.completionformer_root)
    elif args.external_root is not None or \
            args.completionformer_root is not None:
        raise ValueError(
            "external and CompletionFormer roots must be specified together")

    from scripts.nyu_model_runtime import NYUModelRuntime
    from scripts import run_nyu_rtn_quantization as rtn_runner

    runtime = NYUModelRuntime.from_args(_runtime_args(spec, str(device)))
    wall_start = time.time()
    try:
        model = runtime.build_model(runtime.device)
        if args.fixed_inputs is None:
            dataset = runtime.build_dataset("val")
            sample = rtn_runner.seeded_sample(
                dataset, sample_index, runtime.saved_args.seed)
            batch = rtn_runner.batch_from_sample(sample)
            model_args, ground_truth = runtime.model_input(batch, runtime.device)
            del ground_truth, batch, sample, dataset
            input_artifact = None
        else:
            fixed_inputs = Path(args.fixed_inputs).resolve()
            model_args = load_fixed_model_args(
                fixed_inputs, args.model, runtime.device)
            input_artifact = {
                "path": str(fixed_inputs),
                "sha256": _sha256(fixed_inputs),
            }
        audit = audit_fp32_execution(model, model_args)
        for index, tensor in enumerate(
                tensor for value in model_args
                for tensor in _input_tensors(value)):
            if tensor.device != device:
                raise RuntimeError(
                    "input %d is not resident on %s" % (index, device))

        with torch.no_grad():
            validation_output = model(*model_args)
        prediction = runtime.prediction(validation_output)
        if tuple(prediction.shape) != (1, 1, 228, 304):
            raise RuntimeError("unexpected vanilla prediction shape: %s" %
                               (tuple(prediction.shape),))
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("vanilla prediction contains non-finite values")
        prediction_range = {
            "minimum": float(prediction.min().item()),
            "maximum": float(prediction.max().item()),
            "mean": float(prediction.mean().item()),
        }
        del prediction, validation_output
        torch.cuda.synchronize(device)
        telemetry_before = _gpu_telemetry(device)
        measured = measure_forward(
            model, model_args, device,
            warmup_iterations=int(args.warmup_iterations),
            timed_iterations=int(args.timed_iterations),
            rounds=int(args.rounds),
            disable_python_gc=bool(args.disable_python_gc))
        checkpoint = Path(spec["checkpoint"]).resolve()
        energy = None
        if args.energy_output is not None:
            energy = measure_energy(model, model_args, device, args)
            energy.update({
                "architecture_class": type(model).__name__,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "fixed_input_artifact": input_artifact,
                "precision": audit["precision"],
                "batch_size": 1,
                "sample_index": sample_index,
                "input_shapes": audit["input_shapes"],
                "propagation_iterations": int(spec["propagation_iterations"]),
                "source_revision": _source_revision(),
                "gpu_before": telemetry_before,
                "software": {
                    "python_executable": sys.executable,
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "cuda": str(torch.version.cuda),
                    "cudnn": int(torch.backends.cudnn.version()),
                },
            })
            _atomic_json(Path(args.energy_output), energy)
        telemetry_after = _gpu_telemetry(device)
        result = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "architecture_class": type(model).__name__,
            "checkpoint_architecture": spec["checkpoint_architecture"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_revision": _source_revision(),
            "precision": audit["precision"],
            "batch_size": 1,
            "measurement_scope": MEASUREMENT_SCOPE,
            "includes_propagation": True,
            "input_shapes": audit["input_shapes"],
            "sample_index": sample_index,
            "fixed_input_artifact": input_artifact,
            "propagation_iterations": int(spec["propagation_iterations"]),
            "parameter_count": audit["parameter_count"],
            "parameter_size_mb": _mb(audit["parameter_size_bytes"]),
            "tensor_audit": audit,
            "prediction_range": prediction_range,
            "gpu": {
                "index": int(device.index),
                "name": telemetry_before["name"],
                "uuid": telemetry_before["uuid"],
                "driver_version": telemetry_before["driver_version"],
            },
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "software": {
                "python_executable": sys.executable,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "cudnn": int(torch.backends.cudnn.version()),
            },
            "settings": {
                "warmup_iterations": int(args.warmup_iterations),
                "timed_iterations_per_round": int(args.timed_iterations),
                "rounds": int(args.rounds),
                "tf32": False,
                "cudnn_benchmark": True,
                "python_gc_enabled_during_timing": not bool(
                    args.disable_python_gc),
                "event_synchronized_each_iteration": True,
            },
            "wall_seconds_including_setup": time.time() - wall_start,
        }
        if energy is not None:
            result["energy_output"] = str(Path(args.energy_output).resolve())
        result.update(measured)
        _atomic_json(Path(args.output), result)
        return result
    finally:
        runtime.close()


def _input_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _input_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _input_tensors(item)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--warmup-iterations", type=int, required=True)
    parser.add_argument("--timed-iterations", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--disable-python-gc", action="store_true")
    parser.add_argument("--fixed-inputs", type=Path)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--completionformer-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-output", type=Path)
    parser.add_argument("--power-source", choices=("nvml", "jtop"))
    parser.add_argument("--power-sampler-python", type=Path)
    parser.add_argument("--energy-idle-seconds", type=float, default=15.0)
    parser.add_argument("--energy-active-seconds", type=float, default=60.0)
    parser.add_argument("--power-sample-interval-ms", type=float, default=50.0)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    if (args.energy_output is None) != (args.power_source is None):
        raise ValueError(
            "--energy-output and --power-source must be specified together")
    result = benchmark(args)
    print(json.dumps({
        "model": result["model"],
        "mean_ms": result["latency_ms"]["mean"],
        "p95_ms": result["latency_ms"]["p95"],
        "output": str(Path(result["model"] + ".json")),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
