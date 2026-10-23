#!/usr/bin/env python3
"""Strict CPU-only full-forward latency, breakdown, and energy benchmark."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# Fixed-input inference never opens an HDF5 dataset, but the shared training
# module imports h5py eagerly.  Keep this process free of an unused dependency.
sys.modules.setdefault("h5py", types.ModuleType("h5py"))

from scripts.benchmark_nyu_vanilla_gpu_worker import (  # noqa: E402
    _atomic_json,
    _input_tensors,
    _runtime_args,
    _sha256,
    _source_revision,
    _wait_for_first_power_sample,
    build_power_sampler_command,
    configure_runtime_environment,
    load_fixed_model_args,
)
from spn_quant.cpu_benchmark import (  # noqa: E402
    audit_cpu_execution,
    measure_cpu_forward,
    run_cpu_energy_workload,
    summarize_profile_events,
)
from spn_quant.energy import (  # noqa: E402
    parse_power_trace,
    summarize_energy_trace,
    validate_energy_settings,
)
from spn_quant.gpu_latency import (  # noqa: E402
    MEASUREMENT_SCOPE,
    MODEL_ORDER,
    audit_fp32_execution,
)


PROPAGATION_MODULE_PATHS = {
    "cspn": "post_process_layer",
    "dyspn": "dyspn_6_5",
    "nlspn": "prop_layer",
    "completionformer": "prop_layer",
}


def assert_cpu_only_process() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty for CPU-only run")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be unavailable inside the CPU-only process")


def validate_thread_settings(args: argparse.Namespace) -> None:
    if int(args.cpu_threads) <= 0:
        raise ValueError("CPU threads must be positive")


def prime_cpu_convolutions(model: torch.nn.Module,
                           model_args: Sequence[Any]) -> Dict[str, Any]:
    """Prime ARM CPU Conv2d kernels once, outside every measured region.

    The Jetson PyTorch 2.1 NVIDIA build can return invalid data from the first
    NNPACK Conv2d invocation in a fresh process.  A pre-hook executes and
    discards that first kernel for every static Conv2d module; the immediately
    following real invocation supplies the model result used for validation.
    """
    handles = []
    count = 0

    def prime(module, inputs):
        value = inputs[0]
        torch.nn.functional.conv2d(
            value, module.weight, module.bias, module.stride,
            module.padding, module.dilation, module.groups)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_pre_hook(prime))
            count += 1
    try:
        with torch.no_grad():
            output = model(*model_args)
    finally:
        for handle in handles:
            handle.remove()
    return {"primed_conv2d_modules": count, "output": output}


def resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current = model
    for component in path.split("."):
        if not hasattr(current, component):
            raise RuntimeError("model has no module: %s" % path)
        current = getattr(current, component)
    if not isinstance(current, torch.nn.Module):
        raise RuntimeError("target is not a torch module: %s" % path)
    return current


def select_attention_modules(model: torch.nn.Module):
    selected = []
    for path, module in model.named_modules():
        if not path:
            continue
        leaf = path.rsplit(".", 1)[-1].lower()
        class_name = type(module).__name__.lower()
        if leaf in ("attn", "attention") or "attention" in class_name:
            selected.append((path, module))
    return selected


@contextmanager
def annotate_module_forward(module: torch.nn.Module, label: str):
    original = module.forward

    def annotated(*args: Any, **kwargs: Any):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    module.forward = annotated
    try:
        yield
    finally:
        module.forward = original


def cpu_telemetry() -> Dict[str, Any]:
    lscpu = subprocess.run(
        ["lscpu", "--json"], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True)
    fields = {}
    for entry in json.loads(lscpu.stdout).get("lscpu", []):
        key = str(entry.get("field", "")).rstrip(":")
        fields[key] = entry.get("data")
    frequencies = {}
    for name in ("scaling_cur_freq", "scaling_min_freq", "scaling_max_freq",
                 "scaling_governor", "affected_cpus"):
        path = Path("/sys/devices/system/cpu/cpufreq/policy0") / name
        if path.is_file():
            frequencies[name] = path.read_text(encoding="utf-8").strip()
    return {"lscpu": fields, "cpufreq_policy0": frequencies}


def require_cpu_energy_rails(result: Mapping[str, Any]) -> None:
    rails = result["power"]["rails"]
    normalized = {str(name).lower(): name for name in rails}
    if "cpu" not in normalized:
        raise RuntimeError("jtop trace has no CPU rail")
    if "total_module" not in normalized:
        raise RuntimeError("jtop trace has no total_module rail")
    result["cpu_rail_name"] = normalized["cpu"]
    for rail_name in (normalized["cpu"], normalized["total_module"]):
        rail = rails[rail_name]
        if float(rail["active_mean_power_w"]) < float(
                rail["idle"]["mean_power_w"]):
            raise RuntimeError("%s active power is below idle power" % rail_name)


def measure_energy(model: torch.nn.Module, model_args: Sequence[Any],
                   args: argparse.Namespace) -> Dict[str, Any]:
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
    command = build_power_sampler_command(
        "jtop", Path(args.power_sampler_python).resolve(), trace_path,
        float(args.power_sample_interval_ms), 0)
    sampler_log.parent.mkdir(parents=True, exist_ok=True)
    with sampler_log.open("w", encoding="utf-8") as log_handle:
        sampler_start_ns = time.monotonic_ns()
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), stdout=log_handle,
            stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            _wait_for_first_power_sample(process, trace_path, sampler_start_ns)
            idle_start = time.monotonic_ns()
            time.sleep(float(args.energy_idle_seconds))
            idle_end = time.monotonic_ns()
            workload = run_cpu_energy_workload(
                model, model_args, float(args.energy_active_seconds))
            time.sleep(max(0.1, 2.0 * args.power_sample_interval_ms / 1000.0))
        finally:
            if process.poll() is None:
                process.terminate()
            return_code = process.wait(timeout=15.0)
    if return_code != 0:
        raise RuntimeError("power sampler failed; inspect %s" % sampler_log)
    trace = parse_power_trace(trace_path)
    integrated = summarize_energy_trace(
        trace, idle_start, idle_end,
        workload["active_start_monotonic_ns"],
        workload["active_end_monotonic_ns"],
        workload["completed_forwards"])
    result = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "settings": settings,
        "phase_boundaries_monotonic_ns": {
            "idle_start": idle_start, "idle_end": idle_end,
            "active_start": workload["active_start_monotonic_ns"],
            "active_end": workload["active_end_monotonic_ns"],
        },
        "workload": workload,
        "power": integrated,
        "raw_power_trace": {
            "path": str(trace_path), "sha256": _sha256(trace_path),
            "sample_count": len(trace["samples"]),
        },
        "sampler_log": str(sampler_log),
    }
    require_cpu_energy_rails(result)
    _atomic_json(energy_output, result)
    return result


def profile_breakdown(model: torch.nn.Module, model_args: Sequence[Any],
                      model_name: str, warmups: int,
                      iterations: int) -> Dict[str, Any]:
    propagation_path = PROPAGATION_MODULE_PATHS[model_name]
    propagation = resolve_module(model, propagation_path)
    attention = select_attention_modules(model) if model_name == \
        "completionformer" else []
    with torch.inference_mode():
        for _ in range(int(warmups)):
            output = model(*model_args)
        del output
    wall_samples = []
    with ExitStack() as stack:
        stack.enter_context(annotate_module_forward(
            propagation, "SPN_PROPAGATION"))
        for _path, module in attention:
            stack.enter_context(annotate_module_forward(
                module, "SPN_ATTENTION"))
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True) as profiler:
            with torch.inference_mode():
                for _ in range(int(iterations)):
                    start = time.perf_counter_ns()
                    with torch.profiler.record_function("SPN_PROFILE_INFERENCE"):
                        output = model(*model_args)
                    end = time.perf_counter_ns()
                    wall_samples.append((end - start) / 1.0e6)
                    del output
    summary = summarize_profile_events(profiler.events(), int(iterations))
    summary.update({
        "propagation_module": propagation_path,
        "attention_modules": [path for path, _module in attention],
        "profiled_wall_latency_ms": wall_samples,
        "method": "PyTorch CPU profiler exclusive self_cpu_time_total",
    })
    return summary


def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    validate_thread_settings(args)
    assert_cpu_only_process()
    if args.model not in MODEL_ORDER:
        raise ValueError("unsupported SPN model: %s" % args.model)
    torch.set_num_threads(int(args.cpu_threads))
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["models"][args.model]
    configure_runtime_environment(
        Path(spec["data_root"]), args.external_root,
        args.completionformer_root)

    from scripts.nyu_model_runtime import NYUModelRuntime

    runtime = NYUModelRuntime.from_args(_runtime_args(spec, "cpu"))
    wall_start = time.time()
    try:
        model = runtime.build_model(device)
        model_args = load_fixed_model_args(
            args.fixed_inputs, args.model, device)
        fp32_audit = audit_fp32_execution(model, model_args)
        cpu_audit = audit_cpu_execution(model, model_args)
        priming = prime_cpu_convolutions(model, model_args)
        validation_output = priming.pop("output")
        prediction = runtime.prediction(validation_output)
        if tuple(prediction.shape) != (1, 1, 228, 304):
            raise RuntimeError("unexpected prediction shape: %s" %
                               (tuple(prediction.shape),))
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("prediction contains non-finite values")
        prediction_range = {
            "minimum": float(prediction.min().item()),
            "maximum": float(prediction.max().item()),
            "mean": float(prediction.mean().item()),
        }
        del prediction, validation_output
        telemetry_before = cpu_telemetry()
        measured = measure_cpu_forward(
            model, model_args, int(args.warmup_iterations),
            int(args.timed_iterations), int(args.rounds),
            bool(args.disable_python_gc))
        energy = measure_energy(model, model_args, args) \
            if args.energy_output is not None else None
        breakdown = profile_breakdown(
            model, model_args, args.model, int(args.profile_warmup_iterations),
            int(args.profiled_iterations)) \
            if int(args.profiled_iterations) > 0 else None
        checkpoint = Path(spec["checkpoint"]).resolve()
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
            "precision": fp32_audit["precision"],
            "batch_size": 1,
            "measurement_scope": MEASUREMENT_SCOPE,
            "includes_propagation": True,
            "input_shapes": fp32_audit["input_shapes"],
            "sample_index": int(spec["evaluation_indices"][0]),
            "fixed_input_artifact": {
                "path": str(Path(args.fixed_inputs).resolve()),
                "sha256": _sha256(Path(args.fixed_inputs).resolve()),
            },
            "propagation_iterations": int(spec["propagation_iterations"]),
            "parameter_count": fp32_audit["parameter_count"],
            "parameter_size_mb": fp32_audit["parameter_size_bytes"] / 1024**2,
            "tensor_audit": fp32_audit,
            "cpu_execution_audit": cpu_audit,
            "cpu_backend_priming": priming,
            "prediction_range": prediction_range,
            "cpu_before": telemetry_before,
            "cpu_after": cpu_telemetry(),
            "software": {
                "python_executable": sys.executable,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_available": bool(torch.cuda.is_available()),
            },
            "settings": {
                "cpu_threads": int(args.cpu_threads),
                "interop_threads": 1,
                "warmup_iterations": int(args.warmup_iterations),
                "timed_iterations_per_round": int(args.timed_iterations),
                "rounds": int(args.rounds),
                "python_gc_enabled_during_timing": not bool(
                    args.disable_python_gc),
            },
            "breakdown": breakdown,
            "wall_seconds_including_setup": time.time() - wall_start,
        }
        if energy is not None:
            result["energy_output"] = str(Path(args.energy_output).resolve())
        result.update(measured)
        _atomic_json(Path(args.output), result)
        return result
    finally:
        runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--fixed-inputs", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--completionformer-root", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    parser.add_argument("--warmup-iterations", type=int, required=True)
    parser.add_argument("--timed-iterations", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--disable-python-gc", action="store_true")
    parser.add_argument("--profile-warmup-iterations", type=int, default=2)
    parser.add_argument("--profiled-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-output", type=Path)
    parser.add_argument("--power-sampler-python", type=Path,
                        default=Path("/usr/bin/python3"))
    parser.add_argument("--energy-idle-seconds", type=float, default=15.0)
    parser.add_argument("--energy-active-seconds", type=float, default=60.0)
    parser.add_argument("--power-sample-interval-ms", type=float, default=50.0)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    if args.energy_output is None and (args.energy_idle_seconds != 15.0 or
                                       args.energy_active_seconds != 60.0):
        raise ValueError("energy durations require --energy-output")
    result = benchmark(args)
    print(json.dumps({
        "model": result["model"],
        "mean_ms": result["latency_ms"]["mean"],
        "p95_ms": result["latency_ms"]["p95"],
        "cpu_threads": result["settings"]["cpu_threads"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
