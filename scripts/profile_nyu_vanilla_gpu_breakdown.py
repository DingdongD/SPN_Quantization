#!/usr/bin/env python3
"""Capture strict full-forward CUDA/NVTX traces for one vanilla SPN model."""

from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from scripts.benchmark_nyu_vanilla_gpu_worker import (  # noqa: E402
    _configure_fp32_backend,
    _runtime_args,
    load_fixed_model_args,
)
from spn_quant.gpu_latency import MODEL_ORDER  # noqa: E402


PROPAGATION_MODULE_PATHS = {
    "cspn": "post_process_layer",
    "dyspn": "dyspn_6_5",
    "nlspn": "prop_layer",
    "completionformer": "prop_layer",
}


def resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current = model
    for component in path.split("."):
        if not hasattr(current, component):
            raise RuntimeError("model has no propagation module: %s" % path)
        current = getattr(current, component)
    if not isinstance(current, torch.nn.Module):
        raise RuntimeError("propagation target is not a torch module: %s" % path)
    return current


@contextmanager
def annotate_module_forward(module: torch.nn.Module, label: str):
    original = module.forward

    def annotated(*args: Any, **kwargs: Any):
        with torch.cuda.nvtx.range(label):
            return original(*args, **kwargs)

    module.forward = annotated
    try:
        yield
    finally:
        module.forward = original


def capture(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.model not in MODEL_ORDER:
        raise ValueError("unsupported model: %s" % args.model)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("capture requires an explicitly indexed CUDA device")
    _configure_fp32_backend()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["models"][args.model]
    sample_index = int(spec["evaluation_indices"][0])

    from scripts.nyu_model_runtime import NYUModelRuntime

    runtime = NYUModelRuntime.from_args(_runtime_args(spec, str(device)))
    try:
        model = runtime.build_model(runtime.device)
        model_args = load_fixed_model_args(
            args.fixed_inputs, args.model, runtime.device)
        propagation_path = PROPAGATION_MODULE_PATHS[args.model]
        propagation_module = resolve_module(model, propagation_path)
        with torch.no_grad():
            for _ in range(args.warmup_iterations):
                output = model(*model_args)
            del output
        torch.cuda.synchronize(device)

        samples = []
        with annotate_module_forward(propagation_module, "SPN_PROPAGATION"):
            with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                with torch.no_grad():
                    for _ in range(args.profiled_iterations):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.nvtx.range("SPN_PROFILE_INFERENCE"):
                            start.record()
                            output = model(*model_args)
                            end.record()
                            end.synchronize()
                        samples.append(float(start.elapsed_time(end)))
                        del output
        torch.cuda.synchronize(device)
        payload = {
            "model": args.model,
            "board": torch.cuda.get_device_name(device),
            "precision": "fp32",
            "sample_index": sample_index,
            "input_resolution": [228, 304],
            "propagation_module": propagation_path,
            "warmup_iterations": args.warmup_iterations,
            "profiled_iterations": args.profiled_iterations,
            "captured_cuda_event_ms": samples,
            "capture_mean_ms": statistics.mean(samples),
            "capture_p50_ms": statistics.median(samples),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trace_method": (
                "Nsight Systems CUDA timeline; emit_nvtx aten operator scopes; "
                "explicit propagation and full-forward NVTX ranges; CUDA Events"),
        }
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-inputs", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=8)
    parser.add_argument("--profiled-iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup_iterations <= 0 or args.profiled_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    payload = capture(args)
    print(json.dumps({
        "model": payload["model"],
        "capture_mean_ms": payload["capture_mean_ms"],
        "profiled_iterations": payload["profiled_iterations"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
