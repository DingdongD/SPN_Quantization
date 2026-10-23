#!/usr/bin/env python3
"""Measure full-forward vanilla SPN energy on a Jetson CUDA deployment."""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import sys
import types

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.modules.setdefault("h5py", types.ModuleType("h5py"))

from scripts import benchmark_nyu_vanilla_gpu_worker as worker  # noqa: E402


def _configure_model_imports(model_name: str, external_root: Path,
                             completionformer_root: Path) -> None:
    deform_root = (
        external_root / "NLSPN_ECCV20/src/model/deformconv"
        if model_name == "nlspn" else
        completionformer_root / "src/model/deformconv")
    sys.path.insert(0, str(deform_root))
    if model_name == "completionformer":
        mmseg = types.ModuleType("mmseg")
        mmseg_utils = types.ModuleType("mmseg.utils")
        mmseg_utils.get_root_logger = lambda *a, **k: logging.getLogger(
            "completionformer")
        mmcv = types.ModuleType("mmcv")
        mmcv_runner = types.ModuleType("mmcv.runner")
        mmcv_runner.load_checkpoint = lambda *a, **k: None
        mmseg.utils, mmcv.runner = mmseg_utils, mmcv_runner
        sys.modules.update({
            "mmseg": mmseg, "mmseg.utils": mmseg_utils,
            "mmcv": mmcv, "mmcv.runner": mmcv_runner,
        })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixed-inputs", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--completionformer-root", type=Path, required=True)
    parser.add_argument("--energy-output", type=Path, required=True)
    parser.add_argument("--power-sampler-python", type=Path,
                        default=Path("/usr/bin/python3"))
    parser.add_argument("--warmup-iterations", type=int, default=30)
    parser.add_argument("--energy-idle-seconds", type=float, default=15.0)
    parser.add_argument("--energy-active-seconds", type=float, default=60.0)
    parser.add_argument("--power-sample-interval-ms", type=float, default=50.0)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["models"][args.model]
    worker.configure_runtime_environment(
        Path(spec["data_root"]), args.external_root,
        args.completionformer_root)
    _configure_model_imports(
        args.model, args.external_root.resolve(),
        args.completionformer_root.resolve())

    from scripts.nyu_model_runtime import NYUModelRuntime

    device = torch.device("cuda:0")
    worker._configure_fp32_backend()
    runtime = NYUModelRuntime.from_args(worker._runtime_args(spec, str(device)))
    try:
        model = runtime.build_model(device).eval()
        model_args = worker.load_fixed_model_args(
            args.fixed_inputs.resolve(), args.model, device)
        audit = worker.audit_fp32_execution(model, model_args)
        with torch.no_grad():
            prediction = runtime.prediction(model(*model_args))
        torch.cuda.synchronize(device)
        if tuple(prediction.shape) != (1, 1, 228, 304):
            raise RuntimeError("unexpected prediction shape")
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("non-finite prediction")
        del prediction

        # The timed iteration follows the same 30-forward warmup contract and
        # leaves the model/inputs resident before the loaded-idle phase.
        worker.measure_forward(
            model, model_args, device,
            warmup_iterations=args.warmup_iterations,
            timed_iterations=1, rounds=1, disable_python_gc=False)
        energy_args = Namespace(
            model=args.model,
            warmup_iterations=args.warmup_iterations,
            energy_idle_seconds=args.energy_idle_seconds,
            energy_active_seconds=args.energy_active_seconds,
            power_sample_interval_ms=args.power_sample_interval_ms,
            energy_output=args.energy_output,
            power_sampler_python=args.power_sampler_python,
            power_source="jtop",
        )
        energy = worker.measure_energy(model, model_args, device, energy_args)
        checkpoint = Path(spec["checkpoint"]).resolve()
        energy.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": "Jetson AGX Xavier",
            "architecture_class": type(model).__name__,
            "precision": audit["precision"],
            "batch_size": 1,
            "sample_index": int(spec["evaluation_indices"][0]),
            "input_shapes": audit["input_shapes"],
            "propagation_iterations": int(spec["propagation_iterations"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": worker._sha256(checkpoint),
            "config": str(config_path),
            "config_sha256": worker._sha256(config_path),
            "fixed_input_artifact": {
                "path": str(args.fixed_inputs.resolve()),
                "sha256": worker._sha256(args.fixed_inputs.resolve()),
            },
            "software": {
                "python_executable": sys.executable,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "cudnn": int(torch.backends.cudnn.version()),
            },
        })
        worker._atomic_json(args.energy_output.resolve(), energy)
        print(json.dumps({
            "model": args.model,
            "completed_forwards": energy["workload"]["completed_forwards"],
            "energy_output": str(args.energy_output.resolve()),
        }, sort_keys=True), flush=True)
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
