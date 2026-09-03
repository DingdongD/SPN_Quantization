#!/usr/bin/env python3
"""Launch the formal four-GPU CSPN LSQ+ and HAWQ experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from spn_quant.qat.method_config import load_method_config  # noqa: E402


@dataclass(frozen=True)
class GPUJob:
    name: str
    gpu: int
    commands: Tuple[Tuple[str, ...], ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--mixed-source-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args(argv)


def _script(name: str) -> str:
    return str((REPO_ROOT / "scripts" / name).resolve())


def _training_command(args, method: str, gpu: int, assignment=None):
    command = [
        sys.executable,
        _script("train_nyu_cspn_lsqplus_hawq.py"),
        "--method", method,
        "--config", str(Path(args.config).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--calibration-metadata",
        str(Path(args.calibration_metadata).resolve()),
        "--output-root", str(
            (Path(args.output_root) / "training").resolve()),
        "--device", "cuda:%d" % gpu,
    ]
    if assignment is not None:
        command.extend(("--assignment", str(Path(assignment).resolve())))
    return tuple(command)


def _evaluation_command(
        args, configuration: str, gpu: int,
        method_checkpoint=None, assignment=None):
    command = [
        sys.executable,
        _script("evaluate_nyu_cspn_lsqplus_hawq.py"),
        "--configuration", configuration,
        "--config", str(Path(args.config).resolve()),
        "--fp32-checkpoint", str(Path(args.checkpoint).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--calibration-metadata",
        str(Path(args.calibration_metadata).resolve()),
        "--mixed-source-root", str(Path(args.mixed_source_root).resolve()),
        "--output-root", str(
            (Path(args.output_root) / "evaluation").resolve()),
        "--device", "cuda:%d" % gpu,
    ]
    if method_checkpoint is not None:
        command.extend((
            "--method-checkpoint",
            str(Path(method_checkpoint).resolve()),
        ))
    if assignment is not None:
        command.extend(("--assignment", str(Path(assignment).resolve())))
    return tuple(command)


def build_phases(args):
    config = load_method_config(Path(args.config))
    gpus = dict(config.gpus)
    output = Path(args.output_root)
    trace_root = output / "hawq_trace"
    assignment = trace_root / "selected_assignment.json"
    training_root = output / "training"
    trace_command = (
        sys.executable,
        _script("run_nyu_cspn_hawq_trace.py"),
        "--config", str(Path(args.config).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--calibration-metadata",
        str(Path(args.calibration_metadata).resolve()),
        "--output-root", str(trace_root.resolve()),
        "--device", "cuda:%d" % gpus["hawq_mixed_le6"],
    )
    phase_one = (
        GPUJob("baselines", gpus["baselines"], tuple(
            _evaluation_command(args, configuration, gpus["baselines"])
            for configuration in (
                "FP32", "PA_RTN_W4A4", "PA_RTN_W6A6",
                "MIXED_TASK_AWARE_QAT"))),
        GPUJob(
            "lsqplus_w4a4", gpus["lsqplus_w4a4"],
            (_training_command(
                args, "lsqplus_w4a4", gpus["lsqplus_w4a4"]),)),
        GPUJob(
            "lsqplus_w6a6", gpus["lsqplus_w6a6"],
            (_training_command(
                args, "lsqplus_w6a6", gpus["lsqplus_w6a6"]),)),
        GPUJob(
            "hawq", gpus["hawq_mixed_le6"],
            (trace_command, _training_command(
                args, "hawq_mixed_le6", gpus["hawq_mixed_le6"],
                assignment))),
    )
    phase_two = (
        GPUJob("evaluate_lsqplus_w4a4", gpus["lsqplus_w4a4"], (
            _evaluation_command(
                args, "LSQPLUS_W4A4", gpus["lsqplus_w4a4"],
                training_root / "lsqplus_w4a4" / "best.pt"),)),
        GPUJob("evaluate_lsqplus_w6a6", gpus["lsqplus_w6a6"], (
            _evaluation_command(
                args, "LSQPLUS_W6A6", gpus["lsqplus_w6a6"],
                training_root / "lsqplus_w6a6" / "best.pt"),)),
        GPUJob("evaluate_hawq", gpus["hawq_mixed_le6"], (
            _evaluation_command(
                args, "HAWQ_MIXED_LE6", gpus["hawq_mixed_le6"],
                training_root / "hawq_mixed_le6" / "best.pt",
                assignment),)),
    )
    aggregate = (
        sys.executable,
        _script("evaluate_nyu_cspn_lsqplus_hawq.py"),
        "--aggregate",
        "--calibration-metadata",
        str(Path(args.calibration_metadata).resolve()),
        "--output-root", str((output / "evaluation").resolve()),
    )
    plot = (
        sys.executable,
        _script("plot_nyu_cspn_lsqplus_hawq.py"),
        "--output-root", str((output / "evaluation").resolve()),
        "--calibration-metadata",
        str(Path(args.calibration_metadata).resolve()),
        "--detail-samples", str(config.evaluation.detail_samples),
    )
    phase_three = (GPUJob("aggregate_and_plot", -1, (aggregate, plot)),)
    return phase_one, phase_two, phase_three


def _run_job(job: GPUJob, log_root: Path) -> None:
    log_path = log_root / (job.name + ".log")
    with log_path.open("a", encoding="utf-8") as handle:
        for command in job.commands:
            handle.write("command=%s\n" % " ".join(command))
            handle.flush()
            subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )


def _run_phase(jobs, log_root: Path) -> None:
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = tuple(
            executor.submit(_run_job, job, log_root) for job in jobs)
        for future in futures:
            future.result()


def main(argv=None) -> None:
    args = parse_args(argv)
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("formal LSQ+/HAWQ output root must be empty")
    output.mkdir(parents=True, exist_ok=True)
    log_root = output / "logs"
    log_root.mkdir()
    for phase in build_phases(args):
        _run_phase(phase, log_root)


if __name__ == "__main__":
    main()
