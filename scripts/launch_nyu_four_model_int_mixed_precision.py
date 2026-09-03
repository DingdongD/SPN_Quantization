#!/usr/bin/env python3
"""Run strict four-model mixed-precision phases on declared GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")


@dataclass(frozen=True)
class LaunchJob:
    model: str
    device: str
    command: Tuple[str, ...]
    environment: Tuple[Tuple[str, str], ...]
    output: Path
    log: Path


def _load_config(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version", "model_source_config", "output_root", "devices",
        "search", "boundary_order", "interaction_pairs", "hard_deployment",
        "environments", "qat",
    }
    if set(payload) != required:
        raise ValueError("four-model launch configuration fields changed")
    if tuple(payload["devices"]) != MODEL_ORDER or \
            tuple(payload["environments"]) != MODEL_ORDER:
        raise ValueError("four-model launch order changed")
    return payload


def build_jobs(config_path: Path, output: Path,
               phase: str) -> Tuple[LaunchJob, ...]:
    if phase not in ("anchors", "ptq-search"):
        raise ValueError("launch phase must be anchors or ptq-search")
    config_path = Path(config_path).resolve()
    payload = _load_config(config_path)
    phase_root = Path(output).resolve() / phase
    script = REPO_ROOT / "scripts/run_nyu_four_model_int_mixed_precision.py"
    jobs = []
    for model in MODEL_ORDER:
        model_environment = payload["environments"][model]
        python = str(model_environment["python_executable"])
        environment = tuple(sorted(
            (str(name), str(value)) for name, value in
            model_environment.items() if name != "python_executable"))
        model_output = phase_root / model
        command = (
            python, str(script),
            "--config", str(config_path),
            "--model", model,
            "--output", str(model_output),
            "--phase", phase,
        )
        jobs.append(LaunchJob(
            model=model,
            device=str(payload["devices"][model]),
            command=command,
            environment=environment,
            output=model_output,
            log=phase_root / ("%s.log" % model),
        ))
    return tuple(jobs)


def _run_job(job: LaunchJob):
    environment = os.environ.copy()
    environment.update(dict(job.environment))
    environment["PYTHONPATH"] = str(REPO_ROOT)
    with job.log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            job.command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return job, int(completed.returncode)


def _manifest(job: LaunchJob):
    path = job.output / "manifest.json"
    if not path.is_file():
        raise RuntimeError(
            "manifest validation failed for %s: missing file" % job.model)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model", "status", "reference_sample_count",
        "reference_pooled_rmse", "propagation_dtype",
        "pareto_candidate_ids",
    }
    if not required <= set(payload) or payload["model"] != job.model or \
            int(payload["reference_sample_count"]) != 64 or \
            payload["propagation_dtype"] != "fp16":
        raise RuntimeError(
            "manifest validation failed for %s: contract mismatch" %
            job.model)
    return payload


def publish_summary(jobs: Tuple[LaunchJob, ...], phase_root: Path) -> Path:
    manifests = tuple(_manifest(job) for job in jobs)
    rows = tuple({
        "model": job.model,
        "status": manifest["status"],
        "reference_pooled_rmse": manifest["reference_pooled_rmse"],
        "reference_sample_count": manifest["reference_sample_count"],
        "propagation_dtype": manifest["propagation_dtype"],
        "pareto_candidates": len(manifest["pareto_candidate_ids"]),
        "manifest": str((job.output / "manifest.json").resolve()),
    } for job, manifest in zip(jobs, manifests))
    summary_path = Path(phase_root) / "four_model_summary.csv"
    with summary_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = Path(phase_root) / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "phase": Path(phase_root).name,
        "models": [dict(row) for row in rows],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def launch(config_path: Path, output: Path, phase: str) -> Path:
    jobs = build_jobs(config_path, output, phase)
    phase_root = Path(output).resolve() / phase
    phase_root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        results = tuple(executor.map(_run_job, jobs))
    failures = tuple(
        (job.model, code, job.command) for job, code in results if code != 0)
    if failures:
        raise RuntimeError("four-model workers failed: %s" % (failures,))
    return publish_summary(jobs, phase_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch strict four-model mixed-precision evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("anchors", "ptq-search"), required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    summary = launch(args.config, args.output, args.phase)
    print("four-model summary: %s" % summary)


if __name__ == "__main__":
    main()
