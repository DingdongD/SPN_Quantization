#!/usr/bin/env python3
"""Run strict four-model mixed-precision phases on declared GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
QAT_FIELDS = {
    "epochs", "checkpoint_protocol", "maximum_candidates_per_model",
    "train_split", "batch_size", "validation_batch_size",
    "validation_sample_count", "workers",
    "learning_rate", "momentum", "weight_decay", "scheduler_factor",
    "scheduler_patience", "scheduler_threshold", "scheduler_min_lr",
    "max_gradient_norm", "patience", "min_relative_improvement", "seed",
    "hawq_range_momentum", "depth_loss_weight", "boundary_loss_weight",
    "teacher_loss_weight", "initial_depth_loss_weight",
    "propagation_loss_weight", "boundary_threshold_m", "log_interval",
}


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
    if phase not in ("anchors", "ptq-search", "qat"):
        raise ValueError("launch phase must be anchors, ptq-search, or qat")
    config_path = Path(config_path).resolve()
    payload = _load_config(config_path)
    phase_root = Path(output).resolve() / phase
    script = REPO_ROOT / "scripts/run_nyu_four_model_int_mixed_precision.py"
    if phase == "qat":
        script = REPO_ROOT / "scripts/train_nyu_selected_qat.py"
    jobs = []
    for model in MODEL_ORDER:
        model_environment = payload["environments"][model]
        python = str(model_environment["python_executable"])
        environment = tuple(sorted(
            (str(name), str(value)) for name, value in
            model_environment.items() if name != "python_executable"))
        if phase != "qat":
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
            continue
        qat = payload["qat"]
        if set(qat) != QAT_FIELDS or qat["train_split"] != "complete" or \
                qat["checkpoint_protocol"] != "fixed_final_epoch":
            raise ValueError("four-model QAT protocol changed")
        ptq_root = Path(output).resolve() / "ptq-search" / model
        manifest_path = ptq_root / "manifest.json"
        candidates_path = ptq_root / "candidate_assignments.json"
        if not manifest_path.is_file() or not candidates_path.is_file():
            raise FileNotFoundError(
                "QAT requires completed PTQ artifacts for %s" % model)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model"] != model:
            raise ValueError("QAT PTQ manifest model differs")
        candidate_ids = tuple(str(value)
                              for value in manifest["qat_candidate_ids"])
        maximum_candidates = int(qat["maximum_candidates_per_model"])
        if not candidate_ids or len(candidate_ids) > maximum_candidates:
            raise ValueError("QAT candidate count differs from protocol")
        for candidate_id in candidate_ids:
            if Path(candidate_id).name != candidate_id:
                raise ValueError("QAT candidate identity is not path-safe")
            model_output = phase_root / model / candidate_id
            command = [
                python, str(script),
                "--config", str(config_path),
                "--model", model,
                "--method", "mixed_task_aware",
                "--device", str(payload["devices"][model]),
                "--output", str(model_output),
                "--constrained-candidates", str(candidates_path),
                "--constrained-manifest", str(manifest_path),
                "--constrained-candidate-id", candidate_id,
                "--constrained-maximum-relative-loss",
                str(payload["search"]["qat_candidate_loss"]),
            ]
            arguments = (
                ("epochs", "--epochs"),
                ("checkpoint_protocol", "--checkpoint-protocol"),
                ("batch_size", "--batch-size"),
                ("validation_batch_size", "--validation-batch-size"),
                ("validation_sample_count", "--validation-sample-count"),
                ("workers", "--workers"),
                ("learning_rate", "--learning-rate"),
                ("momentum", "--momentum"),
                ("weight_decay", "--weight-decay"),
                ("scheduler_factor", "--scheduler-factor"),
                ("scheduler_patience", "--scheduler-patience"),
                ("scheduler_threshold", "--scheduler-threshold"),
                ("scheduler_min_lr", "--scheduler-min-lr"),
                ("max_gradient_norm", "--max-gradient-norm"),
                ("patience", "--patience"),
                ("min_relative_improvement", "--min-relative-improvement"),
                ("seed", "--seed"),
                ("hawq_range_momentum", "--hawq-range-momentum"),
                ("depth_loss_weight", "--depth-loss-weight"),
                ("boundary_loss_weight", "--boundary-loss-weight"),
                ("teacher_loss_weight", "--teacher-loss-weight"),
                ("initial_depth_loss_weight",
                 "--initial-depth-loss-weight"),
                ("propagation_loss_weight", "--propagation-loss-weight"),
                ("boundary_threshold_m", "--boundary-threshold-m"),
            )
            for name, flag in arguments:
                command.extend((flag, str(qat[name])))
            hard = payload["hard_deployment"]
            command.append("--fold-conv-bn" if hard["fold_conv_bn"] else
                           "--skip-conv-bn-fold")
            command.extend((
                "--fold-max-error", str(hard["fold_max_error"]),
                "--joint-clip-factors",
            ))
            command.extend(str(value)
                           for value in hard["joint_clip_factors"])
            command.extend((
                "--joint-search-rounds", str(hard["joint_search_rounds"]),
                "--joint-cache-sample-limit",
                str(hard["joint_cache_sample_limit"]),
                "--joint-cache-byte-limit",
                str(hard["joint_cache_byte_limit"]),
                "--log-interval", str(qat["log_interval"]),
            ))
            jobs.append(LaunchJob(
                model=model,
                device=str(payload["devices"][model]),
                command=tuple(command),
                environment=environment,
                output=model_output,
                log=phase_root / model / ("%s.log" % candidate_id),
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


def qat_job_groups(jobs: Tuple[LaunchJob, ...]):
    grouped = []
    for model in MODEL_ORDER:
        rows = tuple(job for job in jobs if job.model == model)
        if not rows:
            raise ValueError("QAT model has no published candidates: %s" % model)
        if any(job.device != rows[0].device for job in rows):
            raise ValueError("QAT model candidates use different devices")
        grouped.append(rows)
    if sum(len(group) for group in grouped) != len(jobs):
        raise ValueError("QAT jobs contain an unknown model")
    return tuple(grouped)


def _run_qat_group(jobs):
    results = []
    for job in jobs:
        result = _run_job(job)
        results.append(result)
        if result[1] != 0:
            break
    return tuple(results)


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


def publish_qat_summary(jobs: Tuple[LaunchJob, ...], phase_root: Path) -> Path:
    rows = []
    required_evaluation = {
        "model", "candidate_id", "propagation_dtype", "sample_count",
        "reference_pooled_rmse", "pooled_rmse", "relative_loss",
        "average_weight_bits", "average_activation_bits",
        "fp16_mac_fraction", "fp16_activation_fraction",
        "hard_deployment_validation",
    }
    for job in jobs:
        final_path = job.output / "final.pt"
        manifest_path = job.output / "manifest.json"
        evaluation_path = job.output / "fixed_evaluation.json"
        if not final_path.is_file() or not manifest_path.is_file() or not \
                evaluation_path.is_file():
            raise RuntimeError(
                "QAT manifest validation failed for %s/%s" %
                (job.model, job.output.name))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8"))
        if set(evaluation) != required_evaluation or \
                manifest["model"] != job.model or \
                manifest["method"] != "mixed_task_aware" or \
                evaluation["model"] != job.model or \
                evaluation["candidate_id"] != job.output.name or \
                evaluation["propagation_dtype"] != "fp16" or \
                int(evaluation["sample_count"]) != 64 or \
                int(evaluation["hard_deployment_validation"]["validated"]) != 1:
            raise RuntimeError(
                "QAT manifest validation failed for %s/%s" %
                (job.model, job.output.name))
        reference = float(evaluation["reference_pooled_rmse"])
        pooled = float(evaluation["pooled_rmse"])
        relative = float(evaluation["relative_loss"])
        values = (
            reference, pooled, relative,
            float(evaluation["average_weight_bits"]),
            float(evaluation["average_activation_bits"]),
            float(evaluation["fp16_mac_fraction"]),
            float(evaluation["fp16_activation_fraction"]),
        )
        if any(not math.isfinite(value) for value in values) or \
                reference <= 0.0 or pooled <= 0.0 or not math.isclose(
                    relative, pooled / reference - 1.0,
                    rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(
                "QAT metric validation failed for %s/%s" %
                (job.model, job.output.name))
        rows.append({
            "model": job.model,
            "candidate_id": evaluation["candidate_id"],
            "reference_pooled_rmse": reference,
            "pooled_rmse": pooled,
            "relative_loss": relative,
            "average_weight_bits": evaluation["average_weight_bits"],
            "average_activation_bits": evaluation["average_activation_bits"],
            "fp16_mac_fraction": evaluation["fp16_mac_fraction"],
            "fp16_activation_fraction":
                evaluation["fp16_activation_fraction"],
            "within_one_percent": int(relative <= 0.01),
            "artifact": str(job.output.resolve()),
        })
    summary_path = Path(phase_root) / "four_model_qat_summary.csv"
    with summary_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = Path(phase_root) / "manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "phase": "qat",
        "results": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def launch(config_path: Path, output: Path, phase: str) -> Path:
    jobs = build_jobs(config_path, output, phase)
    phase_root = Path(output).resolve() / phase
    phase_root.mkdir(parents=True, exist_ok=False)
    if phase == "qat":
        groups = qat_job_groups(jobs)
        for group in groups:
            group[0].log.parent.mkdir(parents=False, exist_ok=False)
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            grouped_results = tuple(executor.map(_run_qat_group, groups))
        results = tuple(result for group in grouped_results
                        for result in group)
    else:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            results = tuple(executor.map(_run_job, jobs))
    failures = tuple(
        (job.model, code, job.command) for job, code in results if code != 0)
    if failures:
        raise RuntimeError("four-model workers failed: %s" % (failures,))
    if phase == "qat":
        return publish_qat_summary(jobs, phase_root)
    return publish_summary(jobs, phase_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch strict four-model mixed-precision evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("anchors", "ptq-search", "qat"), required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    summary = launch(args.config, args.output, args.phase)
    print("four-model summary: %s" % summary)


if __name__ == "__main__":
    main()
