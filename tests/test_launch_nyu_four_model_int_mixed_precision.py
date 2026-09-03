import json
from pathlib import Path

import pytest

from scripts import launch_nyu_four_model_int_mixed_precision as launcher


CONFIG = Path(__file__).resolve().parents[1] / \
    "configs/four_model_int_mixed_precision_1pct.json"


def test_launcher_assigns_each_model_to_its_declared_gpu(tmp_path):
    jobs = launcher.build_jobs(CONFIG, tmp_path, "anchors")

    assert tuple((job.model, job.device) for job in jobs) == (
        ("cspn", "cuda:0"),
        ("dyspn", "cuda:2"),
        ("nlspn", "cuda:1"),
        ("completionformer", "cuda:3"),
    )
    assert all(job.command[-2:] == ("--phase", "anchors") for job in jobs)
    assert all(dict(job.environment)["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
               for job in jobs)


def test_summary_rejects_missing_model_manifest(tmp_path):
    jobs = launcher.build_jobs(CONFIG, tmp_path, "anchors")
    for job in jobs[:-1]:
        job.output.mkdir(parents=True)
        (job.output / "manifest.json").write_text(json.dumps({
            "model": job.model,
            "status": "feasible",
            "reference_sample_count": 64,
            "reference_pooled_rmse": 0.16,
            "propagation_dtype": "fp16",
            "pareto_candidate_ids": [],
        }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest validation"):
        launcher.publish_summary(jobs, tmp_path / "anchors")


def test_qat_jobs_use_only_published_candidates_on_each_model_gpu(tmp_path):
    ptq_root = tmp_path / "ptq-search"
    for model in launcher.MODEL_ORDER:
        model_root = ptq_root / model
        model_root.mkdir(parents=True)
        (model_root / "candidate_assignments.json").write_text(
            json.dumps({"model": model, "candidates": []}),
            encoding="utf-8")
        (model_root / "manifest.json").write_text(json.dumps({
            "model": model,
            "qat_candidate_ids": ["candidate-a", "candidate-b"],
        }), encoding="utf-8")

    jobs = launcher.build_jobs(CONFIG, tmp_path, "qat")

    assert len(jobs) == 8
    assert tuple(job.model for job in jobs) == tuple(
        model for model in launcher.MODEL_ORDER for index in range(2))
    assert tuple(job.device for job in jobs) == (
        "cuda:0", "cuda:0", "cuda:2", "cuda:2",
        "cuda:1", "cuda:1", "cuda:3", "cuda:3",
    )
    assert all("--constrained-candidate-id" in job.command for job in jobs)
    assert all("--checkpoint-protocol" in job.command for job in jobs)

    groups = launcher.qat_job_groups(jobs)
    assert tuple(tuple(job.model for job in group) for group in groups) == (
        ("cspn", "cspn"),
        ("dyspn", "dyspn"),
        ("nlspn", "nlspn"),
        ("completionformer", "completionformer"),
    )


def test_qat_summary_uses_fixed_hard_deployment_evaluations(tmp_path):
    ptq_root = tmp_path / "ptq-search"
    for model in launcher.MODEL_ORDER:
        model_root = ptq_root / model
        model_root.mkdir(parents=True)
        (model_root / "candidate_assignments.json").write_text(
            json.dumps({"model": model, "candidates": []}),
            encoding="utf-8")
        (model_root / "manifest.json").write_text(json.dumps({
            "model": model,
            "qat_candidate_ids": ["candidate"],
        }), encoding="utf-8")
    jobs = launcher.build_jobs(CONFIG, tmp_path, "qat")
    qat_root = tmp_path / "qat"
    qat_root.mkdir()
    for job in jobs:
        job.output.mkdir(parents=True)
        (job.output / "final.pt").write_bytes(b"checkpoint")
        (job.output / "manifest.json").write_text(json.dumps({
            "model": job.model,
            "method": "mixed_task_aware",
        }), encoding="utf-8")
        (job.output / "fixed_evaluation.json").write_text(json.dumps({
            "model": job.model,
            "candidate_id": "candidate",
            "propagation_dtype": "fp16",
            "sample_count": 64,
            "reference_pooled_rmse": 0.15,
            "pooled_rmse": 0.151,
            "relative_loss": 0.151 / 0.15 - 1.0,
            "average_weight_bits": 5.0,
            "average_activation_bits": 6.0,
            "fp16_mac_fraction": 0.01,
            "fp16_activation_fraction": 0.02,
            "hard_deployment_validation": {"validated": 1},
        }), encoding="utf-8")

    summary = launcher.publish_qat_summary(jobs, qat_root)

    rows = summary.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5
    assert all("candidate" in row for row in rows[1:])
