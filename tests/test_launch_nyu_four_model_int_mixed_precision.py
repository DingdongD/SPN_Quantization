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
