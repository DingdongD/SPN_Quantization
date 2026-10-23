import json
from pathlib import Path

import pytest

from scripts.summarize_spn_cross_platform import (
    load_run_means,
    summarize_run_means,
    validate_breakdown,
)


def test_summarize_run_means_uses_independent_process_means():
    summary = summarize_run_means([10.0, 11.0, 9.0, 10.5, 9.5])

    assert summary["process_count"] == 5
    assert summary["mean_ms"] == pytest.approx(10.0)
    assert summary["median_ms"] == pytest.approx(10.0)
    assert summary["sample_std_ms"] == pytest.approx(0.790569415)
    assert summary["cv_percent"] == pytest.approx(7.90569415)
    assert summary["ci95_low_ms"] < 10.0 < summary["ci95_high_ms"]


def test_load_run_means_accepts_gpu_and_rknn_latency_schemas(tmp_path: Path):
    gpu = tmp_path / "rep1.json"
    rknn = tmp_path / "rep2.json"
    gpu.write_text(json.dumps({"latency_ms": {"mean": 3.5}}))
    rknn.write_text(json.dumps({"raw_latency_ms": [4.0, 5.0]}))

    assert load_run_means([gpu, rknn]) == [3.5, 4.5]


def test_load_run_means_rejects_nonfinite_rknn_output(tmp_path: Path):
    result = tmp_path / "invalid.json"
    result.write_text(json.dumps({
        "raw_latency_ms": [4.0, 5.0],
        "output": {"all_finite": False, "nan_count": 2},
    }))

    with pytest.raises(ValueError, match="non-finite output"):
        load_run_means([result])


def test_validate_breakdown_rejects_scaled_or_incomplete_categories():
    valid = {
        "gpu_activity_sum_mean_ms": 10.0,
        "category_mean_gpu_activity_ms": {
            "dense_compute": 4.0,
            "attention": 1.0,
            "sampling": 1.0,
            "propagation": 2.0,
            "elementwise_norm": 1.0,
            "layout_memory": 1.0,
            "other_unclassified": 0.0,
        },
    }
    validate_breakdown(valid)

    invalid = dict(valid)
    invalid["category_mean_gpu_activity_ms"] = dict(
        valid["category_mean_gpu_activity_ms"])
    invalid["category_mean_gpu_activity_ms"]["dense_compute"] = 5.0
    with pytest.raises(ValueError, match="category sum"):
        validate_breakdown(invalid)
