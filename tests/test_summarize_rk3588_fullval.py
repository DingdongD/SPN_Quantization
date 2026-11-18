import json

import pytest

from scripts import summarize_rk3588_fullval as summary


def _write_success(root, model, precision, rmse):
    samples = [
        {"index": index, "valid_pixels": 2, "rmse_m": rmse,
         "latency_ms": 10.0}
        for index in range(654)
    ]
    payload = {
        "schema_version": 1,
        "model": model,
        "model_path": "/models/%s_%s.rknn" % (model, precision),
        "manifest_path": "/inputs/manifest.tsv",
        "api_version": "2.5.0",
        "driver_version": "0.9.8",
        "sample_count": 654,
        "valid_pixels": 1308,
        "nonfinite_prediction_pixels": 0,
        "nonpositive_prediction_pixels": 0,
        "mean_sample_rmse_m": rmse,
        "pooled_rmse_m": rmse,
        "mae_m": rmse / 2,
        "abs_rel": rmse / 4,
        "mean_forward_latency_ms": 10.0,
        "samples": samples,
    }
    (root / (model + "_" + precision + ".json")).write_text(
        json.dumps(payload), encoding="utf-8")


def _write_seven_successes(root):
    for model, fp16, w8a8 in (
            ("cspn", 0.10, 0.20),
            ("dyspn", 0.20, 0.30),
            ("nlspn", 0.25, 0.50)):
        _write_success(root, model, "fp16", fp16)
        _write_success(root, model, "w8a8", w8a8)
    _write_success(root, "completionformer", "w8a8", 0.40)


def test_summary_requires_seven_complete_runs_and_preserves_failure(tmp_path):
    _write_seven_successes(tmp_path)
    failure = {
        "schema_version": 1,
        "model": "completionformer",
        "precision": "fp16",
        "status": "invalid_nonfinite_output",
        "completed_samples": 144,
        "first_failing_index": 144,
        "valid_pixels": 69312,
        "nonfinite_prediction_pixels": 41358,
    }
    failure_path = tmp_path / "completionformer_fp16_failure.json"
    failure_path.write_text(json.dumps(failure), encoding="utf-8")

    result = summary.summarize(tmp_path, failure_path)

    assert result["successful_run_count"] == 7
    assert result["expected_run_count"] == 8
    assert result["all_runs_accuracy_qualified"] is False
    assert result["runs"]["completionformer"]["fp16"]["status"] == \
        "invalid_nonfinite_output"
    assert result["runs"]["completionformer"]["w8a8"][
        "mean_sample_rmse_m"] == 0.40
    assert result["w8a8_vs_fp16"]["cspn"]["rmse_increase_pct"] == \
        pytest.approx(100.0)
    assert "completionformer" not in result["w8a8_vs_fp16"]


def test_summary_rejects_partial_or_internally_inconsistent_run(tmp_path):
    _write_seven_successes(tmp_path)
    path = tmp_path / "dyspn_fp16.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    failure_path = tmp_path / "completionformer_fp16_failure.json"
    failure_path.write_text(json.dumps({
        "model": "completionformer", "precision": "fp16",
        "status": "invalid_nonfinite_output", "completed_samples": 144,
        "first_failing_index": 144, "valid_pixels": 69312,
        "nonfinite_prediction_pixels": 41358,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="654 sample rows"):
        summary.summarize(tmp_path, failure_path)
