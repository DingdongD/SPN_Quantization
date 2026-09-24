import pytest

from scripts import report_cspn_inference_acceleration as report


def test_parse_candidate_assignment_keeps_paths():
    assert report.parse_candidate("18-tf32=report.json=latency.json") == (
        "18-tf32", "report.json", "latency.json")

    with pytest.raises(ValueError, match="NAME=REPORT=LATENCY"):
        report.parse_candidate("missing-fields")


def test_select_fastest_uses_only_noninferior_successful_candidates():
    candidates = [
        {"name": "baseline", "noninferior": True, "status": "ok",
         "median_ms": 20.0},
        {"name": "fast-fail", "noninferior": False, "status": "ok",
         "median_ms": 10.0},
        {"name": "runtime-fail", "noninferior": True, "status": "failed"},
        {"name": "winner", "noninferior": True, "status": "ok",
         "median_ms": 12.0},
    ]

    decision = report.select_fastest(candidates, baseline_name="baseline")

    assert decision["selected"] == "winner"
    assert decision["latency_reduction_ratio"] == pytest.approx(0.4)
    assert decision["speedup"] == pytest.approx(20.0 / 12.0)


def test_select_fastest_requires_a_passing_candidate():
    with pytest.raises(ValueError, match="no non-inferior"):
        report.select_fastest([
            {"name": "failed", "noninferior": False, "status": "ok",
             "median_ms": 1.0},
        ], baseline_name="failed")
