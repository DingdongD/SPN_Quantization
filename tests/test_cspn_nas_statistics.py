import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import report_cspn_encoder_nas as report
from spn_quant.nas.statistics import paired_hierarchical_bootstrap


def test_paired_bootstrap_is_deterministic_and_passes_below_margin():
    control = np.array([
        [0.145, 0.150, 0.155, 0.160],
        [0.146, 0.151, 0.154, 0.159],
        [0.144, 0.149, 0.156, 0.161],
    ])
    candidate = control + 0.001

    first = paired_hierarchical_bootstrap(
        control, candidate, margin_ratio=0.02, replicates=10_000, seed=17)
    second = paired_hierarchical_bootstrap(
        control, candidate, margin_ratio=0.02, replicates=10_000, seed=17)

    assert first == second
    assert first["replicates"] == 10_000
    assert first["delta_rmse"] == pytest.approx(0.001)
    assert first["upper_confidence_bound"] <= first["margin"]
    assert first["noninferior"] is True


def test_paired_bootstrap_fails_above_two_percent_margin():
    control = np.full((3, 20), 0.15)
    candidate = control + 0.004

    result = paired_hierarchical_bootstrap(control, candidate, seed=3)

    assert result["margin"] == pytest.approx(0.003)
    assert result["upper_confidence_bound"] > result["margin"]
    assert result["noninferior"] is False


def _write_metrics(path: Path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("seed", "sample_id", "RMSE", "MAE", "ABS_REL",
                        "DELTA1.25"))
        writer.writeheader()
        writer.writerows(rows)


def test_report_validates_pairs_and_writes_all_formats(tmp_path):
    control_rows = []
    candidate_rows = []
    for seed in (1, 2):
        for sample_id in (10, 20, 30):
            control_rows.append({
                "seed": seed, "sample_id": sample_id, "RMSE": 0.15,
                "MAE": 0.05, "ABS_REL": 0.02, "DELTA1.25": 0.95,
            })
            candidate_rows.append({
                "seed": seed, "sample_id": sample_id, "RMSE": 0.151,
                "MAE": 0.051, "ABS_REL": 0.021, "DELTA1.25": 0.949,
            })
    control = tmp_path / "control.csv"
    candidate = tmp_path / "candidate.csv"
    output = tmp_path / "report"
    _write_metrics(control, control_rows)
    _write_metrics(candidate, candidate_rows)

    result = report.generate_report(control, candidate, output, seed=9)

    assert result["noninferiority"]["noninferior"] is True
    assert result["candidate_metrics"]["MAE"] == pytest.approx(0.051)
    assert json.loads((output / "noninferiority.json").read_text()) == result
    assert (output / "noninferiority.csv").is_file()
    assert "PASS" in (output / "noninferiority.md").read_text()

    _write_metrics(candidate, candidate_rows[:-1])
    with pytest.raises(ValueError, match="identical seed/sample pairs"):
        report.generate_report(control, candidate, output, seed=9)


def test_report_rejects_duplicate_and_nonfinite_rows(tmp_path):
    valid = {
        "seed": 1, "sample_id": 10, "RMSE": 0.15, "MAE": 0.05,
        "ABS_REL": 0.02, "DELTA1.25": 0.95,
    }
    control = tmp_path / "control.csv"
    candidate = tmp_path / "candidate.csv"
    _write_metrics(control, [valid, valid])
    _write_metrics(candidate, [valid])
    with pytest.raises(ValueError, match="duplicate"):
        report.generate_report(control, candidate, tmp_path / "out")

    invalid = dict(valid, RMSE="nan")
    _write_metrics(control, [invalid])
    with pytest.raises(ValueError, match="non-finite"):
        report.generate_report(control, candidate, tmp_path / "out")
