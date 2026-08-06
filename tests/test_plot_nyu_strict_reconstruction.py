import csv
from pathlib import Path

from scripts import plot_nyu_strict_reconstruction as analysis


def write_metrics(path, fp32_values, quant_values, invalid_samples=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "config", "sample_index", "RMSE",
        "nonfinite_pixels", "num_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, value in enumerate(fp32_values):
            writer.writerow({
                "model": "cspn", "config": "FP32",
                "sample_index": index, "RMSE": value,
                "nonfinite_pixels": 0, "num_pixels": 10,
            })
        for index, value in enumerate(quant_values):
            writer.writerow({
                "model": "cspn", "config": "HW_W4A8_full",
                "sample_index": index, "RMSE": value,
                "nonfinite_pixels": 1 if index in invalid_samples else 0,
                "num_pixels": 10,
            })


def test_summary_rejects_nonfinite_and_regressing_contracts(tmp_path):
    write_metrics(
        tmp_path / "rtn" / "cspn" / "sample_metrics.csv",
        [0.10, 0.20], [0.30, float("inf")], invalid_samples=(1,))
    write_metrics(
        tmp_path / "adaround" / "cspn" / "sample_metrics.csv",
        [0.10, 0.20], [0.25, 0.30])
    write_metrics(
        tmp_path / "brecq" / "cspn" / "sample_metrics.csv",
        [0.10, 0.20], [0.20, 0.22])

    rows = analysis.summarize_evaluation(
        tmp_path, models=("cspn",), methods=analysis.METHOD_ORDER)

    lookup = {row["method"]: row for row in rows}
    assert lookup["rtn"]["invalid_samples"] == 1
    assert lookup["adaround"]["deployment_status"] == "accepted"
    assert lookup["brecq"]["deployment_status"] == "accepted"
    assert lookup["brecq"]["mean_rmse"] == 0.21


def test_summary_rejects_finite_method_that_is_worse_than_finite_rtn(tmp_path):
    write_metrics(
        tmp_path / "rtn" / "cspn" / "sample_metrics.csv",
        [0.10, 0.20], [0.20, 0.22])
    write_metrics(
        tmp_path / "adaround" / "cspn" / "sample_metrics.csv",
        [0.10, 0.20], [0.24, 0.26])
    write_metrics(
        tmp_path / "brecq" / "cspn" / "sample_metrics.csv",
        [0.10, 0.20], [0.19, 0.21])

    rows = analysis.summarize_evaluation(
        tmp_path, models=("cspn",), methods=analysis.METHOD_ORDER)

    lookup = {row["method"]: row for row in rows}
    assert lookup["adaround"]["deployment_status"] == "rejected_regression"
    assert lookup["brecq"]["deployment_status"] == "accepted"


def test_representative_index_maximizes_cross_method_spread(tmp_path):
    write_metrics(
        tmp_path / "rtn" / "cspn" / "sample_metrics.csv",
        [0.10, 0.10, 0.10], [0.11, 0.12, 0.13])
    write_metrics(
        tmp_path / "adaround" / "cspn" / "sample_metrics.csv",
        [0.10, 0.10, 0.10], [0.11, 1.20, 0.14])
    write_metrics(
        tmp_path / "brecq" / "cspn" / "sample_metrics.csv",
        [0.10, 0.10, 0.10], [0.11, 0.13, 0.18])

    index = analysis.representative_index(
        tmp_path, "cspn", methods=analysis.METHOD_ORDER)

    assert index == 1
