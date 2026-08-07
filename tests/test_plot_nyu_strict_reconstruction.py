import csv
import json
from pathlib import Path

import numpy as np
import pytest

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


def write_prediction(path, sample_index):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((2, 3), float(sample_index), dtype=np.float32)
    np.savez_compressed(
        str(path), pred=values, gt=values, abs_err=np.zeros_like(values),
        valid_gt=np.ones_like(values, dtype=bool),
        nonfinite=np.zeros_like(values, dtype=bool))


def write_metadata(path, method, manifest, sample_indices=(0, 1)):
    payload = {
        "model": "cspn",
        "evaluation_indices": list(sample_indices),
        "model_provenance": {
            "model_class": "CSPN",
            "model_module": "models.cspn",
            "source_sha256": "source",
            "source_git_commit": "commit",
            "checkpoint_sha256": "checkpoint",
        },
    }
    if method != "rtn":
        payload["reconstruction"] = {
            "manifest": str(manifest),
            "method": "%s_strict" % method,
            "weight_bits": 4,
            "activation_bits": 0,
            "targets": ["head"],
            "exact_weight_contract": 1,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_strict_manifest(path, method, include_alignment=True):
    payload = {
        "strict": 1,
        "method": "%s_strict" % method,
        "model": "cspn",
        "targets": ["head"],
        "weight_bits": 4,
        "activation_bits": 0,
    }
    if include_alignment:
        payload.update({
            "steps": 15000 if method == "adaround" else 20000,
            "round_loss_weight": 0.01,
            "warmup_fraction": 0.2 if method == "adaround" else 0.0,
            "beta_schedule": "cosine" if method == "adaround" else "linear",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_valid_evaluation_root(root):
    for method in analysis.METHOD_ORDER:
        model_root = root / method / "cspn"
        write_metrics(
            model_root / "sample_metrics.csv",
            [0.10, 0.20], [0.20, 0.22])
        manifest = root / "contracts" / method / "manifest.json"
        if method != "rtn":
            write_strict_manifest(manifest, method)
        write_metadata(
            model_root / "metadata.json", method, manifest)
        for config in ("FP32", "HW_W4A8_full"):
            for sample_index in (0, 1):
                write_prediction(
                    model_root / "predictions" / config /
                    ("sample_%05d.npz" % sample_index), sample_index)


def test_validate_evaluation_root_rejects_legacy_adaround_manifest(tmp_path):
    write_valid_evaluation_root(tmp_path)
    legacy = tmp_path / "contracts" / "adaround" / "manifest.json"
    write_strict_manifest(legacy, "adaround", include_alignment=False)

    with pytest.raises(ValueError, match="missing aligned fields"):
        analysis.validate_evaluation_root(
            tmp_path, models=("cspn",), methods=analysis.METHOD_ORDER)


def test_validate_evaluation_root_accepts_aligned_strict_contracts(tmp_path):
    write_valid_evaluation_root(tmp_path)

    analysis.validate_evaluation_root(
        tmp_path, models=("cspn",), methods=analysis.METHOD_ORDER)


def test_validate_evaluation_root_rejects_model_source_mismatch(tmp_path):
    write_valid_evaluation_root(tmp_path)
    path = tmp_path / "adaround" / "cspn" / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_provenance"]["source_sha256"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model provenance mismatch"):
        analysis.validate_evaluation_root(
            tmp_path, models=("cspn",), methods=analysis.METHOD_ORDER)
