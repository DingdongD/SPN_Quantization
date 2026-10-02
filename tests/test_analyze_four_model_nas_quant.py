import csv
import json

import pytest

from scripts import analyze_four_model_nas_quant as analysis


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model_result(root, model):
    model_root = root / model
    model_root.mkdir(parents=True)
    manifest = {
        "model": model,
        "reference_pooled_rmse": 1.0,
        "maximum_relative_loss": 0.01,
        "evaluation_indices": list(range(64)),
        "anchor": {
            "candidate_id": "UNIFORM_W8A8", "pooled_rmse": 1.002,
            "relative_loss": 0.002,
            "assignment": {"weight_bits": {"encoder": 8},
                           "activation_bits": {"encoder": 8},
                           "fp16_units": []},
        },
        "precision_costs": {
            "weight_macs": [["encoder", 100]],
            "activation_elements": [["encoder", 200]],
        },
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_csv(model_root / "single_module_ablation.csv", [
        {"candidate_id": "SINGLE_encoder_W4A8", "pooled_rmse": 1.003,
         "relative_loss": 0.003},
        {"candidate_id": "SINGLE_encoder_W8A4", "pooled_rmse": 1.020,
         "relative_loss": 0.020},
    ])
    _write_csv(model_root / "pareto_ptq.csv", [
        {"candidate_id": "WEIGHT", "pooled_rmse": 1.003,
         "relative_loss": 0.003, "average_weight_bits": 5.5,
         "average_activation_bits": 8.0, "valid": "True"},
        {"candidate_id": "ACTIVATION", "pooled_rmse": 1.004,
         "relative_loss": 0.004, "average_weight_bits": 8.0,
         "average_activation_bits": 6.5, "valid": "True"},
    ])


def test_analysis_separates_incremental_anchor_loss_and_search_evidence(tmp_path):
    source = tmp_path / "source"
    for model in analysis.MODEL_ORDER:
        _model_result(source, model)

    report = analysis.run(source, tmp_path / "out")

    rows = list(csv.DictReader(
        (tmp_path / "out" / "quantization_sensitivity.csv").open()))
    weight = next(row for row in rows
                  if row["model"] == "cspn" and row["axis"] == "weight")
    activation = next(row for row in rows
                      if row["model"] == "cspn" and row["axis"] == "activation")
    assert float(weight["increment_vs_anchor_pct"]) == pytest.approx(0.1)
    assert weight["classification"] == "safe"
    assert activation["classification"] == "sensitive"
    text = report.read_text(encoding="utf-8")
    assert "Architecture candidates" in text
    assert "not trained accuracy results" in text


def test_analysis_marks_unmeasured_protected_unit(tmp_path):
    source = tmp_path / "source"
    for model in analysis.MODEL_ORDER:
        _model_result(source, model)
    manifest_path = source / "nlspn" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["anchor"]["assignment"]["weight_bits"]["early_boundary"] = 16
    manifest["anchor"]["assignment"]["activation_bits"]["early_boundary"] = 16
    manifest["anchor"]["assignment"]["fp16_units"] = ["early_boundary"]
    manifest["precision_costs"]["weight_macs"].append(["early_boundary", 10])
    manifest["precision_costs"]["activation_elements"].append(["early_boundary", 20])
    manifest_path.write_text(json.dumps(manifest))

    analysis.run(source, tmp_path / "out")

    rows = list(csv.DictReader(
        (tmp_path / "out" / "quantization_sensitivity.csv").open()))
    protected = [row for row in rows if row["model"] == "nlspn" and
                 row["unit"] == "early_boundary"]
    assert len(protected) == 2
    assert {row["classification"] for row in protected} == {"protected"}
