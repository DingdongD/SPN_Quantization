import json

import numpy as np
import pytest

from scripts import evaluate_nyu_selected_quantization as evaluator


def test_selected_method_order_is_exact():
    assert evaluator.SELECTED_METHODS == (
        "fp32",
        "rtn_w8a8",
        "rtn_w4a4",
        "qdrop_w6a6",
        "brecq_w6a6",
        "hawq_mixed_le6",
        "lsqplus_w6a6",
        "lsqplus_w4a4",
        "mixed_task_aware",
        "p3_t3_mixed_ptq",
    )
    assert evaluator.SELECTED_CONFIGURATION_LABELS == (
        "FP32",
        "RTN W8A8",
        "RTN W4A4",
        "QDrop W6A6",
        "BRECQ W6A6",
        "HAWQ mixed<=6",
        "LSQ++ W6A6",
        "LSQ++ W4A4",
        "mixed task-aware QAT",
        "P3/T3 mixed PTQ",
    )


def test_pooled_rmse_uses_global_squared_error_sum():
    records = (
        {
            "sample_index": 7,
            "gt": np.array([[1.0, 2.0]], dtype=np.float32),
            "pred": np.array([[2.0, 3.0]], dtype=np.float32),
        },
        {
            "sample_index": 11,
            "gt": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "pred": np.array([[4.0, 8.0, 8.0, 8.0]], dtype=np.float32),
        },
    )

    result = evaluator.aggregate_predictions(records)

    assert result.squared_error_sum == pytest.approx(11.0)
    assert result.valid_pixel_count == 3
    assert result.pooled_rmse == pytest.approx(np.sqrt(11.0 / 3.0))
    assert result.mean_sample_rmse == pytest.approx(2.0)


def test_aggregate_predictions_rejects_nonfinite_values_outside_gt_mask():
    records = ({
        "sample_index": 0,
        "gt": np.array([[1.0, 0.0]], dtype=np.float32),
        "pred": np.array([[1.0, np.nan]], dtype=np.float32),
    },)

    with pytest.raises(FloatingPointError, match="prediction"):
        evaluator.aggregate_predictions(records)


def test_summary_rejects_missing_selected_method(tmp_path):
    with pytest.raises(FileNotFoundError, match="fp32"):
        evaluator.build_method_summary(tmp_path, evaluator.SELECTED_METHODS)


def test_summary_rejects_incomplete_or_identity_mismatched_run(tmp_path):
    indices = tuple(range(64))
    identity = evaluator.ordered_evaluation_identity(indices)
    methods = tmp_path / "methods"
    for method in evaluator.SELECTED_METHODS:
        directory = methods / method
        directory.mkdir(parents=True)
        artifact = directory / "source.bin"
        artifact.write_bytes(method.encode("ascii"))
        payload = {
            "format_version": 1,
            "complete": 1,
            "model": "dyspn",
            "method": method,
            "artifact_kind": "official_checkpoint" if method == "fp32"
            else ("terminal_qat_checkpoint"
                  if method in evaluator.QAT_METHODS
                  else "strict_ptq_manifest"),
            "artifact": str(artifact.resolve()),
            "artifact_sha256": evaluator.file_sha256(artifact),
            "evaluation_indices": list(indices),
            "evaluation_identity": identity,
            "prediction_directory": str(
                (tmp_path / "predictions" / method).resolve()),
            "metrics": {
                "pooled_rmse": 1.0,
                "mean_sample_rmse": 1.0,
            },
            "cost": {
                "average_weight_bits": 32.0,
                "average_activation_bits": 32.0,
            },
            "diagnostics": str(
                (directory / "diagnostics.json").resolve()),
        }
        (directory / "diagnostics.json").write_text(
            "{}\n", encoding="utf-8")
        (tmp_path / "predictions" / method).mkdir(parents=True)
        (directory / "formal_run.json").write_text(
            json.dumps(payload), encoding="utf-8")

    changed = methods / "rtn_w4a4" / "formal_run.json"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["evaluation_identity"] = "0" * 64
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation identity"):
        evaluator.build_method_summary(
            tmp_path,
            evaluator.SELECTED_METHODS,
            expected_model="dyspn",
            expected_indices=indices,
        )


def test_cost_row_uses_explicit_weight_and_activation_costs():
    assignment = {
        "weight_bits": (("first", 4), ("second", 8)),
        "activation_bits": (
            (("first-output", "module_output"), 6),
            (("second-output", "module_output"), 8),
        ),
    }
    basis = {
        "weight_macs": (("first", 3), ("second", 1)),
        "activation_elements": (
            (("first-output", "module_output"), 1),
            (("second-output", "module_output"), 3),
        ),
    }

    row = evaluator.assignment_cost_row("rtn_w4a4", assignment, basis)

    assert row["average_weight_bits"] == pytest.approx(5.0)
    assert row["average_activation_bits"] == pytest.approx(7.5)
    assert row["weight_w4_share"] == pytest.approx(0.75)
    assert row["weight_w8_share"] == pytest.approx(0.25)
    assert row["activation_a6_share"] == pytest.approx(0.25)
    assert row["activation_a8_share"] == pytest.approx(0.75)


def test_relative_loss_labels_pooled_and_sample_rmse_separately():
    rows = (
        {
            "model": "dyspn",
            "method": "fp32",
            "configuration": "FP32",
            "pooled_rmse": 0.5,
            "mean_sample_rmse": 0.4,
            "pooled_mae": 0.2,
            "pooled_abs_rel": 0.1,
            "pooled_irmse": 0.25,
        },
        {
            "model": "dyspn",
            "method": "rtn_w8a8",
            "configuration": "RTN W8A8",
            "pooled_rmse": 0.55,
            "mean_sample_rmse": 0.44,
            "pooled_mae": 0.22,
            "pooled_abs_rel": 0.12,
            "pooled_irmse": 0.30,
        },
    )

    result = evaluator.build_relative_loss_rows(rows)

    assert result[1]["pooled_rmse_delta_m"] == pytest.approx(0.05)
    assert result[1]["pooled_rmse_relative_percent"] == pytest.approx(10.0)
    assert result[1]["mean_sample_rmse_delta_m"] == pytest.approx(0.04)
    assert result[1]["mean_sample_rmse_relative_percent"] == \
        pytest.approx(10.0)
    assert result[1]["pooled_mae_relative_percent"] == pytest.approx(10.0)
    assert result[1]["pooled_abs_rel_relative_percent"] == pytest.approx(20.0)
    assert result[1]["pooled_irmse_relative_percent"] == pytest.approx(20.0)


def _write_artifact_index(tmp_path):
    indices = tuple(range(64))
    rows = []
    ptq_manifests = {}
    for method in evaluator.SELECTED_METHODS:
        artifact = tmp_path / (method + ".artifact")
        artifact.write_bytes((method + "\n").encode("ascii"))
        kind = "official_checkpoint" if method == "fp32" else (
            "terminal_qat_checkpoint"
            if method in evaluator.QAT_METHODS else
            "strict_ptq_manifest")
        support_names = {
            "hawq_mixed_le6": (
                "hawq_assignment", "hawq_trace_artifact"),
            "mixed_task_aware": ("p3_t3_assignment",),
            "p3_t3_mixed_ptq": ("p3_t3_assignment",),
        }.get(method, ())
        supports = []
        for name in support_names:
            support = tmp_path / (
                "shared.p3_t3_assignment" if name == "p3_t3_assignment"
                else method + "." + name)
            support.write_bytes(name.encode("ascii"))
            supports.append({
                "name": name,
                "path": str(support.resolve()),
                "sha256": evaluator.file_sha256(support),
            })
        rows.append({
            "method": method,
            "artifact_kind": kind,
            "artifact": str(artifact.resolve()),
            "artifact_sha256": evaluator.file_sha256(artifact),
            "supporting_artifacts": supports,
        })
        if method in evaluator.PTQ_METHODS:
            ptq_manifests[method] = str(artifact.resolve())
    ptq_matrix = tmp_path / "selected_ptq_matrix.json"
    ptq_matrix.write_text(json.dumps({
        "format_version": 1,
        "model": "dyspn",
        "methods": list(evaluator.PTQ_METHODS),
        "calibration_identity": "1" * 64,
        "evaluation_identity": evaluator.ordered_evaluation_identity(indices),
        "hard_deployment_manifests": ptq_manifests,
    }), encoding="utf-8")
    payload = {
        "format_version": 1,
        "model": "dyspn",
        "evaluation_indices": list(indices),
        "evaluation_identity": evaluator.ordered_evaluation_identity(indices),
        "ptq_matrix": {
            "path": str(ptq_matrix.resolve()),
            "sha256": evaluator.file_sha256(ptq_matrix),
        },
        "methods": rows,
        "preparation": {
            "fold_conv_bn": 1,
            "fold_max_error": 1e-5,
            "joint_clip_factors": [1.0],
            "joint_search_rounds": 1,
            "joint_cache_sample_limit": 1,
            "joint_cache_byte_limit": 1024,
        },
    }
    path = tmp_path / "formal_artifacts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_artifact_index_binds_exact_order_and_file_fingerprints(tmp_path):
    path, payload = _write_artifact_index(tmp_path)

    result = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))

    assert tuple(result.methods) == evaluator.SELECTED_METHODS
    assert result.evaluation_identity == payload["evaluation_identity"]
    assert result.methods["rtn_w8a8"].artifact_kind == \
        "strict_ptq_manifest"


def test_artifact_index_rejects_missing_method_and_changed_bytes(tmp_path):
    path, payload = _write_artifact_index(tmp_path)
    payload["methods"] = payload["methods"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="method order"):
        evaluator.load_formal_artifact_index(
            path, "dyspn", tuple(range(64)))

    path, payload = _write_artifact_index(tmp_path)
    artifact = tmp_path / "lsqplus_w4a4.artifact"
    artifact.write_bytes(b"changed\n")

    with pytest.raises(RuntimeError, match="fingerprint"):
        evaluator.load_formal_artifact_index(
            path, "dyspn", tuple(range(64)))


def test_artifact_index_rejects_incomplete_ptq_matrix(tmp_path):
    path, payload = _write_artifact_index(tmp_path)
    matrix = tmp_path / "selected_ptq_matrix.json"
    matrix_payload = json.loads(matrix.read_text(encoding="utf-8"))
    matrix_payload["methods"] = matrix_payload["methods"][:-1]
    matrix.write_text(json.dumps(matrix_payload), encoding="utf-8")
    payload["ptq_matrix"]["sha256"] = evaluator.file_sha256(matrix)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="PTQ method order"):
        evaluator.load_formal_artifact_index(
            path, "dyspn", tuple(range(64)))


def test_artifact_index_rejects_missing_hawq_trace_link(tmp_path):
    path, payload = _write_artifact_index(tmp_path)
    hawq = next(row for row in payload["methods"]
                if row["method"] == "hawq_mixed_le6")
    hawq["supporting_artifacts"] = hawq["supporting_artifacts"][:1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="supporting artifact set"):
        evaluator.load_formal_artifact_index(
            path, "dyspn", tuple(range(64)))
