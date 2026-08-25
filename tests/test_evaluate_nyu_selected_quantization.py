import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

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
    assert result[1]["pooled_irmse_delta_inverse_m"] == pytest.approx(0.05)
    assert "pooled_irmse_delta" not in result[1]
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


def _formal_aggregation():
    rows = tuple({
        "model": "dyspn",
        "method": method,
        "configuration": evaluator.METHOD_LABELS[method],
        "samples": 64,
        "valid_pixel_count": 128,
        "squared_error_sum": float(position + 1),
        "pooled_rmse": float(np.sqrt((position + 1) / 128.0)),
        "mean_sample_rmse": float(position + 1) / 100.0,
        "pooled_mae": float(position + 1) / 200.0,
        "pooled_abs_rel": float(position + 1) / 300.0,
        "pooled_irmse": float(position + 1) / 400.0,
        "nonpositive_pixel_count": 0,
        "nonpositive_ratio": 0.0,
    } for position, method in enumerate(evaluator.SELECTED_METHODS))
    return evaluator.FormalAggregation(
        aggregate_metrics=rows,
        sample_metrics=(),
        relative_fp_loss=evaluator.build_relative_loss_rows(rows),
    )


def _write_formal_run_fixtures(tmp_path, index, aggregation):
    metrics = dict((row["method"], row)
                   for row in aggregation.aggregate_metrics)
    assignment = {
        "weight_bits": (("weight", 32),),
        "activation_bits": ((("activation", "module_output"), 32),),
    }
    basis = {
        "weight_macs": (("weight", 1),),
        "activation_elements": ((("activation", "module_output"), 1),),
    }
    for method in evaluator.SELECTED_METHODS:
        directory = tmp_path / "methods" / method
        directory.mkdir(parents=True)
        predictions = tmp_path / "predictions" / method
        predictions.mkdir(parents=True)
        code_rows = [{
            "model": "dyspn", "method": method,
            "sample_index": sample_index, "iteration": -1,
            "owner": "owner", "owner_kind": "test", "numel": 1,
            "zero_code_rate": 0.0, "saturation_rate": 0.0,
        } for sample_index in index.evaluation_indices]
        block_rows = [{
            "model": "dyspn", "method": method,
            "sample_index": sample_index, "iteration": -1,
            "owner": "block", "owner_kind": "quantization_block",
            "elements": 1, "signal_energy": 1.0, "error_energy": 0.0,
            "mse": 0.0, "sqnr_db": 300.0,
        } for sample_index in index.evaluation_indices]
        semantic_rows = []
        for sample_index in index.evaluation_indices:
            semantic_rows.extend(({
                "model": "dyspn", "method": method,
                "sample_index": sample_index, "iteration": -1,
                "owner": "initial_depth", "owner_kind": "semantic_state",
                "elements": 1, "signal_energy": 1.0,
                "error_energy": 0.0, "mse": 0.0, "sqnr_db": 300.0,
            }, {
                "model": "dyspn", "method": method,
                "sample_index": sample_index, "iteration": 0,
                "owner": "propagation_state",
                "owner_kind": "semantic_state",
                "elements": 1, "signal_energy": 1.0,
                "error_energy": 0.0, "mse": 0.0, "sqnr_db": 300.0,
            }))
        diagnostics = {
            "format_version": 1,
            "model": "dyspn",
            "method": method,
            "hard_deployment": int(method != "fp32"),
            "samples": 64,
            "prediction_finite_ratio": 1.0,
            "prediction_nonpositive_ratio": 0.0,
            "weighted_saturation_ratio": None if method == "fp32" else 0.0,
            "weighted_zero_code_ratio": None if method == "fp32" else 0.0,
            "quantization_statistics": [] if method == "fp32"
            else code_rows,
            "propagation_statistics": [],
            "block_output_statistics": block_rows,
            "semantic_state_statistics": semantic_rows,
        }
        diagnostics_path = directory / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        entry = index.methods[method]
        cost = evaluator.assignment_cost_row(method, assignment, basis)
        payload = {
            "format_version": 1,
            "complete": 1,
            "model": "dyspn",
            "method": method,
            "artifact_kind": entry.artifact_kind,
            "artifact": str(entry.artifact.resolve()),
            "artifact_sha256": entry.artifact_sha256,
            "supporting_artifacts": [{
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256,
            } for name, path, sha256 in entry.supporting_artifacts],
            "artifact_index": str(index.source),
            "artifact_index_sha256": index.fingerprint,
            "evaluation_indices": list(index.evaluation_indices),
            "evaluation_identity": index.evaluation_identity,
            "prediction_directory": str(predictions.resolve()),
            "metrics": metrics[method],
            "cost": cost,
            "assignment": assignment,
            "cost_basis": basis,
            "diagnostics": str(diagnostics_path.resolve()),
        }
        (directory / "formal_run.json").write_text(
            json.dumps(payload), encoding="utf-8")


def test_summary_rejects_missing_selected_method(tmp_path):
    path, _ = _write_artifact_index(tmp_path)
    index = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))

    with pytest.raises(FileNotFoundError, match="fp32"):
        evaluator.build_method_summary(
            tmp_path, evaluator.SELECTED_METHODS, index,
            _formal_aggregation())


def test_summary_rejects_prediction_metric_and_recomputed_cost_tampering(
        tmp_path):
    path, _ = _write_artifact_index(tmp_path)
    index = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))
    aggregation = _formal_aggregation()
    _write_formal_run_fixtures(tmp_path, index, aggregation)
    changed = tmp_path / "methods" / "rtn_w4a4" / "formal_run.json"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["metrics"]["pooled_rmse"] = 99.0
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest metrics"):
        evaluator.build_method_summary(
            tmp_path, evaluator.SELECTED_METHODS, index, aggregation)

    payload["metrics"] = dict(aggregation.aggregate_metrics[2])
    payload["cost"]["average_weight_bits"] = 4.0
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest cost"):
        evaluator.build_method_summary(
            tmp_path, evaluator.SELECTED_METHODS, index, aggregation)


def test_summary_rejects_stale_same_model_artifact_index(tmp_path):
    path, _ = _write_artifact_index(tmp_path)
    index = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))
    aggregation = _formal_aggregation()
    _write_formal_run_fixtures(tmp_path, index, aggregation)
    changed = tmp_path / "methods" / "lsqplus_w6a6" / "formal_run.json"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["artifact_index_sha256"] = "b" * 64
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact index"):
        evaluator.build_method_summary(
            tmp_path, evaluator.SELECTED_METHODS, index, aggregation)


def test_summary_publishes_prediction_aggregation_and_recomputed_costs(
        tmp_path):
    path, _ = _write_artifact_index(tmp_path)
    index = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))
    aggregation = _formal_aggregation()
    _write_formal_run_fixtures(tmp_path, index, aggregation)
    rows = evaluator.build_method_summary(
        tmp_path, evaluator.SELECTED_METHODS, index, aggregation)

    evaluator.write_method_summary(tmp_path, index, aggregation, rows)

    payload = json.loads((tmp_path / "selected_method_summary.json").read_text(
        encoding="utf-8"))
    assert payload["artifact_index_sha256"] == index.fingerprint
    assert payload["aggregate_metrics"] == list(aggregation.aggregate_metrics)
    assert payload["runs"][2]["metrics"] == aggregation.aggregate_metrics[2]
    assert payload["runs"][2]["cost"] == evaluator.assignment_cost_row(
        "rtn_w4a4", rows[2]["assignment"], rows[2]["cost_basis"])


def test_summary_rejects_empty_qat_hard_code_diagnostics(tmp_path):
    path, _ = _write_artifact_index(tmp_path)
    index = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))
    aggregation = _formal_aggregation()
    _write_formal_run_fixtures(tmp_path, index, aggregation)
    diagnostics = tmp_path / "methods" / "hawq_mixed_le6" / \
        "diagnostics.json"
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    payload["quantization_statistics"] = []
    diagnostics.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="diagnostics are empty"):
        evaluator.build_method_summary(
            tmp_path, evaluator.SELECTED_METHODS, index, aggregation)


def test_summary_rejects_hard_code_rows_without_sample_identity(tmp_path):
    path, _ = _write_artifact_index(tmp_path)
    index = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))
    aggregation = _formal_aggregation()
    _write_formal_run_fixtures(tmp_path, index, aggregation)
    diagnostics = tmp_path / "methods" / "lsqplus_w4a4" / \
        "diagnostics.json"
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    payload["quantization_statistics"][0]["sample_index"] = -1
    diagnostics.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hard-code sample coverage"):
        evaluator.build_method_summary(
            tmp_path, evaluator.SELECTED_METHODS, index, aggregation)


def test_artifact_index_binds_exact_order_and_file_fingerprints(tmp_path):
    path, payload = _write_artifact_index(tmp_path)

    result = evaluator.load_formal_artifact_index(
        path, "dyspn", tuple(range(64)))

    assert tuple(result.methods) == evaluator.SELECTED_METHODS
    assert result.evaluation_identity == payload["evaluation_identity"]
    assert result.methods["rtn_w8a8"].artifact_kind == \
        "strict_ptq_manifest"
    assert result.source == path.resolve()
    assert result.fingerprint == evaluator.file_sha256(path)


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


def _write_prediction(tmp_path, *, rgb=None, sparse=None):
    identity = evaluator.ordered_evaluation_identity(tuple(range(64)))
    fingerprint = "a" * 64
    target = np.array([[1.0, 2.0]], dtype=np.float32)
    path = evaluator.write_prediction_export(
        root=tmp_path,
        model="dyspn",
        method="fp32",
        sample_index=0,
        evaluation_identity=identity,
        artifact_index_sha256=fingerprint,
        sparse_depth_max_m=10.0,
        rgb=np.zeros((1, 2, 3), dtype=np.float32) if rgb is None else rgb,
        sparse=np.array([[0.0, 2.0]], dtype=np.float32)
        if sparse is None else sparse,
        gt=target,
        pred=target,
    )
    return path, identity, fingerprint


@pytest.mark.parametrize("field,value,error", (
    ("rgb", np.array([[[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                     dtype=np.float32), FloatingPointError),
    ("rgb", np.array([[[1.01, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                     dtype=np.float32), ValueError),
    ("sparse", np.array([[-0.01, 2.0]], dtype=np.float32), ValueError),
    ("sparse", np.array([[0.0, 10.01]], dtype=np.float32), ValueError),
))
def test_prediction_reload_rejects_invalid_rgb_and_sparse_domain(
        tmp_path, field, value, error):
    path, identity, fingerprint = _write_prediction(tmp_path)
    with np.load(path, allow_pickle=False) as source:
        payload = dict((key, source[key]) for key in source.files)
    payload[field] = value
    np.savez_compressed(path, **payload)

    with pytest.raises(error):
        evaluator.load_prediction_export(
            path, "dyspn", "fp32", 0, identity, fingerprint)


def test_prediction_reload_rejects_stale_artifact_index(tmp_path):
    path, identity, fingerprint = _write_prediction(tmp_path)

    with pytest.raises(ValueError, match="artifact index"):
        evaluator.load_prediction_export(
            path, "dyspn", "fp32", 0, identity, "b" * 64)


def test_aggregate_cli_requires_artifact_index():
    with pytest.raises(SystemExit):
        evaluator.parse_args((
            "aggregate", "--config", "config.json", "--model", "dyspn",
            "--output-root", "results",
        ))

    args = evaluator.parse_args((
        "aggregate", "--config", "config.json", "--model", "dyspn",
        "--artifact-index", "formal_artifacts.json",
        "--output-root", "results",
    ))
    assert args.artifact_index.name == "formal_artifacts.json"


def test_paired_tensor_diagnostic_has_complete_identity_and_finite_sqnr():
    row = evaluator.paired_tensor_diagnostic_row(
        reference=torch.tensor([1.0, 2.0]),
        candidate=torch.tensor([1.0, 3.0]),
        model="dyspn",
        method="lsqplus_w4a4",
        sample_index=7,
        iteration=2,
        owner="propagation_state",
        owner_kind="semantic_state",
    )

    assert row["model"] == "dyspn"
    assert row["method"] == "lsqplus_w4a4"
    assert row["sample_index"] == 7
    assert row["iteration"] == 2
    assert row["owner"] == "propagation_state"
    assert row["owner_kind"] == "semantic_state"
    assert row["mse"] == pytest.approx(0.5)
    assert np.isfinite(row["sqnr_db"])


def test_paired_task_diagnostics_require_iteration_alignment():
    reference = SimpleNamespace(
        initial_depth=torch.ones(1, 1, 2, 2),
        propagation_states=(torch.ones(1, 1, 2, 2),),
    )
    candidate = SimpleNamespace(
        initial_depth=torch.ones(1, 1, 2, 2),
        propagation_states=(
            torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),
    )

    with pytest.raises(ValueError, match="propagation iteration"):
        evaluator.paired_task_diagnostic_rows(
            reference, candidate, "dyspn", "rtn_w4a4", 3)


def test_generic_contract_block_capture_handles_structured_outputs():
    class StructuredBlock(nn.Module):
        def forward(self, value):
            return {"second": value + 1.0, "first": (value,)}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = StructuredBlock()

        def forward(self, value):
            return self.encoder(value)["second"]

    contract = SimpleNamespace(blocks=(SimpleNamespace(name="encoder"),))
    model = Model()
    capture = evaluator.ContractBlockOutputCapture(model, contract)
    capture.begin()
    model(torch.ones(1, 1, 2, 2))

    values = capture.values()
    rows = evaluator.paired_block_diagnostic_rows(
        values, values, "nlspn", "fp32", 4)

    assert len(rows) == 1
    assert rows[0]["owner"] == "encoder"
    assert rows[0]["sample_index"] == 4
    assert rows[0]["mse"] == 0.0
    capture.close()
