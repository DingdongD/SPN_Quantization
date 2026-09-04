import json
import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts import train_nyu_selected_qat as runner
from scripts import run_nyu_model_hawq_trace as hawq_runner
from scripts import train_nyu_iteration_sweep as sweep
from spn_quant.hawq_trace import BlockTraceEstimate
from spn_quant.mixed_precision import BitAssignment, CostBasis
from spn_quant.model_contracts import (
    PrecisionSearchUnit,
    QuantizationBlock,
    QuantizationModelContract,
)
from spn_quant.qdrop_targets import QDropActivationSite, QDropTargetPlan
from spn_quant.qat.task_loss import ModelTaskLoss


def _contract():
    return QuantizationModelContract(
        model_name="nlspn",
        blocks=(
            QuantizationBlock(
                "encoder", ("encoder",),
                (("activation::encoder::input", "module_input"),)),
            QuantizationBlock(
                "decoder", ("decoder",),
                (("activation::decoder::input", "module_input"),)),
        ),
        prefix_groups=(("encoder",),),
        tail_groups=(("decoder",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )


def test_task_forward_loss_validation_reports_nonfinite_component():
    finite = torch.tensor(1.0)
    loss = ModelTaskLoss(
        total=finite,
        depth=finite,
        boundary=finite,
        teacher=torch.tensor(float("inf")),
        initial_depth=finite,
        propagation=finite,
    )

    with pytest.raises(FloatingPointError, match="teacher"):
        runner.require_finite_task_loss(loss)


def test_training_metrics_do_not_read_each_scalar_with_item(monkeypatch):
    original = torch.Tensor.item
    calls = []

    def counted(tensor, *args):
        calls.append(tensor)
        return original(tensor, *args)

    monkeypatch.setattr(torch.Tensor, "item", counted)
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    prediction = target + 0.1

    metrics = sweep.evaluate_error(target, prediction)
    metric_item_calls = len(calls)

    assert metrics["RMSE"] == pytest.approx(0.1)
    assert metric_item_calls == 0


def _search_contract():
    contract = _contract()
    units = tuple(
        PrecisionSearchUnit(
            name=name,
            members=(name,),
            activation_owners=(("activation::%s::input" % name,
                                "module_input"),),
            kind="initial_depth" if name == "decoder" else "encoder",
            minimum_weight_bits=4,
            minimum_activation_bits=4,
            allow_fp16=name == "decoder",
            scale_policy="static_tensor",
        ) for name in ("encoder", "decoder"))
    return QuantizationModelContract(
        model_name=contract.model_name,
        blocks=contract.blocks,
        prefix_groups=contract.prefix_groups,
        tail_groups=contract.tail_groups,
        protected_roles=contract.protected_roles,
        attention_edges=contract.attention_edges,
        concat_edges=contract.concat_edges,
        protected_modules=contract.protected_modules,
        module_roles=contract.module_roles,
        search_units=units,
    )


def _write_constrained_candidate(tmp_path, checkpoint, *, published=True):
    from scripts.run_nyu_model_hawq_trace import capture_checkpoint_identity

    candidate_id = "ANCHOR_FP16_decoder"
    assignment = {
        "weight_bits": {"encoder": 6, "decoder": 16},
        "activation_bits": {"encoder": 6, "decoder": 16},
        "scale_policies": {
            "encoder": "static_tensor",
            "decoder": "static_tensor",
        },
        "fp16_units": ["decoder"],
    }
    candidates = tmp_path / "candidate_assignments.json"
    candidates.write_text(json.dumps({
        "model": "nlspn",
        "candidates": [{
            "candidate_id": candidate_id,
            "pooled_rmse": 0.150995,
            "relative_loss": 0.01,
            "average_weight_bits": 8.5,
            "average_activation_bits": 8.5,
            "fp16_mac_fraction": 0.25,
            "fp16_activation_fraction": 0.25,
            "assignment": assignment,
        }],
    }), encoding="utf-8")
    identity = capture_checkpoint_identity(checkpoint)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format_version": 1,
        "model": "nlspn",
        "status": "feasible",
        "reference_pooled_rmse": 0.1495,
        "reference_sample_count": 64,
        "maximum_relative_loss": 0.01,
        "anchor": None,
        "pareto_candidate_ids": [],
        "qat_candidate_ids": [candidate_id] if published else [],
        "checkpoint": {
            "path": str(identity.path),
            "sha256": identity.sha256,
        },
        "architecture_class": "NLSPNModel",
        "calibration_indices": list(range(128)),
        "evaluation_indices": list(range(128, 192)),
        "propagation_iterations": 18,
        "propagation_dtype": "fp16",
        "precision_costs": {
            "weight_macs": [["encoder", 3], ["decoder", 1]],
            "activation_elements": [["encoder", 3], ["decoder", 1]],
        },
    }), encoding="utf-8")
    return candidates, manifest, candidate_id


def _trace_settings():
    return {
        "batch_size": 128,
        "probes_per_batch": 1,
        "seed": 20260824,
        "depth_mse_weight": 1.0,
        "boundary_mse_weight": 0.25,
        "boundary_threshold_m": 0.1,
    }


def _checkpoint_payload(path):
    from scripts.run_nyu_model_hawq_trace import (
        capture_checkpoint_identity,
    )
    identity = capture_checkpoint_identity(path)
    return {
        "path": str(identity.path),
        "size_bytes": identity.size_bytes,
        "sha256": identity.sha256,
    }


def _valid_hawq_artifacts(tmp_path, checkpoint):
    indices = tuple(range(128))
    settings = hawq_runner.HAWQTraceSettings(**_trace_settings())
    model = nn.Module()
    model.encoder = nn.Linear(2, 1, bias=False)
    model.decoder = nn.Linear(1, 1, bias=False)
    traces = (
        BlockTraceEstimate(
            "encoder", (4.0,), 4.0, 0.0, 2.0, 0.0, 2),
        BlockTraceEstimate(
            "decoder", (1.0,), 1.0, 0.0, 1.0, 0.0, 1),
    )
    identity = hawq_runner.ordered_sample_identity_sha256("train", indices)
    traced = hawq_runner.HAWQTraceRun(
        traces=traces,
        raw_rows=(
            {"batch_start": 0, "block": "encoder", "probe": 0,
             "estimate": 4.0},
            {"batch_start": 0, "block": "decoder", "probe": 0,
             "estimate": 1.0},
        ),
        calibration_indices=indices,
        settings=settings,
        checkpoint_identity=hawq_runner.capture_checkpoint_identity(
            checkpoint),
    )
    trace_root = tmp_path / "trace"
    trace_root.mkdir()
    trace_path = hawq_runner.write_trace_artifact(
        trace_root,
        model=model,
        contract=_contract(),
        traced=traced,
        weight_macs=(("encoder", 3), ("decoder", 1)),
        activation_traffic=(
            (("activation::encoder::input", "module_input"), 3),
            (("activation::decoder::input", "module_input"), 1),
        ),
        bits=(4, 6, 8),
        model_name="nlspn",
        calibration_identity=identity,
    )
    assignment_root = tmp_path / "allocation"
    assignment_root.mkdir()
    assignment_path = hawq_runner.allocate_trace_artifact(
        trace_path,
        assignment_root,
        expected_model_name="nlspn",
        expected_checkpoint=checkpoint,
        expected_calibration_indices=indices,
        expected_calibration_identity=identity,
        expected_trace_settings=settings,
        bits=(4, 6, 8),
        maximum_weight_bits=6.0,
        maximum_activation_bits=6.0,
    )
    return trace_path, json.loads(
        assignment_path.read_text(encoding="utf-8"))


def _p3_costs():
    return CostBasis(
        weight_macs=(("encoder", 3), ("decoder", 1)),
        activation_elements=(
            (("activation::encoder::input", "module_input"), 3),
            (("activation::decoder::input", "module_input"), 1),
        ),
    )


def _assignment_payload(assignment):
    return {
        "model_name": assignment.model_name,
        "weight_bits": [list(row) for row in assignment.weight_bits],
        "activation_bits": [
            [list(owner), bits] for owner, bits in assignment.activation_bits],
    }


def _valid_p3_payload(checkpoint):
    from scripts.run_nyu_model_p3t3_search import (
        build_p3_candidates,
        build_t3_candidates,
    )
    from spn_quant.mixed_precision import build_registry
    contract = _contract()
    costs = _p3_costs()
    registry = build_registry(contract, costs)
    p3_candidates = build_p3_candidates(
        contract, registry, 4, 4, 8, 8)
    candidates = p3_candidates + build_t3_candidates(
        contract, registry, ("encoder",), 4, 4, 8, 8)
    baseline_rmse = 1.0
    rows = []
    for index, candidate in enumerate(candidates):
        rmse = baseline_rmse - index * 0.05
        weight_map = dict(candidate.assignment.weight_bits)
        activation_map = dict(candidate.assignment.activation_bits)
        normalized_weight = sum(
            weight_map[name] * cost for name, cost in costs.weight_macs
        ) / float(4 * sum(cost for name, cost in costs.weight_macs))
        normalized_activation = sum(
            activation_map[owner] * cost
            for owner, cost in costs.activation_elements
        ) / float(4 * sum(
            cost for owner, cost in costs.activation_elements))
        rows.append({
            "name": candidate.name,
            "stage": candidate.stage,
            "prefix": list(candidate.prefix),
            "tail": list(candidate.tail),
            "pooled_rmse": rmse,
            "mean_sample_rmse": rmse,
            "relative_rmse_loss": rmse / 0.7 - 1.0,
            "normalized_weight_cost": normalized_weight,
            "normalized_activation_cost": normalized_activation,
            "valid": True,
            "metrics_finite": True,
            "sample_rmse": [[128, rmse], [129, rmse]],
            "sample_evidence": [{
                "sample_index": sample_index,
                "squared_error_sum": rmse * rmse,
                "valid_pixels": 1,
                "prediction_finite": True,
                "propagation_valid": True,
                "reproducible": True,
            } for sample_index in (128, 129)],
            "paired_sample_differences": [
                rmse - baseline_rmse, rmse - baseline_rmse],
            "assignment": _assignment_payload(candidate.assignment),
        })
    selected = rows[-1]
    costs_payload = {
        "activation_elements": [
            [list(owner), elements]
            for owner, elements in costs.activation_elements],
        "weight_macs": [list(row) for row in costs.weight_macs],
    }
    evaluation_indices = (128, 129)
    identity_payload = json.dumps(
        [["val", index] for index in evaluation_indices],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "format_version": 3,
        "artifact_kind": "nyu_model_p3_t3_assignment",
        "model_name": "nlspn",
        "source_checkpoint": _checkpoint_payload(checkpoint),
        "prefix": selected["prefix"],
        "tail": selected["tail"],
        "selected_candidate": selected["name"],
        "precision": {
            "base_activation_bits": 4,
            "base_weight_bits": 4,
            "promotion_activation_bits": 8,
            "promotion_weight_bits": 8,
        },
        "budgets": {
            "maximum_normalized_activation_cost": 2.0,
            "maximum_normalized_weight_cost": 2.0,
        },
        "selection_policy": {
            "metric_aggregation": "mean_of_per_sample_rmse",
            "maximum_relative_rmse_loss": 0.10,
            "reference_mean_sample_rmse": 0.7,
            "selected_relative_rmse_loss": selected["relative_rmse_loss"],
        },
        "expected_samples": 2,
        "evaluation": {
            "split": "val",
            "count": 2,
            "indices": list(evaluation_indices),
            "identity_sha256": hashlib.sha256(
                identity_payload).hexdigest(),
        },
        "cost_definition": {
            "activation_denominator": 16,
            "activation_formula":
                "sum(activation_bits*elements)/activation_denominator",
            "weight_denominator": 16,
            "weight_formula":
                "sum(weight_bits*macs)/weight_denominator",
        },
        "cost_basis": costs_payload,
        "assignment": deepcopy(selected["assignment"]),
        "candidates": rows,
    }


def test_selected_qat_methods_are_exact():
    assert runner.selected_qat_methods() == (
        "lsqplus_w4a4",
        "lsqplus_w6a6",
        "hawq_mixed_le6",
        "mixed_task_aware",
    )


def test_method_assignment_paths_are_strict():
    runner.validate_method_assignment_paths(
        "hawq_mixed_le6", "hawq.json", None, "trace.json")
    runner.validate_method_assignment_paths(
        "mixed_task_aware", None, "p3.json", None)
    with pytest.raises(ValueError, match="HAWQ"):
        runner.validate_method_assignment_paths(
            "hawq_mixed_le6", None, None, "trace.json")
    with pytest.raises(ValueError, match="trace artifact"):
        runner.validate_method_assignment_paths(
            "hawq_mixed_le6", "hawq.json", None, None)
    with pytest.raises(ValueError, match="P3/T3"):
        runner.validate_method_assignment_paths(
            "mixed_task_aware", None, None, None)
    with pytest.raises(ValueError, match="uniform"):
        runner.validate_method_assignment_paths(
            "lsqplus_w4a4", "hawq.json", None, None)
    with pytest.raises(ValueError, match="trace artifact"):
        runner.validate_method_assignment_paths(
            "mixed_task_aware", None, "p3.json", "trace.json")


def test_uniform_assignment_covers_contract_owners():
    assignment = runner.uniform_qat_assignment(_contract(), 6)

    assert assignment.model_name == "nlspn"
    assert assignment.weight_bits == (("decoder", 6), ("encoder", 6))
    assert assignment.activation_bits == (
        (("activation::decoder::input", "module_input"), 6),
        (("activation::encoder::input", "module_input"), 6),
    )


def test_mixed_task_assignment_keeps_p3_weights_and_enforces_a6_budget():
    p3 = BitAssignment(
        weight_bits=(("encoder", 4), ("decoder", 8)),
        activation_bits=(
            (("activation::encoder::input", "module_input"), 4),
            (("activation::decoder::input", "module_input"), 8),
        ),
        model_name="nlspn",
    )
    activation_bits = (
        (("activation::encoder::input", "module_input"), 4),
        (("activation::decoder::input", "module_input"), 8),
    )
    costs = CostBasis(
        weight_macs=(("encoder", 1), ("decoder", 1)),
        activation_elements=(
            (("activation::encoder::input", "module_input"), 3),
            (("activation::decoder::input", "module_input"), 1),
        ),
    )

    assignment, audit = runner.mixed_task_aware_assignment(
        _contract(), p3, activation_bits, costs, 6.0)

    assert assignment.weight_bits == p3.weight_bits
    assert audit.average_activation_bits == 5.0
    assert audit.feasible

    over_budget = tuple((owner, 8) for owner, bits in activation_bits)
    with pytest.raises(ValueError, match="budget"):
        runner.mixed_task_aware_assignment(
            _contract(), p3, over_budget, costs, 6.0)


def test_constrained_qat_assignment_preserves_integer_and_fp16_units(
        tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    candidates, manifest, candidate_id = _write_constrained_candidate(
        tmp_path, checkpoint)

    assignment, audit = runner.load_constrained_qat_assignment(
        candidates,
        manifest,
        candidate_id,
        _search_contract(),
        checkpoint,
        tuple(range(128)),
        tuple(range(128, 192)),
        "NLSPNModel",
        18,
        0.015,
    )

    assert assignment.weight_bits == (("encoder", 6),)
    assert assignment.activation_bits == (
        (("activation::encoder::input", "module_input"), 6),)
    assert assignment.fp16_weight_modules == ("decoder",)
    assert assignment.fp16_activation_owners == (
        ("activation::decoder::input", "module_input"),)
    assert audit["candidate_id"] == candidate_id
    assert audit["relative_loss"] == pytest.approx(0.01)
    assert audit["reference_pooled_rmse"] == pytest.approx(0.1495)


def test_constrained_qat_assignment_rejects_unpublished_candidate(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    candidates, manifest, candidate_id = _write_constrained_candidate(
        tmp_path, checkpoint, published=False)

    with pytest.raises(ValueError, match="not published for QAT"):
        runner.load_constrained_qat_assignment(
            candidates,
            manifest,
            candidate_id,
            _search_contract(),
            checkpoint,
            tuple(range(128)),
            tuple(range(128, 192)),
            "NLSPNModel",
            18,
            0.015,
        )


@pytest.mark.parametrize(
    "architecture_class,propagation_iterations,error",
    (
        ("OtherModel", 18, "architecture class differs"),
        ("NLSPNModel", 12, "propagation iterations differ"),
    ),
)
def test_constrained_qat_assignment_rejects_model_protocol_mismatch(
        tmp_path, architecture_class, propagation_iterations, error):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    candidates, manifest, candidate_id = _write_constrained_candidate(
        tmp_path, checkpoint)

    with pytest.raises(ValueError, match=error):
        runner.load_constrained_qat_assignment(
            candidates,
            manifest,
            candidate_id,
            _search_contract(),
            checkpoint,
            tuple(range(128)),
            tuple(range(128, 192)),
            architecture_class,
            propagation_iterations,
            0.015,
        )


def test_constrained_qat_initializes_only_integer_activation_owners(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    candidates, manifest, candidate_id = _write_constrained_candidate(
        tmp_path, checkpoint)
    assignment, audit = runner.load_constrained_qat_assignment(
        candidates, manifest, candidate_id, _search_contract(), checkpoint,
        tuple(range(128)), tuple(range(128, 192)), "NLSPNModel", 18, 0.015)
    del audit
    rows = (
        (("activation::encoder::input", "module_input"), torch.ones(2)),
        (("activation::decoder::input", "module_input"), torch.ones(2)),
    )

    filtered = runner.constrained_initialization_rows(rows, assignment)

    assert filtered == (rows[0],)
    payload = runner.constrained_assignment_payload(assignment)
    assert payload["fp16_weight_modules"] == ["decoder"]
    assert payload["fp16_activation_owners"] == [
        ["activation::decoder::input", "module_input"]]


def test_constrained_qat_mode_requires_fixed_epoch_and_no_legacy_inputs():
    args = SimpleNamespace(
        constrained_candidates=Path("candidates.json"),
        constrained_manifest=Path("manifest.json"),
        constrained_candidate_id="candidate",
        constrained_maximum_relative_loss=0.015,
        launch_spec=None,
        method="mixed_task_aware",
        hawq_assignment=None,
        hawq_trace_artifact=None,
        p3_t3_assignment=None,
        checkpoint_protocol="fixed_final_epoch",
    )

    assert runner.constrained_qat_mode(args)
    args.checkpoint_protocol = "validation_best"
    with pytest.raises(ValueError, match="fixed final epoch"):
        runner.constrained_qat_mode(args)


def test_constrained_model_config_includes_official_cspn():
    path = Path(__file__).resolve().parents[1] / \
        "configs/four_model_int_mixed_precision_1pct.json"

    model = runner.load_constrained_model_config(path, "cspn")

    assert model.model == "cspn"
    assert model.device == "cuda:0"
    assert model.runtime_args().propagation_iterations == 24


def test_fixed_evaluation_loader_preserves_declared_64_sample_order():
    prepared = SimpleNamespace(valset=tuple(range(256)))
    model = SimpleNamespace(evaluation_indices=tuple(range(191, 127, -1)))
    training = {"validation_batch_size": 1, "workers": 0}

    loader = runner.build_fixed_evaluation_loader(
        prepared, model, training)

    assert tuple(int(batch[0]) for batch in loader) == \
        model.evaluation_indices


def test_constrained_final_evaluation_uses_pooled_rmse_and_fp_reference():
    evaluation = {"samples": 64, "pooled_RMSE": 0.151}
    audit = {
        "candidate_id": "candidate",
        "reference_pooled_rmse": 0.15,
        "pooled_rmse": 0.152,
        "relative_loss": 0.152 / 0.15 - 1.0,
        "average_weight_bits": 5.0,
        "average_activation_bits": 6.0,
        "fp16_mac_fraction": 0.01,
        "fp16_activation_fraction": 0.02,
    }

    payload = runner.constrained_final_evaluation_payload(
        "nlspn", evaluation, audit)

    assert payload["pooled_rmse"] == pytest.approx(0.151)
    assert payload["relative_loss"] == pytest.approx(0.151 / 0.15 - 1.0)
    assert payload["sample_count"] == 64
    assert payload["candidate_id"] == "candidate"


def test_qat_history_csv_keeps_train_and_validation_rows(tmp_path):
    path = tmp_path / "qat_history.csv"
    history = (
        {"epoch": 1, "split": "train", "RMSE": 0.2, "loss": 0.1},
        {"epoch": 1, "split": "validation", "RMSE": 0.21,
         "hard_deployment_validated": 1},
    )

    runner.write_qat_history(path, history)

    text = path.read_text(encoding="utf-8")
    assert "epoch,split" in text
    assert "train" in text
    assert "validation" in text


def test_checkpoint_requires_canonical_master_and_hard_validation():
    payload = dict((field, object()) for field in runner.CHECKPOINT_FIELDS)
    payload["format_version"] = 2
    payload["model_name"] = "nlspn"
    payload["method"] = "lsqplus_w4a4"
    payload["model_state"] = {"encoder.weight": torch.ones(1)}
    payload["method_state"] = {"activation.step": torch.tensor([0.25])}
    payload["hard_deployment_validation"] = \
        runner.hard_deployment_evaluation_record(
            2,
            {
                "validated": 1,
                "method": "lsqplus",
                "materialized_weight_count": 1,
                "activation_owner_count": 1,
                "canonical_master_weights": 1,
                "protected_scale_roles_excluded": 1,
            },
            {"samples": 2, "RMSE": 0.25},
            {"encoder.weight": torch.ones(1)},
            payload["method_state"],
            {"activation": (), "propagation": None},
        )
    payload["epoch"] = 2
    payload["deterministic_algorithms"] = True
    payload["calibration_indices"] = tuple(range(128))
    payload["validation_indices"] = (128, 129)
    payload["convergence"] = {"reason": "running"}
    payload["run_state"] = {
        "terminal": False,
        "completed": False,
        "reason": "running",
    }

    runner.validate_checkpoint_payload(payload)

    soft = dict(payload)
    soft["model_state"] = {
        "encoder.parametrizations.weight.original": torch.ones(1),
    }
    with pytest.raises(ValueError, match="canonical FP32"):
        runner.validate_checkpoint_payload(soft)
    nonfinite = dict(payload)
    nonfinite["model_state"] = {
        "encoder.weight": torch.tensor([float("nan")]),
    }
    with pytest.raises(ValueError, match="finite FP32"):
        runner.validate_checkpoint_payload(nonfinite)
    quantized = dict(payload)
    quantized["model_state"] = {
        "encoder.weight": torch.ones(1, dtype=torch.int8),
    }
    with pytest.raises(ValueError, match="finite FP32"):
        runner.validate_checkpoint_payload(quantized)
    stale = dict(payload)
    stale["hard_deployment_validation"] = deepcopy(
        payload["hard_deployment_validation"])
    stale["hard_deployment_validation"]["epoch"] = 1
    with pytest.raises(ValueError, match="evaluation epoch"):
        runner.validate_checkpoint_payload(stale)

    detached_qparams = deepcopy(payload)
    detached_qparams["hard_deployment_validation"]["qparams"] = {
        "activation": ({"scale": 99.0},),
        "propagation": None,
    }
    with pytest.raises(ValueError, match="qparam fingerprint"):
        runner.validate_checkpoint_payload(detached_qparams)

    duplicate_identities = dict(payload)
    duplicate_identities["calibration_indices"] = tuple(range(127)) + (0,)
    with pytest.raises(ValueError, match="calibration identities"):
        runner.validate_checkpoint_payload(duplicate_identities)


def test_hawq_assignment_loader_rejects_direct_average_over_six(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    trace_path, payload = _valid_hawq_artifacts(tmp_path, checkpoint)
    payload["average_activation_bits"] = 6.1
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="six-bit"):
        runner.load_hawq_qat_assignment(
            path, trace_path, _contract(), checkpoint,
            _trace_settings(), 6.0, 6.0)


def test_hawq_assignment_loader_recomputes_cost_weighted_averages(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    trace_path, payload = _valid_hawq_artifacts(tmp_path, checkpoint)
    payload["average_weight_bits"] = 6.0
    payload["average_weight_mac_bits"] = 6.0
    payload["average_activation_bits"] = 6.0
    for family in ("weight_block_bits", "activation_block_bits",
                   "weight_bits", "activation_bits"):
        for row in payload["assignment"][family]:
            row["bits"] = 8
    path = tmp_path / "hawq_tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="recomputed"):
        runner.load_hawq_qat_assignment(
            path, trace_path, _contract(), checkpoint,
            _trace_settings(), 6.0, 6.0)


@pytest.mark.parametrize(("mutation", "message"), (
    (lambda payload: payload["provenance"]["checkpoint"].update(
        {"sha256": "0" * 64}), "checkpoint identity"),
    (lambda payload: payload["provenance"]["trace_settings"].update(
        {"seed": 1}), "trace settings"),
    (lambda payload: payload.update({"solver_success": False}),
     "solver success"),
    (lambda payload: payload.update({"solver_status": "solver failed"}),
     "solver success"),
    (lambda payload: payload.update({"solver_status": "unsuccessful"}),
     "solver success"),
    (lambda payload: payload["provenance"].update(
        {"trace_artifact_sha256": "0" * 64}), "trace artifact fingerprint"),
    (lambda payload: payload["objective"]["components"][0].update(
        {"cost": 99.0}), "objective component"),
    (lambda payload: payload["objective"].update({
        "kind": "weight_hessian_times_squared_quantization_error"}),
     "objective identity"),
    (lambda payload: payload["constraints"].update(
        {"weight_mac_residual": 99.0}), "constraint residual"),
    (lambda payload: payload["assignment"]["weight_bits"][0].update(
        {"bits": 8}), "block and owner assignments"),
    (lambda payload: payload["cost_basis"]["weight_macs"].append(
        {"module": "encoder", "macs": 3}), "cost basis assignment"),
))
def test_hawq_assignment_requires_full_trace_and_solver_provenance(
        tmp_path, mutation, message):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    trace_path, payload = _valid_hawq_artifacts(tmp_path, checkpoint)
    mutation(payload)
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner.load_hawq_qat_assignment(
            path, trace_path, _contract(), checkpoint,
            _trace_settings(), 6.0, 6.0)


def test_hawq_assignment_accepts_exact_selected_artifact(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    trace_path, payload = _valid_hawq_artifacts(tmp_path, checkpoint)
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assignment = runner.load_hawq_qat_assignment(
        path, trace_path, _contract(), checkpoint,
        _trace_settings(), 6.0, 6.0)

    assert dict(assignment.weight_bits) == dict(
        (row["module"], row["bits"])
        for row in payload["assignment"]["weight_bits"])


def test_hawq_assignment_rejects_trace_bytes_changed_after_allocation(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    trace_path, payload = _valid_hawq_artifacts(tmp_path, checkpoint)
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    trace_path.write_bytes(trace_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="trace artifact fingerprint"):
        runner.load_hawq_qat_assignment(
            path, trace_path, _contract(), checkpoint,
            _trace_settings(), 6.0, 6.0)


def test_hawq_assignment_rejects_self_consistent_infeasible_constraints(
        tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    trace_path, payload = _valid_hawq_artifacts(tmp_path, checkpoint)
    maximum_weight = min(
        float(payload["average_weight_bits"]),
        float(payload["average_weight_mac_bits"]),
    ) - 0.5
    payload["constraints"].update({
        "maximum_average_weight_bits": maximum_weight,
        "weight_parameter_residual": maximum_weight -
            float(payload["average_weight_bits"]),
        "weight_mac_residual": maximum_weight -
            float(payload["average_weight_mac_bits"]),
    })
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="constraint assignment is infeasible"):
        runner.load_hawq_qat_assignment(
            path, trace_path, _contract(), checkpoint,
            _trace_settings(), maximum_weight, 6.0)


@pytest.mark.parametrize(("mutation", "message"), (
    (lambda payload: payload["source_checkpoint"].update(
        {"sha256": "0" * 64}), "checkpoint identity"),
    (lambda payload: payload["assignment"]["weight_bits"][0].__setitem__(
        1, 4), "selected candidate assignment"),
    (lambda payload: next(
        row for row in payload["candidates"]
        if row["name"] == payload["selected_candidate"]).update(
            {"valid": False}), "validity evidence"),
    (lambda payload: payload["budgets"].update(
        {"maximum_normalized_weight_cost": 1.5}), "weight budget"),
    (lambda payload: next(
        row for row in payload["candidates"]
        if row["name"] == payload["selected_candidate"]).update(
            {"normalized_activation_cost": 1.5}), "activation cost audit"),
    (lambda payload: next(
        row for row in payload["candidates"]
        if row["name"] == payload["selected_candidate"]).update(
            {"pooled_rmse": -1.0}), "pooled RMSE"),
    (lambda payload: next(
        row for row in payload["candidates"]
        if row["name"] == payload["selected_candidate"])[
            "sample_evidence"][0].update(
                {"squared_error_sum": 0.0}), "sample RMSE"),
    (lambda payload: next(
        row for row in payload["candidates"]
        if row["name"] == payload["selected_candidate"])[
            "sample_evidence"][0].update(
                {"reproducible": False}), "validity evidence"),
    (lambda payload: payload.update(
        {"format_version": 1}), "version"),
    (lambda payload: payload.update(
        {"format_version": 2.0}), "version"),
    (lambda payload: payload["selection_policy"].update(
        {"maximum_relative_rmse_loss": 0.0}), "relative RMSE gate"),
))
def test_p3_t3_assignment_requires_selected_candidate_evidence(
        tmp_path, mutation, message):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    payload = _valid_p3_payload(checkpoint)
    mutation(payload)
    path = tmp_path / "p3_t3_assignment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner.load_p3_t3_qat_assignment(
            path,
            _contract(),
            {
                "base_weight_bits": 4,
                "base_activation_bits": 4,
                "promotion_weight_bits": 8,
                "promotion_activation_bits": 8,
            },
            checkpoint,
            (128, 129),
            _p3_costs(),
            2.0,
            2.0,
            0.10,
        )


def test_p3_t3_assignment_returns_both_cost_audits(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    path = tmp_path / "p3_t3_assignment.json"
    path.write_text(
        json.dumps(_valid_p3_payload(checkpoint)), encoding="utf-8")

    assignment, costs, audit = runner.load_p3_t3_qat_assignment(
        path,
        _contract(),
        {
            "base_weight_bits": 4,
            "base_activation_bits": 4,
            "promotion_weight_bits": 8,
            "promotion_activation_bits": 8,
        },
        checkpoint,
        (128, 129),
        _p3_costs(),
        2.0,
        2.0,
        0.10,
    )

    assert costs == _p3_costs()
    assert assignment.weight_bits == (("decoder", 8), ("encoder", 8))
    assert audit["normalized_weight_cost"] == 2.0
    assert audit["normalized_activation_cost"] == 2.0
    assert audit["weight_feasible"] == 1
    assert audit["activation_feasible"] == 1


@pytest.mark.parametrize(("mutation", "message"), (
    (lambda payload: payload["budgets"].update({
        "maximum_normalized_weight_cost": 3.0}), "configured budget"),
    (lambda payload: payload["cost_basis"]["weight_macs"][0].__setitem__(
        1, 4), "configured cost basis"),
))
def test_p3_t3_assignment_rejects_self_consistent_trust_anchor_drift(
        tmp_path, mutation, message):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    payload = _valid_p3_payload(checkpoint)
    mutation(payload)
    path = tmp_path / "p3_t3_assignment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner.load_p3_t3_qat_assignment(
            path,
            _contract(),
            {
                "base_weight_bits": 4,
                "base_activation_bits": 4,
                "promotion_weight_bits": 8,
                "promotion_activation_bits": 8,
            },
            checkpoint,
            (128, 129),
            _p3_costs(),
            2.0,
            2.0,
            0.10,
        )


def test_activation_range_collector_covers_contract_owners_exactly():
    model = nn.Module()
    model.encoder = nn.Conv2d(1, 1, 1, bias=False)
    model.decoder = nn.Conv2d(1, 1, 1, bias=False)
    plan = QDropTargetPlan(
        model="nlspn",
        blocks=("decoder", "encoder"),
        activation_sites=(
            QDropActivationSite(
                "activation::decoder::input", "decoder", "module_input",
                "module_input", True, True),
            QDropActivationSite(
                "activation::encoder::input", "encoder", "module_input",
                "module_input", True, True),
        ),
        excluded_sites=(),
    )
    collector = runner.ModelActivationRangeCollector(
        model, _contract(), plan)

    model.decoder(model.encoder(torch.tensor([[[[-2.0, 3.0]]]])))
    rows = dict(collector.initialization_rows(None))

    assert set(rows) == {
        ("activation::encoder::input", "module_input"),
        ("activation::decoder::input", "module_input"),
    }
    assert torch.equal(
        rows[("activation::encoder::input", "module_input")],
        torch.tensor([-2.0, 3.0]))
    collector.close()


def test_hard_deployment_record_is_bound_to_evaluation_epoch():
    hard_state = {"encoder.weight": torch.tensor([4.0])}
    method_state = {"activation.step": torch.tensor([0.25])}
    qparams = {
        "activation": ({
            "owner": ("activation::encoder::input", "module_input"),
            "bits": 4,
            "unsigned": 0,
            "qmin": -8,
            "qmax": 7,
            "scale": 0.25,
            "offset": 0.0,
        },),
        "propagation": {
            "maximum": (("state", 2.0),),
            "config": {
                "affinity_bits": 8,
                "confidence_bits": 8,
                "offset_bits": 8,
                "state_bits": 8,
                "coefficient_fraction_bits": 13,
            },
            "frozen": True,
        },
    }
    record = runner.hard_deployment_evaluation_record(
        3,
        {
            "validated": 1,
            "method": "lsqplus",
            "materialized_weight_count": 1,
            "activation_owner_count": 1,
            "canonical_master_weights": 1,
            "protected_scale_roles_excluded": 1,
        },
        {"samples": 17, "RMSE": 0.25},
        hard_state,
        method_state,
        qparams,
    )

    assert record["validated"] == 1
    assert record["materialized_weight_count"] == 1
    assert record["epoch"] == 3
    assert record["evaluation_samples"] == 17
    assert record["evaluation_rmse"] == 0.25
    assert len(record["hard_model_state_sha256"]) == 64
    assert len(record["method_state_sha256"]) == 64
    assert len(record["qparams_sha256"]) == 64
    assert len(record["deployment_fingerprint"]) == 64
    runner.validate_hard_deployment_record(record, 3, method_state)
    changed_method = {"activation.step": torch.tensor([0.5])}
    with pytest.raises(ValueError, match="method state fingerprint"):
        runner.validate_hard_deployment_record(record, 3, changed_method)
    changed_metric = deepcopy(record)
    changed_metric["evaluation_rmse"] = 0.5
    with pytest.raises(ValueError, match="deployment fingerprint"):
        runner.validate_hard_deployment_record(
            changed_metric, 3, method_state)
    with pytest.raises(ValueError, match="hard deployment"):
        runner.hard_deployment_evaluation_record(
            3, {"validated": 0}, {"samples": 17, "RMSE": 0.25},
            hard_state, method_state, qparams)


def test_hard_deployment_validation_rejects_evaluation_state_mutation():
    before = {
        "model": {"encoder.weight": torch.tensor([1.0])},
        "method": {"activation.step": torch.tensor([0.25])},
    }
    unchanged = {
        "model": {"encoder.weight": torch.tensor([1.0])},
        "method": {"activation.step": torch.tensor([0.25])},
    }

    runner.validate_hard_deployment_stability(before, unchanged)

    changed = dict(unchanged)
    changed["method"] = {"activation.step": torch.tensor([0.5])}
    with pytest.raises(RuntimeError, match="mutated"):
        runner.validate_hard_deployment_stability(before, changed)


def test_hard_epoch_evaluates_fresh_materialized_context_not_live_model():
    class LiveModel(object):
        def eval(self):
            return self

        def __call__(self, value):
            raise AssertionError("live parametrized model was evaluated")

    class HardModel(object):
        def eval(self):
            return self

        def __call__(self, value):
            return value * 9.0

        @staticmethod
        def state_dict():
            return {"encoder.weight": torch.tensor([9.0])}

    class Runtime(object):
        model_name = "dyspn"

        @staticmethod
        def model_input(sample, device):
            del device
            return (sample["input"],), sample["target"]

        @staticmethod
        def prediction(output):
            return output

    propagation_rows = (
        {"signal": "state", "mse": 0.0},
        {"signal": "affinity_constraints",
         "coefficient_sum_max_error": 0.0,
         "contraction_violation_rate": 0.0},
        {"signal": "anchor", "anchor_max_error": 0.0},
    )
    qparams = {
        "activation": (),
        "propagation": {"maximum": (("state", 1.0),)},
    }
    controller = SimpleNamespace(
        activation_modules=SimpleNamespace(eval=lambda: None),
        hard_model_state_dict=lambda: {
            "encoder.weight": torch.tensor([9.0])},
        method_state_dict=lambda: {
            "activation.step": torch.tensor([0.25])},
        deployment_qparams=lambda: deepcopy(qparams),
        hard_deployment_manifest=lambda: {
            "validated": 1,
            "method": "lsqplus",
            "materialized_weight_count": 1,
            "activation_owner_count": 0,
            "canonical_master_weights": 1,
            "protected_scale_roles_excluded": 1,
        },
    )
    prepared = SimpleNamespace(
        model=LiveModel(),
        controller=controller,
    )
    observed = {}

    def context_factory(prepared_arg, model_config, training, hard_state,
                        method_state, deployment_qparams, device):
        assert prepared_arg is prepared
        assert model_config == "model-config"
        assert training == {
            "fold_conv_bn": False,
            "fold_max_error": 0.0,
        }
        assert hard_state["encoder.weight"].item() == 9.0
        assert method_state["activation.step"].item() == 0.25
        assert deployment_qparams == qparams
        assert str(device) == "cpu"
        context = SimpleNamespace(
            model=HardModel(),
            student_runtime=Runtime(),
            propagation=SimpleNamespace(
                statistics=lambda: propagation_rows),
            controller=SimpleNamespace(
                activation_modules=SimpleNamespace(eval=lambda: None),
                deployment_qparams=lambda: deepcopy(qparams)),
            close=lambda: observed.update({"closed": True}),
        )
        observed["context"] = context
        return context

    loader = ({
        "input": torch.ones(1, 1, 1, 1),
        "target": torch.ones(1, 1, 1, 1),
    },)
    evaluation, record = runner._evaluate_hard_deployment_epoch(
        prepared,
        loader,
        torch.device("cpu"),
        4,
        "model-config",
        {"fold_conv_bn": False, "fold_max_error": 0.0},
        context_factory=context_factory,
    )

    assert evaluation["RMSE"] == 8.0
    assert record["evaluation_rmse"] == 8.0
    assert observed["closed"]


def test_dyspn_qat_train_mode_keeps_stochastic_depth_deterministic():
    class StoDepth_SE_BasicBlock(torch.nn.Module):
        def forward(self, value):
            return value

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 1),
        StoDepth_SE_BasicBlock(),
        torch.nn.BatchNorm2d(1),
    )

    runner.set_model_qat_train_mode("dyspn", model)

    assert model.training
    assert model[0].training
    assert not model[1].training
    assert not model[2].training


def test_cspn_qat_train_mode_uses_common_qat_mode():
    model = torch.nn.Linear(2, 2)
    runner.set_model_qat_train_mode("cspn", model)
    assert model.training


def test_terminal_checkpoint_resume_position_runs_no_new_epoch():
    terminal = {
        "epoch": 4,
        "run_state": {
            "terminal": True,
            "completed": True,
            "reason": "validation_plateau",
        },
    }
    running = {
        "epoch": 4,
        "run_state": {
            "terminal": False,
            "completed": False,
            "reason": "running",
        },
    }

    assert runner.checkpoint_resume_epoch(terminal) is None
    assert runner.checkpoint_resume_epoch(running) == 5


def test_fixed_final_epoch_protocol_ignores_validation_for_control():
    from scripts import train_nyu_cspn_group_a4_qat as qat_base

    training = {
        "checkpoint_protocol": "fixed_final_epoch",
        "epochs": 5,
        "patience": 2,
    }
    tracker = qat_base.QATConvergenceTracker(
        training["epochs"], runner._tracker_patience(training), 0.001)
    stops = tuple(tracker.update(epoch, float(epoch))
                  for epoch in range(1, 6))

    assert stops == (False, False, False, False, True)
    assert runner._scheduler_metric(
        training, {"RMSE": 0.25}, {"RMSE": 9.0}) == 0.25
    assert not runner._publish_best_checkpoint(training)


def test_validation_best_protocol_preserves_legacy_control():
    training = {
        "checkpoint_protocol": "validation_best",
        "epochs": 5,
        "patience": 2,
    }

    assert runner._tracker_patience(training) == 2
    assert runner._scheduler_metric(
        training, {"RMSE": 9.0}, {"RMSE": 0.25}) == 0.25
    assert runner._publish_best_checkpoint(training)


def test_hawq_terminal_validation_freezes_ranges_before_re_evaluation():
    events = []

    class Controller:
        def freeze_activation_ranges(self):
            events.append("freeze")

    prepared = SimpleNamespace(controller=Controller())

    def evaluator(prepared_arg, loader, device, epoch, model_config,
                  training):
        assert prepared_arg is prepared
        assert events == ["freeze"]
        events.append("evaluate")
        return {
            "RMSE": 0.25,
            "samples": 3,
        }, {"deployment_fingerprint": "frozen"}

    evaluation, record = runner._freeze_and_revalidate_terminal_hawq(
        prepared,
        "loader",
        torch.device("cpu"),
        4,
        "model-config",
        {"fold_conv_bn": False, "fold_max_error": 0.0},
        {"RMSE": 0.25, "samples": 3},
        evaluator=evaluator,
    )

    assert events == ["freeze", "evaluate"]
    assert evaluation == {"RMSE": 0.25, "samples": 3}
    assert record == {"deployment_fingerprint": "frozen"}


@pytest.mark.parametrize(("field", "changed"), (
    ("RMSE", 0.5),
    ("samples", 2),
))
def test_hawq_terminal_revalidation_rejects_metric_drift(field, changed):
    prepared = SimpleNamespace(
        controller=SimpleNamespace(freeze_activation_ranges=lambda: None))

    def evaluator(*args, **kwargs):
        del args, kwargs
        evaluation = {"RMSE": 0.25, "samples": 3}
        evaluation[field] = changed
        return evaluation, {"deployment_fingerprint": "frozen"}

    with pytest.raises(RuntimeError, match="terminal HAWQ"):
        runner._freeze_and_revalidate_terminal_hawq(
            prepared,
            "loader",
            torch.device("cpu"),
            4,
            "model-config",
            {"fold_conv_bn": False, "fold_max_error": 0.0},
            {"RMSE": 0.25, "samples": 3},
            evaluator=evaluator,
        )


@pytest.mark.parametrize(("family", "mutation", "message"), (
    ("optimizer_state",
     lambda state: state["param_groups"][0].update({"momentum": 0.1}),
     "optimizer"),
    ("optimizer_state",
     lambda state: state["state"].update({
         0: {"momentum_buffer": torch.ones(1, dtype=torch.float64)}}),
     "momentum state"),
    ("scheduler_state",
     lambda state: state.update({"factor": 0.5}),
     "scheduler"),
    ("convergence",
     lambda state: state.update({"patience": 99}),
     "convergence"),
    ("scheduler_state",
     lambda state: state.update({"_last_lr": [0.5]}),
     "scheduler and optimizer LR"),
    ("scheduler_state",
     lambda state: state.update({"best": "invalid"}),
     "scheduler best"),
    ("convergence",
     lambda state: state.update({"best_epoch": -1}),
     "convergence state"),
))
def test_resume_validates_optimizer_scheduler_tracker_subcontracts(
        family, mutation, message):
    from scripts import train_nyu_cspn_group_a4_qat as qat_base
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        (parameter,), lr=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
    )
    tracker = qat_base.QATConvergenceTracker(10, 4, 0.001)
    payload = {
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "convergence": tracker.state_dict(),
        "run_state": {
            "terminal": False,
            "completed": False,
            "reason": "running",
        },
    }
    changed = deepcopy(payload)
    mutation(changed[family])

    with pytest.raises(ValueError, match=message):
        runner.validate_training_state_subcontracts(
            changed, optimizer, scheduler, tracker)

    runner.validate_training_state_subcontracts(
        payload, optimizer, scheduler, tracker)
