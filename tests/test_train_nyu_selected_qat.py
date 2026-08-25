import json

import pytest
import torch
import torch.nn as nn

from scripts import train_nyu_selected_qat as runner
from spn_quant.mixed_precision import BitAssignment, CostBasis
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)
from spn_quant.qdrop_targets import QDropActivationSite, QDropTargetPlan


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


def test_selected_qat_methods_are_exact():
    assert runner.selected_qat_methods() == (
        "lsqplus_w4a4",
        "lsqplus_w6a6",
        "hawq_mixed_le6",
        "mixed_task_aware",
    )


def test_method_assignment_paths_are_strict():
    runner.validate_method_assignment_paths(
        "hawq_mixed_le6", "hawq.json", None)
    runner.validate_method_assignment_paths(
        "mixed_task_aware", None, "p3.json")
    with pytest.raises(ValueError, match="HAWQ"):
        runner.validate_method_assignment_paths(
            "hawq_mixed_le6", None, None)
    with pytest.raises(ValueError, match="P3/T3"):
        runner.validate_method_assignment_paths(
            "mixed_task_aware", None, None)
    with pytest.raises(ValueError, match="uniform"):
        runner.validate_method_assignment_paths(
            "lsqplus_w4a4", "hawq.json", None)


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


def test_checkpoint_requires_canonical_master_and_hard_validation():
    payload = dict((field, object()) for field in runner.CHECKPOINT_FIELDS)
    payload["format_version"] = 1
    payload["model_name"] = "nlspn"
    payload["method"] = "lsqplus_w4a4"
    payload["model_state"] = {"encoder.weight": torch.ones(1)}
    payload["hard_deployment_validation"] = {
        "epoch": 2, "validated": 1,
    }
    payload["epoch"] = 2
    payload["deterministic_algorithms"] = True
    payload["calibration_indices"] = tuple(range(128))
    payload["validation_indices"] = (128, 129)

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
    stale["hard_deployment_validation"] = {
        "epoch": 1, "validated": 1,
    }
    with pytest.raises(ValueError, match="evaluation epoch"):
        runner.validate_checkpoint_payload(stale)

    duplicate_identities = dict(payload)
    duplicate_identities["calibration_indices"] = tuple(range(127)) + (0,)
    with pytest.raises(ValueError, match="calibration identities"):
        runner.validate_checkpoint_payload(duplicate_identities)


def test_hawq_assignment_loader_rejects_direct_average_over_six(tmp_path):
    contract = _contract()
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps({
        "model_name": "nlspn",
        "average_weight_bits": 5.0,
        "average_activation_bits": 6.1,
        "assignment": {
            "model_name": "nlspn",
            "weight_block_bits": [
                {"block": "encoder", "bits": 4},
                {"block": "decoder", "bits": 6},
            ],
            "activation_block_bits": [
                {"block": "encoder", "bits": 4},
                {"block": "decoder", "bits": 6},
            ],
            "weight_bits": [
                {"module": "encoder", "bits": 4},
                {"module": "decoder", "bits": 6},
            ],
            "activation_bits": [
                {"site": "activation::encoder::input",
                 "role": "module_input", "bits": 4},
                {"site": "activation::decoder::input",
                 "role": "module_input", "bits": 6},
            ],
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="six-bit"):
        runner.load_hawq_qat_assignment(path, contract)


def test_hawq_assignment_loader_recomputes_cost_weighted_averages(tmp_path):
    contract = _contract()
    path = tmp_path / "hawq_tampered.json"
    path.write_text(json.dumps({
        "model_name": "nlspn",
        "calibration": {
            "count": 128, "indices": list(range(128)),
            "identity_sha256": "identity",
        },
        "contract": {
            "blocks": ["encoder", "decoder"],
            "protected_roles": ["propagation_state"],
            "protected_modules": ["propagation"],
            "attention_edges": [], "concat_edges": [],
        },
        "average_weight_bits": 6.0,
        "average_weight_mac_bits": 6.0,
        "average_activation_bits": 6.0,
        "assignment": {
            "model_name": "nlspn",
            "weight_block_bits": [
                {"block": "encoder", "bits": 8},
                {"block": "decoder", "bits": 8},
            ],
            "activation_block_bits": [
                {"block": "encoder", "bits": 8},
                {"block": "decoder", "bits": 8},
            ],
            "weight_bits": [
                {"module": "encoder", "bits": 8},
                {"module": "decoder", "bits": 8},
            ],
            "activation_bits": [
                {"site": "activation::encoder::input",
                 "role": "module_input", "bits": 8},
                {"site": "activation::decoder::input",
                 "role": "module_input", "bits": 8},
            ],
        },
        "objective": {},
        "constraints": {
            "maximum_average_weight_bits": 6.0,
            "maximum_average_activation_bits": 6.0,
            "average_weight_parameter_bits": 6.0,
            "average_weight_mac_bits": 6.0,
            "average_activation_traffic_bits": 6.0,
        },
        "cost_basis": {
            "weight_parameters": [
                {"block": "encoder", "parameters": 3},
                {"block": "decoder", "parameters": 1},
            ],
            "weight_macs": [
                {"module": "encoder", "macs": 3},
                {"module": "decoder", "macs": 1},
            ],
            "activation_traffic": [
                {"site": "activation::encoder::input",
                 "role": "module_input", "elements": 3},
                {"site": "activation::decoder::input",
                 "role": "module_input", "elements": 1},
            ],
        },
        "solver_status": "optimal",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="recomputed"):
        runner.load_hawq_qat_assignment(path, contract)


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
    record = runner.hard_deployment_evaluation_record(
        3,
        {"validated": 1, "materialized_weight_count": 2},
        {"samples": 17, "RMSE": 0.25},
    )

    assert record == {
        "validated": 1,
        "materialized_weight_count": 2,
        "epoch": 3,
        "evaluation_samples": 17,
        "evaluation_rmse": 0.25,
    }
    with pytest.raises(ValueError, match="hard deployment"):
        runner.hard_deployment_evaluation_record(
            3, {"validated": 0}, {"samples": 17, "RMSE": 0.25})


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
