import json
from copy import deepcopy
from types import SimpleNamespace

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


def _trace_settings():
    return {
        "batch_size": 4,
        "probes_per_batch": 8,
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


def _valid_hawq_payload(checkpoint):
    from scripts.run_nyu_qdrop_reconstruction import (
        ordered_sample_identity_sha256,
    )
    indices = tuple(range(128))
    components = []
    for block, normalized_trace in (("encoder", 2.0), ("decoder", 1.0)):
        for bits, error in ((4, 0.3), (6, 0.2), (8, 0.1)):
            components.append({
                "block": block,
                "bits": bits,
                "normalized_trace": normalized_trace,
                "quantization_error": error,
                "cost": normalized_trace * error,
            })
    return {
        "model_name": "nlspn",
        "provenance": {
            "checkpoint": _checkpoint_payload(checkpoint),
            "trace_settings": _trace_settings(),
            "trace_artifact_sha256": "a" * 64,
        },
        "calibration": {
            "count": 128,
            "indices": list(indices),
            "identity_sha256": ordered_sample_identity_sha256(
                "train", indices),
        },
        "contract": {
            "blocks": ["encoder", "decoder"],
            "protected_roles": ["propagation_state"],
            "protected_modules": ["propagation"],
            "attention_edges": [],
            "concat_edges": [],
        },
        "average_weight_bits": 5.0,
        "average_weight_mac_bits": 5.0,
        "average_activation_bits": 5.0,
        "assignment": {
            "model_name": "nlspn",
            "weight_block_bits": [
                {"block": "encoder", "bits": 4},
                {"block": "decoder", "bits": 8},
            ],
            "activation_block_bits": [
                {"block": "encoder", "bits": 4},
                {"block": "decoder", "bits": 8},
            ],
            "weight_bits": [
                {"module": "encoder", "bits": 4},
                {"module": "decoder", "bits": 8},
            ],
            "activation_bits": [
                {"site": "activation::encoder::input",
                 "role": "module_input", "bits": 4},
                {"site": "activation::decoder::input",
                 "role": "module_input", "bits": 8},
            ],
        },
        "objective": {
            "kind": "weight_hessian_times_squared_quantization_error",
            "activation_sensitivity": "not_estimated",
            "total": 0.7,
            "components": components,
            "selected_components": [components[0], components[-1]],
        },
        "constraints": {
            "maximum_average_weight_bits": 6.0,
            "maximum_average_activation_bits": 6.0,
            "average_weight_parameter_bits": 5.0,
            "average_weight_mac_bits": 5.0,
            "average_activation_traffic_bits": 5.0,
            "weight_parameter_residual": 1.0,
            "weight_mac_residual": 1.0,
            "activation_traffic_residual": 1.0,
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
        "solver_success": True,
        "solver_status": "Optimization terminated successfully",
    }


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
        build_p3_t3_candidates,
    )
    from spn_quant.mixed_precision import build_registry
    contract = _contract()
    costs = _p3_costs()
    candidates = build_p3_t3_candidates(
        contract, build_registry(contract, costs), 4, 4, 8, 8)
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
            "normalized_weight_cost": normalized_weight,
            "normalized_activation_cost": normalized_activation,
            "valid": True,
            "metrics_finite": True,
            "sample_rmse": [[128, rmse], [129, rmse]],
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
    return {
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
        "expected_samples": 2,
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
    payload = _valid_hawq_payload(checkpoint)
    payload["average_activation_bits"] = 6.1
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="six-bit"):
        runner.load_hawq_qat_assignment(
            path, _contract(), checkpoint, _trace_settings(), 6.0, 6.0)


def test_hawq_assignment_loader_recomputes_cost_weighted_averages(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    payload = _valid_hawq_payload(checkpoint)
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
            path, _contract(), checkpoint, _trace_settings(), 6.0, 6.0)


@pytest.mark.parametrize(("mutation", "message"), (
    (lambda payload: payload["provenance"]["checkpoint"].update(
        {"sha256": "0" * 64}), "checkpoint identity"),
    (lambda payload: payload["provenance"]["trace_settings"].update(
        {"seed": 1}), "trace settings"),
    (lambda payload: payload.update({"solver_success": False}),
     "solver success"),
    (lambda payload: payload.update({"solver_status": "solver failed"}),
     "solver success"),
    (lambda payload: payload["objective"]["components"][0].update(
        {"cost": 99.0}), "objective component"),
    (lambda payload: payload["constraints"].update(
        {"weight_mac_residual": 0.5}), "constraint residual"),
    (lambda payload: payload["assignment"]["weight_bits"][0].update(
        {"bits": 8}), "block and owner assignments"),
    (lambda payload: payload["cost_basis"]["weight_macs"].append(
        {"module": "encoder", "macs": 3}), "cost basis assignment"),
))
def test_hawq_assignment_requires_full_trace_and_solver_provenance(
        tmp_path, mutation, message):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    payload = _valid_hawq_payload(checkpoint)
    mutation(payload)
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner.load_hawq_qat_assignment(
            path, _contract(), checkpoint, _trace_settings(), 6.0, 6.0)


def test_hawq_assignment_accepts_exact_selected_artifact(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    path = tmp_path / "hawq.json"
    path.write_text(
        json.dumps(_valid_hawq_payload(checkpoint)), encoding="utf-8")

    assignment = runner.load_hawq_qat_assignment(
        path, _contract(), checkpoint, _trace_settings(), 6.0, 6.0)

    assert assignment.weight_bits == (("decoder", 8), ("encoder", 4))


def test_hawq_assignment_rejects_self_consistent_infeasible_constraints(
        tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    payload = _valid_hawq_payload(checkpoint)
    payload["constraints"].update({
        "maximum_average_weight_bits": 4.5,
        "weight_parameter_residual": -0.5,
        "weight_mac_residual": -0.5,
    })
    path = tmp_path / "hawq.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="constraint assignment is infeasible"):
        runner.load_hawq_qat_assignment(
            path, _contract(), checkpoint, _trace_settings(), 4.5, 6.0)


@pytest.mark.parametrize(("mutation", "message"), (
    (lambda payload: payload["source_checkpoint"].update(
        {"sha256": "0" * 64}), "checkpoint identity"),
    (lambda payload: payload["assignment"]["weight_bits"][0].__setitem__(
        1, 4), "selected candidate assignment"),
    (lambda payload: next(
        row for row in payload["candidates"]
        if row["name"] == payload["selected_candidate"]).update(
            {"valid": False}), "stable and finite"),
    (lambda payload: payload["budgets"].update(
        {"maximum_normalized_weight_cost": 1.5}), "weight budget"),
    (lambda payload: next(
        row for row in payload["candidates"]
        if row["name"] == payload["selected_candidate"]).update(
            {"normalized_activation_cost": 1.5}), "activation cost audit"),
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
    )

    assert costs == _p3_costs()
    assert assignment.weight_bits == (("decoder", 8), ("encoder", 8))
    assert audit["normalized_weight_cost"] == 2.0
    assert audit["normalized_activation_cost"] == 2.0
    assert audit["weight_feasible"] == 1
    assert audit["activation_feasible"] == 1


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
        assert training == {"fold_max_error": 0.0}
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
        {"fold_max_error": 0.0},
        context_factory=context_factory,
    )

    assert evaluation["RMSE"] == 8.0
    assert record["evaluation_rmse"] == 8.0
    assert observed["closed"]


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
        {"fold_max_error": 0.0},
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
            {"fold_max_error": 0.0},
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
