import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts import run_nyu_four_model_int_mixed_precision as runner
from spn_quant.constrained_mixed_precision import PrecisionCosts
from spn_quant.model_contracts import (
    PrecisionSearchUnit,
    QuantizationBlock,
    QuantizationModelContract,
)


def _contract():
    blocks = tuple(
        QuantizationBlock(
            name=name,
            weight_modules=("%s.conv" % name,),
            activation_owners=(("activation::%s.conv::input" % name,
                                "module_input"),),
        ) for name in ("encoder", "decoder", "head"))
    units = tuple(
        PrecisionSearchUnit(
            name=name,
            members=("%s.conv" % name,),
            activation_owners=(("activation::%s.conv::input" % name,
                                "module_input"),),
            kind="initial_depth" if name == "head" else name,
            minimum_weight_bits=4,
            minimum_activation_bits=4,
            allow_fp16=name == "head",
            scale_policy="static_tensor",
        ) for name in ("encoder", "decoder", "head"))
    return QuantizationModelContract(
        model_name="cspn",
        blocks=blocks,
        prefix_groups=(),
        tail_groups=(),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("prop",),
        module_roles=(("prop", "propagation_state"),),
        search_units=units,
    )


def _costs():
    return PrecisionCosts(
        weight_macs=(("encoder", 80), ("decoder", 15), ("head", 5)),
        activation_elements=(("encoder", 30), ("decoder", 60), ("head", 10)),
    )


class FakeEvaluator(object):
    def __init__(self, infeasible=False):
        self.calls = []
        self.infeasible = bool(infeasible)

    def reference(self):
        self.calls.append("FP32")
        return {"pooled_rmse": 1.0, "sample_count": 64}

    def evaluate(self, candidate_id, assignment):
        self.calls.append(candidate_id)
        weights = dict(assignment.weight_bits)
        activations = dict(assignment.activation_bits)
        if self.infeasible:
            rmse = 1.03
        elif candidate_id == "UNIFORM_W8A8":
            rmse = 1.02
        elif assignment.fp16_units == ("head",):
            penalty = sum(8 - weights[name] for name in ("encoder", "decoder"))
            penalty += sum(8 - activations[name]
                           for name in ("encoder", "decoder"))
            rmse = 1.007 + 0.0002 * penalty
        else:
            rmse = 1.02
        return {
            "pooled_rmse": rmse,
            "sample_count": 64,
            "finite_positive": True,
            "reproducible": True,
            "propagation_valid": True,
            "owner_counts_valid": True,
        }


def _settings():
    return runner.SearchSettings(
        maximum_relative_loss=0.01,
        anchor_headroom_loss=0.008,
        qat_candidate_loss=0.015,
        beam_width=4,
        maximum_depth=2,
    )


def test_search_measures_fp32_and_w8a8_before_boundary_promotions():
    evaluator = FakeEvaluator()

    result = runner.run_constrained_search(
        contract=_contract(),
        costs=_costs(),
        evaluator=evaluator,
        settings=_settings(),
        boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),),
        phase="ptq-search",
    )

    assert evaluator.calls[:3] == ["FP32", "UNIFORM_W8A8", "ANCHOR_FP16_head"]
    assert result.status == "feasible"
    assert result.anchor.assignment.fp16_units == ("head",)
    assert result.anchor.relative_loss == pytest.approx(0.007)
    assert result.pareto_frontier


def test_beam_reuses_single_ablation_measurements_before_second_demotion():
    result = runner.run_constrained_search(
        contract=_contract(),
        costs=_costs(),
        evaluator=FakeEvaluator(),
        settings=_settings(),
        boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),),
        phase="ptq-search",
    )

    beam_assignments = tuple(
        record.candidate.assignment for record in result.records
        if record.phase == "beam")
    assert beam_assignments
    assert any(
        sum(bits < 8 for name, bits in assignment.weight_bits
            if name != "head") +
        sum(bits < 8 for name, bits in assignment.activation_bits
            if name != "head") >= 2
        for assignment in beam_assignments)


def test_search_records_infeasible_without_accepting_best_failure():
    result = runner.run_constrained_search(
        contract=_contract(),
        costs=_costs(),
        evaluator=FakeEvaluator(infeasible=True),
        settings=_settings(),
        boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),),
        phase="ptq-search",
    )

    assert result.status == "infeasible"
    assert result.anchor is None
    assert result.pareto_frontier == ()


def test_factorial_generator_emits_independent_weight_activation_pairs():
    anchor = runner.uniform_assignment(_contract(), 8, 8)

    rows = runner.single_unit_factorial_assignments(_contract(), anchor)
    encoder = tuple(row for row in rows if row[0].startswith("SINGLE_encoder"))

    assert tuple(name.rpartition("_")[2] for name, assignment in encoder) == (
        "W6A8", "W8A6", "W6A6", "W4A8",
        "W8A4", "W4A6", "W6A4", "W4A4")


def test_interaction_assignment_preserves_unit_precision_floors():
    contract = _contract()
    units = tuple(
        PrecisionSearchUnit(
            name=unit.name,
            members=unit.members,
            activation_owners=unit.activation_owners,
            kind=unit.kind,
            minimum_weight_bits=unit.minimum_weight_bits,
            minimum_activation_bits=8 if unit.name == "head" else
                unit.minimum_activation_bits,
            allow_fp16=unit.allow_fp16,
            scale_policy=unit.scale_policy,
        ) for unit in contract.search_units)
    contract = QuantizationModelContract(
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
    anchor = runner.promote_fp16(
        runner.uniform_assignment(contract, 8, 8), contract, "head")

    candidate_id, assignment = runner.interaction_assignment(
        contract, anchor, "decoder", "head")

    assert candidate_id == \
        "INTERACTION_decoder_W6A6_head_W6A8"
    assert dict(assignment.weight_bits)["head"] == 6
    assert dict(assignment.activation_bits)["head"] == 8
    assert assignment.fp16_units == ()


def test_anchor_phase_does_not_run_factorial_or_beam_candidates():
    evaluator = FakeEvaluator()

    result = runner.run_constrained_search(
        contract=_contract(), costs=_costs(), evaluator=evaluator,
        settings=_settings(), boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),), phase="anchors")

    assert evaluator.calls == ["FP32", "UNIFORM_W8A8", "ANCHOR_FP16_head"]
    assert tuple(record.phase for record in result.records) == (
        "anchor", "anchor")


def test_reference_artifact_uses_search_measurement_without_re_evaluation():
    result = runner.run_constrained_search(
        contract=_contract(), costs=_costs(), evaluator=FakeEvaluator(),
        settings=_settings(), boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),), phase="anchors")

    assert runner.reference_artifact(result) == {
        "pooled_rmse": 1.0,
        "sample_count": 64,
    }


def test_worker_uses_checkpoint_thread_contract(monkeypatch):
    observed = []
    monkeypatch.setattr(
        runner.torch, "set_num_threads", lambda value: observed.append(value))

    runner.configure_runtime_execution(
        SimpleNamespace(saved_args=SimpleNamespace(torch_threads=1)))

    assert observed == [1]


def test_artifacts_persist_explicit_assignments_and_pareto_status(tmp_path):
    result = runner.run_constrained_search(
        contract=_contract(),
        costs=_costs(),
        evaluator=FakeEvaluator(),
        settings=_settings(),
        boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),),
        phase="ptq-search",
    )

    runner.write_search_artifacts(tmp_path, result)

    payload = json.loads(
        (tmp_path / "candidate_assignments.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["candidates"][0]["assignment"]["weight_bits"]
    assert "fp16_units" in payload["candidates"][0]["assignment"]
    assert manifest["status"] == "feasible"
    rows = (tmp_path / "anchor_summary.csv").read_text(encoding="utf-8")
    assert "finite_positive" in rows
    assert "owner_counts_valid" in rows
    assert (tmp_path / "anchor_summary.csv").is_file()
    assert (tmp_path / "single_module_ablation.csv").is_file()
    assert (tmp_path / "interaction_ablation.csv").is_file()
    assert (tmp_path / "pareto_ptq.csv").is_file()


def test_selected_audit_writes_sample_signal_state_and_effective_rows(
        tmp_path):
    evaluation = {
        "candidate_id": "CANDIDATE",
        "sample_rows": ({
            "config": "CANDIDATE", "sample_index": 9,
            "squared_error_sum": 2.0, "valid_pixels": 8,
            "RMSE": 0.5, "prediction_finite": True,
            "prediction_positive": True, "propagation_valid": True,
            "reproducible": True,
        },),
        "signal_rows": (
            {"candidate_id": "CANDIDATE", "sample_index": 9,
             "signal": "state", "iteration": 1, "mse": 0.25},
            {"candidate_id": "CANDIDATE", "sample_index": 9,
             "signal": "affinity_constraints", "iteration": 0,
             "coefficient_sum_max_error": 0.0,
             "contraction_violation_rate": 0.0},
        ),
        "effective_weight_bits": (("encoder.conv", 6),),
        "effective_activation_bits": (
            (("encoder.conv", "input"), 8),),
        "owner_call_counts": ((("encoder.conv", "input"), 1),),
    }

    runner.write_evaluation_audit_artifacts(tmp_path, evaluation)

    assert "sample_index" in (
        tmp_path / "sample_metrics.csv").read_text(encoding="utf-8")
    state = (tmp_path / "propagation_state_metrics.csv").read_text(
        encoding="utf-8")
    assert "state" in state
    assert "affinity_constraints" not in state
    effective = (tmp_path / "effective_quantization.csv").read_text(
        encoding="utf-8")
    assert "encoder.conv,weight,6" in effective
    assert "encoder.conv,input,8" in effective


def test_load_balanced_ptq_candidate_uses_only_published_frontier(tmp_path):
    contract = _contract()
    assignments = []
    for candidate_id, weight, activation, rmse in (
            ("MIN_W", 4, 8, 1.004),
            ("BALANCED", 6, 6, 1.002),
            ("MIN_A", 8, 4, 1.003),
            ("UNPUBLISHED", 4, 4, 1.001)):
        assignment = runner.uniform_assignment(contract, 8, 8)
        for unit in assignment.expected_units:
            assignment = runner._replace_unit(
                assignment, unit, weight, activation, False, contract)
        candidate = runner.MeasuredCandidate(
            candidate_id=candidate_id,
            assignment=assignment,
            pooled_rmse=rmse,
            reference_pooled_rmse=1.0,
            average_weight_bits=float(weight),
            average_activation_bits=float(activation),
            fp16_mac_fraction=0.0,
            fp16_activation_fraction=0.0,
        )
        assignments.append(runner._candidate_payload(candidate))
    (tmp_path / "candidate_assignments.json").write_text(json.dumps({
        "model": "cspn",
        "candidates": assignments,
    }), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model": "cspn",
        "reference_pooled_rmse": 1.0,
        "pareto_candidate_ids": ["MIN_W", "BALANCED", "MIN_A"],
    }), encoding="utf-8")

    selected = runner.load_balanced_ptq_candidate(tmp_path, contract)

    assert selected.candidate_id == "BALANCED"
    assert selected.assignment.expected_units == (
        "encoder", "decoder", "head")


def test_propagation_dtype_comparison_uses_one_fixed_assignment():
    fp16 = {
        "candidate_id": "FIXED", "propagation_dtype": "fp16",
        "pooled_rmse": 0.201, "sample_count": 64,
        "finite_positive": True, "reproducible": True,
        "propagation_valid": True, "owner_counts_valid": True,
    }
    bf16 = dict(fp16)
    bf16["propagation_dtype"] = "bf16"
    bf16["pooled_rmse"] = 0.202

    rows = runner.propagation_dtype_comparison_rows(0.2, fp16, bf16)

    assert tuple(row["propagation_dtype"] for row in rows) == (
        "fp16", "bf16")
    assert rows[0]["relative_loss_from_fp32"] == pytest.approx(0.005)
    assert rows[0]["relative_delta_from_fp16"] == pytest.approx(0.0)
    assert rows[1]["relative_delta_from_fp16"] == pytest.approx(
        0.202 / 0.201 - 1.0)


def test_selected_audit_cli_requires_and_forwards_ptq_root(
        monkeypatch, tmp_path):
    calls = []
    summary = tmp_path / "summary.csv"
    monkeypatch.setattr(
        runner, "run_official_selected_audit",
        lambda config, model, ptq_root, output: calls.append((
            config, model, ptq_root, output)) or summary)

    runner.main((
        "--config", str(tmp_path / "config.json"),
        "--model", "cspn",
        "--output", str(tmp_path / "audit"),
        "--phase", "selected-audit",
        "--ptq-root", str(tmp_path / "ptq"),
    ))

    assert calls == [(
        tmp_path / "config.json", "cspn", tmp_path / "ptq",
        tmp_path / "audit")]


def test_selected_audit_cli_rejects_missing_ptq_root(tmp_path):
    with pytest.raises(ValueError, match="--ptq-root"):
        runner.main((
            "--config", str(tmp_path / "config.json"),
            "--model", "cspn",
            "--output", str(tmp_path / "audit"),
            "--phase", "selected-audit",
        ))


def test_measure_unit_costs_uses_executed_conv_macs_and_input_elements():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.encoder = nn.Conv2d(1, 2, 3, padding=1, bias=False)
            self.head = nn.Conv2d(2, 1, 1, bias=False)

        def forward(self, value):
            return self.head(self.encoder(value))

    model = Model().eval()
    blocks = (
        QuantizationBlock(
            "encoder", ("encoder",),
            (("activation::encoder::input", "module_input"),)),
        QuantizationBlock(
            "head", ("head",),
            (("activation::head::input", "module_input"),)),
    )
    units = tuple(PrecisionSearchUnit(
        name=name, members=(name,),
        activation_owners=(("activation::%s::input" % name,
                            "module_input"),),
        kind=name, minimum_weight_bits=4, minimum_activation_bits=4,
        allow_fp16=False, scale_policy="static_tensor")
        for name in ("encoder", "head"))
    contract = QuantizationModelContract(
        model_name="cspn", blocks=blocks, prefix_groups=(), tail_groups=(),
        protected_roles=("propagation_state",), attention_edges=(),
        concat_edges=(), protected_modules=("prop",),
        module_roles=(("prop", "propagation_state"),), search_units=units)

    costs = runner.measure_unit_costs(
        model, contract, (torch.ones(1, 1, 4, 4),))

    assert dict(costs.weight_macs) == {"encoder": 288, "head": 32}
    assert dict(costs.activation_elements) == {"encoder": 16, "head": 32}


def test_measure_unit_costs_counts_qkv_output_elements():
    class AttentionProjection(nn.Module):
        def __init__(self):
            super(AttentionProjection, self).__init__()
            self.q = nn.Linear(4, 4, bias=False)
            self.kv = nn.Linear(4, 8, bias=False)

        def forward(self, value):
            return self.q(value), self.kv(value)

    model = AttentionProjection().eval()
    owners = (
        ("attention::::q", "attention_q"),
        ("attention::::k", "attention_k"),
        ("attention::::v", "attention_v"),
    )
    block = QuantizationBlock(
        "attention", ("q", "kv"), owners)
    unit = PrecisionSearchUnit(
        name="attention_qkv", members=("q", "kv"),
        activation_owners=owners, kind="attention_qkv",
        minimum_weight_bits=4, minimum_activation_bits=8,
        allow_fp16=False, scale_policy="static_tensor")
    contract = QuantizationModelContract(
        model_name="completionformer", blocks=(block,), prefix_groups=(),
        tail_groups=(), protected_roles=("propagation_state",),
        attention_edges=tuple(row[0] for row in owners), concat_edges=(),
        protected_modules=("prop",),
        module_roles=(("prop", "propagation_state"),),
        search_units=(unit,))

    costs = runner.measure_unit_costs(
        model, contract, (torch.ones(1, 3, 4),))

    assert dict(costs.activation_elements) == {"attention_qkv": 36}
