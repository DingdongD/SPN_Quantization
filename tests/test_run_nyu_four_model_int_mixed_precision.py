import json

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
    )

    assert evaluator.calls[:3] == ["FP32", "UNIFORM_W8A8", "ANCHOR_FP16_head"]
    assert result.status == "feasible"
    assert result.anchor.assignment.fp16_units == ("head",)
    assert result.anchor.relative_loss == pytest.approx(0.007)
    assert result.pareto_frontier


def test_search_records_infeasible_without_accepting_best_failure():
    result = runner.run_constrained_search(
        contract=_contract(),
        costs=_costs(),
        evaluator=FakeEvaluator(infeasible=True),
        settings=_settings(),
        boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),),
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


def test_artifacts_persist_explicit_assignments_and_pareto_status(tmp_path):
    result = runner.run_constrained_search(
        contract=_contract(),
        costs=_costs(),
        evaluator=FakeEvaluator(),
        settings=_settings(),
        boundary_order=("head",),
        interaction_pairs=(("decoder", "head"),),
    )

    runner.write_search_artifacts(tmp_path, result)

    payload = json.loads(
        (tmp_path / "candidate_assignments.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["candidates"][0]["assignment"]["weight_bits"]
    assert "fp16_units" in payload["candidates"][0]["assignment"]
    assert manifest["status"] == "feasible"
    assert (tmp_path / "anchor_summary.csv").is_file()
    assert (tmp_path / "single_module_ablation.csv").is_file()
    assert (tmp_path / "interaction_ablation.csv").is_file()
    assert (tmp_path / "pareto_ptq.csv").is_file()


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
