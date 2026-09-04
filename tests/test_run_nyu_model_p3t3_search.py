import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import run_nyu_model_p3t3_search as runner
from spn_quant import mixed_precision
from spn_quant.constrained_mixed_precision import PrecisionAssignment
from spn_quant.model_contracts import (
    PrecisionSearchUnit,
    QuantizationBlock,
    QuantizationModelContract,
)


def test_strict_candidate_evaluation_forwards_one_sample_at_a_time(
        monkeypatch):
    class Instrumentor(object):
        def expected_execution_call_counts(self):
            return {("conv", "input"): 1}

        def execution_call_counts(self):
            return {("conv", "input"): 1}

    class PropagationAdapter(object):
        def statistics(self):
            return ()

    evaluator = runner.HardDeploymentP3T3Evaluator.__new__(
        runner.HardDeploymentP3T3Evaluator)
    evaluator.evaluation_batches = (
        (3, {"value": torch.ones(1, 1, 1, 1)}),
        (7, {"value": torch.ones(1, 1, 1, 1)}),
    )
    evaluator.instrumentor = Instrumentor()
    evaluator.propagation_adapter = PropagationAdapter()
    evaluator.runtime = SimpleNamespace(
        model_name="cspn", propagation_iterations=24)
    evaluator.preserve_input = False
    evaluator._active_joint_quantizers = ()
    evaluator._configure_candidate = lambda candidate: None
    observed_batch_sizes = []

    def forward(batch):
        observed_batch_sizes.append(batch["value"].shape[0])
        return torch.full((1, 1, 1, 1), 2.0), torch.ones(1, 1, 1, 1)

    evaluator._forward = forward
    monkeypatch.setattr(runner, "_propagation_valid",
                        lambda model_name, preserve_input, rows: True)
    monkeypatch.setattr(runner, "_propagation_iterations_valid",
                        lambda rows, expected_iterations: True)

    rows, owner_counts_valid = evaluator._evaluate_precision_candidate(
        SimpleNamespace(name="STRICT"))

    assert observed_batch_sizes == [1, 1, 1, 1]
    assert tuple(row["sample_index"] for row in rows) == (3, 7)
    assert owner_counts_valid


def test_strict_candidate_evaluation_retains_first_forward_signal_rows(
        monkeypatch):
    class Instrumentor(object):
        def expected_execution_call_counts(self):
            return {("conv", "input"): 1}

        def execution_call_counts(self):
            return {("conv", "input"): 1}

    class PropagationAdapter(object):
        def statistics(self):
            return (
                {"signal": "state", "iteration": 1, "mse": 0.25},
                {"signal": "affinity_constraints", "iteration": 0,
                 "coefficient_sum_max_error": 0.0,
                 "contraction_violation_rate": 0.0},
            )

    evaluator = runner.HardDeploymentP3T3Evaluator.__new__(
        runner.HardDeploymentP3T3Evaluator)
    evaluator.evaluation_batches = (
        (11, {"value": torch.ones(1, 1, 1, 1)}),)
    evaluator.instrumentor = Instrumentor()
    evaluator.propagation_adapter = PropagationAdapter()
    evaluator.runtime = SimpleNamespace(
        model_name="cspn", propagation_iterations=1)
    evaluator.preserve_input = False
    evaluator._active_joint_quantizers = ()
    evaluator._configure_candidate = lambda candidate: None
    evaluator._forward = lambda batch: (
        torch.full((1, 1, 1, 1), 2.0),
        torch.ones(1, 1, 1, 1),
    )
    monkeypatch.setattr(runner, "_propagation_valid",
                        lambda model_name, preserve_input, rows: True)
    monkeypatch.setattr(runner, "_propagation_iterations_valid",
                        lambda rows, expected_iterations: True)

    evaluator._evaluate_precision_candidate(SimpleNamespace(name="STRICT"))

    assert evaluator.last_signal_rows == (
        {"candidate_id": "STRICT", "sample_index": 11,
         "signal": "state", "iteration": 1, "mse": 0.25},
        {"candidate_id": "STRICT", "sample_index": 11,
         "signal": "affinity_constraints", "iteration": 0,
         "coefficient_sum_max_error": 0.0,
         "contraction_violation_rate": 0.0},
    )


def test_fixed_assignment_bf16_evaluation_only_changes_propagation_state():
    calls = []

    class Instrumentor(object):
        def weight_bits_by_module(self):
            return {"conv": 6}

        def manifest(self):
            return ({"module": "conv", "kind": "input", "bits": 8},)

        def execution_call_counts(self):
            return {("conv", "input"): 1}

    class PropagationAdapter(object):
        def configure_float(self, dtype):
            calls.append(("propagation", dtype))

    evaluator = runner.HardDeploymentP3T3Evaluator.__new__(
        runner.HardDeploymentP3T3Evaluator)
    evaluator.instrumentor = Instrumentor()
    evaluator.propagation_adapter = PropagationAdapter()
    evaluator.configure_precision_assignment = \
        lambda assignment, candidate_id: calls.append(
            ("assignment", candidate_id)) or SimpleNamespace(name=candidate_id)
    evaluator._evaluate_configured_precision_candidate = \
        lambda candidate: (({
            "squared_error_sum": 4.0,
            "valid_pixels": 4,
            "prediction_finite": True,
            "prediction_positive": True,
            "reproducible": True,
            "propagation_valid": True,
        },), True)
    evaluator.last_signal_rows = ()

    result = evaluator.evaluate_precision_assignment_with_propagation_dtype(
        object(), "FIXED", "bf16")

    assert calls == [("assignment", "FIXED"), ("propagation", "bf16")]
    assert result["propagation_dtype"] == "bf16"
    assert result["pooled_rmse"] == pytest.approx(1.0)
    assert result["effective_weight_bits"] == (("conv", 6),)


def test_strict_candidate_positivity_uses_ground_truth_valid_mask(
        monkeypatch):
    class Instrumentor(object):
        def expected_execution_call_counts(self):
            return {("conv", "input"): 1}

        def execution_call_counts(self):
            return {("conv", "input"): 1}

    class PropagationAdapter(object):
        def statistics(self):
            return ()

    evaluator = runner.HardDeploymentP3T3Evaluator.__new__(
        runner.HardDeploymentP3T3Evaluator)
    evaluator.evaluation_batches = (
        (3, {"value": torch.ones(1, 1, 1, 2)}),)
    evaluator.instrumentor = Instrumentor()
    evaluator.propagation_adapter = PropagationAdapter()
    evaluator.runtime = SimpleNamespace(
        model_name="dyspn", propagation_iterations=6)
    evaluator.preserve_input = True
    evaluator._active_joint_quantizers = ()
    evaluator._configure_candidate = lambda candidate: None
    evaluator._forward = lambda batch: (
        torch.tensor([[[[2.0, -1.0]]]]),
        torch.tensor([[[[1.0, 0.0]]]]),
    )
    monkeypatch.setattr(runner, "_propagation_valid",
                        lambda model_name, preserve_input, rows: True)
    monkeypatch.setattr(runner, "_propagation_iterations_valid",
                        lambda rows, expected_iterations: True)

    rows, owner_counts_valid = evaluator._evaluate_precision_candidate(
        SimpleNamespace(name="STRICT"))

    assert rows[0]["prediction_positive"]
    assert owner_counts_valid


def test_strict_reference_metrics_forward_one_sample_at_a_time():
    class Disabled(object):
        def disable(self):
            return None

    evaluator = runner.HardDeploymentP3T3Evaluator.__new__(
        runner.HardDeploymentP3T3Evaluator)
    evaluator.evaluation_batches = (
        (3, {"value": torch.ones(1, 1, 1, 1)}),
        (7, {"value": torch.ones(1, 1, 1, 1)}),
    )
    evaluator.instrumentor = Disabled()
    evaluator.concat_adapter = None
    evaluator.propagation_projection_instrumentor = None
    evaluator.propagation_adapter = Disabled()
    evaluator.joint_adapter = None
    observed_batch_sizes = []

    def forward(batch):
        observed_batch_sizes.append(batch["value"].shape[0])
        return torch.full((1, 1, 1, 1), 2.0), torch.ones(1, 1, 1, 1)

    evaluator._forward = forward

    metrics = evaluator.reference_metrics()

    assert observed_batch_sizes == [1, 1]
    assert metrics["sample_count"] == 2
    assert metrics["pooled_rmse"] == pytest.approx(1.0)


def test_hard_joint_quantizer_reports_zero_and_saturation_codes():
    quantizer = runner._SiteSymmetricActivationQuantizer(
        "attention::block::q", 4, 1.0)

    quantizer.quantize_with_codes(torch.tensor([-2.0, 0.0, 2.0]))
    row = quantizer.statistics()[0]

    assert row["module"] == "attention::block::q"
    assert row["calls"] == 1
    assert row["numel"] == 3
    assert row["zero_code_count"] == 1
    assert row["saturation_count"] == 2
    assert row["zero_code_rate"] == pytest.approx(1.0 / 3.0)
    assert row["saturation_rate"] == pytest.approx(2.0 / 3.0)


def test_nlspn_concat_consumer_order_matches_official_forward_order():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            for name in ("id_dec1", "id_dec0", "gd_dec1", "gd_dec0",
                         "cf_dec1", "cf_dec0"):
                setattr(self, name, torch.nn.Sequential(
                    torch.nn.Conv2d(128, 64 if name.endswith("dec1") else 1, 1)))

    names = runner._scale_aware_concat_consumers("nlspn", Model())

    assert names == (None, None, None,
        "id_dec1.0", "id_dec0.0", "gd_dec1.0", "gd_dec0.0",
        "cf_dec1.0", "cf_dec0.0")
    assert runner._scale_aware_concat_consumers("cspn", Model()) == ()


def contract():
    return QuantizationModelContract(
        model_name="model_z",
        blocks=tuple(
            QuantizationBlock(
                name,
                ("weight_%s" % name,),
                (("edge_%s" % name, "input"),),
            )
            for name in ("alpha", "beta", "gamma", "delta")
        ),
        prefix_groups=(
            ("alpha",),
            ("alpha", "beta"),
            ("alpha", "beta", "gamma"),
        ),
        tail_groups=(("gamma",), ("delta",)),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("protected",),
        module_roles=(("protected", "propagation_state"),),
        search_units=tuple(
            PrecisionSearchUnit(
                name=name,
                members=("weight_%s" % name,),
                activation_owners=(("edge_%s" % name, "input"),),
                kind="encoder",
                minimum_weight_bits=4,
                minimum_activation_bits=4,
                allow_fp16=False,
                scale_policy="static_tensor",
            )
            for name in ("alpha", "beta", "gamma", "delta")
        ),
    )


def test_expand_precision_assignment_maps_units_to_exact_hardware_owners():
    assignment = PrecisionAssignment(
        weight_bits=(("alpha", 4), ("beta", 6),
                     ("gamma", 8), ("delta", 4)),
        activation_bits=(("alpha", 8), ("beta", 6),
                         ("gamma", 4), ("delta", 8)),
        scale_policies=tuple((name, "static_tensor")
                             for name in ("alpha", "beta", "gamma", "delta")),
        expected_units=("alpha", "beta", "gamma", "delta"),
    )

    expanded = runner.expand_precision_assignment(contract(), assignment)

    assert dict(expanded.weight_bits) == {
        "weight_alpha": 4,
        "weight_beta": 6,
        "weight_gamma": 8,
        "weight_delta": 4,
    }
    assert dict(expanded.activation_bits) == {
        ("edge_alpha", "input"): 8,
        ("edge_beta", "input"): 6,
        ("edge_gamma", "input"): 4,
        ("edge_delta", "input"): 8,
    }


def test_expand_precision_assignment_enforces_attention_activation_floor():
    search_unit = PrecisionSearchUnit(
        name="qkv",
        members=("block.attn.q",),
        activation_owners=(("attention::block.attn::q", "attention_q"),),
        kind="attention_qkv",
        minimum_weight_bits=4,
        minimum_activation_bits=8,
        allow_fp16=False,
        scale_policy="static_tensor",
    )
    attention_contract = QuantizationModelContract(
        model_name="completionformer",
        blocks=(QuantizationBlock(
            "block", ("block.attn.q",),
            (("attention::block.attn::q", "attention_q"),)),),
        prefix_groups=(), tail_groups=(),
        protected_roles=("propagation_state",),
        attention_edges=("attention::block.attn::q",),
        concat_edges=(), protected_modules=("prop",),
        module_roles=(("prop", "propagation_state"),),
        search_units=(search_unit,),
    )
    assignment = PrecisionAssignment(
        weight_bits=(("qkv", 4),),
        activation_bits=(("qkv", 6),),
        scale_policies=(("qkv", "static_tensor"),),
        expected_units=("qkv",),
    )

    with pytest.raises(ValueError, match="activation precision floor"):
        runner.expand_precision_assignment(attention_contract, assignment)


def costs():
    return mixed_precision.CostBasis(
        weight_macs=tuple(
            ("weight_%s" % name, value)
            for name, value in zip(
                ("alpha", "beta", "gamma", "delta"), (1, 2, 4, 3))
        ),
        activation_elements=tuple(
            (("edge_%s" % name, "input"), value)
            for name, value in zip(
                ("alpha", "beta", "gamma", "delta"), (1, 2, 4, 3))
        ),
    )


def score(candidate):
    prefix_scores = {
        (): 1.00,
        ("alpha",): 0.80,
        ("alpha", "beta"): 0.50,
        ("alpha", "beta", "gamma"): 0.49,
    }
    if candidate.stage == "single_block":
        return 0.95
    value = prefix_scores[candidate.prefix]
    if candidate.tail == ("gamma",):
        value -= 0.03
    elif candidate.tail == ("delta",):
        value -= 0.10
    elif candidate.tail == ("gamma", "delta"):
        value -= 0.12
    return value


class MeasuredEvaluator(object):
    def __init__(self):
        self.candidates = ()
        self.calls = []
        self.closed = False

    def __call__(self, candidates):
        self.candidates = tuple(candidates)
        self.calls.append(self.candidates)
        rows = []
        for candidate in candidates:
            measured = score(candidate)
            for sample_index, pixels in enumerate((1, 2, 3)):
                sample_rmse = measured + sample_index * 0.01
                rows.append({
                    "config": candidate.name,
                    "sample_index": sample_index,
                    "squared_error_sum": sample_rmse ** 2 * pixels,
                    "valid_pixels": pixels,
                    "RMSE": sample_rmse,
                    "prediction_finite": True,
                    "propagation_valid": True,
                    "reproducible": True,
                })
        return tuple(rows)

    def close(self):
        self.closed = True

    def reference_sample_rmse(self):
        return tuple(
            (sample_index, 1.0 + sample_index * 0.01)
            for sample_index in range(3))


def test_p3t3_selects_lowest_protection_cost_within_fp32_rmse_gate(
        monkeypatch):
    class GateEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = []
            for candidate in candidates:
                if candidate.stage == "interaction":
                    measured = {
                        "INTERACTION_P2_T01": 1.08,
                        "INTERACTION_P2_T02": 1.09,
                        "INTERACTION_P2_T03": 1.20,
                    }[candidate.name]
                else:
                    measured = 1.0
                for sample_index, pixels in enumerate((1, 2, 3)):
                    sample_rmse = measured + sample_index * 0.001
                    rows.append({
                        "config": candidate.name,
                        "sample_index": sample_index,
                        "squared_error_sum": sample_rmse ** 2 * pixels,
                        "valid_pixels": pixels,
                        "RMSE": sample_rmse,
                        "prediction_finite": True,
                        "propagation_valid": True,
                        "reproducible": True,
                    })
            return tuple(rows)

    monkeypatch.setattr(
        runner, "_prefix_knee",
        lambda rows: next(row for row in rows
                          if row.prefix == ("alpha", "beta")))
    result = runner.search_p3_t3(
        contract(), costs(), GateEvaluator(),
        4, 4, 8, 8, 1.65, 1.65, 0.10, 3)

    assert result.selected_candidate == "INTERACTION_P2_T02"
    selected = next(row for row in result.candidates
                    if row.name == result.selected_candidate)
    assert selected.normalized_weight_cost < next(
        row.normalized_weight_cost for row in result.candidates
        if row.name == "INTERACTION_P2_T01")
    assert selected.mean_sample_rmse / 1.01 - 1.0 <= 0.10
    assert all(
        bits in (4, 8)
        for bits in tuple(dict(selected.assignment.weight_bits).values()) +
        tuple(dict(selected.assignment.activation_bits).values()))


def test_p3t3_rejects_search_without_candidate_in_fp32_rmse_gate(
        monkeypatch):
    class FailingGateEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = []
            for candidate in candidates:
                measured = 1.0 if candidate.stage != "interaction" else 1.5
                for sample_index, pixels in enumerate((1, 2, 3)):
                    sample_rmse = measured + sample_index * 0.001
                    rows.append({
                        "config": candidate.name,
                        "sample_index": sample_index,
                        "squared_error_sum": sample_rmse ** 2 * pixels,
                        "valid_pixels": pixels,
                        "RMSE": sample_rmse,
                        "prediction_finite": True,
                        "propagation_valid": True,
                        "reproducible": True,
                    })
            return tuple(rows)

    monkeypatch.setattr(
        runner, "_prefix_knee",
        lambda rows: next(row for row in rows
                          if row.prefix == ("alpha", "beta")))
    with pytest.raises(RuntimeError, match="relative RMSE gate"):
        runner.search_p3_t3(
            contract(), costs(), FailingGateEvaluator(),
            4, 4, 8, 8, 1.65, 1.65, 0.10, 3)


def test_hard_evaluator_caches_fixed_evaluation_batches_once():
    evaluator = object.__new__(runner.HardDeploymentP3T3Evaluator)
    evaluator.valset = object()
    evaluator.settings = type("Settings", (), {
        "evaluation_indices": (386, 572),
    })()
    calls = []
    evaluator._sample_batch = lambda dataset, index: calls.append(index) or {
        "rgbd": torch.full((1, 4, 2, 2), float(index)),
        "depth": torch.full((1, 1, 2, 2), float(index)),
    }

    evaluator._cache_evaluation_batches()

    assert calls == [386, 572]
    assert tuple(index for index, batch in evaluator.evaluation_batches) == \
        (386, 572)
    assert evaluator.evaluation_batch["rgbd"].shape == (2, 4, 2, 2)
    assert evaluator.evaluation_batch["depth"].shape == (2, 1, 2, 2)


def test_candidate_matrix_uses_contract_prefixes_tail_combinations_and_blocks():
    registry = mixed_precision.build_registry(contract(), costs())

    candidates = runner.build_p3_candidates(
        contract(), registry, 4, 4, 8, 8)
    interactions = runner.build_t3_candidates(
        contract(), registry, ("alpha", "beta"), 4, 4, 8, 8)

    assert len(candidates) == 11
    assert tuple(candidate.stage for candidate in candidates).count(
        "single_block") == 4
    assert tuple(candidate.stage for candidate in candidates).count("prefix") == 3
    assert tuple(candidate.stage for candidate in candidates).count("tail") == 3
    assert not any(candidate.stage == "interaction" for candidate in candidates)
    assert len(interactions) == 3
    assert all(candidate.stage == "interaction" for candidate in interactions)
    assert all(candidate.prefix == ("alpha", "beta")
               for candidate in interactions)
    assert all(set(candidate.prefix + candidate.tail) <= set(registry.blocks)
               for candidate in candidates + interactions)
    assert all("stem" not in candidate.name
               for candidate in candidates + interactions)


def test_p3t3_search_selects_model_specific_prefix_and_budget_valid_tail():
    measured = MeasuredEvaluator()

    result = runner.search_p3_t3(
        contract=contract(),
        costs=costs(),
        evaluator=measured,
        base_weight_bits=4,
        base_activation_bits=4,
        promotion_weight_bits=8,
        promotion_activation_bits=8,
        maximum_normalized_weight_cost=1.65,
        maximum_normalized_activation_cost=1.65,
        maximum_relative_rmse_loss=0.10,
        expected_samples=3,
    )

    assert result.assignment.model_name == "model_z"
    assert result.prefix == ("alpha", "beta")
    assert result.prefix in contract().prefix_groups
    assert result.tail == ("delta",)
    assert result.tail in contract().tail_groups
    assert tuple(len(call) for call in measured.calls) == (11, 3)
    assert all(candidate.stage == "interaction"
               for candidate in measured.calls[1])
    assert all(candidate.prefix == result.prefix
               for candidate in measured.calls[1])
    assert len(result.candidates) == 14
    assert not any(
        row.stage == "interaction" and row.prefix != result.prefix
        for row in result.candidates)
    selected = next(row for row in result.candidates
                    if row.name == result.selected_candidate)
    assert selected.pooled_rmse == pytest.approx(
        ((0.40 ** 2 + 2 * 0.41 ** 2 + 3 * 0.42 ** 2) / 6) ** 0.5)
    assert selected.mean_sample_rmse == pytest.approx(0.41)
    assert selected.normalized_weight_cost == pytest.approx(1.6)
    assert selected.normalized_activation_cost == pytest.approx(1.6)
    assert selected.paired_sample_differences == pytest.approx(
        (-0.60, -0.60, -0.60))


def test_search_rejects_incomplete_measured_coverage_without_estimating_accuracy():
    class IncompleteEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            return super().__call__(candidates)[:-1]

    with pytest.raises(ValueError, match="measured sample coverage mismatch"):
        runner.search_p3_t3(
            contract(), costs(), IncompleteEvaluator(),
            4, 4, 8, 8, 2.0, 2.0, 0.10, 3)


@pytest.mark.parametrize("field,value,match", (
    ("propagation_valid", "yes", "validity flags"),
    ("RMSE", 9.0, "RMSE and squared error"),
    ("valid_pixels", 1.5, "valid pixel count"),
    ("squared_error_sum", -1.0, "squared error"),
    ("squared_error_sum", True, "squared error"),
))
def test_search_rejects_malformed_measured_rows(field, value, match):
    class MalformedEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = list(super().__call__(candidates))
            rows[0][field] = value
            return tuple(rows)

    with pytest.raises(ValueError, match=match):
        runner.search_p3_t3(
            contract(), costs(), MalformedEvaluator(),
            4, 4, 8, 8, 2.0, 2.0, 0.10, 3)


def test_invalid_or_unreproducible_rows_cannot_be_selected():
    class InvalidBestEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = list(super().__call__(candidates))
            for row in rows:
                if row["config"] == "INTERACTION_P2_T02":
                    row["squared_error_sum"] = 0.0
                    row["RMSE"] = 0.0
                    row["reproducible"] = False
            return tuple(rows)

    result = runner.search_p3_t3(
        contract(), costs(), InvalidBestEvaluator(),
        4, 4, 8, 8, 1.75, 1.75, 0.10, 3)

    assert result.selected_candidate == "INTERACTION_P2_T01"
    rejected = next(row for row in result.candidates
                    if row.name == "INTERACTION_P2_T02")
    assert not rejected.valid


def test_search_rejects_nonfinite_baseline_before_paired_selection():
    class NonfiniteBaselineEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = list(super().__call__(candidates))
            for row in rows:
                if row["config"] == "UNIFORM_W4A4":
                    row["squared_error_sum"] = float("inf")
                    row["RMSE"] = float("inf")
                    row["prediction_finite"] = False
            return tuple(rows)

    with pytest.raises(RuntimeError, match="stable finite baseline"):
        runner.search_p3_t3(
            contract(), costs(), NonfiniteBaselineEvaluator(),
            4, 4, 8, 8, 1.65, 1.65, 0.10, 3)


@pytest.mark.parametrize("anchor_signal", ("anchor", "anchor_injection"))
def test_propagation_valid_accepts_established_anchor_signals(anchor_signal):
    rows = (
        {"signal": "state", "mse": 0.01},
        {"signal": "affinity_constraints",
         "coefficient_sum_max_error": 0.0,
         "contraction_violation_rate": 0.0},
        {"signal": anchor_signal, "anchor_max_error": 0.0},
    )

    assert runner._propagation_valid("dyspn", True, rows)


def test_propagation_valid_requires_cspn_float_diagnostics():
    rows = (
        {"signal": "state", "mse": 0.0},
        {"signal": "affinity_constraints",
         "coefficient_sum_max_error": 1e-7,
         "contraction_violation_rate": 0.0},
        {"signal": "anchor", "anchor_max_error": 0.0},
    )

    assert not runner._propagation_valid("cspn", False, ())
    assert runner._propagation_valid("cspn", False, rows)


def test_propagation_valid_rejects_coefficient_error_above_fp32_roundoff():
    rows = (
        {"signal": "state", "mse": 0.0},
        {"signal": "affinity_constraints",
         "coefficient_sum_max_error": 1e-4,
         "contraction_violation_rate": 0.0},
        {"signal": "anchor", "anchor_max_error": 0.0},
    )

    assert not runner._propagation_valid("cspn", False, rows)


def test_propagation_valid_accepts_official_nlspn_without_anchor_injection():
    rows = (
        {"signal": "state", "mse": 0.01},
        {"signal": "affinity_constraints",
         "coefficient_sum_max_error": 0.0,
         "contraction_violation_rate": 0.0},
    )

    assert runner._propagation_valid("nlspn", False, rows)
    assert runner._propagation_valid("completionformer", False, rows)


def test_propagation_valid_requires_model_specific_anchor_evidence():
    no_anchor = (
        {"signal": "state", "mse": 0.01},
        {"signal": "affinity_constraints",
         "coefficient_sum_max_error": 0.0,
         "contraction_violation_rate": 0.0},
    )
    unexpected_anchor = no_anchor + (
        {"signal": "anchor_injection", "anchor_max_error": 0.0},)

    assert not runner._propagation_valid("dyspn", False, no_anchor)
    assert not runner._propagation_valid("nlspn", True, no_anchor)
    assert not runner._propagation_valid(
        "completionformer", False, unexpected_anchor)


def test_propagation_iteration_audit_requires_every_official_step():
    rows = tuple(
        {"signal": "state", "iteration": iteration, "mse": 0.0}
        for iteration in range(1, 5))

    assert runner._propagation_iterations_valid(rows, 4)
    assert not runner._propagation_iterations_valid(rows[:-1], 4)


def test_assignment_artifact_persists_measured_evidence_and_tuple_payload():
    result = runner.search_p3_t3(
        contract(), costs(), MeasuredEvaluator(),
        4, 4, 8, 8, 1.65, 1.65, 0.10, 3)
    root = Path(__file__).resolve().parent / ".model_p3t3_output"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    checkpoint = root / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")

    identity = runner.capture_checkpoint_identity(checkpoint)
    path = runner.write_p3_t3_assignment(root, result, identity)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == root / "p3_t3_assignment.json"
    assert payload["format_version"] == 3
    assert payload["artifact_kind"] == "nyu_model_p3_t3_assignment"
    assert payload["model_name"] == "model_z"
    assert payload["source_checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "size_bytes": len(b"official-checkpoint"),
        "sha256": identity.sha256,
    }
    assert payload["prefix"] == ["alpha", "beta"]
    assert payload["tail"] == ["delta"]
    assert payload["assignment"]["weight_bits"][0] == ["weight_alpha", 8]
    assert payload["assignment"]["activation_bits"][0] == [
        ["edge_alpha", "input"], 8]
    assert payload["cost_definition"]["weight_denominator"] == 40
    assert payload["cost_definition"]["activation_denominator"] == 40
    assert payload["budgets"] == {
        "maximum_normalized_activation_cost": 1.65,
        "maximum_normalized_weight_cost": 1.65,
    }
    assert payload["selection_policy"]["metric_aggregation"] == \
        "mean_of_per_sample_rmse"
    assert payload["selection_policy"]["maximum_relative_rmse_loss"] == 0.10
    assert "relative_rmse_loss" in payload["candidates"][0]
    assert payload["evaluation"]["split"] == "val"
    assert payload["evaluation"]["count"] == 3
    assert payload["evaluation"]["indices"] == [0, 1, 2]
    assert len(payload["evaluation"]["identity_sha256"]) == 64
    assert payload["cost_basis"]["weight_macs"][0] == ["weight_alpha", 1]
    assert len(payload["candidates"]) == 14
    assert all("pooled_rmse" in row and "paired_sample_differences" in row
               for row in payload["candidates"])
    assert payload["candidates"][0]["sample_evidence"] == [
        {
            "sample_index": 0,
            "squared_error_sum": 1.0,
            "valid_pixels": 1,
            "prediction_finite": True,
            "propagation_valid": True,
            "reproducible": True,
        },
        {
            "sample_index": 1,
            "squared_error_sum": pytest.approx(1.01 ** 2 * 2),
            "valid_pixels": 2,
            "prediction_finite": True,
            "propagation_valid": True,
            "reproducible": True,
        },
        {
            "sample_index": 2,
            "squared_error_sum": pytest.approx(1.02 ** 2 * 3),
            "valid_pixels": 3,
            "prediction_finite": True,
            "propagation_valid": True,
            "reproducible": True,
        },
    ]
    shutil.rmtree(root)


def test_assignment_artifact_retains_nonfinite_invalid_candidate_as_json_null():
    class NonfiniteCandidateEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = list(super().__call__(candidates))
            for row in rows:
                if row["config"] == "SINGLE_B001":
                    row["squared_error_sum"] = float("inf")
                    row["RMSE"] = float("inf")
                    row["prediction_finite"] = False
            return tuple(rows)

    result = runner.search_p3_t3(
        contract(), costs(), NonfiniteCandidateEvaluator(),
        4, 4, 8, 8, 1.65, 1.65, 0.10, 3)
    root = Path(__file__).resolve().parent / ".model_p3t3_nonfinite_output"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    checkpoint = root / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")

    path = runner.write_p3_t3_assignment(
        root, result, runner.capture_checkpoint_identity(checkpoint))
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    invalid = next(row for row in payload["candidates"]
                   if row["name"] == "SINGLE_B001")
    selected = next(row for row in payload["candidates"]
                    if row["name"] == payload["selected_candidate"])

    assert "Infinity" not in text
    assert "NaN" not in text
    assert invalid["pooled_rmse"] is None
    assert invalid["mean_sample_rmse"] is None
    assert invalid["sample_rmse"][0][1] is None
    assert invalid["paired_sample_differences"][0] is None
    assert not invalid["metrics_finite"]
    assert selected["metrics_finite"]
    assert isinstance(selected["pooled_rmse"], float)
    shutil.rmtree(root)


def test_runtime_search_builds_the_official_contract_and_closes_runtime(monkeypatch):
    class FakeRuntime(object):
        model_name = "model_z"
        device = "cuda:7"

        def __init__(self):
            self.model = object()
            self.closed = False

        def build_model(self, device):
            assert device == "cuda:7"
            return self.model

        def close(self):
            self.closed = True

    runtime = FakeRuntime()
    monkeypatch.setattr(
        runner,
        "build_model_quantization_contract",
        lambda model_name, model: contract(),
    )
    observed = {}

    def evaluator_factory(observed_runtime, model, observed_contract, registry):
        assert observed_runtime is runtime
        assert model is runtime.model
        assert observed_contract == contract()
        assert registry.model_name == "model_z"
        observed["evaluator"] = MeasuredEvaluator()
        return observed["evaluator"]

    result = runner.run_runtime_search(
        runtime, costs(), evaluator_factory,
        4, 4, 8, 8, 1.65, 1.65, 0.10, 3)

    assert result.assignment.model_name == "model_z"
    assert runtime.closed
    assert observed["evaluator"].closed


def test_hard_evaluator_closes_partial_resources_when_calibration_fails(
        monkeypatch):
    class Closer(object):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class JointCloser(Closer):
        def unbind_qdrop_sites(self):
            return None

    joint = JointCloser()
    propagation = Closer()
    instrumentor = Closer()

    def fail_after_installation(self):
        self.joint_adapter = joint
        self.propagation_adapter = propagation
        self.instrumentor = instrumentor
        raise RuntimeError("calibration failed")

    runtime = type("Runtime", (), {
        "model_name": "model_z",
        "device": torch.device("cuda:0"),
        "saved_args": type("Args", (), {"seed": 1})(),
        "build_dataset": lambda self, split: (object(),),
    })()
    settings = runner.HardDeploymentSettings(
        device="cuda:0",
        calibration_metadata=Path("calibration.json"),
        calibration_count=1,
        evaluation_indices=(0,),
        base_weight_bits=4,
        base_activation_bits=4,
        promotion_weight_bits=8,
        promotion_activation_bits=8,
        fold_conv_bn=True,
        fold_max_error=0.0,
        joint_clip_factors=(1.0,),
        joint_search_rounds=1,
        joint_cache_sample_limit=1,
        joint_cache_byte_limit=1,
    )
    monkeypatch.setattr(runner, "_calibration_indices",
                        lambda *args: (0,))
    monkeypatch.setattr(
        runner, "resolve_qdrop_targets",
        lambda model_name, model: type("Plan", (), {
            "activation_sites": (),
        })())
    monkeypatch.setattr(
        runner.HardDeploymentP3T3Evaluator,
        "_prepare_and_calibrate", fail_after_installation)

    with pytest.raises(RuntimeError, match="calibration failed"):
        runner.HardDeploymentP3T3Evaluator(
            runtime, torch.nn.Module(),
            QuantizationModelContract(
                model_name="model_z",
                blocks=(QuantizationBlock(
                    "block", ("weight",), ()),),
                prefix_groups=(("block",),),
                tail_groups=(("block",),),
                protected_roles=("propagation_state",),
                attention_edges=(),
                concat_edges=(),
                protected_modules=(),
                module_roles=(),
            ),
            mixed_precision.AllocationRegistry(
                weights_by_block={"block": ("weight",)},
                activations_by_block={"block": ()}, blocks=("block",),
                model_name="model_z"),
            settings)

    assert joint.closed
    assert propagation.closed
    assert instrumentor.closed


def test_cli_runs_selected_model_search_and_writes_assignment():
    root = Path(__file__).resolve().parent / ".model_p3t3_cli"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    weight_rows = root / "weight_cost_rows.csv"
    activation_rows = root / "activation_cost_rows.csv"
    weight_rows.write_text(
        "module,macs\n" + "".join(
            "weight_%s,%d\n" % row for row in zip(
                ("alpha", "beta", "gamma", "delta"), (1, 2, 4, 3))),
        encoding="utf-8")
    activation_rows.write_text(
        "site,role,elements\n" + "".join(
            "edge_%s,input,%d\n" % row for row in zip(
                ("alpha", "beta", "gamma", "delta"), (1, 2, 4, 3))),
        encoding="utf-8")

    class FakeRuntime(object):
        def __init__(self, model_config):
            self.model_name = "dyspn"
            self.device = model_config.device
            self.model = object()
            self.closed = False

        def build_model(self, device):
            assert str(device) == "cuda:0"
            return self.model

        def close(self):
            self.closed = True

    observed = {}

    def runtime_factory(model_config):
        observed["model_config"] = model_config
        observed["runtime"] = FakeRuntime(model_config)
        return observed["runtime"]

    def contract_builder(model_name, model):
        assert model_name == "dyspn"
        assert model is observed["runtime"].model
        return contract()

    def evaluator_factory(runtime, model, observed_contract, registry, settings):
        assert runtime is observed["runtime"]
        assert model is runtime.model
        assert observed_contract == contract()
        assert registry.model_name == "model_z"
        assert settings.device == "cuda:0"

        class FixedEvaluationEvaluator(object):
            def __init__(self):
                self.closed = False

            def reference_sample_rmse(self):
                return tuple(
                    (sample_index, 1.0 + sample_index * 0.0001)
                    for sample_index in settings.evaluation_indices)

            def __call__(self, candidates):
                rows = []
                for candidate in candidates:
                    measured = score(candidate)
                    for sample_index in settings.evaluation_indices:
                        pixels = sample_index + 1
                        sample_rmse = measured + sample_index * 0.0001
                        rows.append({
                            "config": candidate.name,
                            "sample_index": sample_index,
                            "squared_error_sum":
                                sample_rmse ** 2 * pixels,
                            "valid_pixels": pixels,
                            "RMSE": sample_rmse,
                            "prediction_finite": True,
                            "propagation_valid": True,
                            "reproducible": True,
                        })
                return tuple(rows)

            def close(self):
                self.closed = True

        observed["evaluator"] = FixedEvaluationEvaluator()
        return observed["evaluator"]

    dependencies = runner.RunnerDependencies(
        runtime_factory=runtime_factory,
        contract_builder=contract_builder,
        evaluator_factory=evaluator_factory,
    )
    config_path = Path(__file__).resolve().parents[1] / \
        "configs/three_model_selected_quantization.json"
    artifact = runner.run_cli((
        "--config", str(config_path),
        "--model", "dyspn",
        "--device", "cuda:0",
        "--maximum-normalized-weight-cost", "1.65",
        "--maximum-normalized-activation-cost", "1.65",
        "--maximum-relative-rmse-loss", "0.1",
        "--weight-cost-rows", str(weight_rows),
        "--activation-cost-rows", str(activation_rows),
        "--output", str(root),
        "--fold-conv-bn",
        "--fold-max-error", "0.000001",
        "--joint-clip-factors", "1.0",
        "--joint-search-rounds", "1",
        "--joint-cache-sample-limit", "1",
        "--joint-cache-byte-limit", "1024",
    ), dependencies=dependencies)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact == root / "p3_t3_assignment.json"
    assert payload["selected_candidate"] == "INTERACTION_P2_T02"
    assert observed["model_config"].model == "dyspn"
    assert observed["runtime"].closed
    assert observed["evaluator"].closed
    shutil.rmtree(root)


def test_direct_execution_requires_all_cli_arguments():
    script = Path(runner.__file__).resolve()

    completed = subprocess.run(
        (sys.executable, str(script)),
        cwd=str(script.parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "the following arguments are required" in completed.stderr
    assert "--config" in completed.stderr
