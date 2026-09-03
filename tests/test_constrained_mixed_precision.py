import math

import pytest

from spn_quant.constrained_mixed_precision import (
    MeasuredCandidate,
    PrecisionAssignment,
    PrecisionCosts,
    balanced_knee,
    feasible_pareto_frontier,
    relative_loss,
    weighted_average_bits,
)


def _assignment(encoder=(4, 8), head=(8, 4)):
    return PrecisionAssignment(
        weight_bits=(("encoder", encoder[0]), ("head", head[0])),
        activation_bits=(("encoder", encoder[1]), ("head", head[1])),
        scale_policies=(("encoder", "static_tensor"),
                        ("head", "static_tensor")),
        expected_units=("encoder", "head"),
    )


def _candidate(candidate_id, weight_bits, activation_bits, rmse):
    assignment = _assignment(
        encoder=(weight_bits, activation_bits),
        head=(weight_bits, activation_bits),
    )
    return MeasuredCandidate(
        candidate_id=candidate_id,
        assignment=assignment,
        pooled_rmse=rmse,
        reference_pooled_rmse=1.0,
        average_weight_bits=float(weight_bits),
        average_activation_bits=float(activation_bits),
        fp16_mac_fraction=0.0,
        fp16_activation_fraction=0.0,
    )


def test_assignment_requires_complete_independent_weight_and_activation_maps():
    with pytest.raises(ValueError, match="weight assignment coverage"):
        PrecisionAssignment(
            weight_bits=(("encoder", 8),),
            activation_bits=(("encoder", 8), ("decoder", 8)),
            scale_policies=(("encoder", "static_tensor"),
                            ("decoder", "static_tensor")),
            expected_units=("encoder", "decoder"),
        )


def test_assignment_rejects_non_integer_search_precision():
    with pytest.raises(ValueError, match="4, 6, or 8"):
        PrecisionAssignment(
            weight_bits=(("encoder", 5),),
            activation_bits=(("encoder", 8),),
            scale_policies=(("encoder", "static_tensor"),),
            expected_units=("encoder",),
        )


def test_assignment_serializes_in_canonical_unit_order():
    assignment = PrecisionAssignment(
        weight_bits=(("head", 8), ("encoder", 4)),
        activation_bits=(("head", 4), ("encoder", 8)),
        scale_policies=(("head", "static_tensor"),
                        ("encoder", "dynamic_group8")),
        expected_units=("encoder", "head"),
    )

    assert assignment.canonical_payload() == {
        "weight_bits": {"encoder": 4, "head": 8},
        "activation_bits": {"encoder": 8, "head": 4},
        "scale_policies": {
            "encoder": "dynamic_group8",
            "head": "static_tensor",
        },
        "fp16_units": [],
    }


def test_assignment_requires_explicit_fp16_unit_and_counts_its_cost():
    assignment = PrecisionAssignment(
        weight_bits=(("encoder", 4), ("head", 16)),
        activation_bits=(("encoder", 8), ("head", 16)),
        scale_policies=(("encoder", "static_tensor"),
                        ("head", "static_tensor")),
        expected_units=("encoder", "head"),
        fp16_units=("head",),
    )
    costs = PrecisionCosts(
        weight_macs=(("encoder", 90), ("head", 10)),
        activation_elements=(("encoder", 10), ("head", 90)),
    )

    assert weighted_average_bits(assignment, costs) == pytest.approx(
        (5.2, 15.2))
    assert assignment.canonical_payload()["fp16_units"] == ["head"]


def test_assignment_rejects_implicit_or_partial_fp16():
    with pytest.raises(ValueError, match="FP16 units require W16A16"):
        PrecisionAssignment(
            weight_bits=(("head", 16),),
            activation_bits=(("head", 8),),
            scale_policies=(("head", "static_tensor"),),
            expected_units=("head",),
            fp16_units=("head",),
        )


def test_weighted_costs_use_macs_and_activation_elements():
    costs = PrecisionCosts(
        weight_macs=(("encoder", 90), ("head", 10)),
        activation_elements=(("encoder", 10), ("head", 90)),
    )

    assert weighted_average_bits(_assignment(), costs) == pytest.approx(
        (4.4, 4.4))


def test_relative_loss_uses_pooled_rmse_ratio():
    assert relative_loss(1.01, 1.0) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="finite and positive"):
        relative_loss(math.inf, 1.0)


def test_pareto_frontier_enforces_one_percent_before_dominance():
    candidates = (
        _candidate("min-a", 8, 4, 1.009),
        _candidate("balanced", 6, 6, 1.005),
        _candidate("dominated", 8, 8, 1.001),
        _candidate("infeasible", 4, 4, 1.011),
    )

    frontier = feasible_pareto_frontier(
        candidates, maximum_relative_loss=0.01)

    assert tuple(row.candidate_id for row in frontier) == (
        "balanced", "min-a")


def test_balanced_knee_normalizes_only_feasible_frontier():
    frontier = (
        _candidate("min-w", 4, 8, 1.009),
        _candidate("knee", 6, 6, 1.005),
        _candidate("min-a", 8, 4, 1.001),
    )

    assert balanced_knee(frontier).candidate_id == "knee"
