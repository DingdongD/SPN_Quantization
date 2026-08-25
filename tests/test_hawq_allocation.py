import pytest
import torch

from spn_quant.hawq_allocation import (
    HAWQBlock,
    HAWQCandidate,
    HAWQIndependentBlock,
    candidate_cost,
    solve_independent_hawq_assignment,
    solve_hawq_assignment,
)


def _candidates(costs):
    return tuple(
        HAWQCandidate(block, bits, cost)
        for block, rows in costs
        for bits, cost in rows)


def test_hawq_solver_selects_sensitive_block_at_eight_bits():
    blocks = (
        HAWQBlock("a", 10, 10, False),
        HAWQBlock("b", 10, 10, False),
    )
    candidates = _candidates((
        ("a", ((4, 100.0), (6, 10.0), (8, 0.0))),
        ("b", ((4, 2.0), (6, 1.0), (8, 0.0))),
    ))

    assignment = solve_hawq_assignment(
        blocks, candidates, 6.0, 6.0, ())

    assert assignment.block_bits == (("a", 8), ("b", 4))
    assert assignment.average_weight_bits == 6.0
    assert assignment.average_activation_bits == 6.0


def test_fixed_stem_and_depth_head_are_counted_in_budget():
    blocks = (
        HAWQBlock("encoder_stem", 1, 1, True),
        HAWQBlock("body", 3, 3, False),
        HAWQBlock("initial_depth", 1, 1, True),
    )
    candidates = _candidates((
        ("encoder_stem", ((4, 8.0), (6, 4.0), (8, 0.0))),
        ("body", ((4, 8.0), (6, 4.0), (8, 0.0))),
        ("initial_depth", ((4, 8.0), (6, 4.0), (8, 0.0))),
    ))

    assignment = solve_hawq_assignment(
        blocks, candidates, 6.0, 6.0, ())

    assert assignment.block_bits == (
        ("encoder_stem", 8), ("body", 4), ("initial_depth", 8))
    assert assignment.average_weight_bits == 5.6
    assert assignment.average_activation_bits == 5.6


def test_coupled_blocks_receive_identical_bits():
    blocks = (
        HAWQBlock("main", 1, 1, False),
        HAWQBlock("skip", 1, 1, False),
        HAWQBlock("other", 2, 2, False),
    )
    candidates = _candidates((
        ("main", ((4, 100.0), (6, 2.0), (8, 0.0))),
        ("skip", ((4, 1.0), (6, 0.5), (8, 0.0))),
        ("other", ((4, 1.0), (6, 0.5), (8, 0.0))),
    ))

    assignment = solve_hawq_assignment(
        blocks, candidates, 6.0, 6.0, (("main", "skip"),))

    selected = dict(assignment.block_bits)
    assert selected["main"] == selected["skip"]


def test_missing_candidate_bit_is_rejected():
    blocks = (HAWQBlock("a", 1, 1, False),)
    candidates = (
        HAWQCandidate("a", 4, 2.0),
        HAWQCandidate("a", 6, 1.0),
    )

    with pytest.raises(ValueError, match="candidate coverage"):
        solve_hawq_assignment(blocks, candidates, 6.0, 6.0, ())


def test_impossible_fixed_budget_is_rejected():
    blocks = (HAWQBlock("fixed", 1, 1, True),)
    candidates = _candidates((
        ("fixed", ((4, 2.0), (6, 1.0), (8, 0.0))),
    ))

    with pytest.raises(RuntimeError, match="solver"):
        solve_hawq_assignment(blocks, candidates, 6.0, 6.0, ())


def test_candidate_cost_uses_trace_times_weight_error():
    weight = torch.tensor([[[[1.0, 0.25]]]])
    cost = candidate_cost(2.0, weight, bits=4, channel_dim=0)
    qmax = 7.0
    scale = weight.abs().amax() / qmax
    quantized = torch.round(weight / scale).clamp(-7, 7) * scale
    expected = 2.0 * float(
        (weight - quantized).to(torch.float64).square().sum())

    assert cost == expected


def test_independent_solver_does_not_tie_activation_bits_to_weight_bits():
    blocks = (
        HAWQIndependentBlock("sensitive", 10, 10, 30),
        HAWQIndependentBlock("ordinary", 10, 10, 10),
    )
    candidates = _candidates((
        ("sensitive", ((4, 100.0), (6, 10.0), (8, 0.0))),
        ("ordinary", ((4, 2.0), (6, 1.0), (8, 0.0))),
    ))

    assignment = solve_independent_hawq_assignment(
        blocks, candidates, 6.0, 6.0)

    assert assignment.weight_block_bits == (
        ("sensitive", 8), ("ordinary", 4))
    assert assignment.activation_block_bits == (
        ("sensitive", 4), ("ordinary", 4))
    assert assignment.average_weight_bits == 6.0
    assert assignment.average_weight_mac_bits == 6.0
    assert assignment.average_activation_bits == 4.0
    assert assignment.weight_parameter_budget_residual == 0.0
    assert assignment.weight_mac_budget_residual == 0.0
    assert assignment.activation_budget_residual == 2.0
    assert tuple(
        (term.block, term.bits, term.cost)
        for term in assignment.objective_components) == (
            ("sensitive", 8, 0.0),
            ("ordinary", 4, 2.0),
        )


def test_independent_solver_enforces_explicit_weight_mac_budget():
    blocks = (
        HAWQIndependentBlock("mac_heavy", 1, 9, 1),
        HAWQIndependentBlock("parameter_heavy", 9, 1, 1),
    )
    candidates = _candidates((
        ("mac_heavy", ((4, 100.0), (6, 10.0), (8, 0.0))),
        ("parameter_heavy", ((4, 0.0), (6, 0.0), (8, 0.0))),
    ))

    assignment = solve_independent_hawq_assignment(
        blocks, candidates, 6.0, 6.0)

    assert dict(assignment.weight_block_bits)["mac_heavy"] == 6
    assert assignment.average_weight_bits == 4.2
    assert assignment.average_weight_mac_bits == 5.8
    assert assignment.average_weight_mac_bits <= 6.0


def test_independent_block_requires_all_explicit_positive_costs():
    with pytest.raises(ValueError, match="positive"):
        HAWQIndependentBlock("block", 1, 0, 1)
