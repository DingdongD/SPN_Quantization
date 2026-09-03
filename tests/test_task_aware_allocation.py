from __future__ import annotations

import pytest

from spn_quant.task_aware_allocation import (
    boundary_budget_allocation,
    greedy_budget_allocation,
)


def _scores(names):
    return {
        name: {4: 4.0, 6: 1.0, 8: 0.0}
        for name in names
    }


def test_greedy_allocation_meets_weighted_budget():
    result = greedy_budget_allocation(
        {"a": 1, "b": 3}, _scores(("a", "b")), 5.0, fixed_bits={})
    assignment = dict(result)
    assert assignment["a"] == 6
    assert assignment["b"] == 4


def test_fixed_units_are_not_demoted():
    result = greedy_budget_allocation(
        {"a": 1, "b": 1}, _scores(("a", "b")), 6.0,
        fixed_bits={"a": 8})
    assert dict(result) == {"a": 8, "b": 4}


def test_infeasible_protected_budget_is_explicit():
    with pytest.raises(ValueError, match="infeasible"):
        greedy_budget_allocation(
            {"a": 1}, _scores(("a",)), 4.0, fixed_bits={"a": 8})


def test_boundary_allocation_preserves_protected_units_and_uses_four_bit_tail():
    result = boundary_budget_allocation(
        {"a": 1, "b": 1, "c": 2}, _scores(("a", "b", "c")), 6.0,
        fixed_bits={"a": 8}, boundary_bits=6)
    assignment = dict(result)
    assert assignment["a"] == 8
    assert sum(assignment[name] * cost for name, cost in
               (("a", 1), ("b", 1), ("c", 2))) / 4.0 <= 6.0
    assert 4 in assignment.values()
