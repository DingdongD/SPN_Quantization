import pytest

from spn_quant.task_aware_allocation import greedy_promotion_allocation


def test_promotion_allocation_starts_at_four_and_meets_budget():
    costs = {"sensitive": 1, "wide": 3}
    scores = {
        "sensitive": {4: 9.0, 6: 6.0, 8: 0.0},
        "wide": {4: 8.0, 6: 7.0, 8: 0.0},
    }

    result = greedy_promotion_allocation(
        costs, scores, maximum_average_bits=6.0, fixed_bits={})

    assignment = dict(result)
    assert assignment["sensitive"] == 8
    assert assignment["wide"] == 4
    assert sum(assignment[name] * costs[name]
               for name in costs) / 4.0 <= 6.0


def test_promotion_allocation_rejects_incomplete_scores():
    with pytest.raises(ValueError, match="score levels"):
        greedy_promotion_allocation(
            {"a": 1}, {"a": {4: 1.0, 6: 0.5}},
            maximum_average_bits=6.0, fixed_bits={})
