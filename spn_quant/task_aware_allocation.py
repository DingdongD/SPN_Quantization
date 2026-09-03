"""Task-aware greedy allocation over independent weight and activation budgets."""

from __future__ import annotations

import math
from typing import Mapping, Tuple


BIT_LEVELS = (4, 6, 8)


def _validated_inputs(costs, scores, maximum_average_bits, fixed_bits):
    names = tuple(sorted(costs, key=str))
    if not names:
        raise ValueError("allocation costs must not be empty")
    if set(scores) != set(names):
        raise ValueError("allocation score coverage differs from costs")
    limit = float(maximum_average_bits)
    if not math.isfinite(limit) or limit < BIT_LEVELS[0]:
        raise ValueError("allocation budget must be finite and at least 4 bits")
    fixed = dict((name, int(bits)) for name, bits in dict(fixed_bits).items())
    if not set(fixed) <= set(names):
        raise ValueError("fixed allocation unit is unknown")
    if any(bits not in BIT_LEVELS for bits in fixed.values()):
        raise ValueError("fixed allocation bits must be 4, 6, or 8")
    for name in names:
        if set(scores[name]) != set(BIT_LEVELS):
            raise ValueError("allocation score levels are incomplete: %s" % name)
        if any(not math.isfinite(float(scores[name][bits]))
               for bits in BIT_LEVELS):
            raise ValueError("allocation scores must be finite: %s" % name)
        if int(costs[name]) <= 0:
            raise ValueError("allocation costs must be positive: %s" % name)
    return names, fixed, limit


def _average(assignment, costs, names) -> float:
    denominator = float(sum(int(costs[name]) for name in names))
    return sum(assignment[name] * int(costs[name]) for name in names) / \
        denominator


def greedy_budget_allocation(
        costs: Mapping[str, int],
        scores: Mapping[str, Mapping[int, float]],
        maximum_average_bits: float,
        fixed_bits: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    """Demote the least task-sensitive unit until its weighted budget is met."""
    names, fixed, limit = _validated_inputs(
        costs, scores, maximum_average_bits, fixed_bits)

    assignment = dict((name, fixed[name] if name in fixed else 8)
                      for name in names)

    while _average(assignment, costs, names) > limit:
        moves = []
        for name in names:
            current = assignment[name]
            if name in fixed or current == BIT_LEVELS[0]:
                continue
            next_bits = BIT_LEVELS[1] if current == BIT_LEVELS[2] else \
                BIT_LEVELS[0]
            saved = (current - next_bits) * int(costs[name])
            penalty = float(scores[name][next_bits]) - \
                float(scores[name][current])
            if saved <= 0:
                raise RuntimeError("allocation demotion saves no budget: %s" % name)
            moves.append((penalty / float(saved), penalty, name, next_bits))
        if not moves:
            raise ValueError("bit budget is infeasible with protected units")
        _, _, name, next_bits = min(moves)
        assignment[name] = next_bits

    return tuple((name, assignment[name]) for name in names)


def greedy_promotion_allocation(
        costs: Mapping[str, int],
        scores: Mapping[str, Mapping[int, float]],
        maximum_average_bits: float,
        fixed_bits: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    """Promote the most sensitive units from 4 bits under a hard budget."""
    names, fixed, limit = _validated_inputs(
        costs, scores, maximum_average_bits, fixed_bits)
    assignment = dict((name, fixed[name] if name in fixed else 4)
                      for name in names)
    if _average(assignment, costs, names) > limit:
        raise ValueError("bit budget is infeasible with protected units")

    while True:
        moves = []
        for name in names:
            current = assignment[name]
            if name in fixed or current == BIT_LEVELS[-1]:
                continue
            next_bits = 6 if current == 4 else 8
            saved = (next_bits - current) * int(costs[name])
            if saved <= 0:
                raise RuntimeError("allocation promotion costs no budget: %s" % name)
            if _average(assignment, costs, names) + saved / float(
                    sum(int(costs[item]) for item in names)) > limit:
                continue
            gain = float(scores[name][current]) - \
                float(scores[name][next_bits])
            if gain <= 0.0:
                continue
            moves.append((gain / float(saved), gain, name, next_bits))
        if not moves:
            break
        _, _, name, next_bits = min(
            moves, key=lambda row: (-row[0], -row[1], row[2], row[3]))
        assignment[name] = next_bits

    return tuple((name, assignment[name]) for name in names)


def boundary_budget_allocation(
        costs: Mapping[str, int],
        scores: Mapping[str, Mapping[int, float]],
        maximum_average_bits: float,
        fixed_bits: Mapping[str, int],
        boundary_bits: int) -> Tuple[Tuple[str, int], ...]:
    """Allocate around a fixed precision boundary, normally W6/A6.

    Protected units start at their requested precision, while all other units
    start at the boundary. Only non-protected units are demoted when the
    protected units consume budget. This prevents a direct 8-to-4 collapse
    from turning the W6 operating point into an unstable mixed assignment.
    """
    names, fixed, limit = _validated_inputs(
        costs, scores, maximum_average_bits, fixed_bits)
    boundary = int(boundary_bits)
    if boundary not in BIT_LEVELS:
        raise ValueError("allocation boundary bits must be 4, 6, or 8")
    assignment = dict((name, fixed[name] if name in fixed else boundary)
                      for name in names)

    while _average(assignment, costs, names) > limit:
        moves = []
        for name in names:
            current = assignment[name]
            if name in fixed or current == BIT_LEVELS[0]:
                continue
            next_bits = BIT_LEVELS[1] if current == BIT_LEVELS[2] else \
                BIT_LEVELS[0]
            saved = (current - next_bits) * int(costs[name])
            penalty = float(scores[name][next_bits]) - \
                float(scores[name][current])
            if saved <= 0:
                raise RuntimeError("allocation demotion saves no budget: %s" % name)
            moves.append((penalty / float(saved), penalty, name, next_bits))
        if not moves:
            raise ValueError("bit budget is infeasible with protected units")
        _, _, name, next_bits = min(moves)
        assignment[name] = next_bits

    return tuple((name, assignment[name]) for name in names)
