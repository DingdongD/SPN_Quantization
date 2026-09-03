"""Strict cost and Pareto rules for task-aware mixed precision."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Tuple


INTEGER_BITS = (4, 6, 8)
SCALE_POLICIES = (
    "branch_independent",
    "dynamic_group8",
    "static_tensor",
)


def _unique_mapping(rows, name):
    keys = tuple(row[0] for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate %s unit" % name)
    return dict(rows)


@dataclass(frozen=True)
class PrecisionAssignment:
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[str, int], ...]
    scale_policies: Tuple[Tuple[str, str], ...]
    expected_units: Tuple[str, ...]
    fp16_units: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.expected_units:
            raise ValueError("precision assignment requires units")
        if len(set(self.expected_units)) != len(self.expected_units):
            raise ValueError("duplicate expected precision unit")
        expected = set(self.expected_units)
        if len(set(self.fp16_units)) != len(self.fp16_units):
            raise ValueError("duplicate FP16 precision unit")
        if not set(self.fp16_units) <= expected:
            raise ValueError("FP16 precision unit is unknown")
        weights = _unique_mapping(self.weight_bits, "weight assignment")
        activations = _unique_mapping(
            self.activation_bits, "activation assignment")
        policies = _unique_mapping(self.scale_policies, "scale policy")
        if set(weights) != expected:
            raise ValueError("weight assignment coverage differs from units")
        if set(activations) != expected:
            raise ValueError(
                "activation assignment coverage differs from units")
        if set(policies) != expected:
            raise ValueError("scale policy coverage differs from units")
        fp16 = set(self.fp16_units)
        for name in self.expected_units:
            weight_bits = int(weights[name])
            activation_bits = int(activations[name])
            if name in fp16:
                if weight_bits != 16 or activation_bits != 16:
                    raise ValueError("FP16 units require W16A16")
            elif weight_bits not in INTEGER_BITS:
                raise ValueError("weight bits must be 4, 6, or 8")
            elif activation_bits not in INTEGER_BITS:
                raise ValueError("activation bits must be 4, 6, or 8")
        unknown_policies = sorted(
            set(policies.values()) - set(SCALE_POLICIES))
        if unknown_policies:
            raise ValueError("unknown scale policies: %s" % unknown_policies)

    def canonical_payload(self) -> Dict[str, Dict[str, object]]:
        weights = dict(self.weight_bits)
        activations = dict(self.activation_bits)
        policies = dict(self.scale_policies)
        return {
            "weight_bits": dict(
                (name, int(weights[name])) for name in self.expected_units),
            "activation_bits": dict(
                (name, int(activations[name])) for name in self.expected_units),
            "scale_policies": dict(
                (name, policies[name]) for name in self.expected_units),
            "fp16_units": list(self.fp16_units),
        }


@dataclass(frozen=True)
class PrecisionCosts:
    weight_macs: Tuple[Tuple[str, int], ...]
    activation_elements: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        weights = _unique_mapping(self.weight_macs, "weight cost")
        activations = _unique_mapping(
            self.activation_elements, "activation cost")
        if not weights:
            raise ValueError("precision costs require units")
        if set(weights) != set(activations):
            raise ValueError("weight and activation cost coverage differs")
        if any(int(value) <= 0 for value in weights.values()):
            raise ValueError("weight MAC costs must be positive")
        if any(int(value) <= 0 for value in activations.values()):
            raise ValueError("activation element costs must be positive")


@dataclass(frozen=True)
class MeasuredCandidate:
    candidate_id: str
    assignment: PrecisionAssignment
    pooled_rmse: float
    reference_pooled_rmse: float
    average_weight_bits: float
    average_activation_bits: float
    fp16_mac_fraction: float
    fp16_activation_fraction: float

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("measured candidate requires an ID")
        relative_loss(self.pooled_rmse, self.reference_pooled_rmse)
        for value in (
                self.average_weight_bits,
                self.average_activation_bits,
                self.fp16_mac_fraction,
                self.fp16_activation_fraction):
            if not math.isfinite(float(value)):
                raise ValueError("candidate costs must be finite")
        if self.average_weight_bits <= 0.0 or \
                self.average_activation_bits <= 0.0:
            raise ValueError("candidate average bits must be positive")
        if not 0.0 <= self.fp16_mac_fraction <= 1.0:
            raise ValueError("FP16 MAC fraction must be within [0, 1]")
        if not 0.0 <= self.fp16_activation_fraction <= 1.0:
            raise ValueError(
                "FP16 activation fraction must be within [0, 1]")

    @property
    def relative_loss(self) -> float:
        return relative_loss(self.pooled_rmse, self.reference_pooled_rmse)


def relative_loss(pooled_rmse: float, reference_pooled_rmse: float) -> float:
    values = (float(pooled_rmse), float(reference_pooled_rmse))
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("pooled RMSE values must be finite and positive")
    return values[0] / values[1] - 1.0


def weighted_average_bits(assignment: PrecisionAssignment,
                          costs: PrecisionCosts) -> Tuple[float, float]:
    weight_bits = dict(assignment.weight_bits)
    activation_bits = dict(assignment.activation_bits)
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    expected = set(assignment.expected_units)
    if set(weight_costs) != expected:
        raise ValueError("weight cost coverage differs from assignment")
    if set(activation_costs) != expected:
        raise ValueError("activation cost coverage differs from assignment")
    total_macs = float(sum(weight_costs.values()))
    total_elements = float(sum(activation_costs.values()))
    average_weight = sum(
        int(weight_bits[name]) * weight_costs[name]
        for name in assignment.expected_units) / total_macs
    average_activation = sum(
        int(activation_bits[name]) * activation_costs[name]
        for name in assignment.expected_units) / total_elements
    return average_weight, average_activation


def dominates(left: MeasuredCandidate, right: MeasuredCandidate) -> bool:
    no_worse = left.average_weight_bits <= right.average_weight_bits and \
        left.average_activation_bits <= right.average_activation_bits
    strictly_better = left.average_weight_bits < right.average_weight_bits or \
        left.average_activation_bits < right.average_activation_bits
    return no_worse and strictly_better


def feasible_pareto_frontier(
        candidates: Iterable[MeasuredCandidate],
        maximum_relative_loss: float) -> Tuple[MeasuredCandidate, ...]:
    limit = float(maximum_relative_loss)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("relative-loss limit must be finite and nonnegative")
    feasible = tuple(candidate for candidate in candidates
                     if candidate.relative_loss <= limit)
    frontier = tuple(candidate for candidate in feasible
                     if not any(dominates(other, candidate)
                                for other in feasible if other != candidate))
    return tuple(sorted(
        frontier,
        key=lambda row: (
            row.average_weight_bits,
            row.average_activation_bits,
            row.pooled_rmse,
            row.candidate_id,
        )))


def balanced_knee(
        frontier: Iterable[MeasuredCandidate]) -> MeasuredCandidate:
    rows = tuple(frontier)
    if not rows:
        raise ValueError("balanced knee requires a nonempty frontier")
    weight_values = tuple(row.average_weight_bits for row in rows)
    activation_values = tuple(row.average_activation_bits for row in rows)
    weight_span = max(weight_values) - min(weight_values)
    activation_span = max(activation_values) - min(activation_values)

    def normalized(value, minimum, span):
        return 0.0 if span == 0.0 else (value - minimum) / span

    return min(rows, key=lambda row: (
        math.hypot(
            normalized(row.average_weight_bits,
                       min(weight_values), weight_span),
            normalized(row.average_activation_bits,
                       min(activation_values), activation_span),
        ),
        row.pooled_rmse,
        row.candidate_id,
    ))
