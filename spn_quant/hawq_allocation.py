"""Mixed-precision allocation from CSPN HAWQ trace costs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np
from scipy import optimize
import torch


BITS = (4, 6, 8)


@dataclass(frozen=True)
class HAWQBlock:
    name: str
    weight_elements: int
    activation_elements: int
    fixed_eight: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "weight_elements", int(self.weight_elements))
        object.__setattr__(
            self, "activation_elements", int(self.activation_elements))
        object.__setattr__(self, "fixed_eight", bool(self.fixed_eight))
        if not self.name:
            raise ValueError("HAWQ block name must be nonempty")
        if self.weight_elements <= 0 or self.activation_elements <= 0:
            raise ValueError("HAWQ block costs must be positive")


@dataclass(frozen=True)
class HAWQCandidate:
    block: str
    bits: int
    cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "block", str(self.block))
        object.__setattr__(self, "bits", int(self.bits))
        object.__setattr__(self, "cost", float(self.cost))
        if self.bits not in BITS:
            raise ValueError("HAWQ candidate bits must be 4, 6, or 8")
        if not math.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("HAWQ candidate cost must be finite and nonnegative")


@dataclass(frozen=True)
class HAWQAssignment:
    block_bits: Tuple[Tuple[str, int], ...]
    average_weight_bits: float
    average_activation_bits: float
    objective: float
    solver_status: str


def candidate_cost(trace: float, weight: torch.Tensor, bits: int,
                   channel_dim: int) -> float:
    trace = float(trace)
    bits = int(bits)
    channel_dim = int(channel_dim)
    if not math.isfinite(trace) or trace < 0.0:
        raise ValueError("HAWQ trace must be finite and nonnegative")
    if bits not in BITS:
        raise ValueError("HAWQ perturbation bits must be 4, 6, or 8")
    if not torch.is_tensor(weight) or weight.numel() == 0 or \
            not bool(torch.isfinite(weight).all().item()):
        raise ValueError("HAWQ weight must be a finite nonempty tensor")
    if channel_dim < 0 or channel_dim >= weight.ndim:
        raise ValueError("HAWQ weight channel dimension is invalid")
    qmax = (1 << (bits - 1)) - 1
    flat = weight.detach().movedim(channel_dim, 0).reshape(
        weight.shape[channel_dim], -1)
    maximum = flat.abs().amax(dim=1)
    scale = torch.where(
        maximum > 0.0, maximum / float(qmax), torch.ones_like(maximum))
    shape = [1] * weight.ndim
    shape[channel_dim] = int(weight.shape[channel_dim])
    scale = scale.reshape(shape)
    quantized = torch.round(weight.detach() / scale).clamp(
        -qmax, qmax) * scale
    error = float((weight.detach() - quantized).to(
        torch.float64).square().sum().item())
    return trace * error


def _validate_inputs(blocks, candidates, coupled_blocks):
    declared_blocks = tuple(blocks)
    declared_candidates = tuple(candidates)
    if not declared_blocks:
        raise ValueError("HAWQ allocation requires blocks")
    names = tuple(block.name for block in declared_blocks)
    if len(names) != len(set(names)):
        raise ValueError("HAWQ block names contain duplicates")
    lookup = {}
    for candidate in declared_candidates:
        key = (candidate.block, candidate.bits)
        if key in lookup:
            raise ValueError("HAWQ candidate rows contain duplicates")
        lookup[key] = candidate
    expected = set((name, bits) for name in names for bits in BITS)
    if set(lookup) != expected:
        raise ValueError("HAWQ candidate coverage mismatch")
    couples = tuple((str(left), str(right))
                    for left, right in coupled_blocks)
    if any(left not in names or right not in names or left == right
           for left, right in couples):
        raise ValueError("HAWQ coupled block contract is invalid")
    return declared_blocks, lookup, couples


def _constraint_rows(blocks, variable_index, maximum_weight_bits,
                     maximum_activation_bits, coupled_blocks):
    variable_count = len(variable_index)
    rows = []
    lower = []
    upper = []
    for block in blocks:
        row = np.zeros(variable_count, dtype=np.float64)
        for bits in BITS:
            row[variable_index[(block.name, bits)]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
        if block.fixed_eight:
            fixed = np.zeros(variable_count, dtype=np.float64)
            fixed[variable_index[(block.name, 8)]] = 1.0
            rows.append(fixed)
            lower.append(1.0)
            upper.append(1.0)
    for left, right in coupled_blocks:
        for bits in BITS:
            row = np.zeros(variable_count, dtype=np.float64)
            row[variable_index[(left, bits)]] = 1.0
            row[variable_index[(right, bits)]] = -1.0
            rows.append(row)
            lower.append(0.0)
            upper.append(0.0)
    weight = np.zeros(variable_count, dtype=np.float64)
    activation = np.zeros(variable_count, dtype=np.float64)
    for block in blocks:
        for bits in BITS:
            index = variable_index[(block.name, bits)]
            weight[index] = bits * block.weight_elements
            activation[index] = bits * block.activation_elements
    rows.extend((weight, activation))
    lower.extend((-np.inf, -np.inf))
    upper.extend((
        float(maximum_weight_bits) * sum(
            block.weight_elements for block in blocks),
        float(maximum_activation_bits) * sum(
            block.activation_elements for block in blocks),
    ))
    return rows, lower, upper


def _solve(objective, rows, lower, upper):
    matrix = np.stack(rows, axis=0)
    return optimize.milp(
        c=np.asarray(objective, dtype=np.float64),
        integrality=np.ones(len(objective), dtype=np.int8),
        bounds=optimize.Bounds(0.0, 1.0),
        constraints=optimize.LinearConstraint(
            matrix, np.asarray(lower), np.asarray(upper)),
        options={"mip_rel_gap": 0.0},
    )


def solve_hawq_assignment(
        blocks: Sequence[HAWQBlock],
        candidates: Sequence[HAWQCandidate],
        maximum_weight_bits: float,
        maximum_activation_bits: float,
        coupled_blocks: Sequence[Tuple[str, str]]) -> HAWQAssignment:
    maximum_weight_bits = float(maximum_weight_bits)
    maximum_activation_bits = float(maximum_activation_bits)
    if not math.isfinite(maximum_weight_bits) or maximum_weight_bits <= 0.0:
        raise ValueError("HAWQ weight budget must be finite and positive")
    if not math.isfinite(maximum_activation_bits) or \
            maximum_activation_bits <= 0.0:
        raise ValueError("HAWQ activation budget must be finite and positive")
    blocks, lookup, couples = _validate_inputs(
        blocks, candidates, coupled_blocks)
    variables = tuple(
        (block.name, bits) for block in blocks for bits in BITS)
    variable_index = dict(
        (variable, index) for index, variable in enumerate(variables))
    objective = np.asarray(
        [lookup[variable].cost for variable in variables], dtype=np.float64)
    rows, lower, upper = _constraint_rows(
        blocks, variable_index, maximum_weight_bits,
        maximum_activation_bits, couples)
    primary = _solve(objective, rows, lower, upper)
    if not primary.success:
        raise RuntimeError("HAWQ primary solver failed: %s" % primary.message)
    primary_values = np.rint(primary.x)
    optimum = float(np.dot(objective, primary_values))
    tolerance = max(1.0, abs(optimum)) * 1e-10
    tie_rows = list(rows) + [objective]
    tie_lower = list(lower) + [optimum - tolerance]
    tie_upper = list(upper) + [optimum + tolerance]
    tie_objective = np.arange(1, len(variables) + 1, dtype=np.float64)
    secondary = _solve(
        tie_objective, tie_rows, tie_lower, tie_upper)
    if not secondary.success:
        raise RuntimeError(
            "HAWQ secondary solver failed: %s" % secondary.message)
    values = np.rint(secondary.x).astype(np.int64)
    if not np.allclose(secondary.x, values, rtol=0.0, atol=1e-7):
        raise RuntimeError("HAWQ solver returned non-integral assignment")
    selected = []
    for block in blocks:
        bits = tuple(
            bits for bits in BITS
            if values[variable_index[(block.name, bits)]] == 1)
        if len(bits) != 1:
            raise RuntimeError("HAWQ solver block selection is invalid")
        selected.append((block.name, bits[0]))
    selected_map = dict(selected)
    total_weight = sum(block.weight_elements for block in blocks)
    total_activation = sum(block.activation_elements for block in blocks)
    average_weight = sum(
        selected_map[block.name] * block.weight_elements
        for block in blocks) / float(total_weight)
    average_activation = sum(
        selected_map[block.name] * block.activation_elements
        for block in blocks) / float(total_activation)
    selected_objective = sum(
        lookup[(name, bits)].cost for name, bits in selected)
    return HAWQAssignment(
        block_bits=tuple(selected),
        average_weight_bits=average_weight,
        average_activation_bits=average_activation,
        objective=selected_objective,
        solver_status=str(secondary.message),
    )
