#!/usr/bin/env python3
"""Run measured, model-relative P3/T3 mixed-precision selection."""

from __future__ import annotations

from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from spn_quant import mixed_precision  # noqa: E402
from spn_quant.model_contracts import (  # noqa: E402
    QuantizationModelContract,
    build_model_quantization_contract,
)


P3T3Candidate = mixed_precision.P3T3Candidate
P3T3CandidateResult = mixed_precision.P3T3CandidateResult
P3T3SearchResult = mixed_precision.P3T3SearchResult


def _ordered_union(blocks: Sequence[str], registry) -> Tuple[str, ...]:
    selected = set(blocks)
    return tuple(block for block in registry.blocks if block in selected)


def _candidate(
        name: str,
        stage: str,
        prefix: Sequence[str],
        tail: Sequence[str],
        registry: mixed_precision.AllocationRegistry,
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int) -> P3T3Candidate:
    normalized_prefix = tuple(str(block) for block in prefix)
    normalized_tail = tuple(str(block) for block in tail)
    promoted = _ordered_union(
        normalized_prefix + normalized_tail, registry)
    assignment = mixed_precision.promoted_assignment(
        registry,
        promoted,
        base_weight_bits,
        base_activation_bits,
        promotion_weight_bits,
        promotion_activation_bits,
    )
    return P3T3Candidate(
        name=name,
        stage=stage,
        prefix=normalized_prefix,
        tail=normalized_tail,
        promoted_blocks=promoted,
        assignment=assignment,
    )


def _tail_combinations(contract, registry):
    combinations = []
    group_count = len(contract.tail_groups)
    width = max(2, len(str((1 << group_count) - 1)))
    for mask in range(1, 1 << group_count):
        groups = tuple(
            contract.tail_groups[index]
            for index in range(group_count)
            if mask & (1 << index))
        tail = _ordered_union(tuple(itertools.chain.from_iterable(groups)), registry)
        combinations.append((mask, width, tail))
    return tuple(combinations)


def build_p3_t3_candidates(
        contract: QuantizationModelContract,
        registry: mixed_precision.AllocationRegistry,
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int) -> Tuple[P3T3Candidate, ...]:
    """Build every measured candidate from the model contract topology."""
    if contract.model_name != registry.model_name:
        raise ValueError("contract and allocation registry model mismatch")
    if contract.block_names != registry.blocks:
        raise ValueError("contract and allocation registry blocks differ")
    output = [_candidate(
        "UNIFORM_W%dA%d" % (base_weight_bits, base_activation_bits),
        "baseline", (), (), registry,
        base_weight_bits, base_activation_bits,
        promotion_weight_bits, promotion_activation_bits)]
    for index, block in enumerate(registry.blocks, 1):
        output.append(_candidate(
            "SINGLE_B%03d" % index,
            "single_block", (), (block,), registry,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits))
    for prefix_index, prefix in enumerate(contract.prefix_groups, 1):
        output.append(_candidate(
            "PREFIX_P%d" % prefix_index,
            "prefix", prefix, (), registry,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits))
    tail_combinations = _tail_combinations(contract, registry)
    for tail_mask, width, tail in tail_combinations:
        output.append(_candidate(
            "TAIL_T%0*d" % (width, tail_mask),
            "tail", (), tail, registry,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits))
    for prefix_index, prefix in enumerate(contract.prefix_groups, 1):
        for tail_mask, width, tail in tail_combinations:
            output.append(_candidate(
                "INTERACTION_P%d_T%0*d" % (
                    prefix_index, width, tail_mask),
                "interaction", prefix, tail, registry,
                base_weight_bits, base_activation_bits,
                promotion_weight_bits, promotion_activation_bits))
    names = tuple(candidate.name for candidate in output)
    if len(names) != len(set(names)):
        raise ValueError("P3/T3 candidate names contain duplicates")
    return tuple(output)


def _normalized_cost(assignment, costs, base_weight_bits, base_activation_bits):
    weight_bits = dict(assignment.weight_bits)
    activation_bits = dict(assignment.activation_bits)
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    if set(weight_bits) != set(weight_costs):
        raise ValueError("weight assignment and cost coverage mismatch")
    if set(activation_bits) != set(activation_costs):
        raise ValueError("activation assignment and cost coverage mismatch")
    weight_denominator = int(base_weight_bits) * sum(weight_costs.values())
    activation_denominator = (
        int(base_activation_bits) * sum(activation_costs.values()))
    if weight_denominator <= 0 or activation_denominator <= 0:
        raise ValueError("normalized precision cost denominator must be positive")
    weight = sum(weight_bits[name] * weight_costs[name]
                 for name in weight_costs) / float(weight_denominator)
    activation = sum(activation_bits[owner] * activation_costs[owner]
                     for owner in activation_costs) / float(
                         activation_denominator)
    return weight, activation


def _measured_rows(candidates, rows, costs, base_weight_bits,
                   base_activation_bits, expected_samples):
    if int(expected_samples) <= 0:
        raise ValueError("expected sample count must be positive")
    by_name = dict((candidate.name, []) for candidate in candidates)
    for row in rows:
        name = str(row["config"])
        if name not in by_name:
            raise ValueError("measured candidate coverage mismatch")
        by_name[name].append(row)

    sample_ids = None
    partial = []
    for candidate in candidates:
        selected = tuple(sorted(
            by_name[candidate.name], key=lambda row: int(row["sample_index"])))
        identities = tuple(int(row["sample_index"]) for row in selected)
        if len(selected) != int(expected_samples) or \
                len(identities) != len(set(identities)):
            raise ValueError("measured sample coverage mismatch")
        if sample_ids is None:
            sample_ids = identities
        elif identities != sample_ids:
            raise ValueError("paired measured sample identities differ")
        squared_error_sum = 0.0
        valid_pixels = 0
        sample_rmse = []
        flags = []
        finite = True
        for row in selected:
            squared = float(row["squared_error_sum"])
            pixels = int(row["valid_pixels"])
            rmse = float(row["RMSE"])
            if isinstance(row["sample_index"], bool) or \
                    isinstance(row["valid_pixels"], bool) or pixels <= 0:
                raise ValueError("measured valid pixel count must be positive")
            validity = (
                row["prediction_finite"],
                row["propagation_valid"],
                row["reproducible"],
            )
            if not all(isinstance(value, bool) for value in validity):
                raise ValueError("measured validity flags must be booleans")
            if math.isfinite(squared) and squared < 0.0:
                raise ValueError("measured squared error must be nonnegative")
            if math.isfinite(squared) and math.isfinite(rmse) and not \
                    math.isclose(
                        rmse * rmse * pixels, squared,
                        rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("measured RMSE and squared error disagree")
            finite = finite and math.isfinite(squared) and math.isfinite(rmse)
            squared_error_sum += squared
            valid_pixels += pixels
            sample_rmse.append((int(row["sample_index"]), rmse))
            flags.append(all(validity))
        pooled_rmse = math.sqrt(squared_error_sum / float(valid_pixels)) \
            if finite and squared_error_sum >= 0.0 else float("inf")
        mean_sample_rmse = sum(value for index, value in sample_rmse) / \
            float(len(sample_rmse)) if finite else float("inf")
        weight_cost, activation_cost = _normalized_cost(
            candidate.assignment, costs,
            base_weight_bits, base_activation_bits)
        partial.append(P3T3CandidateResult(
            name=candidate.name,
            stage=candidate.stage,
            prefix=candidate.prefix,
            tail=candidate.tail,
            assignment=candidate.assignment,
            pooled_rmse=pooled_rmse,
            mean_sample_rmse=mean_sample_rmse,
            normalized_weight_cost=weight_cost,
            normalized_activation_cost=activation_cost,
            valid=finite and all(flags),
            sample_rmse=tuple(sample_rmse),
            paired_sample_differences=(),
        ))
    baseline = partial[0]
    baseline_samples = dict(baseline.sample_rmse)
    return tuple(replace(
        row,
        paired_sample_differences=tuple(
            value - baseline_samples[index]
            for index, value in row.sample_rmse),
    ) for row in partial)


def _dominates(left, right):
    no_worse = (
        left.pooled_rmse <= right.pooled_rmse and
        left.normalized_weight_cost <= right.normalized_weight_cost and
        left.normalized_activation_cost <= right.normalized_activation_cost)
    strictly_better = (
        left.pooled_rmse < right.pooled_rmse or
        left.normalized_weight_cost < right.normalized_weight_cost or
        left.normalized_activation_cost < right.normalized_activation_cost)
    return no_worse and strictly_better


def _prefix_knee(rows):
    stable = tuple(row for row in rows if row.stage == "prefix" and row.valid)
    if not stable:
        raise RuntimeError("P3/T3 search has no stable prefix candidate")
    frontier = tuple(
        row for row in stable
        if not any(_dominates(other, row) for other in stable if other != row))
    ordered = tuple(sorted(
        frontier,
        key=lambda row: (
            (row.normalized_weight_cost + row.normalized_activation_cost) / 2.0,
            row.pooled_rmse,
            row.prefix,
        )))
    if len(ordered) <= 2:
        return ordered[0]
    costs = tuple(
        (row.normalized_weight_cost + row.normalized_activation_cost) / 2.0
        for row in ordered)
    errors = tuple(row.pooled_rmse for row in ordered)
    cost_range = costs[-1] - costs[0]
    error_high = max(errors)
    error_low = min(errors)
    error_range = error_high - error_low
    if cost_range == 0.0 or error_range == 0.0:
        return ordered[0]
    points = tuple(
        ((cost - costs[0]) / cost_range,
         (error_high - error) / error_range)
        for cost, error in zip(costs, errors))
    x0, y0 = points[0]
    x1, y1 = points[-1]
    denominator = math.hypot(y1 - y0, x1 - x0)
    ranked = []
    for index, (x_value, y_value) in enumerate(points):
        distance = abs(
            (y1 - y0) * x_value - (x1 - x0) * y_value +
            x1 * y0 - y1 * x0) / denominator
        ranked.append((-distance, costs[index], ordered[index].prefix, ordered[index]))
    return min(ranked)[3]


def search_p3_t3(
        contract: QuantizationModelContract,
        costs: mixed_precision.CostBasis,
        evaluator: Callable[[Sequence[P3T3Candidate]], Sequence[Mapping[str, object]]],
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int,
        maximum_normalized_weight_cost: float,
        maximum_normalized_activation_cost: float,
        expected_samples: int) -> P3T3SearchResult:
    """Measure every candidate and select a stable model-relative P3/T3."""
    maximum_weight = float(maximum_normalized_weight_cost)
    maximum_activation = float(maximum_normalized_activation_cost)
    if not math.isfinite(maximum_weight) or maximum_weight <= 0.0 or \
            not math.isfinite(maximum_activation) or maximum_activation <= 0.0:
        raise ValueError("normalized precision budgets must be finite and positive")
    registry = mixed_precision.build_registry(contract, costs)
    candidates = build_p3_t3_candidates(
        contract, registry,
        base_weight_bits, base_activation_bits,
        promotion_weight_bits, promotion_activation_bits)
    rows = tuple(evaluator(candidates))
    measured = _measured_rows(
        candidates, rows, costs,
        base_weight_bits, base_activation_bits, expected_samples)
    prefix = _prefix_knee(measured)
    tails = tuple(
        row for row in measured
        if row.stage == "interaction" and row.prefix == prefix.prefix and
        row.valid and row.normalized_weight_cost <= maximum_weight and
        row.normalized_activation_cost <= maximum_activation)
    if not tails:
        raise RuntimeError("P3/T3 search has no stable budget-valid tail")
    selected = min(tails, key=lambda row: (
        row.pooled_rmse,
        row.mean_sample_rmse,
        row.normalized_weight_cost,
        row.normalized_activation_cost,
        row.tail,
        row.name,
    ))
    return P3T3SearchResult(
        assignment=selected.assignment,
        prefix=prefix.prefix,
        tail=selected.tail,
        selected_candidate=selected.name,
        candidates=measured,
        cost_basis=costs,
        base_weight_bits=int(base_weight_bits),
        base_activation_bits=int(base_activation_bits),
        promotion_weight_bits=int(promotion_weight_bits),
        promotion_activation_bits=int(promotion_activation_bits),
        maximum_normalized_weight_cost=maximum_weight,
        maximum_normalized_activation_cost=maximum_activation,
        expected_samples=int(expected_samples),
    )


def run_runtime_search(
        runtime: NYUModelRuntime,
        costs: mixed_precision.CostBasis,
        evaluator_factory,
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int,
        maximum_normalized_weight_cost: float,
        maximum_normalized_activation_cost: float,
        expected_samples: int) -> P3T3SearchResult:
    """Bind an official NYU runtime to the generic measured search."""
    try:
        model = runtime.build_model(runtime.device)
        contract = build_model_quantization_contract(runtime.model_name, model)
        registry = mixed_precision.build_registry(contract, costs)
        evaluator = evaluator_factory(runtime, model, contract, registry)
        return search_p3_t3(
            contract, costs, evaluator,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits,
            maximum_normalized_weight_cost,
            maximum_normalized_activation_cost,
            expected_samples,
        )
    finally:
        runtime.close()


def _assignment_payload(assignment):
    return {
        "model_name": assignment.model_name,
        "weight_bits": [[module, bits]
                        for module, bits in assignment.weight_bits],
        "activation_bits": [[[owner[0], owner[1]], bits]
                            for owner, bits in assignment.activation_bits],
    }


def _candidate_payload(row):
    return {
        "name": row.name,
        "stage": row.stage,
        "prefix": list(row.prefix),
        "tail": list(row.tail),
        "pooled_rmse": row.pooled_rmse,
        "mean_sample_rmse": row.mean_sample_rmse,
        "normalized_weight_cost": row.normalized_weight_cost,
        "normalized_activation_cost": row.normalized_activation_cost,
        "valid": row.valid,
        "sample_rmse": [[index, value] for index, value in row.sample_rmse],
        "paired_sample_differences": list(row.paired_sample_differences),
        "assignment": _assignment_payload(row.assignment),
    }


def write_p3_t3_assignment(
        output: Path,
        result: P3T3SearchResult) -> Path:
    """Persist the selected tuple assignment and all measured search evidence."""
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("P3/T3 output directory is missing: %s" % root)
    path = root / "p3_t3_assignment.json"
    if path.exists():
        raise FileExistsError("P3/T3 assignment already exists: %s" % path)
    payload = {
        "model_name": result.assignment.model_name,
        "prefix": list(result.prefix),
        "tail": list(result.tail),
        "selected_candidate": result.selected_candidate,
        "precision": {
            "base_activation_bits": result.base_activation_bits,
            "base_weight_bits": result.base_weight_bits,
            "promotion_activation_bits": result.promotion_activation_bits,
            "promotion_weight_bits": result.promotion_weight_bits,
        },
        "budgets": {
            "maximum_normalized_activation_cost":
                result.maximum_normalized_activation_cost,
            "maximum_normalized_weight_cost":
                result.maximum_normalized_weight_cost,
        },
        "expected_samples": result.expected_samples,
        "cost_definition": {
            "activation_denominator": result.base_activation_bits * sum(
                elements for owner, elements
                in result.cost_basis.activation_elements),
            "activation_formula":
                "sum(activation_bits*elements)/activation_denominator",
            "weight_denominator": result.base_weight_bits * sum(
                macs for module, macs in result.cost_basis.weight_macs),
            "weight_formula": "sum(weight_bits*macs)/weight_denominator",
        },
        "cost_basis": {
            "activation_elements": [
                [[owner[0], owner[1]], elements]
                for owner, elements in result.cost_basis.activation_elements],
            "weight_macs": [[module, macs]
                            for module, macs in result.cost_basis.weight_macs],
        },
        "assignment": _assignment_payload(result.assignment),
        "candidates": [_candidate_payload(row) for row in result.candidates],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
