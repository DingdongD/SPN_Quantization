"""Selective CSPN W4A4/W4A8 activation search contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

from spn_quant.cspn_encoder_prefix import UnitRegistry


Owner = Tuple[str, str]

ACTIVATION_UNIT_ORDER = (
    "stem",
    "encoder_layer1",
    "encoder_layer2",
    "decoder_layer4",
)
INITIAL_DEPTH_WEIGHT = "gud_up_proj_layer5.conv1"
RMSE_LIMIT = 0.1773


@dataclass(frozen=True)
class ActivationCandidate:
    name: str
    mask: int
    selected_units: Tuple[str, ...]
    activation_owners: Tuple[Owner, ...]
    weight_modules: Tuple[str, ...]


@dataclass(frozen=True)
class BoundaryDemotion:
    name: str
    owner: Owner
    activation_owners: Tuple[Owner, ...]
    weight_modules: Tuple[str, ...]


@dataclass(frozen=True)
class CumulativeCandidate:
    name: str
    step: int
    demoted_owners: Tuple[Owner, ...]
    activation_owners: Tuple[Owner, ...]
    weight_modules: Tuple[str, ...]


def _require_unique(values, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicates" % name)


def _ordered_union(sequences) -> Tuple[Owner, ...]:
    output = []
    for sequence in sequences:
        for value in sequence:
            if value not in output:
                output.append(value)
    return tuple(output)


def _validate_registry(registry: UnitRegistry) -> None:
    if not isinstance(registry, UnitRegistry):
        raise TypeError("selective W4A8 search requires UnitRegistry")
    for unit in ACTIVATION_UNIT_ORDER:
        registry.activations_by_unit[unit]
        owners = tuple(registry.activations_by_unit[unit])
        _require_unique(owners, "%s activation owners" % unit)


def build_stage1_candidates(
        registry: UnitRegistry) -> Tuple[ActivationCandidate, ...]:
    _validate_registry(registry)
    output = []
    for mask in range(1 << len(ACTIVATION_UNIT_ORDER)):
        selected = tuple(
            unit for index, unit in enumerate(ACTIVATION_UNIT_ORDER)
            if mask & (1 << index))
        owners = _ordered_union(
            registry.activations_by_unit[unit] for unit in selected)
        output.append(ActivationCandidate(
            name="ACT_MASK_%02d" % mask,
            mask=mask,
            selected_units=selected,
            activation_owners=owners,
            weight_modules=(INITIAL_DEPTH_WEIGHT,),
        ))
    _require_unique(tuple(candidate.name for candidate in output),
                    "Stage-1 candidate names")
    return tuple(output)


def candidate_is_feasible(
        row: Mapping[str, object], rmse_limit: float,
        require_rerun: bool) -> bool:
    values = (
        float(row["RMSE"]),
        float(row["nonfinite_ratio"]),
        float(row["nonpositive_ratio"]),
        float(row["coefficient_sum_max_error"]),
        float(row["contraction_violation_rate"]),
        float(row["anchor_max_error"]),
    )
    if not all(math.isfinite(value) for value in values):
        return False
    if int(row["samples"]) != 64:
        return False
    if values[0] > float(rmse_limit):
        return False
    if any(value != 0.0 for value in values[1:]):
        return False
    if require_rerun:
        rerun = float(row["rerun_RMSE"])
        if not math.isfinite(rerun) or rerun != values[0]:
            return False
    return True


def _validated_candidate_rows(
        rows: Sequence[Mapping[str, object]],
        candidates: Sequence[ActivationCandidate]) -> list[Dict[str, object]]:
    expected = tuple(candidate.name for candidate in candidates)
    _require_unique(expected, "candidate names")
    output = []
    names = []
    for source in rows:
        row = dict(source)
        name = str(row["config"])
        names.append(name)
        output.append(row)
    _require_unique(tuple(names), "candidate metric rows")
    if set(names) != set(expected) or len(names) != len(expected):
        raise ValueError("Stage-1 metric coverage mismatch")
    return output


def select_stage1_anchor(
        rows: Sequence[Mapping[str, object]],
        candidates: Sequence[ActivationCandidate],
        rmse_limit: float) -> Dict[str, object]:
    validated = _validated_candidate_rows(rows, candidates)
    feasible = [
        row for row in validated
        if candidate_is_feasible(row, rmse_limit, require_rerun=False)]
    if not feasible:
        raise RuntimeError("no Stage-1 candidate satisfies the target")
    return min(feasible, key=lambda row: (
        float(row["normalized_added_bit_cost"]),
        float(row["a8_activation_element_fraction"]),
        float(row["RMSE"]),
        str(row["config"])))


def build_single_demotions(
        anchor: ActivationCandidate) -> Tuple[BoundaryDemotion, ...]:
    if not isinstance(anchor, ActivationCandidate):
        raise TypeError("boundary anchor must be ActivationCandidate")
    _require_unique(anchor.activation_owners, "anchor activation owners")
    return tuple(
        BoundaryDemotion(
            name="DEMOTE_%03d" % index,
            owner=owner,
            activation_owners=tuple(
                current for current in anchor.activation_owners
                if current != owner),
            weight_modules=anchor.weight_modules,
        )
        for index, owner in enumerate(anchor.activation_owners))


def rank_demotions(
        anchor_row: Mapping[str, object],
        rows: Sequence[Mapping[str, object]],
        demotions: Sequence[BoundaryDemotion]) -> list[Dict[str, object]]:
    expected = dict((demotion.name, demotion) for demotion in demotions)
    if len(expected) != len(demotions):
        raise ValueError("single demotions contain duplicate names")
    anchor_mse = float(anchor_row["propagation_mse"])
    if not math.isfinite(anchor_mse) or anchor_mse < 0.0:
        raise ValueError("anchor propagation MSE must be finite and nonnegative")
    values = []
    names = set()
    for source in rows:
        row = dict(source)
        name = str(row["config"])
        if name in names:
            raise ValueError("demotion calibration rows contain duplicates")
        if name not in expected:
            raise ValueError("unknown demotion calibration row: %s" % name)
        demotion = expected[name]
        owner = str(row["module"]), str(row["kind"])
        if owner != demotion.owner:
            raise ValueError("demotion owner does not match calibration row")
        propagation_mse = float(row["propagation_mse"])
        downstream_mse = float(row["downstream_mse"])
        anchor_downstream_mse = float(row["anchor_downstream_mse"])
        saved_cost = float(row["saved_cost"])
        numeric = (
            propagation_mse, downstream_mse,
            anchor_downstream_mse, saved_cost)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("demotion calibration values must be finite")
        if propagation_mse < 0.0 or downstream_mse < 0.0 or \
                anchor_downstream_mse < 0.0 or saved_cost <= 0.0:
            raise ValueError("demotion calibration values are invalid")
        increase = max(0.0, propagation_mse - anchor_mse)
        downstream_increase = max(
            0.0, downstream_mse - anchor_downstream_mse)
        values.append({
            "config": name,
            "module": owner[0],
            "kind": owner[1],
            "propagation_mse": propagation_mse,
            "propagation_mse_increase": increase,
            "downstream_mse": downstream_mse,
            "anchor_downstream_mse": anchor_downstream_mse,
            "downstream_mse_increase": downstream_increase,
            "saved_cost": saved_cost,
            "score": increase / saved_cost,
        })
        names.add(name)
    if names != set(expected):
        raise ValueError("demotion calibration coverage mismatch")
    values.sort(key=lambda row: (
        float(row["score"]),
        float(row["downstream_mse_increase"]),
        -float(row["saved_cost"]),
        str(row["module"]), str(row["kind"])))
    output = []
    for rank, source in enumerate(values, start=1):
        row = dict(source)
        row["rank"] = rank
        output.append(row)
    return output


def build_cumulative_path(
        anchor: ActivationCandidate,
        ranking: Sequence[Mapping[str, object]]
        ) -> Tuple[CumulativeCandidate, ...]:
    expected = set(anchor.activation_owners)
    ordered = []
    ranks = []
    for row in ranking:
        ranks.append(int(row["rank"]))
        ordered.append((str(row["module"]), str(row["kind"])))
    if ranks != list(range(1, len(ranking) + 1)):
        raise ValueError("boundary ranking is not contiguous")
    _require_unique(tuple(ordered), "boundary ranking owners")
    if set(ordered) != expected or len(ordered) != len(expected):
        raise ValueError("boundary ranking coverage mismatch")
    output = []
    for step in range(len(ordered) + 1):
        demoted = tuple(ordered[:step])
        active = tuple(
            owner for owner in anchor.activation_owners
            if owner not in set(demoted))
        output.append(CumulativeCandidate(
            name="PATH_%03d" % step,
            step=step,
            demoted_owners=demoted,
            activation_owners=active,
            weight_modules=anchor.weight_modules,
        ))
    return tuple(output)


def select_winner(
        rows: Sequence[Mapping[str, object]],
        rmse_limit: float) -> Dict[str, object]:
    names = tuple(str(row["config"]) for row in rows)
    _require_unique(names, "winner rows")
    feasible = [
        dict(row) for row in rows
        if candidate_is_feasible(row, rmse_limit, require_rerun=True)]
    if not feasible:
        raise RuntimeError("no selective W4A8 candidate satisfies the target")
    return min(feasible, key=lambda row: (
        float(row["normalized_added_bit_cost"]),
        float(row["a8_activation_element_fraction"]),
        float(row["RMSE"]), str(row["config"])))


def pareto_rows(
        rows: Sequence[Mapping[str, object]],
        cost_field: str) -> list[Dict[str, object]]:
    if cost_field not in (
            "normalized_added_bit_cost",
            "a8_activation_element_fraction"):
        raise ValueError("unsupported Pareto cost field: %s" % cost_field)
    values = []
    names = set()
    for source in rows:
        row = dict(source)
        name = str(row["config"])
        cost = float(row[cost_field])
        rmse = float(row["RMSE"])
        if name in names:
            raise ValueError("Pareto rows contain duplicate configurations")
        if not math.isfinite(cost) or not math.isfinite(rmse) or cost < 0.0:
            raise ValueError("Pareto values must be finite and nonnegative")
        names.add(name)
        values.append((cost, rmse, row))
    output = []
    for cost, rmse, row in values:
        dominated = any(
            other_cost <= cost and other_rmse <= rmse and
            (other_cost < cost or other_rmse < rmse)
            for other_cost, other_rmse, _ in values)
        if not dominated:
            output.append(row)
    return sorted(output, key=lambda row: (
        float(row[cost_field]), float(row["RMSE"]), str(row["config"])))
