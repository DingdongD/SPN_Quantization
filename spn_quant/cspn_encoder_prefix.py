"""CSPN encoder-prefix and sensitive-tail W8A8 candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

from spn_quant.cspn_sensitivity import (
    ACTIVATION_OWNERS_BY_BLOCK,
    WEIGHT_MODULES_BY_BLOCK,
)


Owner = Tuple[str, str]

ENCODER_UNIT_ORDER = (
    "stem",
    "encoder_layer1",
    "encoder_layer2",
    "encoder_layer3",
    "encoder_layer4",
)
TAIL_UNIT_ORDER = (
    "decoder_layer4",
    "initial_depth",
)
ALL_UNIT_ORDER = ENCODER_UNIT_ORDER + TAIL_UNIT_ORDER

WEIGHT_MODULES_BY_UNIT = {
    "stem": ("conv1_1",),
    "encoder_layer1": (
        "layer1.0.conv1",
        "layer1.0.conv2",
        "layer1.1.conv1",
        "layer1.1.conv2",
    ),
    "encoder_layer2": (
        "layer2.0.conv1",
        "layer2.0.conv2",
        "layer2.0.downsample.0",
        "layer2.1.conv1",
        "layer2.1.conv2",
    ),
    "encoder_layer3": (
        "layer3.0.conv1",
        "layer3.0.conv2",
        "layer3.0.downsample.0",
        "layer3.1.conv1",
        "layer3.1.conv2",
    ),
    "encoder_layer4": (
        "layer4.0.conv1",
        "layer4.0.conv2",
        "layer4.0.downsample.0",
        "layer4.1.conv1",
        "layer4.1.conv2",
    ),
    "decoder_layer4": WEIGHT_MODULES_BY_BLOCK["decoder_layer4"],
    "initial_depth": WEIGHT_MODULES_BY_BLOCK["initial_depth"],
}

ACTIVATION_OWNERS_BY_UNIT = {
    "stem": (
        ("conv1_1", "input"),
        ("relu#0", "relu_output"),
        ("boundary_controller.layer4_signed_skip", "boundary"),
    ),
    "encoder_layer1": (
        ("layer1.0.conv1", "input"),
        ("layer1.0.conv2", "input"),
        ("layer1.0.relu#0", "relu_output"),
        ("layer1.0.relu#1", "relu_output"),
        ("layer1.1.conv1", "input"),
        ("layer1.1.conv2", "input"),
        ("layer1.1.relu#0", "relu_output"),
        ("layer1.1.relu#1", "relu_output"),
    ),
    "encoder_layer2": (
        ("layer2.0.conv1", "input"),
        ("layer2.0.conv2", "input"),
        ("layer2.0.downsample.0", "input"),
        ("layer2.0.downsample.0", "output"),
        ("layer2.0.relu#0", "relu_output"),
        ("layer2.0.relu#1", "relu_output"),
        ("layer2.1.conv1", "input"),
        ("layer2.1.conv2", "input"),
        ("layer2.1.relu#0", "relu_output"),
        ("layer2.1.relu#1", "relu_output"),
    ),
    "encoder_layer3": (
        ("layer3.0.conv1", "input"),
        ("layer3.0.conv2", "input"),
        ("layer3.0.downsample.0", "input"),
        ("layer3.0.downsample.0", "output"),
        ("layer3.0.relu#0", "relu_output"),
        ("layer3.0.relu#1", "relu_output"),
        ("layer3.1.conv1", "input"),
        ("layer3.1.conv2", "input"),
        ("layer3.1.relu#0", "relu_output"),
        ("layer3.1.relu#1", "relu_output"),
    ),
    "encoder_layer4": (
        ("layer4.0.conv1", "input"),
        ("layer4.0.conv2", "input"),
        ("layer4.0.downsample.0", "input"),
        ("layer4.0.downsample.0", "output"),
        ("layer4.0.relu#0", "relu_output"),
        ("layer4.0.relu#1", "relu_output"),
        ("layer4.1.conv1", "input"),
        ("layer4.1.conv2", "input"),
        ("layer4.1.relu#0", "relu_output"),
        ("layer4.1.relu#1", "relu_output"),
    ),
    "decoder_layer4": ACTIVATION_OWNERS_BY_BLOCK["decoder_layer4"],
    "initial_depth": ACTIVATION_OWNERS_BY_BLOCK["initial_depth"],
}

ENCODER_PREFIXES = tuple(
    ENCODER_UNIT_ORDER[:index] for index in range(len(ENCODER_UNIT_ORDER) + 1))
TAIL_STATES = (
    (),
    ("decoder_layer4",),
    ("initial_depth",),
    ("decoder_layer4", "initial_depth"),
)


@dataclass(frozen=True)
class UnitRegistry:
    weights_by_unit: Mapping[str, Tuple[str, ...]]
    activations_by_unit: Mapping[str, Tuple[Owner, ...]]


@dataclass(frozen=True)
class PrefixTailCandidate:
    name: str
    prefix_index: int
    tail_index: int
    encoder_units: Tuple[str, ...]
    tail_units: Tuple[str, ...]
    weight_modules: Tuple[str, ...]
    activation_owners: Tuple[Owner, ...]
    stem_w8a8: bool


def _ordered_union(sequences):
    output = []
    for sequence in sequences:
        for value in sequence:
            if value not in output:
                output.append(value)
    return tuple(output)


def _require_unique(values, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicates" % name)


def build_unit_registry(
        executed_modules: Sequence[str],
        activation_owners: Sequence[Owner]) -> UnitRegistry:
    modules = tuple(str(name) for name in executed_modules)
    owners = tuple((str(owner[0]), str(owner[1]))
                   for owner in activation_owners)
    _require_unique(modules, "executed weight registry")
    _require_unique(owners, "executed activation registry")
    for unit in ALL_UNIT_ORDER:
        _require_unique(
            WEIGHT_MODULES_BY_UNIT[unit], "%s weight registry" % unit)
        _require_unique(
            ACTIVATION_OWNERS_BY_UNIT[unit],
            "%s activation registry" % unit)
    expected_modules = _ordered_union(
        WEIGHT_MODULES_BY_UNIT[unit] for unit in ALL_UNIT_ORDER)
    expected_owners = _ordered_union(
        ACTIVATION_OWNERS_BY_UNIT[unit] for unit in ALL_UNIT_ORDER)
    if set(modules) != set(expected_modules):
        raise ValueError(
            "weight registry mismatch: missing=%s extra=%s" % (
                sorted(set(expected_modules) - set(modules)),
                sorted(set(modules) - set(expected_modules))))
    if set(owners) != set(expected_owners):
        raise ValueError(
            "activation registry mismatch: missing=%s extra=%s" % (
                sorted(set(expected_owners) - set(owners)),
                sorted(set(owners) - set(expected_owners))))
    return UnitRegistry(
        weights_by_unit=dict(
            (unit, tuple(WEIGHT_MODULES_BY_UNIT[unit]))
            for unit in ALL_UNIT_ORDER),
        activations_by_unit=dict(
            (unit, tuple(ACTIVATION_OWNERS_BY_UNIT[unit]))
            for unit in ALL_UNIT_ORDER),
    )


def build_candidates(
        registry: UnitRegistry) -> Tuple[PrefixTailCandidate, ...]:
    if tuple(registry.weights_by_unit) != ALL_UNIT_ORDER or \
            tuple(registry.activations_by_unit) != ALL_UNIT_ORDER:
        raise ValueError("unit registry order differs from the contract")
    output = []
    for prefix_index, encoder_units in enumerate(ENCODER_PREFIXES):
        for tail_index, tail_units in enumerate(TAIL_STATES):
            units = encoder_units + tail_units
            weights = _ordered_union(
                registry.weights_by_unit[unit] for unit in units)
            owners = _ordered_union(
                registry.activations_by_unit[unit] for unit in units)
            output.append(PrefixTailCandidate(
                name="PREFIX_P%d__TAIL_T%d" % (
                    prefix_index, tail_index),
                prefix_index=prefix_index,
                tail_index=tail_index,
                encoder_units=tuple(encoder_units),
                tail_units=tuple(tail_units),
                weight_modules=weights,
                activation_owners=owners,
                stem_w8a8="stem" in encoder_units,
            ))
    _require_unique(
        tuple(candidate.name for candidate in output), "candidate names")
    return tuple(output)


def precision_cost(
        weight_rows: Sequence[Mapping[str, object]],
        activation_rows: Sequence[Mapping[str, object]],
        candidate: PrefixTailCandidate) -> Dict[str, float]:
    weights = {}
    total_macs = 0
    total_weight_elements = 0
    for row in weight_rows:
        module = str(row["module"])
        if module in weights:
            raise ValueError("weight cost rows contain duplicates")
        weight_elements = int(row["weight_elements"])
        macs = int(row["macs"])
        if weight_elements <= 0 or macs <= 0:
            raise ValueError("weight cost values must be positive")
        weights[module] = weight_elements, macs
        total_weight_elements += weight_elements
        total_macs += macs
    activations = {}
    total_activation_elements = 0
    for row in activation_rows:
        owner = str(row["module"]), str(row["kind"])
        if owner in activations:
            raise ValueError("activation cost rows contain duplicates")
        elements = int(row["elements"])
        if elements <= 0:
            raise ValueError("activation cost values must be positive")
        activations[owner] = elements
        total_activation_elements += elements
    unknown_weights = set(candidate.weight_modules) - set(weights)
    unknown_owners = set(candidate.activation_owners) - set(activations)
    if unknown_weights or unknown_owners:
        raise ValueError("candidate precision cost coverage mismatch")
    promoted_weight_elements = sum(
        weights[name][0] for name in candidate.weight_modules)
    promoted_macs = sum(
        weights[name][1] for name in candidate.weight_modules)
    promoted_activation_elements = sum(
        activations[owner] for owner in candidate.activation_owners)
    total_elements = total_weight_elements + total_activation_elements
    promoted_elements = promoted_weight_elements + \
        promoted_activation_elements
    return {
        "normalized_added_bit_cost":
        promoted_elements / float(total_elements),
        "w8_weight_mac_fraction": promoted_macs / float(total_macs),
        "w8_weight_element_fraction":
        promoted_weight_elements / float(total_weight_elements),
        "a8_activation_element_fraction":
        promoted_activation_elements / float(total_activation_elements),
    }


def interaction_rows(
        aggregate_rows: Sequence[Mapping[str, object]]
        ) -> list[Dict[str, object]]:
    expected_cells = set(
        (prefix_index, tail_index)
        for prefix_index in range(len(ENCODER_PREFIXES))
        for tail_index in range(len(TAIL_STATES)))
    cells = {}
    names = set()
    for source in aggregate_rows:
        row = dict(source)
        name = str(row["config"])
        cell = int(row["prefix_index"]), int(row["tail_index"])
        rmse = float(row["RMSE"])
        if name in names or cell in cells:
            raise ValueError("interaction rows contain duplicates")
        if not math.isfinite(rmse):
            raise ValueError("interaction RMSE must be finite")
        names.add(name)
        cells[cell] = row
    if set(cells) != expected_cells:
        raise ValueError("interaction matrix coverage mismatch")
    baseline = float(cells[(0, 0)]["RMSE"])
    output = []
    for prefix_index in range(len(ENCODER_PREFIXES)):
        for tail_index in range(len(TAIL_STATES)):
            current = cells[(prefix_index, tail_index)]
            interaction = float(current["RMSE"]) - \
                float(cells[(prefix_index, 0)]["RMSE"]) - \
                float(cells[(0, tail_index)]["RMSE"]) + baseline
            output.append({
                "config": str(current["config"]),
                "prefix_index": prefix_index,
                "tail_index": tail_index,
                "RMSE": float(current["RMSE"]),
                "interaction_rmse": interaction,
            })
    return output


def pareto_rows(
        rows: Sequence[Mapping[str, object]],
        cost_field: str) -> list[Dict[str, object]]:
    if cost_field not in (
            "normalized_added_bit_cost", "w8_weight_mac_fraction"):
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
    return sorted(
        output,
        key=lambda row: (
            float(row[cost_field]), float(row["RMSE"]),
            str(row["config"])))


def prediction_candidate_names(
        aggregate_rows: Sequence[Mapping[str, object]],
        normalized_pareto_rows: Sequence[Mapping[str, object]]
        ) -> Tuple[str, ...]:
    rows = dict((str(row["config"]), row) for row in aggregate_rows)
    if len(rows) != len(aggregate_rows):
        raise ValueError("aggregate rows contain duplicate configurations")
    strict_name = "PREFIX_P0__TAIL_T0"
    full_name = "PREFIX_P5__TAIL_T3"
    if strict_name not in rows or full_name not in rows:
        raise ValueError("prediction selection lacks required anchors")
    strict_rmse = float(rows[strict_name]["RMSE"])
    improving = [
        row for name, row in rows.items()
        if name != strict_name and float(row["RMSE"]) < strict_rmse
    ]
    if not improving:
        raise ValueError("prediction selection has no improving candidate")
    cheapest = min(
        improving,
        key=lambda row: (
            float(row["normalized_added_bit_cost"]),
            float(row["RMSE"]), str(row["config"])))
    ordered = [strict_name, str(cheapest["config"])]
    ordered.extend(str(row["config"]) for row in normalized_pareto_rows)
    ordered.append(full_name)
    output = []
    for name in ordered:
        if name not in rows:
            raise ValueError("prediction candidate is absent: %s" % name)
        if name not in output:
            output.append(name)
    return tuple(output)
