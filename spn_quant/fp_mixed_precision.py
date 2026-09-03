"""Strict FP4/FP8 format assignments for contract-owned model operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Tuple

from spn_quant.fp_formats import FORMAT_SPECS
from spn_quant.model_contracts import QuantizationModelContract


Owner = Tuple[str, str]
FORMAT_NAMES = tuple(sorted(FORMAT_SPECS))
FORMAT_FP4 = "fp4_e2m1"
FORMAT_FP6 = "fp6_e3m2"
FORMAT_FP8 = "fp8_e4m3fn"
GROUP_NAMES = ("encoder", "decoder", "fusion", "attention", "concat")
FORMAT_BY_BITS = {
    4: FORMAT_FP4,
    6: FORMAT_FP6,
    8: FORMAT_FP8,
}


@dataclass(frozen=True)
class FPFormatAssignment:
    weight_formats: Mapping[str, str]
    activation_formats: Mapping[Owner, str]


def classify_weight_module(module_name: str) -> str:
    """Classify one contract weight module into one execution semantic group."""
    name = str(module_name)
    if not name:
        raise ValueError("weight module name must not be empty")
    if "concat" in name:
        return "concat"
    if ".former." in ".%s." % name or ".attn." in ".%s." % name:
        return "attention"
    if any(token in name for token in
           ("fusion", "skip", "merge", "up_proj", "gd_dec", "id_dec",
            "dep_dec")):
        return "fusion"
    if re.search(r"(^|\.)dec[0-9]|(^|\.)decoder", name):
        return "decoder"
    if re.search(r"(^|\.)conv[0-9]|(^|\.)layer[0-9]|(^|\.)base\.", name):
        return "encoder"
    raise ValueError("weight module has no semantic group: %s" % name)


def classify_activation_owner(owner: Owner) -> str:
    """Classify one contract activation owner without a default group."""
    site, role = owner
    del role
    parts = str(site).split("::")
    if len(parts) != 3:
        raise ValueError("activation owner site is malformed: %s" % (owner,))
    if parts[0] == "attention":
        return "attention"
    if parts[0] == "concat":
        return "concat"
    if parts[0] != "activation":
        raise ValueError("activation owner family is unsupported: %s" % (owner,))
    return classify_weight_module(parts[1])


def _score_gain(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("format promotion sensitivity must be finite")
    return value


def _allocate_grouped_formats(units_by_group, costs, scores,
                              maximum_average_bits, minimum_fp8_fractions,
                              global_average_bits):
    groups = tuple(units_by_group)
    if set(groups) != set(GROUP_NAMES):
        raise KeyError("semantic group allocation coverage differs")
    if set(minimum_fp8_fractions) != set(GROUP_NAMES) or \
            set(maximum_average_bits) != set(GROUP_NAMES):
        raise KeyError("semantic group budget fields differ")
    names = tuple(sorted(
        (name for group in groups for name in units_by_group[group]), key=str))
    if not names:
        return {}
    if set(costs) != set(names) or set(scores) != set(names):
        raise ValueError("grouped format cost and sensitivity coverage differs")
    if any(int(costs[name]) <= 0 for name in names):
        raise ValueError("grouped format costs must be positive")
    global_limit = float(global_average_bits)
    if not math.isfinite(global_limit) or not 4.0 <= global_limit <= 8.0:
        raise ValueError("global format average bits must be in [4, 8]")
    group_of = dict(
        (name, group) for group in groups for name in units_by_group[group])
    group_limits = {}
    group_floors = {}
    for group in groups:
        limit = float(maximum_average_bits[group])
        floor = float(minimum_fp8_fractions[group])
        if not math.isfinite(limit) or not 4.0 <= limit <= 8.0:
            raise ValueError("semantic group average bits must be in [4, 8]")
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError("semantic group FP8 floor must be in [0, 1]")
        if 4.0 + 4.0 * floor > limit:
            raise ValueError("semantic group FP8 floor exceeds its budget: %s" %
                             group)
        group_limits[group] = limit
        group_floors[group] = floor
    group_costs = dict(
        (group, float(sum(int(costs[name]) for name in units_by_group[group])))
        for group in groups)
    assignment = dict((name, FORMAT_FP4) for name in names)

    def average(selected, selected_names):
        denominator = float(sum(int(costs[name]) for name in selected_names))
        return sum((8.0 if selected[name] == FORMAT_FP8 else 4.0) *
                   float(costs[name]) for name in selected_names) / denominator

    ranked_by_group = dict(
        (group, tuple(sorted(
            units_by_group[group],
            key=lambda name: (-_score_gain(scores[name]) /
                              float(costs[name]), str(name)))))
        for group in groups)
    for group in groups:
        if not units_by_group[group]:
            continue
        target_cost = group_costs[group] * group_floors[group]
        covered_cost = 0.0
        for name in ranked_by_group[group]:
            if covered_cost >= target_cost:
                break
            assignment[name] = FORMAT_FP8
            covered_cost += float(costs[name])
        group_average = average(assignment, units_by_group[group])
        if group_average > group_limits[group]:
            raise ValueError("semantic group FP8 floor is infeasible: %s" % group)
    if average(assignment, names) > global_limit:
        raise ValueError("global format budget is infeasible with FP8 floors")

    ranked = sorted(
        (name for name in names if assignment[name] == FORMAT_FP4),
        key=lambda name: (-_score_gain(scores[name]) /
                          float(costs[name]), str(name)))
    for name in ranked:
        if _score_gain(scores[name]) <= 0.0:
            continue
        group = group_of[name]
        assignment[name] = FORMAT_FP8
        if average(assignment, units_by_group[group]) > group_limits[group] or \
                average(assignment, names) > global_limit:
            assignment[name] = FORMAT_FP4
    return assignment


def _group_audit(formats, costs):
    names = tuple(sorted(formats, key=str))
    total = float(sum(int(costs[name]) for name in names))
    format_costs = dict(
        (format_name, sum(int(costs[name]) for name in names
                          if formats[name] == format_name))
        for format_name in FORMAT_BY_BITS.values())
    format_fractions = dict(
        (format_name, float(cost) / total)
        for format_name, cost in format_costs.items() if cost > 0)
    format_unit_counts = dict(
        (format_name, sum(1 for name in names
                          if formats[name] == format_name))
        for format_name in FORMAT_BY_BITS.values()
        if any(formats[name] == format_name for name in names))
    average = sum(
        float(FORMAT_SPECS[format_name].bits) * fraction
        for format_name, fraction in format_fractions.items())
    result = {
        "average_bits": average,
        "format_fractions": format_fractions,
        "format_unit_counts": format_unit_counts,
    }
    for format_name in FORMAT_BY_BITS.values():
        result["%s_fraction" % format_name[:3]] = \
            format_fractions.get(format_name, 0.0)
        result["%s_unit_count" % format_name[:3]] = \
            format_unit_counts.get(format_name, 0)
        result["%s_units" % format_name[:3]] = tuple(
            name for name in names if formats[name] == format_name)
    return result


def build_grouped_format_assignment(
        contract: QuantizationModelContract,
        weight_scores: Mapping[str, float],
        activation_scores: Mapping[Owner, float],
        weight_costs: Mapping[str, int],
        activation_costs: Mapping[Owner, int],
        group_budgets: Mapping[str, Mapping[str, float]],
        global_budgets: Mapping[str, float]):
    """Allocate FP4/FP8 independently for every semantic group and tensor side."""
    modules = tuple(contract.weight_modules)
    owners = _owners(contract)
    if set(weight_scores) != set(modules) or set(weight_costs) != set(modules):
        raise ValueError("grouped weight coverage differs from contract")
    if set(activation_scores) != set(owners) or \
            set(activation_costs) != set(owners):
        raise ValueError("grouped activation coverage differs from contract")
    if set(group_budgets) != set(GROUP_NAMES):
        raise KeyError("semantic group budget coverage differs")
    if set(global_budgets) != {"weight_average_bits", "activation_average_bits"}:
        raise KeyError("global format budget fields differ")
    for group in GROUP_NAMES:
        if set(group_budgets[group]) != {
                "weight_average_bits", "activation_average_bits",
                "activation_minimum_fp8_fraction"}:
            raise KeyError("semantic group budget fields differ: %s" % group)
    weight_groups = dict((group, tuple(
        name for name in modules if classify_weight_module(name) == group))
        for group in GROUP_NAMES)
    activation_groups = dict((group, tuple(
        owner for owner in owners
        if classify_activation_owner(owner) == group)) for group in GROUP_NAMES)
    weight_formats = _allocate_grouped_formats(
        weight_groups,
        weight_costs,
        weight_scores,
        dict((group, group_budgets[group]["weight_average_bits"])
             for group in GROUP_NAMES),
        dict((group, 0.0) for group in GROUP_NAMES),
        global_budgets["weight_average_bits"])
    activation_formats = _allocate_grouped_formats(
        activation_groups,
        activation_costs,
        activation_scores,
        dict((group, group_budgets[group]["activation_average_bits"])
             for group in GROUP_NAMES),
        dict((group, group_budgets[group]["activation_minimum_fp8_fraction"])
             for group in GROUP_NAMES),
        global_budgets["activation_average_bits"])
    audit = {}
    for group in GROUP_NAMES:
        budget = group_budgets[group]
        budget["weight_average_bits"]
        budget["activation_average_bits"]
        group_weights = weight_groups[group]
        group_activations = activation_groups[group]
        if not group_weights and not group_activations:
            audit[group] = {
                "present": False,
                "weight_average_bits": None,
                "activation_average_bits": None,
                "weight": {},
                "activation": {},
            }
            continue
        selected_weights = dict((name, weight_formats[name])
                                for name in group_weights)
        selected_activations = dict((owner, activation_formats[owner])
                                    for owner in group_activations)
        weight_audit = _group_audit(
            selected_weights,
            dict((name, weight_costs[name]) for name in group_weights)) \
            if group_weights else {}
        activation_audit = _group_audit(
            selected_activations,
            dict((owner, activation_costs[owner])
                 for owner in group_activations)) if group_activations else {}
        audit[group] = {
            "present": True,
            "weight_average_bits": weight_audit["average_bits"]
            if group_weights else None,
            "activation_average_bits": activation_audit["average_bits"]
            if group_activations else None,
            "weight": weight_audit,
            "activation": activation_audit,
            "activation_minimum_fp8_fraction":
            budget["activation_minimum_fp8_fraction"],
            "activation_floor_satisfied":
            not group_activations or
            activation_audit["fp8_fraction"] >=
            float(budget["activation_minimum_fp8_fraction"]),
        }
    if set(weight_formats) != set(modules) or \
            set(activation_formats) != set(owners):
        raise RuntimeError("grouped format assignment is incomplete")
    return FPFormatAssignment(weight_formats, activation_formats), audit


def _allocate_three_level_formats(units_by_group, costs, scores,
                                  maximum_average_bits,
                                  minimum_fp8_fractions,
                                  global_average_bits):
    groups = tuple(units_by_group)
    if set(groups) != set(GROUP_NAMES):
        raise KeyError("semantic group allocation coverage differs")
    if set(minimum_fp8_fractions) != set(GROUP_NAMES) or \
            set(maximum_average_bits) != set(GROUP_NAMES):
        raise KeyError("semantic group budget fields differ")
    names = tuple(sorted(
        (name for group in groups for name in units_by_group[group]), key=str))
    if not names:
        return {}
    if set(costs) != set(names) or set(scores) != set(names):
        raise ValueError("three-level format cost and sensitivity coverage differs")
    if any(int(costs[name]) <= 0 for name in names):
        raise ValueError("three-level format costs must be positive")
    global_limit = float(global_average_bits)
    if not math.isfinite(global_limit) or not 4.0 <= global_limit <= 8.0:
        raise ValueError("global format average bits must be in [4, 8]")
    group_of = dict(
        (name, group) for group in groups for name in units_by_group[group])
    group_limits = {}
    group_floors = {}
    for group in groups:
        limit = float(maximum_average_bits[group])
        floor = float(minimum_fp8_fractions[group])
        if not math.isfinite(limit) or not 4.0 <= limit <= 8.0:
            raise ValueError("semantic group average bits must be in [4, 8]")
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError("semantic group FP8 floor must be in [0, 1]")
        if 4.0 + 4.0 * floor > limit:
            raise ValueError("semantic group FP8 floor exceeds its budget: %s" %
                             group)
        group_limits[group] = limit
        group_floors[group] = floor

    def validate_scores(name):
        if set(scores[name]) != {4, 6, 8}:
            raise KeyError("three-level format sensitivity must cover 4, 6, 8")
        if any(not math.isfinite(float(scores[name][bits]))
               for bits in (4, 6, 8)):
            raise ValueError("three-level format sensitivity must be finite")

    for name in names:
        validate_scores(name)
    assignment = dict((name, FORMAT_FP4) for name in names)

    def average(selected_names):
        denominator = float(sum(int(costs[name]) for name in selected_names))
        return sum(
            float(FORMAT_SPECS[assignment[name]].bits) * float(costs[name])
            for name in selected_names) / denominator

    def promote_to_fp8(group):
        ranked = sorted(
            units_by_group[group],
            key=lambda name: (
                -(float(scores[name][4]) - float(scores[name][8])) /
                float(costs[name]), str(name)))
        target_cost = sum(int(costs[name]) for name in units_by_group[group]) * \
            group_floors[group]
        covered_cost = 0.0
        for name in ranked:
            if covered_cost >= target_cost:
                break
            assignment[name] = FORMAT_FP8
            covered_cost += float(costs[name])
        if units_by_group[group] and \
                average(units_by_group[group]) > group_limits[group]:
            raise ValueError("semantic group FP8 floor is infeasible: %s" %
                             group)

    for group in groups:
        if units_by_group[group]:
            promote_to_fp8(group)
    if average(names) > global_limit:
        raise ValueError("global format budget is infeasible with FP8 floors")

    while True:
        options = []
        for name in names:
            current_bits = FORMAT_SPECS[assignment[name]].bits
            if current_bits == 8:
                continue
            next_bits = 6 if current_bits == 4 else 8
            gain = float(scores[name][current_bits]) - \
                float(scores[name][next_bits])
            if gain <= 0.0:
                continue
            candidate = dict(assignment)
            candidate[name] = FORMAT_BY_BITS[next_bits]
            previous = assignment
            assignment = candidate
            group = group_of[name]
            valid = average(units_by_group[group]) <= group_limits[group] and \
                average(names) <= global_limit
            assignment = previous
            if valid:
                options.append((-gain / float(costs[name]), str(name), name,
                                next_bits))
        if not options:
            break
        _, _, name, next_bits = min(options)
        assignment[name] = FORMAT_BY_BITS[next_bits]
    return assignment


def build_grouped_three_level_format_assignment(
        contract: QuantizationModelContract,
        weight_scores: Mapping[str, Mapping[int, float]],
        activation_scores: Mapping[Owner, Mapping[int, float]],
        weight_costs: Mapping[str, int],
        activation_costs: Mapping[Owner, int],
        group_budgets: Mapping[str, Mapping[str, float]],
        global_budgets: Mapping[str, float]):
    """Allocate FP4/FP6/FP8 with staged task-sensitive promotions."""
    modules = tuple(contract.weight_modules)
    owners = _owners(contract)
    if set(weight_scores) != set(modules) or set(weight_costs) != set(modules):
        raise ValueError("three-level weight coverage differs from contract")
    if set(activation_scores) != set(owners) or \
            set(activation_costs) != set(owners):
        raise ValueError("three-level activation coverage differs from contract")
    if set(group_budgets) != set(GROUP_NAMES):
        raise KeyError("semantic group budget coverage differs")
    if set(global_budgets) != {"weight_average_bits", "activation_average_bits"}:
        raise KeyError("global format budget fields differ")
    for group in GROUP_NAMES:
        if set(group_budgets[group]) != {
                "weight_average_bits", "activation_average_bits",
                "activation_minimum_fp8_fraction"}:
            raise KeyError("semantic group budget fields differ: %s" % group)

    weight_groups = dict((group, tuple(
        name for name in modules if classify_weight_module(name) == group))
        for group in GROUP_NAMES)
    activation_groups = dict((group, tuple(
        owner for owner in owners
        if classify_activation_owner(owner) == group)) for group in GROUP_NAMES)
    weight_formats = _allocate_three_level_formats(
        weight_groups,
        weight_costs,
        weight_scores,
        dict((group, group_budgets[group]["weight_average_bits"])
             for group in GROUP_NAMES),
        dict((group, 0.0) for group in GROUP_NAMES),
        global_budgets["weight_average_bits"])
    activation_formats = _allocate_three_level_formats(
        activation_groups,
        activation_costs,
        activation_scores,
        dict((group, group_budgets[group]["activation_average_bits"])
             for group in GROUP_NAMES),
        dict((group, group_budgets[group]["activation_minimum_fp8_fraction"])
             for group in GROUP_NAMES),
        global_budgets["activation_average_bits"])

    audit = {}
    for group in GROUP_NAMES:
        budget = group_budgets[group]
        group_weights = weight_groups[group]
        group_activations = activation_groups[group]
        if not group_weights and not group_activations:
            audit[group] = {
                "present": False,
                "weight_average_bits": None,
                "activation_average_bits": None,
                "weight": {},
                "activation": {},
            }
            continue
        selected_weights = dict((name, weight_formats[name])
                                for name in group_weights)
        selected_activations = dict((owner, activation_formats[owner])
                                    for owner in group_activations)
        weight_audit = _group_audit(
            selected_weights,
            dict((name, weight_costs[name]) for name in group_weights)) \
            if group_weights else {}
        activation_audit = _group_audit(
            selected_activations,
            dict((owner, activation_costs[owner])
                 for owner in group_activations)) if group_activations else {}
        audit[group] = {
            "present": True,
            "weight_average_bits": weight_audit["average_bits"]
            if group_weights else None,
            "activation_average_bits": activation_audit["average_bits"]
            if group_activations else None,
            "weight": weight_audit,
            "activation": activation_audit,
            "activation_minimum_fp8_fraction":
            budget["activation_minimum_fp8_fraction"],
            "activation_floor_satisfied":
            not group_activations or
            activation_audit["fp8_fraction"] >=
            float(budget["activation_minimum_fp8_fraction"]),
        }
    if set(weight_formats) != set(modules) or \
            set(activation_formats) != set(owners):
        raise RuntimeError("three-level format assignment is incomplete")
    return FPFormatAssignment(weight_formats, activation_formats), audit


def _owners(contract: QuantizationModelContract):
    return tuple((owner, role) for block in contract.blocks
                 for owner, role in block.activation_owners)


def _validate_formats(formats, allowed):
    if set(formats) - set(allowed):
        raise KeyError("unsupported floating-point format assignment")
    if any(value not in FORMAT_NAMES for value in formats.values()):
        raise ValueError("floating-point format assignment is unsupported")


def build_format_assignment(contract: QuantizationModelContract,
                            weight_format: str, activation_format: str,
                            sensitivity: Mapping[Owner, float],
                            promotion_fraction: float,
                            activation_costs=None) -> FPFormatAssignment:
    if weight_format not in FORMAT_NAMES or activation_format not in FORMAT_NAMES:
        raise KeyError("unsupported floating-point format")
    fraction = float(promotion_fraction)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("activation promotion fraction must be in [0, 1]")
    weights = dict((name, weight_format) for name in contract.weight_modules)
    owners = _owners(contract)
    if activation_format == "fp8_e4m3fn":
        activations = dict((owner, activation_format) for owner in owners)
    elif fraction == 0.0:
        activations = dict((owner, activation_format) for owner in owners)
    else:
        if set(sensitivity) != set(owners):
            raise ValueError("mixed activation sensitivity coverage differs")
        if activation_costs is None or set(activation_costs) != set(owners):
            raise ValueError("mixed activation costs coverage differs")
        if any(float(value) <= 0.0 for value in activation_costs.values()):
            raise ValueError("mixed activation costs must be positive")
        total = sum(float(activation_costs[owner]) for owner in owners)
        target = total * fraction
        ranked = sorted(owners, key=lambda owner: (-float(sensitivity[owner]), owner))
        promoted = []
        covered = 0.0
        for owner in ranked:
            if covered >= target:
                break
            promoted.append(owner)
            covered += float(activation_costs[owner])
        promoted = set(promoted)
        activations = dict(
            (owner, "fp8_e4m3fn" if owner in promoted else activation_format)
            for owner in owners)
    _validate_formats(weights, contract.weight_modules)
    _validate_formats(activations, owners)
    return FPFormatAssignment(weights, activations)


def weighted_format_fractions(formats: Mapping, costs: Mapping):
    if set(formats) != set(costs):
        raise ValueError("format fraction coverage differs from costs")
    if any(float(value) <= 0.0 for value in costs.values()):
        raise ValueError("format fraction costs must be positive")
    total = sum(float(value) for value in costs.values())
    weighted = {}
    for key, format_name in formats.items():
        if format_name not in weighted:
            weighted[format_name] = 0.0
        weighted[format_name] += float(costs[key])
    return dict((name, value / total) for name, value in sorted(weighted.items()))
