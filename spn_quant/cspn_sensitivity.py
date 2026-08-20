"""Candidate definitions for CSPN decoder precision sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple


Owner = Tuple[str, str]

BLOCK_ORDER = (
    "decoder_layer1",
    "decoder_layer2",
    "decoder_layer3",
    "decoder_layer4",
    "initial_depth",
)

WEIGHT_MODULES_BY_BLOCK = {
    "decoder_layer1": (
        "gud_up_proj_layer1.conv1",
        "gud_up_proj_layer1.conv2",
        "gud_up_proj_layer1.sc_conv1",
    ),
    "decoder_layer2": (
        "gud_up_proj_layer2.conv1",
        "gud_up_proj_layer2.conv1_1",
        "gud_up_proj_layer2.conv2",
        "gud_up_proj_layer2.sc_conv1",
    ),
    "decoder_layer3": (
        "gud_up_proj_layer3.conv1",
        "gud_up_proj_layer3.conv1_1",
        "gud_up_proj_layer3.conv2",
        "gud_up_proj_layer3.sc_conv1",
    ),
    "decoder_layer4": (
        "gud_up_proj_layer4.conv1",
        "gud_up_proj_layer4.conv1_1",
        "gud_up_proj_layer4.conv2",
        "gud_up_proj_layer4.sc_conv1",
    ),
    "initial_depth": ("gud_up_proj_layer5.conv1",),
}

ACTIVATION_OWNERS_BY_BLOCK = {
    "decoder_layer1": (
        ("conv2", "input"),
        ("boundary_controller.decoder_entry", "boundary"),
        ("gud_up_proj_layer1.conv2", "input"),
        ("gud_up_proj_layer1.relu#0", "relu_output"),
        ("gud_up_proj_layer1.relu#1", "relu_output"),
        ("gud_up_proj_layer1.sc_conv1", "output"),
    ),
    "decoder_layer2": (
        ("gud_up_proj_layer2.conv1", "input"),
        ("gud_up_proj_layer2.conv1_1", "input"),
        ("gud_up_proj_layer2.conv2", "input"),
        ("gud_up_proj_layer2.relu#0", "relu_output"),
        ("gud_up_proj_layer2.relu#1", "relu_output"),
        ("gud_up_proj_layer2.relu#2", "relu_output"),
        ("gud_up_proj_layer2.sc_conv1", "input"),
        ("gud_up_proj_layer2.sc_conv1", "output"),
    ),
    "decoder_layer3": (
        ("gud_up_proj_layer3.conv1", "input"),
        ("gud_up_proj_layer3.conv1_1", "input"),
        ("gud_up_proj_layer3.conv2", "input"),
        ("gud_up_proj_layer3.relu#0", "relu_output"),
        ("gud_up_proj_layer3.relu#1", "relu_output"),
        ("gud_up_proj_layer3.relu#2", "relu_output"),
        ("gud_up_proj_layer3.sc_conv1", "input"),
        ("gud_up_proj_layer3.sc_conv1", "output"),
    ),
    "decoder_layer4": (
        ("boundary_controller.layer4_signed_skip", "boundary"),
        ("gud_up_proj_layer4.conv1", "input"),
        ("gud_up_proj_layer4.conv2", "input"),
        ("gud_up_proj_layer4.relu#0", "relu_output"),
        ("gud_up_proj_layer4.relu#1", "relu_output"),
        ("gud_up_proj_layer4.relu#2", "relu_output"),
        ("gud_up_proj_layer4.sc_conv1", "input"),
        ("gud_up_proj_layer4.sc_conv1", "output"),
    ),
    "initial_depth": (("gud_up_proj_layer5.conv1", "input"),),
}

INPUT_DEPENDENCIES = {
    "gud_up_proj_layer1.conv1": (
        ("boundary_controller.decoder_entry", "boundary"),),
    "gud_up_proj_layer1.conv2": (
        ("gud_up_proj_layer1.conv2", "input"),),
    "gud_up_proj_layer1.sc_conv1": (
        ("boundary_controller.decoder_entry", "boundary"),),
    "gud_up_proj_layer2.conv1": (
        ("gud_up_proj_layer2.conv1", "input"),),
    "gud_up_proj_layer2.conv1_1": (
        ("gud_up_proj_layer2.conv1_1", "input"),),
    "gud_up_proj_layer2.conv2": (
        ("gud_up_proj_layer2.conv2", "input"),),
    "gud_up_proj_layer2.sc_conv1": (
        ("gud_up_proj_layer2.sc_conv1", "input"),),
    "gud_up_proj_layer3.conv1": (
        ("gud_up_proj_layer3.conv1", "input"),),
    "gud_up_proj_layer3.conv1_1": (
        ("gud_up_proj_layer3.conv1_1", "input"),),
    "gud_up_proj_layer3.conv2": (
        ("gud_up_proj_layer3.conv2", "input"),),
    "gud_up_proj_layer3.sc_conv1": (
        ("gud_up_proj_layer3.sc_conv1", "input"),),
    "gud_up_proj_layer4.conv1": (
        ("gud_up_proj_layer4.conv1", "input"),),
    "gud_up_proj_layer4.conv1_1": (
        ("gud_up_proj_layer4.relu#0", "relu_output"),
        ("boundary_controller.layer4_signed_skip", "boundary"),
    ),
    "gud_up_proj_layer4.conv2": (
        ("gud_up_proj_layer4.conv2", "input"),),
    "gud_up_proj_layer4.sc_conv1": (
        ("gud_up_proj_layer4.sc_conv1", "input"),),
    "gud_up_proj_layer5.conv1": (
        ("gud_up_proj_layer5.conv1", "input"),),
}


@dataclass(frozen=True)
class SensitivityCandidate:
    name: str
    stage: str
    block: str
    mode: str
    weight_modules: Tuple[str, ...]
    activation_owners: Tuple[Owner, ...]


@dataclass(frozen=True)
class CandidateRegistry:
    weights_by_block: Mapping[str, Tuple[str, ...]]
    activations_by_block: Mapping[str, Tuple[Owner, ...]]
    input_dependencies: Mapping[str, Tuple[Owner, ...]]


def _flatten(mapping, order):
    return tuple(value for name in order for value in mapping[name])


def _validate_unique(values, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicates" % name)


def build_candidate_registry(
        executed_modules: Sequence[str],
        activation_owners: Sequence[Owner]) -> CandidateRegistry:
    modules = tuple(str(name) for name in executed_modules)
    owners = tuple((str(owner[0]), str(owner[1]))
                   for owner in activation_owners)
    expected_modules = _flatten(WEIGHT_MODULES_BY_BLOCK, BLOCK_ORDER)
    expected_owners = _flatten(ACTIVATION_OWNERS_BY_BLOCK, BLOCK_ORDER)
    _validate_unique(modules, "weight registry")
    _validate_unique(owners, "activation registry")
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
    dependency_modules = set(INPUT_DEPENDENCIES)
    if dependency_modules != set(expected_modules):
        raise RuntimeError("Conv input dependency registry is incomplete")
    for name in expected_modules:
        dependencies = INPUT_DEPENDENCIES[name]
        if not dependencies or not set(dependencies) <= set(expected_owners):
            raise RuntimeError("Conv input dependency is invalid: %s" % name)
    return CandidateRegistry(
        weights_by_block=dict(
            (block, tuple(WEIGHT_MODULES_BY_BLOCK[block]))
            for block in BLOCK_ORDER),
        activations_by_block=dict(
            (block, tuple(ACTIVATION_OWNERS_BY_BLOCK[block]))
            for block in BLOCK_ORDER),
        input_dependencies=dict(
            (name, tuple(INPUT_DEPENDENCIES[name]))
            for name in expected_modules),
    )


def build_stage1_candidates(
        registry: CandidateRegistry) -> Tuple[SensitivityCandidate, ...]:
    candidates = [SensitivityCandidate(
        "STRICT_W4A4", "baseline", "all", "W4A4", (), ())]
    for block in BLOCK_ORDER:
        weights = registry.weights_by_block[block]
        activations = registry.activations_by_block[block]
        candidates.extend((
            SensitivityCandidate(
                "BLOCK_%s_W4A8" % block, "block", block, "W4A8",
                (), activations),
            SensitivityCandidate(
                "BLOCK_%s_W8A4" % block, "block", block, "W8A4",
                weights, ()),
            SensitivityCandidate(
                "BLOCK_%s_W8A8" % block, "block", block, "W8A8",
                weights, activations),
        ))
    _validate_unique(tuple(candidate.name for candidate in candidates),
                     "Stage-1 candidate names")
    return tuple(candidates)


def _metric_by_name(rows: Sequence[Mapping[str, object]]):
    metrics = {}
    for row in rows:
        name = str(row["config"])
        value = float(row["RMSE"])
        if not math.isfinite(value):
            raise ValueError("candidate RMSE must be finite")
        if name in metrics:
            raise ValueError("candidate metrics contain duplicates: %s" % name)
        metrics[name] = value
    return metrics


def select_sensitive_blocks(
        rows: Sequence[Mapping[str, object]],
        stage1_candidates: Sequence[SensitivityCandidate]) -> Tuple[str, str]:
    metrics = _metric_by_name(rows)
    expected = {candidate.name for candidate in stage1_candidates}
    if set(metrics) != expected:
        raise ValueError("Stage-1 metric coverage mismatch")
    ranking = []
    for order, block in enumerate(BLOCK_ORDER):
        candidates = [candidate for candidate in stage1_candidates
                      if candidate.block == block]
        best = min(metrics[candidate.name] for candidate in candidates)
        ranking.append((best, order, block))
    ranking.sort()
    return ranking[0][2], ranking[1][2]


def _slug(value: str) -> str:
    return value.replace(".", "_").replace("#", "_")


def _owner_slug(owner: Owner) -> str:
    return "%s_%s" % (_slug(owner[0]), _slug(owner[1]))


def build_stage2_candidates(
        registry: CandidateRegistry,
        blocks: Sequence[str]) -> Tuple[SensitivityCandidate, ...]:
    selected = tuple(str(block) for block in blocks)
    _validate_unique(selected, "selected blocks")
    if not selected or not set(selected) <= set(BLOCK_ORDER):
        raise ValueError("selected blocks are invalid")
    candidates = []
    for block in selected:
        for owner in registry.activations_by_block[block]:
            candidates.append(SensitivityCandidate(
                "SITE_%s_W4A8_%s" % (block, _owner_slug(owner)),
                "site", block, "W4A8", (), (owner,)))
        for module in registry.weights_by_block[block]:
            candidates.extend((
                SensitivityCandidate(
                    "SITE_%s_W8A4_%s" % (block, _slug(module)),
                    "site", block, "W8A4", (module,), ()),
                SensitivityCandidate(
                    "SITE_%s_W8A8_%s" % (block, _slug(module)),
                    "site", block, "W8A8", (module,),
                    registry.input_dependencies[module]),
            ))
    _validate_unique(tuple(candidate.name for candidate in candidates),
                     "Stage-2 candidate names")
    return tuple(candidates)


def build_cumulative_candidates(
        rows: Sequence[Mapping[str, object]],
        site_candidates: Sequence[SensitivityCandidate],
        strict_rmse: float) -> Tuple[SensitivityCandidate, ...]:
    strict = float(strict_rmse)
    if not math.isfinite(strict):
        raise ValueError("strict RMSE must be finite")
    metrics = _metric_by_name(rows)
    by_name = dict((candidate.name, candidate) for candidate in site_candidates)
    if set(metrics) != set(by_name):
        raise ValueError("Stage-2 metric coverage mismatch")
    ranked = sorted(
        (candidate for candidate in site_candidates
         if metrics[candidate.name] < strict),
        key=lambda candidate: (
            metrics[candidate.name] - strict,
            float(next(row["normalized_added_bit_cost"] for row in rows
                       if str(row["config"]) == candidate.name)),
            candidate.name))
    weights = set()
    owners = set()
    output = []
    for candidate in ranked:
        previous = (len(weights), len(owners))
        weights.update(candidate.weight_modules)
        owners.update(candidate.activation_owners)
        if previous == (len(weights), len(owners)):
            continue
        output.append(SensitivityCandidate(
            "CUMULATIVE_%02d" % (len(output) + 1),
            "cumulative", "multiple", "MIXED",
            tuple(sorted(weights)), tuple(sorted(owners))))
    return tuple(output)


def precision_cost(
        weight_rows: Sequence[Mapping[str, object]],
        activation_rows: Sequence[Mapping[str, object]],
        candidate: SensitivityCandidate) -> Dict[str, float]:
    weights = {}
    total_macs = 0
    total_weight_elements = 0
    for row in weight_rows:
        module = str(row["module"])
        if module in weights:
            raise ValueError("weight cost rows contain duplicates")
        values = int(row["weight_elements"]), int(row["macs"])
        if min(values) <= 0:
            raise ValueError("weight cost values must be positive")
        weights[module] = values
        total_weight_elements += values[0]
        total_macs += values[1]
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
    promoted_macs = sum(weights[name][1] for name in candidate.weight_modules)
    promoted_activation_elements = sum(
        activations[owner] for owner in candidate.activation_owners)
    total_elements = total_weight_elements + total_activation_elements
    promoted_elements = promoted_weight_elements + promoted_activation_elements
    return {
        "normalized_added_bit_cost": promoted_elements / float(total_elements),
        "w8_weight_mac_fraction": promoted_macs / float(total_macs),
        "w8_weight_element_fraction": promoted_weight_elements /
        float(total_weight_elements),
        "a8_activation_element_fraction": promoted_activation_elements /
        float(total_activation_elements),
    }


def pareto_rows(rows: Sequence[Mapping[str, object]]):
    values = []
    names = set()
    for source in rows:
        row = dict(source)
        name = str(row["config"])
        cost = float(row["normalized_added_bit_cost"])
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
            other_cost <= cost and other_rmse <= rmse
            and (other_cost < cost or other_rmse < rmse)
            for other_cost, other_rmse, _ in values)
        if not dominated:
            output.append(row)
    return sorted(
        output,
        key=lambda row: (
            float(row["normalized_added_bit_cost"]),
            float(row["RMSE"]), str(row["config"])))
