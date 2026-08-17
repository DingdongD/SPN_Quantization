#!/usr/bin/env python3
"""Search task-sensitive mixed-bit assignments for official CSPN."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_activation_resolution as base
from spn_quant import cspn_task_sensitive_bits as allocation


STEM_WEIGHT_MODULE = "conv1_1"
STEM_INPUT_OWNER = "conv1_1", "input"
SEARCH_STAGES = (
    "single_block",
    "joint",
    "local",
    "refinement",
    "final",
)
DEPTH_METRIC_FIELDS = (
    "RMSE",
    "MAE",
    "ABS_REL",
    "IRMSE",
    "flat_RMSE",
    "boundary_RMSE",
)


@dataclass(frozen=True)
class RuntimeCandidate:
    name: str
    stage: str
    assignment: allocation.BitAssignment


@dataclass(frozen=True)
class CandidateStatus:
    valid: bool
    reasons: Tuple[str, ...]
    budget_excess: bool


def _ordered_union(sequences):
    output = []
    for sequence in sequences:
        for value in sequence:
            if value not in output:
                output.append(value)
    return tuple(output)


def expected_registry() -> allocation.AllocationRegistry:
    modules = _ordered_union(
        allocation.WEIGHT_MODULES_BY_BLOCK[block]
        for block in allocation.BLOCK_ORDER)
    owners = _ordered_union(
        allocation.ACTIVATION_OWNERS_BY_BLOCK[block]
        for block in allocation.BLOCK_ORDER)
    return allocation.build_registry(modules, owners)


def validate_assignment_contract(
        assignment: allocation.BitAssignment) -> None:
    registry = expected_registry()
    expected_modules = set(_ordered_union(registry.weights_by_block.values()))
    expected_owners = set(_ordered_union(
        registry.activations_by_block.values()))
    actual_modules = set(module for module, bits in assignment.weight_bits)
    actual_owners = set(owner for owner, bits in assignment.activation_bits)
    if actual_modules != expected_modules:
        raise ValueError(
            "weight assignment coverage mismatch: missing=%s extra=%s" % (
                sorted(expected_modules - actual_modules),
                sorted(actual_modules - expected_modules)))
    if actual_owners != expected_owners:
        raise ValueError(
            "activation assignment coverage mismatch: missing=%s extra=%s" % (
                sorted(expected_owners - actual_owners, key=str),
                sorted(actual_owners - expected_owners, key=str)))


def runtime_configuration(candidate: RuntimeCandidate):
    if candidate.stage not in SEARCH_STAGES:
        raise ValueError("unknown mixed-bit search stage: %s" % candidate.stage)
    validate_assignment_contract(candidate.assignment)
    generic_weights = tuple(
        (module, bits)
        for module, bits in candidate.assignment.weight_bits
        if module != STEM_WEIGHT_MODULE)
    generic_activations = tuple(
        (owner, bits)
        for owner, bits in candidate.assignment.activation_bits
        if owner != STEM_INPUT_OWNER)
    return base._configuration(
        candidate.name,
        base.ORDINARY_GROUPS,
        base.ORDINARY_GROUPS,
        base.PROPAGATION_A8_Q13,
        granularity="hybrid_group",
        group_size=8,
        weight_bit_overrides=generic_weights,
        activation_bit_overrides=generic_activations,
    )


def validate_configured_precision(
        candidate: RuntimeCandidate,
        weight_bits: Mapping[str, int],
        specs,
        rotation_specs,
        stem_contract: Mapping[str, object]) -> None:
    validate_assignment_contract(candidate.assignment)
    expected_weights = dict(candidate.assignment.weight_bits)
    expected_activations = dict(candidate.assignment.activation_bits)
    expected_stem_weight = expected_weights.pop(STEM_WEIGHT_MODULE)
    expected_stem_activation = expected_activations.pop(STEM_INPUT_OWNER)

    actual_weights = dict(
        (str(module), int(weight_bits[module])) for module in weight_bits)
    if actual_weights != expected_weights:
        raise RuntimeError("configured weight bits differ from assignment")

    actual_activations = {}
    for key in specs:
        owner = base.activation_owner(key)
        if owner in actual_activations:
            raise RuntimeError("configured activation owners contain duplicates")
        actual_activations[owner] = int(specs[key].bits)
    for owner in rotation_specs:
        normalized = tuple(owner)
        if normalized in actual_activations:
            raise RuntimeError("configured activation owners contain duplicates")
        actual_activations[normalized] = int(rotation_specs[owner].bits)
    if actual_activations != expected_activations:
        raise RuntimeError("configured activation bits differ from assignment")

    expected_config = "STEM_W%dA%d" % (
        expected_stem_weight, expected_stem_activation)
    if str(stem_contract["config"]) != expected_config:
        raise RuntimeError("configured stem precision differs from assignment")
    if int(stem_contract["weight_bits"]) != expected_stem_weight or \
            int(stem_contract["activation_bits"]) != \
            expected_stem_activation:
        raise RuntimeError("configured stem precision differs from assignment")


def configure_runtime_context(
        candidate: RuntimeCandidate,
        instrumentor,
        rotation,
        propagation,
        stem):
    config = runtime_configuration(candidate)
    specs, rotation_specs, active_merge = base._configure_quantized(
        config, instrumentor, rotation, propagation, {})
    if active_merge is not None:
        raise RuntimeError("mixed-bit CSPN search forbids merge adapters")
    stem_bits = dict(candidate.assignment.weight_bits)[STEM_WEIGHT_MODULE]
    stem_activation_bits = dict(
        candidate.assignment.activation_bits)[STEM_INPUT_OWNER]
    stem.configure_integer(stem_bits, stem_activation_bits)
    validate_configured_precision(
        candidate,
        instrumentor.weight_bits_by_module(),
        specs,
        rotation_specs,
        stem.contract(),
    )
    return config, specs, rotation_specs


def build_cost_basis(
        registry: allocation.AllocationRegistry,
        weight_rows: Sequence[Mapping[str, object]],
        activation_rows: Sequence[Mapping[str, object]],
        ) -> allocation.CostBasis:
    expected_modules = set(_ordered_union(registry.weights_by_block.values()))
    expected_owners = set(_ordered_union(
        registry.activations_by_block.values()))
    modules = tuple(str(row["module"]) for row in weight_rows)
    owners = tuple(
        (str(row["module"]), str(row["kind"]))
        for row in activation_rows)
    if len(modules) != len(set(modules)):
        raise ValueError("weight cost rows contain duplicates")
    if len(owners) != len(set(owners)):
        raise ValueError("activation cost rows contain duplicates")
    if set(modules) != expected_modules:
        raise ValueError("weight cost coverage mismatch")
    if set(owners) != expected_owners:
        raise ValueError("activation cost coverage mismatch")
    return allocation.CostBasis(
        weight_macs=tuple(
            (str(row["module"]), int(row["macs"]))
            for row in weight_rows),
        activation_elements=tuple(
            ((str(row["module"]), str(row["kind"])),
             int(row["elements"]))
            for row in activation_rows),
    )


def candidate_status(
        metrics: Mapping[str, object],
        budget,
        stage: str) -> CandidateStatus:
    if stage not in SEARCH_STAGES:
        raise ValueError("unknown mixed-bit search stage: %s" % stage)
    reasons = []
    depth_metrics = tuple(float(metrics[field]) for field in DEPTH_METRIC_FIELDS)
    if not all(math.isfinite(value) for value in depth_metrics):
        reasons.append("nonfinite metric")
    if float(metrics["nonfinite_ratio"]) != 0.0:
        reasons.append("nonfinite prediction")
    if float(metrics["nonpositive_ratio"]) != 0.0:
        reasons.append("nonpositive depth")
    if float(metrics["coefficient_sum_max_error"]) != 0.0:
        reasons.append("coefficient sum")
    if float(metrics["contraction_violation_ratio"]) != 0.0:
        reasons.append("contraction")
    if float(metrics["anchor_max_error"]) != 0.0:
        reasons.append("anchor")
    budget_excess = not bool(budget.feasible)
    if stage != "single_block" and budget_excess:
        reasons.append("precision budget")
    return CandidateStatus(
        valid=not reasons,
        reasons=tuple(reasons),
        budget_excess=budget_excess,
    )
