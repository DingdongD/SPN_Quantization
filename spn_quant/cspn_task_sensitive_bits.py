"""CSPN compatibility surface for generic mixed-precision allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from spn_quant import cspn_encoder_prefix as encoder
from spn_quant import cspn_sensitivity as decoder
from spn_quant.mixed_precision import (
    ActivationBudgetAudit,
    AllocationCandidate,
    BIT_OPTIONS,
    BitAssignment,
    BudgetAudit,
    CostBasis,
    MIXED_ACTIVATION_BITS,
    Owner,
    SearchState,
    SensitivityEntry,
    assignment_key,
    audit_activation_budget,
    audit_budget,
    build_budget_preserving_neighbors,
    build_cheapest_block_demotions,
    build_promoted_activation_candidates,
    build_refinement_candidates,
    build_sensitivity_table,
    build_single_block_probes,
    promoted_assignment,
    prune_dominated_states,
    rank_refinement_blocks,
    search_block_assignments,
    search_state_key,
    select_local_improvement,
    select_measured_activation_candidate,
    uniform_assignment,
)


BLOCK_ORDER = (
    "stem",
    "encoder_layer1",
    "encoder_layer2",
    "encoder_layer3",
    "encoder_layer4",
    "decoder_layer1",
    "decoder_layer2",
    "decoder_layer3",
    "decoder_layer4",
    "initial_depth",
)
P3_T3_PROTECTED_BLOCKS = (
    "stem",
    "encoder_layer1",
    "encoder_layer2",
    "decoder_layer4",
    "initial_depth",
)

WEIGHT_MODULES_BY_BLOCK = {
    "stem": encoder.WEIGHT_MODULES_BY_UNIT["stem"],
    "encoder_layer1": encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer1"],
    "encoder_layer2": encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer2"],
    "encoder_layer3": encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer3"],
    "encoder_layer4": encoder.WEIGHT_MODULES_BY_UNIT["encoder_layer4"],
    "decoder_layer1": (
        ("conv2",) + decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer1"]),
    "decoder_layer2": decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer2"],
    "decoder_layer3": decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer3"],
    "decoder_layer4": decoder.WEIGHT_MODULES_BY_BLOCK["decoder_layer4"],
    "initial_depth": decoder.WEIGHT_MODULES_BY_BLOCK["initial_depth"],
}

ACTIVATION_OWNERS_BY_BLOCK = {
    "stem": encoder.ACTIVATION_OWNERS_BY_UNIT["stem"],
    "encoder_layer1": encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer1"],
    "encoder_layer2": encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer2"],
    "encoder_layer3": encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer3"],
    "encoder_layer4": encoder.ACTIVATION_OWNERS_BY_UNIT["encoder_layer4"],
    "decoder_layer1": decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer1"],
    "decoder_layer2": decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer2"],
    "decoder_layer3": decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer3"],
    "decoder_layer4": decoder.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer4"],
    "initial_depth": decoder.ACTIVATION_OWNERS_BY_BLOCK["initial_depth"],
}


@dataclass(frozen=True)
class AllocationRegistry:
    """Historical two-field CSPN allocation registry."""

    weights_by_block: Mapping[str, Tuple[str, ...]]
    activations_by_block: Mapping[str, Tuple[Owner, ...]]

    @property
    def blocks(self):
        return BLOCK_ORDER

    @property
    def model_name(self):
        return ""


def _require_unique(values, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicates" % name)


def _ordered_unique_by_block(mapping, prefer_last: bool = False):
    output = {}
    if prefer_last:
        owner_block = {}
        for block in BLOCK_ORDER:
            for value in mapping[block]:
                owner_block[value] = block
        for block in BLOCK_ORDER:
            output[block] = tuple(
                value for value in mapping[block]
                if owner_block[value] == block)
        return output
    owned = set()
    for block in BLOCK_ORDER:
        values = []
        for value in mapping[block]:
            if value not in owned:
                values.append(value)
                owned.add(value)
        output[block] = tuple(values)
    return output


def _ordered_union(mapping):
    return tuple(value for block in BLOCK_ORDER for value in mapping[block])


def build_registry(
        executed_weight_modules: Sequence[str],
        activation_owners: Sequence[Owner]) -> AllocationRegistry:
    """Build the historical CSPN registry with its exact ownership rules."""
    modules = tuple(str(module) for module in executed_weight_modules)
    owners = tuple(
        (str(owner[0]), str(owner[1])) for owner in activation_owners)
    _require_unique(modules, "weight registry")
    _require_unique(owners, "activation registry")

    weights_by_block = _ordered_unique_by_block(WEIGHT_MODULES_BY_BLOCK)
    activations_by_block = _ordered_unique_by_block(
        ACTIVATION_OWNERS_BY_BLOCK, prefer_last=True)
    expected_modules = _ordered_union(weights_by_block)
    expected_owners = _ordered_union(activations_by_block)
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
    return AllocationRegistry(
        weights_by_block=weights_by_block,
        activations_by_block=activations_by_block,
    )


def p3_t3_assignment(registry: AllocationRegistry) -> BitAssignment:
    return promoted_assignment(registry, P3_T3_PROTECTED_BLOCKS)


def build_p3_t3_activation_candidates(
        registry: AllocationRegistry,
        basis: CostBasis,
        maximum_bits: float):
    return build_promoted_activation_candidates(
        registry, basis, P3_T3_PROTECTED_BLOCKS, maximum_bits)
