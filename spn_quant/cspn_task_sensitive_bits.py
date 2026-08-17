"""CSPN task-sensitive mixed-bit allocation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from spn_quant import cspn_encoder_prefix as encoder
from spn_quant import cspn_sensitivity as decoder


Owner = Tuple[str, str]

BIT_OPTIONS = (2, 4, 6, 8)
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
    weights_by_block: Mapping[str, Tuple[str, ...]]
    activations_by_block: Mapping[str, Tuple[Owner, ...]]


@dataclass(frozen=True)
class BitAssignment:
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[Owner, int], ...]

    def __post_init__(self) -> None:
        weights = tuple(sorted(
            (str(module), int(bits)) for module, bits in self.weight_bits))
        activations = tuple(sorted(
            ((str(owner[0]), str(owner[1])), int(bits))
            for owner, bits in self.activation_bits))
        _require_unique(
            tuple(module for module, bits in weights),
            "weight bit assignment")
        _require_unique(
            tuple(owner for owner, bits in activations),
            "activation bit assignment")
        _validate_bits(tuple(bits for module, bits in weights))
        _validate_bits(tuple(bits for owner, bits in activations))
        object.__setattr__(self, "weight_bits", weights)
        object.__setattr__(self, "activation_bits", activations)


@dataclass(frozen=True)
class AllocationCandidate:
    name: str
    stage: str
    block: str
    assignment: BitAssignment
    weight_bits: int
    activation_bits: int


@dataclass(frozen=True)
class CostBasis:
    weight_macs: Tuple[Tuple[str, int], ...]
    activation_elements: Tuple[Tuple[Owner, int], ...]

    def __post_init__(self) -> None:
        weights = tuple(sorted(
            (str(module), int(macs)) for module, macs in self.weight_macs))
        activations = tuple(sorted(
            ((str(owner[0]), str(owner[1])), int(elements))
            for owner, elements in self.activation_elements))
        _require_unique(
            tuple(module for module, macs in weights), "weight cost")
        _require_unique(
            tuple(owner for owner, elements in activations),
            "activation cost")
        if any(macs <= 0 for module, macs in weights) or any(
                elements <= 0 for owner, elements in activations):
            raise ValueError("cost values must be positive")
        object.__setattr__(self, "weight_macs", weights)
        object.__setattr__(self, "activation_elements", activations)


@dataclass(frozen=True)
class BudgetAudit:
    weight_numerator: int
    weight_denominator: int
    activation_numerator: int
    activation_denominator: int
    average_weight_bits: float
    average_activation_bits: float
    weight_feasible: bool
    activation_feasible: bool
    feasible: bool
    weight_mac_fractions: Tuple[Tuple[int, float], ...]
    activation_element_fractions: Tuple[Tuple[int, float], ...]


def _require_unique(values: Sequence[object], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicates" % name)


def _validate_bits(values: Sequence[int]) -> None:
    invalid = tuple(sorted(set(values) - set(BIT_OPTIONS)))
    if invalid:
        raise ValueError("bit assignment contains unsupported values: %s" % (
            invalid,))


def _ordered_unique_by_block(mapping):
    output = {}
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
    return tuple(
        value for block in BLOCK_ORDER for value in mapping[block])


def build_registry(
        executed_weight_modules: Sequence[str],
        activation_owners: Sequence[Owner]) -> AllocationRegistry:
    modules = tuple(str(module) for module in executed_weight_modules)
    owners = tuple(
        (str(owner[0]), str(owner[1])) for owner in activation_owners)
    _require_unique(modules, "weight registry")
    _require_unique(owners, "activation registry")

    weights_by_block = _ordered_unique_by_block(WEIGHT_MODULES_BY_BLOCK)
    activations_by_block = _ordered_unique_by_block(
        ACTIVATION_OWNERS_BY_BLOCK)
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
    return AllocationRegistry(weights_by_block, activations_by_block)


def uniform_assignment(
        registry: AllocationRegistry,
        weight_bits: int,
        activation_bits: int) -> BitAssignment:
    _validate_bits((int(weight_bits), int(activation_bits)))
    return BitAssignment(
        weight_bits=tuple(
            (module, int(weight_bits))
            for module in _ordered_union(registry.weights_by_block)),
        activation_bits=tuple(
            (owner, int(activation_bits))
            for owner in _ordered_union(registry.activations_by_block)),
    )


def build_single_block_probes(
        registry: AllocationRegistry) -> Tuple[AllocationCandidate, ...]:
    baseline = uniform_assignment(registry, 4, 4)
    output = [AllocationCandidate(
        name="UNIFORM_W4A4",
        stage="baseline",
        block="all",
        assignment=baseline,
        weight_bits=4,
        activation_bits=4,
    )]
    baseline_weights = dict(baseline.weight_bits)
    baseline_activations = dict(baseline.activation_bits)
    for block in BLOCK_ORDER:
        for weight_bits in BIT_OPTIONS:
            for activation_bits in BIT_OPTIONS:
                if (weight_bits, activation_bits) == (4, 4):
                    continue
                weights = dict(baseline_weights)
                activations = dict(baseline_activations)
                for module in registry.weights_by_block[block]:
                    weights[module] = weight_bits
                for owner in registry.activations_by_block[block]:
                    activations[owner] = activation_bits
                output.append(AllocationCandidate(
                    name="PROBE_%s_W%dA%d" % (
                        block, weight_bits, activation_bits),
                    stage="block_probe",
                    block=block,
                    assignment=BitAssignment(
                        tuple(weights.items()), tuple(activations.items())),
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                ))
    _require_unique(
        tuple(candidate.name for candidate in output), "probe names")
    return tuple(output)


def audit_budget(
        assignment: BitAssignment,
        basis: CostBasis) -> BudgetAudit:
    weight_bits = dict(assignment.weight_bits)
    activation_bits = dict(assignment.activation_bits)
    weight_macs = dict(basis.weight_macs)
    activation_elements = dict(basis.activation_elements)
    if set(weight_bits) != set(weight_macs):
        raise ValueError("weight assignment and cost coverage mismatch")
    if set(activation_bits) != set(activation_elements):
        raise ValueError("activation assignment and cost coverage mismatch")

    weight_denominator = sum(weight_macs.values())
    activation_denominator = sum(activation_elements.values())
    weight_numerator = sum(
        weight_bits[module] * weight_macs[module] for module in weight_macs)
    activation_numerator = sum(
        activation_bits[owner] * activation_elements[owner]
        for owner in activation_elements)
    weight_feasible = weight_numerator <= 4 * weight_denominator
    activation_feasible = (
        activation_numerator <= 4 * activation_denominator)
    weight_mac_fractions = tuple(
        (bits, sum(
            macs for module, macs in basis.weight_macs
            if weight_bits[module] == bits) / float(weight_denominator))
        for bits in BIT_OPTIONS)
    activation_element_fractions = tuple(
        (bits, sum(
            elements for owner, elements in basis.activation_elements
            if activation_bits[owner] == bits) /
         float(activation_denominator))
        for bits in BIT_OPTIONS)
    return BudgetAudit(
        weight_numerator=weight_numerator,
        weight_denominator=weight_denominator,
        activation_numerator=activation_numerator,
        activation_denominator=activation_denominator,
        average_weight_bits=weight_numerator / float(weight_denominator),
        average_activation_bits=(
            activation_numerator / float(activation_denominator)),
        weight_feasible=weight_feasible,
        activation_feasible=activation_feasible,
        feasible=weight_feasible and activation_feasible,
        weight_mac_fractions=weight_mac_fractions,
        activation_element_fractions=activation_element_fractions,
    )
