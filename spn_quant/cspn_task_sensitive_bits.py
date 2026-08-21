"""CSPN task-sensitive mixed-bit allocation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
from typing import Mapping, Optional, Sequence, Tuple

from spn_quant import cspn_encoder_prefix as encoder
from spn_quant import cspn_sensitivity as decoder


Owner = Tuple[str, str]

BIT_OPTIONS = (2, 4, 6, 8)
MIXED_ACTIVATION_BITS = (4, 6, 8)
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


@dataclass(frozen=True)
class ActivationBudgetAudit:
    activation_numerator: int
    activation_denominator: int
    average_activation_bits: float
    maximum_activation_bits: float
    feasible: bool
    activation_element_fractions: Tuple[Tuple[int, float], ...]


@dataclass(frozen=True)
class SensitivityEntry:
    name: str
    block: str
    weight_bits: int
    activation_bits: int
    calibration_rmse: float
    boundary_rmse: float
    propagation_mse: float
    rmse_delta: float
    boundary_delta: float
    propagation_delta: float
    valid: bool


@dataclass(frozen=True)
class SearchState:
    block_bits: Tuple[Tuple[str, int, int], ...]
    estimated_rmse: float
    estimated_boundary_rmse: float
    estimated_propagation_mse: float
    weight_numerator: int
    activation_numerator: int
    assignment: Optional[BitAssignment]


def _require_unique(values: Sequence[object], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicates" % name)


def _validate_bits(values: Sequence[int]) -> None:
    invalid = tuple(sorted(set(values) - set(BIT_OPTIONS)))
    if invalid:
        raise ValueError("bit assignment contains unsupported values: %s" % (
            invalid,))


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


def p3_t3_assignment(registry: AllocationRegistry) -> BitAssignment:
    protected = set(P3_T3_PROTECTED_BLOCKS)
    return BitAssignment(
        weight_bits=tuple(
            (module, 8 if block in protected else 4)
            for block in BLOCK_ORDER
            for module in registry.weights_by_block[block]),
        activation_bits=tuple(
            (owner, 8 if block in protected else 4)
            for block in BLOCK_ORDER
            for owner in registry.activations_by_block[block]),
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


def audit_activation_budget(
        assignment: BitAssignment,
        basis: CostBasis,
        maximum_bits: float) -> ActivationBudgetAudit:
    activation_bits = dict(assignment.activation_bits)
    activation_elements = dict(basis.activation_elements)
    if set(activation_bits) != set(activation_elements):
        raise ValueError("activation assignment and cost coverage mismatch")
    invalid_bits = tuple(sorted(
        set(activation_bits.values()) - set(MIXED_ACTIVATION_BITS)))
    if invalid_bits:
        raise ValueError(
            "mixed activation assignment contains unsupported values: %s" %
            (invalid_bits,))
    limit = float(maximum_bits)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("activation budget must be finite and positive")
    denominator = sum(activation_elements.values())
    if denominator <= 0:
        raise ValueError("activation cost denominator must be positive")
    numerator = sum(
        activation_bits[owner] * activation_elements[owner]
        for owner in activation_elements)
    average = numerator / float(denominator)
    fractions = tuple(
        (bits, sum(
            elements for owner, elements in basis.activation_elements
            if activation_bits[owner] == bits) / float(denominator))
        for bits in MIXED_ACTIVATION_BITS)
    return ActivationBudgetAudit(
        activation_numerator=numerator,
        activation_denominator=denominator,
        average_activation_bits=average,
        maximum_activation_bits=limit,
        feasible=average <= limit,
        activation_element_fractions=fractions,
    )


def assignment_key(assignment: BitAssignment):
    return assignment.weight_bits, assignment.activation_bits


def build_p3_t3_activation_candidates(
        registry: AllocationRegistry,
        basis: CostBasis,
        maximum_bits: float) -> Tuple[BitAssignment, ...]:
    seed = p3_t3_assignment(registry)
    seed_activations = dict(seed.activation_bits)
    candidates = []
    for precision in itertools.product(
            MIXED_ACTIVATION_BITS,
            repeat=len(P3_T3_PROTECTED_BLOCKS)):
        activations = dict(seed_activations)
        for block, bits in zip(P3_T3_PROTECTED_BLOCKS, precision):
            for owner in registry.activations_by_block[block]:
                activations[owner] = bits
        candidate = BitAssignment(
            weight_bits=seed.weight_bits,
            activation_bits=tuple(activations.items()),
        )
        if audit_activation_budget(candidate, basis, maximum_bits).feasible:
            candidates.append(candidate)
    return tuple(sorted(set(candidates), key=assignment_key))


def select_measured_activation_candidate(
        candidates: Sequence[BitAssignment],
        rows: Sequence[Mapping[str, object]],
        basis: CostBasis,
        maximum_bits: float) -> BitAssignment:
    assignments = tuple(candidates)
    _require_unique(assignments, "mixed activation candidates")
    if not assignments:
        raise ValueError("mixed activation candidates must not be empty")
    for assignment in assignments:
        if not audit_activation_budget(
                assignment, basis, maximum_bits).feasible:
            raise ValueError("mixed activation candidate exceeds the budget")

    evaluation_fields = {
        "validation_RMSE",
        "evaluation_RMSE",
        "fixed64_RMSE",
    }
    required_fields = {
        "assignment",
        "calibration_RMSE",
        "boundary_RMSE",
        "propagation_MSE",
        "nonfinite_ratio",
        "nonpositive_ratio",
    }
    measured = {}
    for row in rows:
        if evaluation_fields.intersection(row):
            raise ValueError(
                "measured assignment rows contain evaluation-only fields")
        if not required_fields.issubset(row):
            raise ValueError("measured assignment row is incomplete")
        assignment = row["assignment"]
        if assignment in measured:
            raise ValueError("measured assignment rows contain duplicates")
        metrics = tuple(float(row[name]) for name in (
            "calibration_RMSE",
            "boundary_RMSE",
            "propagation_MSE",
            "nonfinite_ratio",
            "nonpositive_ratio",
        ))
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("measured assignment metrics must be finite")
        if metrics[3] < 0.0 or metrics[4] < 0.0:
            raise ValueError("numerical failure ratios must be nonnegative")
        measured[assignment] = metrics
    if set(measured) != set(assignments):
        raise ValueError("measured assignment coverage mismatch")

    ranked = tuple(sorted(
        assignments,
        key=lambda assignment: (
            measured[assignment][3] != 0.0,
            measured[assignment][4] != 0.0,
            measured[assignment][0],
            measured[assignment][1],
            measured[assignment][2],
            audit_activation_budget(
                assignment, basis, maximum_bits).average_activation_bits,
            assignment_key(assignment),
        )))
    best = ranked[0]
    if measured[best][3] != 0.0 or measured[best][4] != 0.0:
        raise ValueError("all mixed activation candidates are numerically invalid")
    return best


def search_state_key(state: SearchState):
    canonical = (
        state.block_bits if state.assignment is None
        else assignment_key(state.assignment))
    return (
        state.estimated_rmse,
        state.estimated_propagation_mse,
        -state.weight_numerator,
        -state.activation_numerator,
        canonical,
    )


def build_sensitivity_table(
        probes: Sequence[AllocationCandidate],
        rows: Sequence[Mapping[str, object]]) -> Tuple[SensitivityEntry, ...]:
    probe_names = tuple(probe.name for probe in probes)
    _require_unique(probe_names, "probe candidates")
    measured_names = tuple(str(row["config"]) for row in rows)
    _require_unique(measured_names, "sensitivity rows")
    if set(measured_names) != set(probe_names):
        raise ValueError("sensitivity row coverage mismatch")
    measured = dict((str(row["config"]), row) for row in rows)
    values = []
    validity = []
    for probe in probes:
        row = measured[probe.name]
        valid = bool(row["sensitivity_valid"])
        if valid:
            metrics = (
                float(row["calibration_RMSE"]),
                float(row["boundary_RMSE"]),
                float(row["propagation_MSE"]),
            )
            if not all(math.isfinite(value) for value in metrics):
                raise ValueError("sensitivity metrics must be finite")
        else:
            metrics = (float("inf"), float("inf"), float("inf"))
        values.append(metrics)
        validity.append(valid)
    baseline_index = probe_names.index("UNIFORM_W4A4")
    baseline = values[baseline_index]
    if not validity[baseline_index]:
        raise ValueError("uniform W4A4 sensitivity baseline is invalid")
    return tuple(
        SensitivityEntry(
            name=probe.name,
            block=probe.block,
            weight_bits=probe.weight_bits,
            activation_bits=probe.activation_bits,
            calibration_rmse=metrics[0],
            boundary_rmse=metrics[1],
            propagation_mse=metrics[2],
            rmse_delta=metrics[0] - baseline[0],
            boundary_delta=metrics[1] - baseline[1],
            propagation_delta=metrics[2] - baseline[2],
            valid=valid,
        )
        for probe, metrics, valid in zip(probes, values, validity))


def _block_costs(registry: AllocationRegistry, basis: CostBasis):
    weight_macs = dict(basis.weight_macs)
    activation_elements = dict(basis.activation_elements)
    expected_modules = set(_ordered_union(registry.weights_by_block))
    expected_owners = set(_ordered_union(registry.activations_by_block))
    if set(weight_macs) != expected_modules:
        raise ValueError("weight registry and cost coverage mismatch")
    if set(activation_elements) != expected_owners:
        raise ValueError("activation registry and cost coverage mismatch")
    return (
        dict((block, sum(
            weight_macs[module]
            for module in registry.weights_by_block[block]))
             for block in BLOCK_ORDER),
        dict((block, sum(
            activation_elements[owner]
            for owner in registry.activations_by_block[block]))
             for block in BLOCK_ORDER),
    )


def _assignment_from_block_bits(
        registry: AllocationRegistry,
        block_bits: Sequence[Tuple[str, int, int]]) -> BitAssignment:
    rows = tuple(block_bits)
    if tuple(block for block, weight_bits, activation_bits in rows) != \
            BLOCK_ORDER:
        raise ValueError("block bit assignment coverage mismatch")
    weight_values = []
    activation_values = []
    for block, weight_bits, activation_bits in rows:
        weight_values.extend(
            (module, weight_bits)
            for module in registry.weights_by_block[block])
        activation_values.extend(
            (owner, activation_bits)
            for owner in registry.activations_by_block[block])
    return BitAssignment(tuple(weight_values), tuple(activation_values))


def prune_dominated_states(
        states: Sequence[SearchState]) -> Tuple[SearchState, ...]:
    unique = {}
    for state in states:
        key = state.block_bits
        if key not in unique or search_state_key(state) < \
                search_state_key(unique[key]):
            unique[key] = state
    ordered = tuple(sorted(unique.values(), key=search_state_key))
    output = []
    for candidate in ordered:
        dominated = False
        for other in ordered:
            if other == candidate:
                continue
            no_worse = (
                other.estimated_rmse <= candidate.estimated_rmse and
                other.estimated_propagation_mse <=
                candidate.estimated_propagation_mse and
                other.weight_numerator <= candidate.weight_numerator and
                other.activation_numerator <=
                candidate.activation_numerator)
            strictly_better = (
                other.estimated_rmse < candidate.estimated_rmse or
                other.estimated_propagation_mse <
                candidate.estimated_propagation_mse or
                other.weight_numerator < candidate.weight_numerator or
                other.activation_numerator < candidate.activation_numerator)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            output.append(candidate)
    return tuple(output)


def search_block_assignments(
        registry: AllocationRegistry,
        basis: CostBasis,
        measured_rows: Sequence[Mapping[str, object]],
        beam_width: int,
        candidate_limit: int) -> Tuple[SearchState, ...]:
    if int(beam_width) <= 0 or int(candidate_limit) <= 0:
        raise ValueError("Beam width and candidate limit must be positive")
    probes = build_single_block_probes(registry)
    sensitivities = build_sensitivity_table(probes, measured_rows)
    baseline = sensitivities[0]
    lookup = dict(
        ((entry.block, entry.weight_bits, entry.activation_bits), entry)
        for entry in sensitivities[1:])
    weight_costs, activation_costs = _block_costs(registry, basis)
    total_weight = sum(value for module, value in basis.weight_macs)
    total_activation = sum(
        value for owner, value in basis.activation_elements)
    states = (SearchState(
        block_bits=(),
        estimated_rmse=baseline.calibration_rmse,
        estimated_boundary_rmse=baseline.boundary_rmse,
        estimated_propagation_mse=baseline.propagation_mse,
        weight_numerator=0,
        activation_numerator=0,
        assignment=None,
    ),)
    for block_index, block in enumerate(BLOCK_ORDER):
        expanded = []
        remaining = BLOCK_ORDER[block_index + 1:]
        minimum_remaining_weight = 2 * sum(
            weight_costs[name] for name in remaining)
        minimum_remaining_activation = 2 * sum(
            activation_costs[name] for name in remaining)
        for state in states:
            for weight_bits in BIT_OPTIONS:
                for activation_bits in BIT_OPTIONS:
                    if (weight_bits, activation_bits) == (4, 4):
                        rmse_delta = 0.0
                        boundary_delta = 0.0
                        propagation_delta = 0.0
                    else:
                        entry = lookup[(block, weight_bits, activation_bits)]
                        rmse_delta = entry.rmse_delta
                        boundary_delta = entry.boundary_delta
                        propagation_delta = entry.propagation_delta
                    weight_numerator = (
                        state.weight_numerator +
                        weight_bits * weight_costs[block])
                    activation_numerator = (
                        state.activation_numerator +
                        activation_bits * activation_costs[block])
                    if weight_numerator + minimum_remaining_weight > \
                            4 * total_weight:
                        continue
                    if activation_numerator + minimum_remaining_activation > \
                            4 * total_activation:
                        continue
                    expanded.append(SearchState(
                        block_bits=(
                            state.block_bits +
                            ((block, weight_bits, activation_bits),)),
                        estimated_rmse=state.estimated_rmse + rmse_delta,
                        estimated_boundary_rmse=(
                            state.estimated_boundary_rmse + boundary_delta),
                        estimated_propagation_mse=(
                            state.estimated_propagation_mse +
                            propagation_delta),
                        weight_numerator=weight_numerator,
                        activation_numerator=activation_numerator,
                        assignment=None,
                    ))
        states = tuple(sorted(
            prune_dominated_states(expanded),
            key=search_state_key)[:int(beam_width)])
    completed = []
    for state in states:
        assignment = _assignment_from_block_bits(registry, state.block_bits)
        if not audit_budget(assignment, basis).feasible:
            continue
        completed.append(SearchState(
            block_bits=state.block_bits,
            estimated_rmse=state.estimated_rmse,
            estimated_boundary_rmse=state.estimated_boundary_rmse,
            estimated_propagation_mse=state.estimated_propagation_mse,
            weight_numerator=state.weight_numerator,
            activation_numerator=state.activation_numerator,
            assignment=assignment,
        ))
    ranked = list(sorted(completed, key=search_state_key)[:
                       int(candidate_limit)])
    baseline_assignment = uniform_assignment(registry, 4, 4)
    if baseline_assignment not in tuple(
            state.assignment for state in ranked):
        baseline_audit = audit_budget(baseline_assignment, basis)
        baseline_state = SearchState(
            block_bits=tuple((block, 4, 4) for block in BLOCK_ORDER),
            estimated_rmse=baseline.calibration_rmse,
            estimated_boundary_rmse=baseline.boundary_rmse,
            estimated_propagation_mse=baseline.propagation_mse,
            weight_numerator=baseline_audit.weight_numerator,
            activation_numerator=baseline_audit.activation_numerator,
            assignment=baseline_assignment,
        )
        ranked[-1] = baseline_state
    return tuple(sorted(ranked, key=search_state_key))


def _block_precision(
        assignment: BitAssignment,
        registry: AllocationRegistry):
    weights = dict(assignment.weight_bits)
    activations = dict(assignment.activation_bits)
    output = {}
    for block in BLOCK_ORDER:
        weight_values = {
            weights[module] for module in registry.weights_by_block[block]}
        activation_values = {
            activations[owner]
            for owner in registry.activations_by_block[block]}
        if len(weight_values) != 1 or len(activation_values) != 1:
            raise ValueError("local search requires block-uniform precision")
        output[block] = (
            next(iter(weight_values)), next(iter(activation_values)))
    return output


def _replace_block_precision(
        current: BitAssignment,
        registry: AllocationRegistry,
        updates: Mapping[str, Tuple[int, int]]) -> BitAssignment:
    weights = dict(current.weight_bits)
    activations = dict(current.activation_bits)
    for block in updates:
        weight_bits, activation_bits = updates[block]
        _validate_bits((weight_bits, activation_bits))
        for module in registry.weights_by_block[block]:
            weights[module] = weight_bits
        for owner in registry.activations_by_block[block]:
            activations[owner] = activation_bits
    return BitAssignment(tuple(weights.items()), tuple(activations.items()))


def build_budget_preserving_neighbors(
        current: BitAssignment,
        registry: AllocationRegistry,
        basis: CostBasis) -> Tuple[BitAssignment, ...]:
    precision = _block_precision(current, registry)
    weight_moves = []
    activation_moves = []
    for source in BLOCK_ORDER:
        for destination in BLOCK_ORDER:
            if source == destination:
                continue
            source_weight, source_activation = precision[source]
            destination_weight, destination_activation = precision[destination]
            if source_weight > BIT_OPTIONS[0] and \
                    destination_weight < BIT_OPTIONS[-1]:
                weight_moves.append((source, destination))
            if source_activation > BIT_OPTIONS[0] and \
                    destination_activation < BIT_OPTIONS[-1]:
                activation_moves.append((source, destination))
    candidates = []
    for source, destination in weight_moves:
        updates = {
            source: (precision[source][0] - 2, precision[source][1]),
            destination: (
                precision[destination][0] + 2,
                precision[destination][1]),
        }
        candidates.append(_replace_block_precision(
            current, registry, updates))
    for source, destination in activation_moves:
        updates = {
            source: (precision[source][0], precision[source][1] - 2),
            destination: (
                precision[destination][0],
                precision[destination][1] + 2),
        }
        candidates.append(_replace_block_precision(
            current, registry, updates))
    activation_move_set = set(activation_moves)
    for source, destination in weight_moves:
        if (source, destination) not in activation_move_set:
            continue
        updates = {
            source: (
                precision[source][0] - 2,
                precision[source][1] - 2),
            destination: (
                precision[destination][0] + 2,
                precision[destination][1] + 2),
        }
        candidates.append(_replace_block_precision(
            current, registry, updates))
    unique = tuple(sorted(set(candidates), key=assignment_key))
    return tuple(
        candidate for candidate in unique
        if audit_budget(candidate, basis).feasible)


def _measured_assignment_rows(
        assignments: Sequence[BitAssignment],
        rows: Sequence[Mapping[str, object]]):
    expected = set(assignments)
    measured = {}
    for row in rows:
        assignment = row["assignment"]
        if assignment in measured:
            raise ValueError("measured assignment rows contain duplicates")
        metrics = (
            float(row["calibration_RMSE"]),
            float(row["boundary_RMSE"]),
            float(row["propagation_MSE"]),
        )
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("measured assignment metrics must be finite")
        measured[assignment] = metrics
    if set(measured) != expected:
        raise ValueError("measured assignment coverage mismatch")
    return measured


def select_local_improvement(
        current: BitAssignment,
        current_rmse: float,
        neighbors: Sequence[BitAssignment],
        measured_rows: Sequence[Mapping[str, object]],
        basis: CostBasis) -> BitAssignment:
    if not math.isfinite(float(current_rmse)):
        raise ValueError("current RMSE must be finite")
    candidates = tuple(neighbors)
    _require_unique(candidates, "local neighbor assignments")
    measured = _measured_assignment_rows(candidates, measured_rows)
    ranked = sorted(
        candidates,
        key=lambda assignment: (
            measured[assignment][0],
            measured[assignment][1],
            measured[assignment][2],
            audit_budget(assignment, basis).weight_numerator,
            audit_budget(assignment, basis).activation_numerator,
            assignment_key(assignment),
        ))
    if ranked and measured[ranked[0]][0] < float(current_rmse):
        return ranked[0]
    return current


def build_cheapest_block_demotions(
        current: BitAssignment,
        registry: AllocationRegistry,
        basis: CostBasis) -> Mapping[str, BitAssignment]:
    if not audit_budget(current, basis).feasible:
        raise ValueError("current assignment exceeds the precision budget")
    precision = _block_precision(current, registry)
    weight_costs, activation_costs = _block_costs(registry, basis)
    total_weight = sum(value for module, value in basis.weight_macs)
    total_activation = sum(
        value for owner, value in basis.activation_elements)
    output = {}
    for block in BLOCK_ORDER:
        weight_bits, activation_bits = precision[block]
        choices = []
        if weight_bits > BIT_OPTIONS[0]:
            choices.append((
                Fraction(2 * weight_costs[block], total_weight),
                0,
                (weight_bits - 2, activation_bits),
            ))
        if activation_bits > BIT_OPTIONS[0]:
            choices.append((
                Fraction(2 * activation_costs[block], total_activation),
                1,
                (weight_bits, activation_bits - 2),
            ))
        if choices:
            selected = min(choices)
            output[block] = _replace_block_precision(
                current, registry, {block: selected[2]})
    return output


def rank_refinement_blocks(
        current: BitAssignment,
        registry: AllocationRegistry,
        basis: CostBasis,
        current_rmse: float,
        measured_rows: Sequence[Mapping[str, object]],
        block_limit: int) -> Tuple[str, ...]:
    if not math.isfinite(float(current_rmse)):
        raise ValueError("current RMSE must be finite")
    demotions = build_cheapest_block_demotions(current, registry, basis)
    measured_blocks = tuple(str(row["block"]) for row in measured_rows)
    _require_unique(measured_blocks, "block demotion rows")
    if set(measured_blocks) != set(demotions):
        raise ValueError("block demotion row coverage mismatch")
    rows = dict((str(row["block"]), row) for row in measured_rows)
    ranking = []
    for order, block in enumerate(BLOCK_ORDER):
        if block not in demotions:
            continue
        row = rows[block]
        if row["assignment"] != demotions[block]:
            raise ValueError("block demotion assignment mismatch")
        metrics = (
            float(row["calibration_RMSE"]),
            float(row["boundary_RMSE"]),
            float(row["propagation_MSE"]),
        )
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("block demotion metrics must be finite")
        ranking.append((-(metrics[0] - float(current_rmse)), order, block))
    if int(block_limit) <= 0 or int(block_limit) > len(ranking):
        raise ValueError("refinement block limit is invalid")
    ranking.sort()
    return tuple(row[2] for row in ranking[:int(block_limit)])


def build_refinement_candidates(
        current: BitAssignment,
        registry: AllocationRegistry,
        basis: CostBasis,
        selected_blocks: Sequence[str],
        measured_rows: Sequence[Mapping[str, object]],
        beam_width: int,
        candidate_limit: int) -> Tuple[SearchState, ...]:
    selected = tuple(str(block) for block in selected_blocks)
    _require_unique(selected, "refinement blocks")
    if not selected or not set(selected) <= set(BLOCK_ORDER):
        raise ValueError("refinement blocks are invalid")
    if int(beam_width) <= 0 or int(candidate_limit) <= 0:
        raise ValueError("refinement width and candidate limit must be positive")
    precision = _block_precision(current, registry)
    table = build_sensitivity_table(
        build_single_block_probes(registry), measured_rows)
    sensitivity = dict(
        ((entry.block, entry.weight_bits, entry.activation_bits), entry)
        for entry in table[1:])
    weight_macs = dict(basis.weight_macs)
    activation_elements = dict(basis.activation_elements)
    current_audit = audit_budget(current, basis)

    sites = []
    selected_weight_cost = 0
    selected_activation_cost = 0
    for block in BLOCK_ORDER:
        if block not in selected:
            continue
        for module in registry.weights_by_block[block]:
            sites.append(("weight", module, block, weight_macs[module]))
            selected_weight_cost += (
                dict(current.weight_bits)[module] * weight_macs[module])
        for owner in registry.activations_by_block[block]:
            sites.append((
                "activation", owner, block, activation_elements[owner]))
            selected_activation_cost += (
                dict(current.activation_bits)[owner] *
                activation_elements[owner])

    fixed_weight = current_audit.weight_numerator - selected_weight_cost
    fixed_activation = (
        current_audit.activation_numerator - selected_activation_cost)
    total_weight = current_audit.weight_denominator
    total_activation = current_audit.activation_denominator
    states = ((0.0, 0.0, 0.0, fixed_weight, fixed_activation, ()),)
    for site_index, site in enumerate(sites):
        kind, identity, block, cost = site
        weight_bits, activation_bits = precision[block]
        remaining_weight = 2 * sum(
            row[3] for row in sites[site_index + 1:]
            if row[0] == "weight")
        remaining_activation = 2 * sum(
            row[3] for row in sites[site_index + 1:]
            if row[0] == "activation")
        if kind == "weight":
            current_entry = (
                None if (weight_bits, activation_bits) == (4, 4)
                else sensitivity[(block, weight_bits, activation_bits)])
            divisor = len(registry.weights_by_block[block])
        else:
            current_entry = (
                None if (weight_bits, activation_bits) == (4, 4)
                else sensitivity[(block, weight_bits, activation_bits)])
            divisor = len(registry.activations_by_block[block])
        current_metrics = (
            0.0 if current_entry is None else current_entry.rmse_delta,
            0.0 if current_entry is None else current_entry.boundary_delta,
            0.0 if current_entry is None else current_entry.propagation_delta,
        )
        expanded = []
        for state in states:
            for bits in BIT_OPTIONS:
                pair = (
                    (bits, activation_bits)
                    if kind == "weight" else (weight_bits, bits))
                candidate_entry = (
                    None if pair == (4, 4)
                    else sensitivity[(block, pair[0], pair[1])])
                candidate_metrics = (
                    0.0 if candidate_entry is None
                    else candidate_entry.rmse_delta,
                    0.0 if candidate_entry is None
                    else candidate_entry.boundary_delta,
                    0.0 if candidate_entry is None
                    else candidate_entry.propagation_delta,
                )
                weight_numerator = (
                    state[3] + bits * cost
                    if kind == "weight" else state[3])
                activation_numerator = (
                    state[4] + bits * cost
                    if kind == "activation" else state[4])
                if weight_numerator + remaining_weight > 4 * total_weight:
                    continue
                if activation_numerator + remaining_activation > \
                        4 * total_activation:
                    continue
                expanded.append((
                    state[0] +
                    (candidate_metrics[0] - current_metrics[0]) / divisor,
                    state[1] +
                    (candidate_metrics[1] - current_metrics[1]) / divisor,
                    state[2] +
                    (candidate_metrics[2] - current_metrics[2]) / divisor,
                    weight_numerator,
                    activation_numerator,
                    state[5] + ((kind, identity, bits),),
                ))
        states = tuple(sorted(
            expanded,
            key=lambda row: (
                row[0], row[2], -row[3], -row[4], row[5]))[
                    :int(beam_width)])

    current_weights = dict(current.weight_bits)
    current_activations = dict(current.activation_bits)
    output = []
    for state in states:
        weights = dict(current_weights)
        activations = dict(current_activations)
        for kind, identity, bits in state[5]:
            if kind == "weight":
                weights[identity] = bits
            else:
                activations[identity] = bits
        assignment = BitAssignment(
            tuple(weights.items()), tuple(activations.items()))
        if not audit_budget(assignment, basis).feasible:
            continue
        output.append(SearchState(
            block_bits=(),
            estimated_rmse=state[0],
            estimated_boundary_rmse=state[1],
            estimated_propagation_mse=state[2],
            weight_numerator=state[3],
            activation_numerator=state[4],
            assignment=assignment,
        ))
    unique = {}
    for state in output:
        if state.assignment not in unique or search_state_key(state) < \
                search_state_key(unique[state.assignment]):
            unique[state.assignment] = state
    return tuple(sorted(unique.values(), key=search_state_key)[
                 :int(candidate_limit)])
