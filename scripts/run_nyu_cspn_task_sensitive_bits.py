#!/usr/bin/env python3
"""Search task-sensitive mixed-bit assignments for official CSPN."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class SearchProtocol:
    beam_width: int
    joint_measured_limit: int
    local_round_limit: int
    refinement_block_limit: int
    refinement_width: int
    refinement_measured_limit: int

    def __post_init__(self) -> None:
        values = (
            self.beam_width,
            self.joint_measured_limit,
            self.local_round_limit,
            self.refinement_block_limit,
            self.refinement_width,
            self.refinement_measured_limit,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("search protocol values must be positive")


@dataclass(frozen=True)
class ValidationCandidate:
    name: str
    assignment: Optional[allocation.BitAssignment]


@dataclass(frozen=True)
class SearchResult:
    single_block_rows: Tuple[Mapping[str, object], ...]
    joint_rows: Tuple[Mapping[str, object], ...]
    local_rounds: Tuple[Tuple[Mapping[str, object], ...], ...]
    demotion_rows: Tuple[Mapping[str, object], ...]
    refined_rows: Tuple[Mapping[str, object], ...]
    validation_rows: Tuple[Mapping[str, object], ...]
    validation_candidates: Tuple[ValidationCandidate, ...]
    refinement_blocks: Tuple[str, ...]
    final_assignment: allocation.BitAssignment
    final_budget: allocation.BudgetAudit


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


def aggregate_candidate_result(
        candidate: RuntimeCandidate,
        sample_rows: Sequence[Mapping[str, object]],
        propagation_rows: Sequence[Mapping[str, object]],
        expected_samples: int):
    samples = int(expected_samples)
    if samples <= 0 or len(sample_rows) != samples:
        raise ValueError("candidate sample count differs from the protocol")
    identities = tuple(int(row["sample_index"]) for row in sample_rows)
    if len(identities) != len(set(identities)):
        raise ValueError("candidate sample identities contain duplicates")
    output = {
        "config": candidate.name,
        "assignment": candidate.assignment,
        "samples": samples,
    }
    for field in DEPTH_METRIC_FIELDS + (
            "nonfinite_ratio", "nonpositive_ratio"):
        values = tuple(float(row[field]) for row in sample_rows)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate sample metrics must be finite")
        output[field] = sum(values) / float(samples)
    output["calibration_RMSE"] = output["RMSE"]

    constraints = tuple(
        row for row in propagation_rows
        if str(row["signal"]) == "affinity_constraints")
    anchors = tuple(
        row for row in propagation_rows
        if str(row["signal"]) == "anchor")
    if not constraints or not anchors:
        raise ValueError("candidate propagation statistics are incomplete")
    coefficient_errors = tuple(
        float(row["coefficient_sum_max_error"]) for row in constraints)
    contraction_rates = tuple(
        float(row["contraction_violation_rate"]) for row in constraints)
    anchor_errors = tuple(float(row["anchor_max_error"]) for row in anchors)
    invariant_values = (
        coefficient_errors + contraction_rates + anchor_errors)
    if not all(math.isfinite(value) for value in invariant_values):
        raise ValueError("candidate propagation metrics must be finite")
    output["coefficient_sum_max_error"] = max(coefficient_errors)
    output["contraction_violation_ratio"] = max(contraction_rates)
    output["anchor_max_error"] = max(anchor_errors)
    return output


def _evaluate_phase(evaluator, phase: str, candidates):
    candidate_names = tuple(candidate.name for candidate in candidates)
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("phase candidate names contain duplicates")
    source_rows = tuple(evaluator.calibration(phase, tuple(candidates)))
    row_names = tuple(str(row["config"]) for row in source_rows)
    if len(row_names) != len(set(row_names)):
        raise ValueError("phase result names contain duplicates")
    if set(row_names) != set(candidate_names):
        raise ValueError("phase result coverage mismatch")
    candidates_by_name = dict(
        (candidate.name, candidate) for candidate in candidates)
    rows_by_name = dict((str(row["config"]), row) for row in source_rows)
    output = []
    for name in candidate_names:
        row = rows_by_name[name]
        if row["assignment"] != candidates_by_name[name].assignment:
            raise RuntimeError("phase result assignment mismatch")
        output.append(row)
    return tuple(output)


def _valid_measured_rows(rows, basis, stage: str):
    output = []
    for row in rows:
        assignment = row["assignment"]
        budget = allocation.audit_budget(assignment, basis)
        status = candidate_status(row, budget, stage)
        if status.valid:
            output.append((row, budget))
    return tuple(output)


def _measured_key(row, budget):
    return (
        float(row["calibration_RMSE"]),
        float(row["boundary_RMSE"]),
        float(row["propagation_MSE"]),
        budget.weight_numerator,
        budget.activation_numerator,
        allocation.assignment_key(row["assignment"]),
    )


def _best_measured(rows, basis, stage: str):
    valid = _valid_measured_rows(rows, basis, stage)
    if not valid:
        raise RuntimeError("phase contains no valid measured candidate")
    return min(valid, key=lambda item: _measured_key(item[0], item[1]))


def _p3_t3_assignment(
        registry: allocation.AllocationRegistry) -> allocation.BitAssignment:
    baseline = allocation.uniform_assignment(registry, 4, 4)
    weights = dict(baseline.weight_bits)
    activations = dict(baseline.activation_bits)
    promoted_blocks = (
        "stem",
        "encoder_layer1",
        "encoder_layer2",
        "decoder_layer4",
        "initial_depth",
    )
    for block in promoted_blocks:
        for module in registry.weights_by_block[block]:
            weights[module] = 8
        for owner in registry.activations_by_block[block]:
            activations[owner] = 8
    return allocation.BitAssignment(
        tuple(weights.items()), tuple(activations.items()))


def run_search(
        protocol: SearchProtocol,
        registry: allocation.AllocationRegistry,
        basis: allocation.CostBasis,
        evaluator) -> SearchResult:
    probes = allocation.build_single_block_probes(registry)
    probe_candidates = tuple(RuntimeCandidate(
        probe.name, "single_block", probe.assignment) for probe in probes)
    single_block_rows = _evaluate_phase(
        evaluator, "single_block", probe_candidates)

    beam = allocation.search_block_assignments(
        registry,
        basis,
        single_block_rows,
        protocol.beam_width,
        protocol.joint_measured_limit,
    )
    if len(beam) != protocol.joint_measured_limit:
        raise RuntimeError("joint Beam did not produce the required coverage")
    joint_candidates = tuple(RuntimeCandidate(
        "JOINT_%03d" % index,
        "joint",
        state.assignment,
    ) for index, state in enumerate(beam))
    joint_rows = _evaluate_phase(evaluator, "joint", joint_candidates)
    current_row, current_budget = _best_measured(joint_rows, basis, "joint")
    current = current_row["assignment"]
    current_rmse = float(current_row["calibration_RMSE"])

    local_rounds = []
    for round_index in range(protocol.local_round_limit):
        neighbors = allocation.build_budget_preserving_neighbors(
            current, registry, basis)
        if not neighbors:
            break
        candidates = tuple(RuntimeCandidate(
            "LOCAL_R%d_%04d" % (round_index + 1, index),
            "local",
            assignment,
        ) for index, assignment in enumerate(neighbors))
        rows = _evaluate_phase(evaluator, "local", candidates)
        local_rounds.append(rows)
        selected = allocation.select_local_improvement(
            current, current_rmse, neighbors, rows, basis)
        if selected == current:
            break
        selected_row = next(
            row for row in rows if row["assignment"] == selected)
        current = selected
        current_rmse = float(selected_row["calibration_RMSE"])
        current_budget = allocation.audit_budget(current, basis)

    demotions = allocation.build_cheapest_block_demotions(
        current, registry, basis)
    demotion_candidates = tuple(RuntimeCandidate(
        "DEMOTION_%s" % block,
        "refinement",
        demotions[block],
    ) for block in allocation.BLOCK_ORDER if block in demotions)
    raw_demotion_rows = _evaluate_phase(
        evaluator, "demotion", demotion_candidates)
    demotion_rows = []
    for block, row in zip(
            tuple(block for block in allocation.BLOCK_ORDER
                  if block in demotions), raw_demotion_rows):
        current_demotion = dict(row)
        current_demotion["block"] = block
        demotion_rows.append(current_demotion)
    refinement_blocks = allocation.rank_refinement_blocks(
        current,
        registry,
        basis,
        current_rmse,
        tuple(demotion_rows),
        protocol.refinement_block_limit,
    )
    refinement_beam = allocation.build_refinement_candidates(
        current,
        registry,
        basis,
        refinement_blocks,
        single_block_rows,
        protocol.refinement_width,
        protocol.refinement_measured_limit,
    )
    if len(refinement_beam) != protocol.refinement_measured_limit:
        raise RuntimeError(
            "refinement Beam did not produce the required coverage")
    refinement_candidates = tuple(RuntimeCandidate(
        "REFINEMENT_%03d" % index,
        "refinement",
        state.assignment,
    ) for index, state in enumerate(refinement_beam))
    refined_rows = _evaluate_phase(
        evaluator, "refinement", refinement_candidates)
    final_row, final_budget = _best_measured(
        refined_rows, basis, "refinement")
    final_assignment = final_row["assignment"]

    validation_candidates = (
        ValidationCandidate("FP32", None),
        ValidationCandidate(
            "UNIFORM_W4A4", allocation.uniform_assignment(registry, 4, 4)),
        ValidationCandidate(
            "CONTEXT_P3_T3_W8A8", _p3_t3_assignment(registry)),
        ValidationCandidate("FINAL", final_assignment),
    )
    validation_rows = tuple(evaluator.validation(validation_candidates))
    validation_names = tuple(str(row["config"]) for row in validation_rows)
    expected_validation_names = tuple(
        candidate.name for candidate in validation_candidates)
    if validation_names != expected_validation_names:
        raise ValueError("validation result coverage mismatch")
    for candidate, row in zip(validation_candidates, validation_rows):
        if row["assignment"] != candidate.assignment:
            raise RuntimeError("validation result assignment mismatch")

    return SearchResult(
        single_block_rows=single_block_rows,
        joint_rows=joint_rows,
        local_rounds=tuple(local_rounds),
        demotion_rows=tuple(demotion_rows),
        refined_rows=refined_rows,
        validation_rows=validation_rows,
        validation_candidates=validation_candidates,
        refinement_blocks=refinement_blocks,
        final_assignment=final_assignment,
        final_budget=final_budget,
    )
