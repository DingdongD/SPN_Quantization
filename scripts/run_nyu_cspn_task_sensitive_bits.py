#!/usr/bin/env python3
"""Search task-sensitive mixed-bit assignments for official CSPN."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_activation_resolution as base
from scripts import run_nyu_cspn_decoder_sensitivity as decoder_runner
from scripts import run_nyu_cspn_encoder_prefix_joint as prefix_runner
from scripts import run_nyu_cspn_selective_w4a8 as selective_runner
from scripts import run_nyu_cspn_stem_precision as stem_runner
from scripts.run_nyu_rtn_quantization import (
    calibration_dataset,
    evaluation_dataset,
    prediction_payload,
    prepare_prediction_dir,
    seeded_sample,
    write_csv,
    write_json,
    write_prediction_payload,
)
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


@dataclass(frozen=True)
class _MetricCandidate:
    name: str
    weight_modules: Tuple[str, ...]


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
        weight_modules=tuple(module for module, bits in generic_weights),
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


def validate_production_protocol(protocol: SearchProtocol) -> None:
    observed = (
        protocol.beam_width,
        protocol.joint_measured_limit,
        protocol.local_round_limit,
        protocol.refinement_block_limit,
        protocol.refinement_width,
        protocol.refinement_measured_limit,
    )
    expected = (512, 128, 3, 4, 128, 128)
    if observed != expected:
        raise ValueError(
            "production protocol differs: expected=%s observed=%s" %
            (expected, observed))


def final_allocation_rows(
        registry: allocation.AllocationRegistry,
        basis: allocation.CostBasis,
        assignment: allocation.BitAssignment):
    allocation.audit_budget(assignment, basis)
    weight_bits = dict(assignment.weight_bits)
    activation_bits = dict(assignment.activation_bits)
    weight_macs = dict(basis.weight_macs)
    activation_elements = dict(basis.activation_elements)
    weight_denominator = sum(weight_macs.values())
    activation_denominator = sum(activation_elements.values())
    weight_blocks = {}
    activation_blocks = {}
    for block in allocation.BLOCK_ORDER:
        for module in registry.weights_by_block[block]:
            if module in weight_blocks:
                raise ValueError("weight ownership contains duplicates")
            weight_blocks[module] = block
        for owner in registry.activations_by_block[block]:
            if owner in activation_blocks:
                raise ValueError("activation ownership contains duplicates")
            activation_blocks[owner] = block
    rows = []
    for module, bits in assignment.weight_bits:
        cost = weight_macs[module]
        rows.append({
            "tensor": "weight",
            "block": weight_blocks[module],
            "module": module,
            "kind": "weight",
            "bits": bits,
            "cost": cost,
            "cost_fraction": cost / float(weight_denominator),
            "weighted_bits": bits * cost,
        })
    for owner, bits in assignment.activation_bits:
        cost = activation_elements[owner]
        rows.append({
            "tensor": "activation",
            "block": activation_blocks[owner],
            "module": owner[0],
            "kind": owner[1],
            "bits": bits,
            "cost": cost,
            "cost_fraction": cost / float(activation_denominator),
            "weighted_bits": bits * cost,
        })
    return tuple(rows)


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
    numeric_values = []
    for field in DEPTH_METRIC_FIELDS + (
            "nonfinite_ratio", "nonpositive_ratio"):
        values = tuple(float(row[field]) for row in sample_rows)
        numeric_values.extend(values)
        output[field] = sum(values) / float(samples)
    output["calibration_RMSE"] = output["RMSE"]

    states = tuple(
        row for row in propagation_rows if str(row["signal"]) == "state")
    constraints = tuple(
        row for row in propagation_rows
        if str(row["signal"]) == "affinity_constraints")
    anchors = tuple(
        row for row in propagation_rows
        if str(row["signal"]) == "anchor")
    if not states or not constraints or not anchors:
        raise ValueError("candidate propagation statistics are incomplete")
    state_mse = tuple(float(row["mse"]) for row in states)
    coefficient_errors = tuple(
        float(row["coefficient_sum_max_error"]) for row in constraints)
    contraction_rates = tuple(
        float(row["contraction_violation_rate"]) for row in constraints)
    anchor_errors = tuple(float(row["anchor_max_error"]) for row in anchors)
    invariant_values = (
        state_mse + coefficient_errors + contraction_rates + anchor_errors)
    numeric_values.extend(invariant_values)
    output["propagation_MSE"] = sum(state_mse) / float(len(state_mse))
    output["coefficient_sum_max_error"] = max(coefficient_errors)
    output["contraction_violation_ratio"] = max(contraction_rates)
    output["anchor_max_error"] = max(anchor_errors)
    output["sensitivity_valid"] = all(
        math.isfinite(value) for value in numeric_values)
    output["valid"] = (
        output["sensitivity_valid"] and
        output["nonfinite_ratio"] == 0.0 and
        output["nonpositive_ratio"] == 0.0 and
        output["coefficient_sum_max_error"] == 0.0 and
        output["contraction_violation_ratio"] == 0.0 and
        output["anchor_max_error"] == 0.0)
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
    if basis is None:
        basis = evaluator.cost_basis()

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


def _runtime_registry(instrumentor, activation_rows):
    modules = prefix_runner.operation_module_names(instrumentor)
    owners = tuple(
        (str(row["module"]), str(row["kind"]))
        for row in activation_rows)
    return allocation.build_registry(modules, owners)


def _run_quantized_candidate(
        candidate: RuntimeCandidate,
        expected_registry: allocation.AllocationRegistry,
        reference_model,
        architecture,
        reference_load,
        reference_preparation,
        saved_args,
        checkpoint: Path,
        preparation_args,
        calibration_dataset,
        evaluation_dataset,
        calibration_indices,
        evaluation_indices,
        seed: int,
        device: torch.device,
        fold_max_error: float,
        prediction_root=None):
    model, current_architecture, load_report, preparation = \
        stem_runner._prepare_model(
            saved_args, checkpoint, device, preparation_args,
            fold_max_error)
    if current_architecture != architecture:
        raise RuntimeError("fresh CSPN architecture changed")
    if load_report != reference_load:
        raise RuntimeError("fresh CSPN checkpoint load changed")
    if preparation["folded_pairs"] != reference_preparation["folded_pairs"]:
        raise RuntimeError("fresh CSPN fold manifest changed")
    instrumentor, rotation, propagation, stem = \
        stem_runner._build_quantization_context(model, preparation, seed)
    stem_runner._calibrate(
        model,
        saved_args,
        calibration_dataset,
        calibration_indices,
        device,
        seed,
        instrumentor,
        rotation,
        propagation,
        stem,
        candidate.name,
    )
    stem_runner._validate_site_contract(instrumentor, rotation)
    first_sample = seeded_sample(
        calibration_dataset, calibration_indices[0], seed)
    activation_rows = decoder_runner.activation_cost_rows(
        instrumentor,
        rotation,
        len(calibration_indices),
        int(first_sample["rgbd"].numel()),
    )
    current_registry = _runtime_registry(instrumentor, activation_rows)
    if current_registry != expected_registry:
        raise RuntimeError("fresh CSPN allocation registry changed")
    config, specs, rotation_specs = configure_runtime_context(
        candidate, instrumentor, rotation, propagation, stem)
    weight_map = dict(candidate.assignment.weight_bits)
    metric_candidate = _MetricCandidate(
        candidate.name,
        tuple(module for module in weight_map if weight_map[module] == 8),
    )
    result = selective_runner._evaluate_candidate(
        metric_candidate,
        reference_model,
        model,
        saved_args,
        evaluation_dataset,
        evaluation_indices,
        device,
        seed,
        instrumentor,
        propagation,
        stem,
        prediction_root,
        retain_invalid_prediction=True,
    )
    for row in result["operation_rows"]:
        row["weight_bits"] = weight_map[str(row["module"])]
    result["activation_rows"] = activation_rows
    result["hardware_configuration"] = config
    result["site_counts"] = {
        "ordinary": len(specs),
        "rotation": len(rotation_specs),
    }
    result["stem_contract"] = stem.contract()
    stem.close()
    propagation.close()
    rotation.close()
    instrumentor.close()
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return result


class CSPNEvaluator(object):
    def __init__(
            self,
            saved_args,
            checkpoint: Path,
            trainset,
            valset,
            calibration_indices,
            evaluation_indices,
            seed: int,
            device: torch.device,
            fold_max_error: float,
            prediction_root=None) -> None:
        if device.type != "cuda":
            raise ValueError("CSPN mixed-bit evaluator requires CUDA")
        self.saved_args = saved_args
        self.checkpoint = Path(checkpoint)
        self.trainset = trainset
        self.valset = valset
        self.calibration_indices = tuple(int(index) for index in calibration_indices)
        self.evaluation_indices = tuple(int(index) for index in evaluation_indices)
        self.seed = int(seed)
        self.device = device
        self.fold_max_error = float(fold_max_error)
        self.prediction_root = prediction_root
        preparation_sample = seeded_sample(
            trainset, self.calibration_indices[0], self.seed)
        self.preparation_args = base._model_args(
            saved_args, preparation_sample, device)
        self.reference_model, self.architecture, self.reference_load, \
            self.reference_preparation = stem_runner._prepare_model(
                saved_args,
                self.checkpoint,
                device,
                self.preparation_args,
                self.fold_max_error,
            )
        self.registry = expected_registry()
        self._basis = None

    def _quantized(self, candidate, dataset, indices, prediction_root):
        result = _run_quantized_candidate(
            candidate,
            self.registry,
            self.reference_model,
            self.architecture,
            self.reference_load,
            self.reference_preparation,
            self.saved_args,
            self.checkpoint,
            self.preparation_args,
            self.trainset,
            dataset,
            self.calibration_indices,
            indices,
            self.seed,
            self.device,
            self.fold_max_error,
            prediction_root,
        )
        basis = build_cost_basis(
            self.registry,
            result["operation_rows"],
            result["activation_rows"],
        )
        if self._basis is None:
            self._basis = basis
        elif basis != self._basis:
            raise RuntimeError("candidate precision cost basis changed")
        row = aggregate_candidate_result(
            candidate,
            result["sample_rows"],
            result["propagation_rows"],
            len(indices),
        )
        return row, result

    def calibration(self, phase, candidates):
        rows = []
        for candidate in candidates:
            row, result = self._quantized(
                candidate,
                self.trainset,
                self.calibration_indices,
                None,
            )
            del result
            rows.append(row)
        return tuple(rows)

    def cost_basis(self):
        if self._basis is None:
            raise RuntimeError("cost basis is unavailable before calibration")
        return self._basis

    def _fp32_validation(self, candidate):
        prediction_dir = prepare_prediction_dir(
            self.prediction_root, candidate.name)
        capture = base.ModuleOutputCapture(
            self.reference_model, base.CSPN_BLOCK_SITES)
        rows = []
        with torch.no_grad():
            for index in self.evaluation_indices:
                sample = seeded_sample(self.valset, index, self.seed)
                prediction, blocks = base._forward(
                    self.reference_model,
                    self.saved_args,
                    sample,
                    self.device,
                    capture,
                )
                del blocks
                pred = prediction.numpy()
                gt = sample["depth"][0].numpy()
                sparse = sample["rgbd"][3].numpy()
                metrics, regions = base.depth_sample_metrics(gt, pred, sparse)
                del regions
                metrics["nonpositive_ratio"] = \
                    selective_runner._prediction_nonpositive_ratio(gt, pred)
                metrics["sample_index"] = int(index)
                rows.append(metrics)
                payload = prediction_payload(
                    gt,
                    pred,
                    pred,
                    int(index),
                    "cspn",
                    candidate.name,
                    sparse=sparse,
                    rgb=sample["rgbd"][:3].permute(1, 2, 0).numpy(),
                )
                write_prediction_payload(prediction_dir, payload)
        capture.close()
        output = {
            "config": candidate.name,
            "assignment": None,
            "samples": len(rows),
            "calibration_RMSE": sum(
                float(row["RMSE"]) for row in rows) / float(len(rows)),
            "coefficient_sum_max_error": 0.0,
            "contraction_violation_ratio": 0.0,
            "anchor_max_error": 0.0,
        }
        for field in DEPTH_METRIC_FIELDS + (
                "nonfinite_ratio", "nonpositive_ratio"):
            output[field] = sum(
                float(row[field]) for row in rows) / float(len(rows))
        return output

    def validation(self, candidates):
        if self.prediction_root is None:
            raise RuntimeError("validation prediction root is not configured")
        rows = []
        for source in candidates:
            if source.assignment is None:
                rows.append(self._fp32_validation(source))
                continue
            candidate = RuntimeCandidate(source.name, "final", source.assignment)
            row, result = self._quantized(
                candidate,
                self.valset,
                self.evaluation_indices,
                self.prediction_root,
            )
            del result
            rows.append(row)
        return tuple(rows)

    def close(self) -> None:
        self.reference_model.cpu()
        del self.reference_model
        torch.cuda.empty_cache()


class ParallelEvaluator(object):
    def __init__(self, workers: Sequence[CSPNEvaluator]) -> None:
        self.workers = tuple(workers)
        if not self.workers:
            raise ValueError("parallel evaluator requires CUDA workers")

    def _batches(self, candidates):
        buckets = [[] for worker in self.workers]
        for index, candidate in enumerate(candidates):
            buckets[index % len(self.workers)].append(candidate)
        return tuple(tuple(bucket) for bucket in buckets)

    @staticmethod
    def _ordered_rows(candidates, rows):
        names = tuple(candidate.name for candidate in candidates)
        row_names = tuple(str(row["config"]) for row in rows)
        if len(row_names) != len(set(row_names)):
            raise ValueError("parallel worker rows contain duplicates")
        if set(row_names) != set(names):
            raise ValueError("parallel worker row coverage mismatch")
        rows_by_name = dict((str(row["config"]), row) for row in rows)
        return tuple(rows_by_name[name] for name in names)

    def calibration(self, phase, candidates):
        declared = tuple(candidates)
        batches = self._batches(declared)
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = tuple(
                executor.submit(worker.calibration, phase, batch)
                for worker, batch in zip(self.workers, batches)
                if batch)
            rows = tuple(
                row for future in futures for row in future.result())
        return self._ordered_rows(declared, rows)

    def validation(self, candidates):
        declared = tuple(candidates)
        batches = self._batches(declared)
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = tuple(
                executor.submit(worker.validation, batch)
                for worker, batch in zip(self.workers, batches)
                if batch)
            rows = tuple(
                row for future in futures for row in future.result())
        return self._ordered_rows(declared, rows)

    def cost_basis(self):
        bases = tuple(worker.cost_basis() for worker in self.workers)
        reference = bases[0]
        if any(basis != reference for basis in bases[1:]):
            raise RuntimeError("CUDA worker cost basis differs")
        return reference

    def close(self) -> None:
        for worker in self.workers:
            worker.close()


def validate_output_directories(output: Path) -> Path:
    final = Path(output)
    staging = Path(str(final) + ".incomplete")
    if final.exists():
        raise FileExistsError("output directory already exists: %s" % final)
    if staging.exists():
        raise FileExistsError(
            "staging directory already exists: %s" % staging)
    return staging


def coordinator_device(devices: Sequence[str]) -> str:
    declared = tuple(str(device) for device in devices)
    if not declared:
        raise ValueError("coordinator requires declared CUDA devices")
    return declared[0]


def assignment_payload(assignment: allocation.BitAssignment):
    return {
        "weight_bits": [
            {"module": module, "bits": bits}
            for module, bits in assignment.weight_bits],
        "activation_bits": [
            {"module": owner[0], "kind": owner[1], "bits": bits}
            for owner, bits in assignment.activation_bits],
    }


def _persisted_metric_row(row, stage: str, local_round: int):
    assignment = row["assignment"]
    output = dict(
        (key, row[key]) for key in row if key != "assignment")
    output["stage"] = stage
    output["local_round"] = local_round
    output["assignment"] = "FP32" if assignment is None else json.dumps(
        assignment_payload(assignment), sort_keys=True)
    return output


def publish_search_result(
        staging: Path,
        output: Path,
        result: SearchResult,
        registry: allocation.AllocationRegistry,
        basis: allocation.CostBasis,
        protocol: SearchProtocol) -> None:
    root = Path(staging)
    final = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("staging directory is missing: %s" % root)
    if final.exists():
        raise FileExistsError("output directory already exists: %s" % final)
    phase_rows = []
    phase_rows.extend(
        _persisted_metric_row(row, "single_block", 0)
        for row in result.single_block_rows)
    phase_rows.extend(
        _persisted_metric_row(row, "joint", 0)
        for row in result.joint_rows)
    for round_index, rows in enumerate(result.local_rounds, 1):
        phase_rows.extend(
            _persisted_metric_row(row, "local", round_index)
            for row in rows)
    phase_rows.extend(
        _persisted_metric_row(row, "demotion", 0)
        for row in result.demotion_rows)
    phase_rows.extend(
        _persisted_metric_row(row, "refinement", 0)
        for row in result.refined_rows)
    validation_rows = tuple(
        _persisted_metric_row(row, "validation", 0)
        for row in result.validation_rows)
    allocation_rows = final_allocation_rows(
        registry, basis, result.final_assignment)
    write_csv(
        root / "calibration_metrics.csv",
        phase_rows,
        ("stage", "local_round", "config", "calibration_RMSE",
         "boundary_RMSE", "propagation_MSE", "RMSE"),
    )
    write_csv(
        root / "validation_metrics.csv",
        validation_rows,
        ("stage", "config", "RMSE", "MAE", "ABS_REL", "IRMSE",
         "flat_RMSE", "boundary_RMSE"),
    )
    write_csv(
        root / "final_allocation.csv",
        allocation_rows,
        ("tensor", "block", "module", "kind", "bits", "cost",
         "cost_fraction", "weighted_bits"),
    )
    write_csv(
        root / "weight_cost_basis.csv",
        tuple({"module": module, "macs": macs}
              for module, macs in basis.weight_macs),
        ("module", "macs"),
    )
    write_csv(
        root / "activation_cost_basis.csv",
        tuple({"module": owner[0], "kind": owner[1], "elements": elements}
              for owner, elements in basis.activation_elements),
        ("module", "kind", "elements"),
    )
    write_json(
        root / "final_assignment.json",
        assignment_payload(result.final_assignment),
    )
    audit = result.final_budget
    manifest = {
        "protocol": {
            "beam_width": protocol.beam_width,
            "joint_measured_limit": protocol.joint_measured_limit,
            "local_round_limit": protocol.local_round_limit,
            "refinement_block_limit": protocol.refinement_block_limit,
            "refinement_width": protocol.refinement_width,
            "refinement_measured_limit": protocol.refinement_measured_limit,
        },
        "phase_counts": {
            "single_block": len(result.single_block_rows),
            "joint": len(result.joint_rows),
            "local_rounds": len(result.local_rounds),
            "local_candidates": sum(len(rows) for rows in result.local_rounds),
            "demotion": len(result.demotion_rows),
            "refinement": len(result.refined_rows),
            "validation": len(result.validation_rows),
        },
        "refinement_blocks": list(result.refinement_blocks),
        "budget": {
            "weight_numerator": audit.weight_numerator,
            "weight_denominator": audit.weight_denominator,
            "activation_numerator": audit.activation_numerator,
            "activation_denominator": audit.activation_denominator,
            "average_weight_bits": audit.average_weight_bits,
            "average_activation_bits": audit.average_activation_bits,
            "weight_feasible": audit.weight_feasible,
            "activation_feasible": audit.activation_feasible,
        },
        "artifacts": stem_runner._artifact_hashes(root),
    }
    write_json(root / "manifest.json", manifest)
    if stem_runner._artifact_hashes(root) != manifest["artifacts"]:
        raise RuntimeError("staging artifact hashes changed before publication")
    root.rename(final)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    parser.add_argument("--beam-width", type=int, required=True)
    parser.add_argument("--joint-measured-limit", type=int, required=True)
    parser.add_argument("--local-round-limit", type=int, required=True)
    parser.add_argument("--refinement-block-limit", type=int, required=True)
    parser.add_argument("--refinement-width", type=int, required=True)
    parser.add_argument("--refinement-measured-limit", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    protocol = SearchProtocol(
        args.beam_width,
        args.joint_measured_limit,
        args.local_round_limit,
        args.refinement_block_limit,
        args.refinement_width,
        args.refinement_measured_limit,
    )
    validate_production_protocol(protocol)
    if not math.isfinite(args.fold_max_error) or args.fold_max_error <= 0.0:
        raise ValueError("fold error threshold must be finite and positive")
    devices = tuple(
        value.strip() for value in str(args.devices).split(",")
        if value.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("CUDA devices must be nonempty and unique")
    if any(not value.startswith("cuda:") for value in devices):
        raise ValueError("mixed-bit search requires explicit CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checkpoint = Path(args.checkpoint)
    calibration_indices_path = Path(args.calibration_indices)
    calibration_metadata_path = Path(args.calibration_metadata)
    evaluation_protocol_path = Path(args.evaluation_protocol)
    calibration_payload = json.loads(calibration_indices_path.read_text())
    calibration_metadata = json.loads(calibration_metadata_path.read_text())
    evaluation_metadata = json.loads(evaluation_protocol_path.read_text())
    index_protocol = stem_runner.index_protocol(
        calibration_payload, evaluation_metadata)
    if int(args.seed) != index_protocol.seed:
        raise ValueError("runner seed differs from evaluation protocol")
    checkpoint_sha256 = stem_runner._sha256(checkpoint)
    stem_runner._validate_source_metadata(
        args, calibration_metadata, evaluation_metadata, checkpoint_sha256)
    output = Path(args.out_dir)
    staging = validate_output_directories(output)
    staging.mkdir(parents=True)
    prediction_root = staging / "predictions"
    prediction_root.mkdir()
    args.device = coordinator_device(devices)
    saved_args = stem_runner._saved_args(args)
    trainset = calibration_dataset(saved_args)
    valset = evaluation_dataset(saved_args)
    if max(index_protocol.calibration_indices) >= len(trainset):
        raise ValueError("calibration index exceeds the train split")
    if max(index_protocol.evaluation_indices) >= len(valset):
        raise ValueError("evaluation index exceeds the validation split")
    workers = tuple(CSPNEvaluator(
        saved_args,
        checkpoint,
        trainset,
        valset,
        index_protocol.calibration_indices,
        index_protocol.evaluation_indices,
        index_protocol.seed,
        torch.device(device),
        args.fold_max_error,
        prediction_root,
    ) for device in devices)
    evaluator = ParallelEvaluator(workers)
    registry = expected_registry()
    result = run_search(protocol, registry, None, evaluator)
    basis = evaluator.cost_basis()
    evaluator.close()
    publish_search_result(
        staging, output, result, registry, basis, protocol)


if __name__ == "__main__":
    main()
