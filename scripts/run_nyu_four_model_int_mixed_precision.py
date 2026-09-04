#!/usr/bin/env python3
"""Measured constrained INT mixed-precision search for official SPN models."""

from __future__ import annotations

import argparse
from argparse import Namespace
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from spn_quant.constrained_mixed_precision import (
    balanced_knee,
    MeasuredCandidate,
    PrecisionAssignment,
    PrecisionCosts,
    feasible_pareto_frontier,
    weighted_average_bits,
)
from spn_quant.model_contracts import QuantizationModelContract

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_model_p3t3_search import (  # noqa: E402
    HardDeploymentP3T3Evaluator,
    HardDeploymentSettings,
)
from scripts import run_nyu_rtn_quantization as rtn_runner  # noqa: E402
from spn_quant.model_contracts import (  # noqa: E402
    build_model_quantization_contract,
)


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")


FACTORIAL_BITS = (
    (6, 8), (8, 6), (6, 6), (4, 8),
    (8, 4), (4, 6), (6, 4), (4, 4),
)


def measure_unit_costs(model: nn.Module,
                       contract: QuantizationModelContract,
                       model_args: Tuple[Any, ...]) -> PrecisionCosts:
    modules = dict(model.named_modules())
    member_units = dict(
        (member, unit.name) for unit in contract.search_units
        for member in unit.members)
    unit_kinds = dict(
        (unit.name, unit.kind) for unit in contract.search_units)
    weight_macs = dict((unit.name, 0) for unit in contract.search_units)
    activation_elements = dict(
        (unit.name, 0) for unit in contract.search_units)
    call_counts = dict((member, 0) for member in member_units)
    handles = []

    def hook_for(name):
        def hook(module, inputs, output):
            if not inputs or not torch.is_tensor(inputs[0]) or \
                    not torch.is_tensor(output):
                raise TypeError("precision cost hook requires tensor I/O: %s" %
                                name)
            batch = int(inputs[0].shape[0])
            if batch <= 0:
                raise ValueError("precision cost hook requires a batch")
            unit = member_units[name]
            activation = output if unit_kinds[unit] == "attention_qkv" \
                else inputs[0]
            activation_elements[unit] += int(activation.numel()) // batch
            if isinstance(module, nn.Conv2d):
                operations = int(output.numel()) // batch * \
                    (module.in_channels // module.groups) * \
                    module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.ConvTranspose2d):
                operations = int(inputs[0].numel()) // batch * \
                    (module.out_channels // module.groups) * \
                    module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Linear):
                operations = int(output.numel()) // batch * module.in_features
            else:
                raise TypeError("unsupported precision cost module: %s" % name)
            weight_macs[unit] += int(operations)
            call_counts[name] += 1
        return hook

    for name in member_units:
        if name not in modules:
            raise KeyError("precision cost module is missing: %s" % name)
        handles.append(modules[name].register_forward_hook(hook_for(name)))
    try:
        with torch.no_grad():
            model(*model_args)
    finally:
        for handle in handles:
            handle.remove()
    missing = tuple(name for name in member_units if call_counts[name] == 0)
    if missing:
        raise RuntimeError("precision cost modules did not execute: %s" %
                           (missing,))
    return PrecisionCosts(
        weight_macs=tuple((unit.name, weight_macs[unit.name])
                          for unit in contract.search_units),
        activation_elements=tuple(
            (unit.name, activation_elements[unit.name])
            for unit in contract.search_units),
    )


@dataclass(frozen=True)
class SearchSettings:
    maximum_relative_loss: float
    anchor_headroom_loss: float
    qat_candidate_loss: float
    beam_width: int
    maximum_depth: int

    def __post_init__(self) -> None:
        values = (
            self.maximum_relative_loss,
            self.anchor_headroom_loss,
            self.qat_candidate_loss,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0
               for value in values):
            raise ValueError("search loss limits must be finite and nonnegative")
        if not self.anchor_headroom_loss <= self.maximum_relative_loss < \
                self.qat_candidate_loss:
            raise ValueError("search loss limits are not ordered")
        if int(self.beam_width) <= 0 or int(self.maximum_depth) <= 0:
            raise ValueError("beam dimensions must be positive")


@dataclass(frozen=True)
class CandidateRecord:
    phase: str
    candidate: MeasuredCandidate
    sample_count: int
    finite_positive: bool
    reproducible: bool
    propagation_valid: bool
    owner_counts_valid: bool

    @property
    def valid(self) -> bool:
        return self.finite_positive and self.reproducible and \
            self.propagation_valid and self.owner_counts_valid


@dataclass(frozen=True)
class ConstrainedSearchResult:
    model_name: str
    status: str
    reference_pooled_rmse: float
    reference_sample_count: int
    anchor: Optional[MeasuredCandidate]
    records: Tuple[CandidateRecord, ...]
    pareto_frontier: Tuple[MeasuredCandidate, ...]
    qat_candidates: Tuple[MeasuredCandidate, ...]
    maximum_relative_loss: float


@dataclass(frozen=True)
class _RegistryView:
    blocks: Tuple[str, ...]


def uniform_assignment(contract: QuantizationModelContract,
                       weight_bits: int,
                       activation_bits: int) -> PrecisionAssignment:
    units = contract.search_units
    if not units:
        raise ValueError("uniform assignment requires precision search units")
    return PrecisionAssignment(
        weight_bits=tuple(
            (unit.name, max(int(weight_bits), unit.minimum_weight_bits))
            for unit in units),
        activation_bits=tuple(
            (unit.name, max(int(activation_bits),
                            unit.minimum_activation_bits))
            for unit in units),
        scale_policies=tuple((unit.name, unit.scale_policy) for unit in units),
        expected_units=tuple(unit.name for unit in units),
    )


def _replace_unit(assignment: PrecisionAssignment, name: str,
                  weight_bits: int, activation_bits: int,
                  fp16: bool,
                  contract: QuantizationModelContract) -> PrecisionAssignment:
    if name not in assignment.expected_units:
        raise KeyError("unknown precision unit: %s" % name)
    units = dict((unit.name, unit) for unit in contract.search_units)
    unit = units[name]
    if fp16:
        if not unit.allow_fp16 or (int(weight_bits), int(activation_bits)) != \
                (16, 16):
            raise ValueError("FP16 is not permitted for unit: %s" % name)
    elif int(weight_bits) < unit.minimum_weight_bits or \
            int(activation_bits) < unit.minimum_activation_bits:
        raise ValueError("precision is below unit minimum: %s" % name)
    weights = dict(assignment.weight_bits)
    activations = dict(assignment.activation_bits)
    weights[name] = int(weight_bits)
    activations[name] = int(activation_bits)
    fp16_units = tuple(
        unit for unit in assignment.expected_units
        if (unit == name and fp16) or
        (unit != name and unit in set(assignment.fp16_units)))
    return PrecisionAssignment(
        weight_bits=tuple((unit, weights[unit])
                          for unit in assignment.expected_units),
        activation_bits=tuple((unit, activations[unit])
                              for unit in assignment.expected_units),
        scale_policies=assignment.scale_policies,
        expected_units=assignment.expected_units,
        fp16_units=fp16_units,
    )


def promote_fp16(assignment: PrecisionAssignment,
                 contract: QuantizationModelContract,
                 unit_name: str) -> PrecisionAssignment:
    units = dict((unit.name, unit) for unit in contract.search_units)
    unit = units[unit_name]
    if not unit.allow_fp16:
        raise ValueError("FP16 is not permitted for unit: %s" % unit_name)
    return _replace_unit(assignment, unit_name, 16, 16, True, contract)


def single_unit_factorial_assignments(
        contract: QuantizationModelContract,
        anchor: PrecisionAssignment) -> Tuple[Tuple[str, PrecisionAssignment], ...]:
    rows = []
    for unit in contract.search_units:
        if unit.name in set(anchor.fp16_units):
            pairs = ((8, 8), (6, 6))
        else:
            pairs = FACTORIAL_BITS
        for weight_bits, activation_bits in pairs:
            if weight_bits < unit.minimum_weight_bits or \
                    activation_bits < unit.minimum_activation_bits:
                continue
            name = "SINGLE_%s_W%dA%d" % (
                unit.name, weight_bits, activation_bits)
            rows.append((name, _replace_unit(
                anchor, unit.name, weight_bits, activation_bits, False,
                contract)))
    return tuple(rows)


def _fp16_fractions(assignment: PrecisionAssignment,
                    costs: PrecisionCosts) -> Tuple[float, float]:
    fp16 = set(assignment.fp16_units)
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    weight = sum(weight_costs[name] for name in fp16) / float(
        sum(weight_costs.values()))
    activation = sum(activation_costs[name] for name in fp16) / float(
        sum(activation_costs.values()))
    return weight, activation


def _measure(candidate_id: str, phase: str,
             assignment: PrecisionAssignment,
             evaluator: Any, costs: PrecisionCosts,
             reference_rmse: float,
             reference_samples: int) -> CandidateRecord:
    result = evaluator.evaluate(candidate_id, assignment)
    if int(result["sample_count"]) != int(reference_samples):
        raise ValueError("candidate sample count differs from FP32 reference")
    pooled_rmse = float(result["pooled_rmse"])
    average_weight, average_activation = weighted_average_bits(
        assignment, costs)
    fp16_weight, fp16_activation = _fp16_fractions(assignment, costs)
    candidate = MeasuredCandidate(
        candidate_id=str(candidate_id),
        assignment=assignment,
        pooled_rmse=pooled_rmse,
        reference_pooled_rmse=reference_rmse,
        average_weight_bits=average_weight,
        average_activation_bits=average_activation,
        fp16_mac_fraction=fp16_weight,
        fp16_activation_fraction=fp16_activation,
    )
    flags = tuple(result[name] for name in (
        "finite_positive", "reproducible", "propagation_valid",
        "owner_counts_valid"))
    if any(type(flag) is not bool for flag in flags):
        raise TypeError("candidate validity fields must be bool")
    return CandidateRecord(
        phase=phase,
        candidate=candidate,
        sample_count=int(result["sample_count"]),
        finite_positive=flags[0],
        reproducible=flags[1],
        propagation_valid=flags[2],
        owner_counts_valid=flags[3],
    )


def _next_precision(bits: int, minimum: int) -> Optional[int]:
    levels = tuple(level for level in (4, 6, 8) if level >= int(minimum))
    lower = tuple(level for level in levels if level < int(bits))
    return max(lower) if lower else None


def _demotions(contract: QuantizationModelContract,
               assignment: PrecisionAssignment,
               prefix: str) -> Tuple[Tuple[str, PrecisionAssignment], ...]:
    units = dict((unit.name, unit) for unit in contract.search_units)
    weights = dict(assignment.weight_bits)
    activations = dict(assignment.activation_bits)
    rows = []
    for name in assignment.expected_units:
        unit = units[name]
        if name in set(assignment.fp16_units):
            rows.append(("%s_%s_W8A8" % (prefix, name),
                         _replace_unit(
                             assignment, name, 8, 8, False, contract)))
            continue
        next_weight = _next_precision(weights[name], unit.minimum_weight_bits)
        next_activation = _next_precision(
            activations[name], unit.minimum_activation_bits)
        if next_weight is not None:
            rows.append(("%s_%s_W%d" % (prefix, name, next_weight),
                         _replace_unit(assignment, name, next_weight,
                                       activations[name], False, contract)))
        if next_activation is not None:
            rows.append(("%s_%s_A%d" % (prefix, name, next_activation),
                         _replace_unit(assignment, name, weights[name],
                                       next_activation, False, contract)))
    return tuple(rows)


def interaction_assignment(
        contract: QuantizationModelContract,
        anchor: PrecisionAssignment,
        left: str,
        right: str) -> Tuple[str, PrecisionAssignment]:
    if left == right:
        raise ValueError("interaction units must differ")
    units = dict((unit.name, unit) for unit in contract.search_units)
    left_unit = units[left]
    right_unit = units[right]
    left_bits = (
        max(6, left_unit.minimum_weight_bits),
        max(6, left_unit.minimum_activation_bits),
    )
    right_bits = (
        max(6, right_unit.minimum_weight_bits),
        max(6, right_unit.minimum_activation_bits),
    )
    assignment = _replace_unit(
        anchor, left, left_bits[0], left_bits[1], False, contract)
    assignment = _replace_unit(
        assignment, right, right_bits[0], right_bits[1], False, contract)
    candidate_id = "INTERACTION_%s_W%dA%d_%s_W%dA%d" % (
        left, left_bits[0], left_bits[1],
        right, right_bits[0], right_bits[1])
    return candidate_id, assignment


def run_constrained_search(
        contract: QuantizationModelContract,
        costs: PrecisionCosts,
        evaluator: Any,
        settings: SearchSettings,
        boundary_order: Sequence[str],
        interaction_pairs: Sequence[Tuple[str, str]],
        phase: str) -> ConstrainedSearchResult:
    if phase not in ("anchors", "ptq-search"):
        raise ValueError("search phase must be anchors or ptq-search")
    reference = evaluator.reference()
    reference_rmse = float(reference["pooled_rmse"])
    reference_samples = int(reference["sample_count"])
    if not math.isfinite(reference_rmse) or reference_rmse <= 0.0:
        raise ValueError("FP32 pooled RMSE must be finite and positive")
    if reference_samples != 64:
        raise ValueError("strict evaluation requires 64 reference samples")

    records = []
    anchor_assignment = uniform_assignment(contract, 8, 8)
    uniform = _measure(
        "UNIFORM_W8A8", "anchor", anchor_assignment,
        evaluator, costs, reference_rmse, reference_samples)
    records.append(uniform)
    anchor = uniform if uniform.valid and \
        uniform.candidate.relative_loss <= settings.anchor_headroom_loss else None
    feasible_anchor = uniform if uniform.valid and \
        uniform.candidate.relative_loss <= settings.maximum_relative_loss else None
    if anchor is None:
        for boundary in boundary_order:
            anchor_assignment = promote_fp16(
                anchor_assignment, contract, boundary)
            record = _measure(
                "ANCHOR_FP16_%s" % boundary, "anchor", anchor_assignment,
                evaluator, costs, reference_rmse, reference_samples)
            records.append(record)
            if record.valid and record.candidate.relative_loss <= \
                    settings.maximum_relative_loss:
                feasible_anchor = record
            if record.valid and record.candidate.relative_loss <= \
                    settings.anchor_headroom_loss:
                anchor = record
                break
    if anchor is None:
        anchor = feasible_anchor
    if anchor is None:
        return ConstrainedSearchResult(
            model_name=contract.model_name,
            status="infeasible",
            reference_pooled_rmse=reference_rmse,
            reference_sample_count=reference_samples,
            anchor=None,
            records=tuple(records),
            pareto_frontier=(),
            qat_candidates=(),
            maximum_relative_loss=settings.maximum_relative_loss,
        )
    if phase == "anchors":
        valid_candidates = tuple(
            record.candidate for record in records if record.valid)
        frontier = feasible_pareto_frontier(
            valid_candidates, settings.maximum_relative_loss)
        return ConstrainedSearchResult(
            model_name=contract.model_name,
            status="feasible" if frontier else "infeasible",
            reference_pooled_rmse=reference_rmse,
            reference_sample_count=reference_samples,
            anchor=anchor.candidate,
            records=tuple(records),
            pareto_frontier=frontier,
            qat_candidates=(),
            maximum_relative_loss=settings.maximum_relative_loss,
        )

    measured_payloads = {}
    for record in records:
        identity = json.dumps(
            record.candidate.assignment.canonical_payload(), sort_keys=True)
        measured_payloads[identity] = record

    def measure_new(candidate_id, phase, assignment):
        identity = json.dumps(
            assignment.canonical_payload(), sort_keys=True)
        if identity in measured_payloads:
            return measured_payloads[identity]
        record = _measure(
            candidate_id, phase, assignment, evaluator, costs,
            reference_rmse, reference_samples)
        records.append(record)
        measured_payloads[identity] = record
        return record

    for candidate_id, assignment in single_unit_factorial_assignments(
            contract, anchor.candidate.assignment):
        measure_new(candidate_id, "single", assignment)

    for left, right in interaction_pairs:
        candidate_id, interaction = interaction_assignment(
            contract, anchor.candidate.assignment, left, right)
        measure_new(candidate_id, "interaction", interaction)

    beam = (anchor.candidate.assignment,)
    for depth in range(1, settings.maximum_depth + 1):
        next_rows = {}
        for index, assignment in enumerate(beam):
            for candidate_id, candidate_assignment in _demotions(
                    contract, assignment, "BEAM_D%d_B%d" % (depth, index)):
                record = measure_new(
                    candidate_id, "beam", candidate_assignment)
                if record.valid and \
                        record.candidate.relative_loss <= \
                        settings.qat_candidate_loss:
                    identity = json.dumps(
                        record.candidate.assignment.canonical_payload(),
                        sort_keys=True)
                    next_rows[identity] = record.candidate
        if not next_rows:
            break
        ordered_rows = sorted(next_rows.values(), key=lambda row: (
            row.average_weight_bits + row.average_activation_bits,
            row.pooled_rmse,
            row.candidate_id,
        ))
        beam = tuple(row.assignment
                     for row in ordered_rows[:settings.beam_width])

    valid_candidates = tuple(record.candidate for record in records
                             if record.valid)
    frontier = feasible_pareto_frontier(
        valid_candidates, settings.maximum_relative_loss)
    qat_candidates = tuple(sorted(
        (candidate for candidate in valid_candidates
         if candidate.relative_loss <= settings.qat_candidate_loss),
        key=lambda row: (
            row.average_weight_bits + row.average_activation_bits,
            row.pooled_rmse,
            row.candidate_id,
        )))[:3]
    return ConstrainedSearchResult(
        model_name=contract.model_name,
        status="feasible" if frontier else "infeasible",
        reference_pooled_rmse=reference_rmse,
        reference_sample_count=reference_samples,
        anchor=anchor.candidate,
        records=tuple(records),
        pareto_frontier=frontier,
        qat_candidates=qat_candidates,
        maximum_relative_loss=settings.maximum_relative_loss,
    )


def _candidate_payload(candidate: MeasuredCandidate) -> Mapping[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "pooled_rmse": candidate.pooled_rmse,
        "relative_loss": candidate.relative_loss,
        "average_weight_bits": candidate.average_weight_bits,
        "average_activation_bits": candidate.average_activation_bits,
        "fp16_mac_fraction": candidate.fp16_mac_fraction,
        "fp16_activation_fraction": candidate.fp16_activation_fraction,
        "assignment": candidate.assignment.canonical_payload(),
    }


def reference_artifact(result: ConstrainedSearchResult) -> Mapping[str, Any]:
    return {
        "pooled_rmse": result.reference_pooled_rmse,
        "sample_count": result.reference_sample_count,
    }


def load_balanced_ptq_candidate(
        ptq_root: Path,
        contract: QuantizationModelContract) -> MeasuredCandidate:
    root = Path(ptq_root)
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8"))
    candidates = json.loads(
        (root / "candidate_assignments.json").read_text(encoding="utf-8"))
    if manifest["model"] != contract.model_name or \
            candidates["model"] != contract.model_name:
        raise ValueError("PTQ audit model differs from contract")
    published = tuple(str(value)
                      for value in manifest["pareto_candidate_ids"])
    if not published or len(published) != len(set(published)):
        raise ValueError("PTQ audit Pareto identities are invalid")
    rows = tuple(
        row for row in candidates["candidates"]
        if str(row["candidate_id"]) in set(published))
    if len(rows) != len(published) or \
            set(str(row["candidate_id"]) for row in rows) != set(published):
        raise ValueError("PTQ audit Pareto candidate coverage differs")
    units = tuple(unit.name for unit in contract.search_units)
    reference = float(manifest["reference_pooled_rmse"])
    measured = []
    for row in rows:
        payload = row["assignment"]
        assignment = PrecisionAssignment(
            weight_bits=tuple(
                (name, int(payload["weight_bits"][name])) for name in units),
            activation_bits=tuple(
                (name, int(payload["activation_bits"][name]))
                for name in units),
            scale_policies=tuple(
                (name, str(payload["scale_policies"][name]))
                for name in units),
            expected_units=units,
            fp16_units=tuple(str(name) for name in payload["fp16_units"]),
        )
        if assignment.canonical_payload() != payload:
            raise ValueError("PTQ audit assignment is not canonical")
        measured.append(MeasuredCandidate(
            candidate_id=str(row["candidate_id"]),
            assignment=assignment,
            pooled_rmse=float(row["pooled_rmse"]),
            reference_pooled_rmse=reference,
            average_weight_bits=float(row["average_weight_bits"]),
            average_activation_bits=float(row["average_activation_bits"]),
            fp16_mac_fraction=float(row["fp16_mac_fraction"]),
            fp16_activation_fraction=float(
                row["fp16_activation_fraction"]),
        ))
    return balanced_knee(tuple(measured))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "phase", "candidate_id", "pooled_rmse", "relative_loss",
        "average_weight_bits", "average_activation_bits",
        "fp16_mac_fraction", "fp16_activation_fraction",
        "finite_positive", "reproducible", "propagation_valid",
        "owner_counts_valid", "valid",
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((name, row[name]) for name in fieldnames))


def _write_audit_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = tuple(rows)
    if not rows:
        raise ValueError("audit CSV requires at least one row")
    available = set(name for row in rows for name in row)
    preferred = (
        "candidate_id", "config", "sample_index", "module", "kind",
        "bits", "calls", "signal", "iteration",
    )
    fieldnames = tuple(name for name in preferred if name in available) + \
        tuple(sorted(available - set(preferred)))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(
                (name, row[name] if name in row else "")
                for name in fieldnames))


def write_evaluation_audit_artifacts(
        output: Path, evaluation: Mapping[str, Any]) -> None:
    required = {
        "candidate_id", "sample_rows", "signal_rows",
        "effective_weight_bits", "effective_activation_bits",
        "owner_call_counts",
    }
    if not required <= set(evaluation):
        raise ValueError("selected evaluation audit fields are incomplete")
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError(
            "selected evaluation audit directory is missing: %s" % root)
    sample_rows = tuple(dict(row) for row in evaluation["sample_rows"])
    signal_rows = tuple(dict(row) for row in evaluation["signal_rows"])
    state_rows = tuple(
        row for row in signal_rows if row["signal"] == "state")
    call_counts = dict(evaluation["owner_call_counts"])
    activation_bits = dict(evaluation["effective_activation_bits"])
    if set(call_counts) != set(activation_bits):
        raise ValueError(
            "effective activation and execution-count coverage differs")
    effective_rows = tuple({
        "module": str(name),
        "kind": "weight",
        "bits": int(bits),
        "calls": "",
    } for name, bits in evaluation["effective_weight_bits"]) + tuple({
        "module": str(owner[0]),
        "kind": str(owner[1]),
        "bits": int(bits),
        "calls": int(call_counts[owner]),
    } for owner, bits in evaluation["effective_activation_bits"])
    _write_audit_csv(root / "sample_metrics.csv", sample_rows)
    _write_audit_csv(root / "signal_metrics.csv", signal_rows)
    _write_audit_csv(
        root / "propagation_state_metrics.csv", state_rows)
    _write_audit_csv(
        root / "effective_quantization.csv", effective_rows)


def write_search_artifacts(output: Path,
                           result: ConstrainedSearchResult) -> None:
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("search output directory is missing: %s" % root)
    records = tuple({
        "phase": record.phase,
        "candidate_id": record.candidate.candidate_id,
        "pooled_rmse": record.candidate.pooled_rmse,
        "relative_loss": record.candidate.relative_loss,
        "average_weight_bits": record.candidate.average_weight_bits,
        "average_activation_bits": record.candidate.average_activation_bits,
        "fp16_mac_fraction": record.candidate.fp16_mac_fraction,
        "fp16_activation_fraction": record.candidate.fp16_activation_fraction,
        "finite_positive": record.finite_positive,
        "reproducible": record.reproducible,
        "propagation_valid": record.propagation_valid,
        "owner_counts_valid": record.owner_counts_valid,
        "valid": record.valid,
    } for record in result.records)
    phase_files = (
        ("anchor_summary.csv", ("anchor",)),
        ("single_module_ablation.csv", ("single",)),
        ("interaction_ablation.csv", ("interaction",)),
    )
    for filename, phases in phase_files:
        _write_csv(root / filename, tuple(
            row for row in records if row["phase"] in phases))
    frontier_ids = set(row.candidate_id for row in result.pareto_frontier)
    _write_csv(root / "pareto_ptq.csv", tuple(
        row for row in records if row["candidate_id"] in frontier_ids))
    assignments = {
        "model": result.model_name,
        "candidates": [_candidate_payload(record.candidate)
                       for record in result.records],
    }
    (root / "candidate_assignments.json").write_text(
        json.dumps(assignments, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest = {
        "format_version": 1,
        "model": result.model_name,
        "status": result.status,
        "reference_pooled_rmse": result.reference_pooled_rmse,
        "reference_sample_count": result.reference_sample_count,
        "maximum_relative_loss": result.maximum_relative_loss,
        "anchor": None if result.anchor is None else
            _candidate_payload(result.anchor),
        "pareto_candidate_ids": [row.candidate_id
                                 for row in result.pareto_frontier],
        "qat_candidate_ids": [row.candidate_id
                              for row in result.qat_candidates],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_run_config(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version", "model_source_config", "output_root", "devices",
        "search", "boundary_order", "interaction_pairs", "hard_deployment",
        "environments", "qat",
    }
    if set(payload) != required:
        raise ValueError("four-model run configuration fields changed")
    if int(payload["format_version"]) != 1:
        raise ValueError("unsupported four-model run configuration version")
    if tuple(payload["devices"]) != MODEL_ORDER or \
            tuple(payload["boundary_order"]) != MODEL_ORDER or \
            tuple(payload["interaction_pairs"]) != MODEL_ORDER or \
            tuple(payload["environments"]) != MODEL_ORDER:
        raise ValueError("four-model configuration order changed")
    return payload


def _model_source(config_path: Path,
                  payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source_path = Path(payload["model_source_config"])
    if not source_path.is_absolute():
        source_path = config_path.resolve().parent.parent / source_path
    source = json.loads(source_path.read_text(encoding="utf-8"))
    models = source["models"]
    if tuple(models) != MODEL_ORDER:
        raise ValueError("source model configuration order changed")
    return source


def _runtime_args(model_payload: Mapping[str, Any], device: str) -> Namespace:
    return Namespace(
        model=str(model_payload["model"]),
        run_dir=Path(model_payload["run_dir"]),
        checkpoint=Path(model_payload["checkpoint"]),
        expected_architecture_class=str(
            model_payload["expected_architecture_class"]),
        required_cuda_extension=str(model_payload["required_cuda_extension"]),
        propagation_iterations=int(model_payload["propagation_iterations"]),
        data_root=Path(model_payload["data_root"]),
        device=str(device),
        checkpoint_architecture=str(model_payload["checkpoint_architecture"]),
        native_cuda_operator=model_payload["native_cuda_operator"],
    )


def _hard_settings(model_payload: Mapping[str, Any], device: str,
                   hard: Mapping[str, Any]) -> HardDeploymentSettings:
    return HardDeploymentSettings(
        device=str(device),
        calibration_metadata=Path(model_payload["calibration_metadata"]),
        calibration_count=int(model_payload["calibration_count"]),
        evaluation_indices=tuple(
            int(index) for index in model_payload["evaluation_indices"]),
        base_weight_bits=8,
        base_activation_bits=8,
        promotion_weight_bits=8,
        promotion_activation_bits=8,
        fold_conv_bn=bool(hard["fold_conv_bn"]),
        fold_max_error=float(hard["fold_max_error"]),
        joint_clip_factors=tuple(float(value)
                                 for value in hard["joint_clip_factors"]),
        joint_search_rounds=int(hard["joint_search_rounds"]),
        joint_cache_sample_limit=int(hard["joint_cache_sample_limit"]),
        joint_cache_byte_limit=int(hard["joint_cache_byte_limit"]),
    )


def _search_settings(payload: Mapping[str, Any]) -> SearchSettings:
    return SearchSettings(
        maximum_relative_loss=float(payload["maximum_relative_loss"]),
        anchor_headroom_loss=float(payload["anchor_headroom_loss"]),
        qat_candidate_loss=float(payload["qat_candidate_loss"]),
        beam_width=int(payload["beam_width"]),
        maximum_depth=int(payload["maximum_depth"]),
    )


def configure_runtime_execution(runtime) -> None:
    threads = int(runtime.saved_args.torch_threads)
    if threads <= 0:
        raise ValueError("official runtime torch threads must be positive")
    torch.set_num_threads(threads)


def run_official_model(config_path: Path, model_name: str,
                       output: Path, phase: str) -> ConstrainedSearchResult:
    config_path = Path(config_path)
    config = _load_run_config(config_path)
    if model_name not in MODEL_ORDER:
        raise ValueError("unsupported four-model search target: %s" % model_name)
    source = _model_source(config_path, config)
    model_payload = source["models"][model_name]
    device = str(config["devices"][model_name])
    runtime = NYUModelRuntime.from_args(_runtime_args(model_payload, device))
    configure_runtime_execution(runtime)
    evaluator = None
    try:
        model = runtime.build_model(runtime.device)
        contract = build_model_quantization_contract(model_name, model)
        calibration = json.loads(Path(
            model_payload["calibration_metadata"]).read_text(encoding="utf-8"))
        first_index = int(calibration["calibration_indices"][0])
        trainset = runtime.build_dataset("train")
        sample = rtn_runner.seeded_sample(
            trainset, first_index, int(runtime.saved_args.seed))
        batch = rtn_runner.batch_from_sample(sample)
        model_args, ground_truth = runtime.model_input(batch, runtime.device)
        del ground_truth
        costs = measure_unit_costs(model, contract, model_args)
        evaluator = HardDeploymentP3T3Evaluator(
            runtime, model, contract, _RegistryView(contract.block_names),
            _hard_settings(model_payload, device, config["hard_deployment"]),
        )
        result = run_constrained_search(
            contract=contract,
            costs=costs,
            evaluator=evaluator,
            settings=_search_settings(config["search"]),
            boundary_order=tuple(config["boundary_order"][model_name]),
            interaction_pairs=tuple(
                tuple(pair) for pair in config["interaction_pairs"][model_name]),
            phase=phase,
        )
        root = Path(output)
        root.mkdir(parents=False, exist_ok=False)
        write_search_artifacts(root, result)
        (root / "fp32_reference.json").write_text(
            json.dumps(reference_artifact(result), indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(model_payload["checkpoint"]).resolve()
        manifest["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": _file_sha256(checkpoint),
        }
        manifest["architecture_class"] = type(model).__name__
        manifest["calibration_indices"] = list(
            calibration["calibration_indices"])
        manifest["evaluation_indices"] = list(
            model_payload["evaluation_indices"])
        manifest["propagation_iterations"] = int(
            model_payload["propagation_iterations"])
        manifest["propagation_dtype"] = "fp16"
        manifest["precision_costs"] = {
            "weight_macs": [list(row) for row in costs.weight_macs],
            "activation_elements": [list(row)
                                    for row in costs.activation_elements],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return result
    finally:
        if evaluator is not None:
            evaluator.close()
        runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict four-model INT mixed-precision search")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("anchors", "ptq-search"), required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    result = run_official_model(
        args.config, args.model, args.output, args.phase)
    print("model=%s status=%s pareto=%d" % (
        result.model_name, result.status, len(result.pareto_frontier)))


if __name__ == "__main__":
    main()
