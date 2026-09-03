#!/usr/bin/env python3
"""Run measured, model-relative P3/T3 mixed-precision selection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_model_hawq_trace import (  # noqa: E402
    CheckpointIdentity,
    capture_checkpoint_identity,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    SymmetricActivationQuantizer,
    prepare_hardware_model,
)
from scripts.hardware_merge_adapters import (  # noqa: E402
    CallIndexedConcatConvAdapter,
)
from scripts import run_nyu_rtn_quantization as rtn_runner  # noqa: E402
from spn_quant import mixed_precision  # noqa: E402
from spn_quant.adapters.completionformer_joint import (  # noqa: E402
    CompletionFormerJointAdapter,
)
from spn_quant.experiment_config import (  # noqa: E402
    MODEL_ORDER,
    load_selected_quantization_config,
)
from spn_quant.model_contracts import (  # noqa: E402
    QuantizationModelContract,
    build_model_quantization_contract,
)
from spn_quant.propagation import (  # noqa: E402
    PropagationQuantConfig,
    install_propagation_adapter,
    propagation_projection_outputs,
)
from spn_quant.qdrop_targets import resolve_qdrop_targets  # noqa: E402


P3T3Candidate = mixed_precision.P3T3Candidate
P3T3CandidateResult = mixed_precision.P3T3CandidateResult
P3T3SampleEvidence = mixed_precision.P3T3SampleEvidence
P3T3SearchResult = mixed_precision.P3T3SearchResult

P3_T3_ASSIGNMENT_VERSION = 3
P3_T3_ASSIGNMENT_KIND = "nyu_model_p3_t3_assignment"


@dataclass(frozen=True)
class HardDeploymentSettings:
    device: str
    calibration_metadata: Path
    calibration_count: int
    evaluation_indices: Tuple[int, ...]
    base_weight_bits: int
    base_activation_bits: int
    promotion_weight_bits: int
    promotion_activation_bits: int
    fold_conv_bn: bool
    fold_max_error: float
    joint_clip_factors: Tuple[float, ...]
    joint_search_rounds: int
    joint_cache_sample_limit: int
    joint_cache_byte_limit: int


@dataclass(frozen=True)
class RunnerDependencies:
    runtime_factory: Callable
    contract_builder: Callable
    evaluator_factory: Callable


class _SiteSymmetricActivationQuantizer(object):
    """Attach a contract site identity to hardware-aligned symmetric QDQ."""

    def __init__(self, site: str, bits: int, maximum: float) -> None:
        self.site = str(site)
        self.quantizer = SymmetricActivationQuantizer(bits, maximum)
        self.calls = 0
        self.numel = 0
        self.zero_codes = 0
        self.saturated_codes = 0

    def quantize_with_codes(self, tensor):
        output, codes = self.quantizer.quantize_with_codes(tensor)
        self.calls += 1
        self.numel += int(codes.numel())
        self.zero_codes += int((codes == 0).sum().item())
        self.saturated_codes += int(torch.logical_or(
            codes == self.quantizer.qmin,
            codes == self.quantizer.qmax).sum().item())
        return output, codes

    def statistics(self):
        if self.calls <= 0 or self.numel <= 0:
            raise RuntimeError(
                "hard joint quantizer has no code statistics: %s" % self.site)
        return self.counter_snapshot()

    def counter_snapshot(self):
        return ({
            "module": self.site,
            "owner_kind": "joint_activation",
            "bits": self.quantizer.bits,
            "calls": self.calls,
            "numel": self.numel,
            "zero_code_count": self.zero_codes,
            "saturation_count": self.saturated_codes,
            "zero_code_rate": self.zero_codes / float(max(self.numel, 1)),
            "saturation_rate": self.saturated_codes /
                float(max(self.numel, 1)),
        },)


def _ordered_union(blocks: Sequence[str], registry) -> Tuple[str, ...]:
    selected = set(blocks)
    return tuple(block for block in registry.blocks if block in selected)


def _candidate(
        name: str,
        stage: str,
        prefix: Sequence[str],
        tail: Sequence[str],
        registry: mixed_precision.AllocationRegistry,
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int) -> P3T3Candidate:
    normalized_prefix = tuple(str(block) for block in prefix)
    normalized_tail = tuple(str(block) for block in tail)
    promoted = _ordered_union(
        normalized_prefix + normalized_tail, registry)
    assignment = mixed_precision.promoted_assignment(
        registry,
        promoted,
        base_weight_bits,
        base_activation_bits,
        promotion_weight_bits,
        promotion_activation_bits,
    )
    return P3T3Candidate(
        name=name,
        stage=stage,
        prefix=normalized_prefix,
        tail=normalized_tail,
        promoted_blocks=promoted,
        assignment=assignment,
    )


def _tail_combinations(contract, registry):
    combinations = []
    group_count = len(contract.tail_groups)
    width = max(2, len(str((1 << group_count) - 1)))
    for mask in range(1, 1 << group_count):
        groups = tuple(
            contract.tail_groups[index]
            for index in range(group_count)
            if mask & (1 << index))
        tail = _ordered_union(tuple(itertools.chain.from_iterable(groups)), registry)
        combinations.append((mask, width, tail))
    return tuple(combinations)


def build_p3_candidates(
        contract: QuantizationModelContract,
        registry: mixed_precision.AllocationRegistry,
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int) -> Tuple[P3T3Candidate, ...]:
    """Build baseline, single-block, prefix, and tail P3 candidates."""
    if contract.model_name != registry.model_name:
        raise ValueError("contract and allocation registry model mismatch")
    if contract.block_names != registry.blocks:
        raise ValueError("contract and allocation registry blocks differ")
    output = [_candidate(
        "UNIFORM_W%dA%d" % (base_weight_bits, base_activation_bits),
        "baseline", (), (), registry,
        base_weight_bits, base_activation_bits,
        promotion_weight_bits, promotion_activation_bits)]
    for index, block in enumerate(registry.blocks, 1):
        output.append(_candidate(
            "SINGLE_B%03d" % index,
            "single_block", (), (block,), registry,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits))
    for prefix_index, prefix in enumerate(contract.prefix_groups, 1):
        output.append(_candidate(
            "PREFIX_P%d" % prefix_index,
            "prefix", prefix, (), registry,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits))
    tail_combinations = _tail_combinations(contract, registry)
    for tail_mask, width, tail in tail_combinations:
        output.append(_candidate(
            "TAIL_T%0*d" % (width, tail_mask),
            "tail", (), tail, registry,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits))
    names = tuple(candidate.name for candidate in output)
    if len(names) != len(set(names)):
        raise ValueError("P3/T3 candidate names contain duplicates")
    return tuple(output)


def build_t3_candidates(
        contract: QuantizationModelContract,
        registry: mixed_precision.AllocationRegistry,
        prefix: Sequence[str],
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int) -> Tuple[P3T3Candidate, ...]:
    selected_prefix = tuple(str(block) for block in prefix)
    if selected_prefix not in contract.prefix_groups:
        raise ValueError("T3 prefix is not a declared contract prefix")
    prefix_index = contract.prefix_groups.index(selected_prefix) + 1
    output = tuple(
        _candidate(
            "INTERACTION_P%d_T%0*d" % (prefix_index, width, tail_mask),
            "interaction", selected_prefix, tail, registry,
            base_weight_bits, base_activation_bits,
            promotion_weight_bits, promotion_activation_bits)
        for tail_mask, width, tail in _tail_combinations(contract, registry))
    if len(set(candidate.name for candidate in output)) != len(output):
        raise RuntimeError("T3 candidate names contain duplicates")
    return output


def _normalized_cost(assignment, costs, base_weight_bits, base_activation_bits):
    weight_bits = dict(assignment.weight_bits)
    activation_bits = dict(assignment.activation_bits)
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    if set(weight_bits) != set(weight_costs):
        raise ValueError("weight assignment and cost coverage mismatch")
    if set(activation_bits) != set(activation_costs):
        raise ValueError("activation assignment and cost coverage mismatch")
    weight_denominator = int(base_weight_bits) * sum(weight_costs.values())
    activation_denominator = (
        int(base_activation_bits) * sum(activation_costs.values()))
    if weight_denominator <= 0 or activation_denominator <= 0:
        raise ValueError("normalized precision cost denominator must be positive")
    weight = sum(weight_bits[name] * weight_costs[name]
                 for name in weight_costs) / float(weight_denominator)
    activation = sum(activation_bits[owner] * activation_costs[owner]
                     for owner in activation_costs) / float(
                         activation_denominator)
    return weight, activation


def _measured_rows(candidates, rows, costs, base_weight_bits,
                   base_activation_bits, expected_samples,
                   reference_sample_rmse=None):
    if int(expected_samples) <= 0:
        raise ValueError("expected sample count must be positive")
    by_name = dict((candidate.name, []) for candidate in candidates)
    for row in rows:
        name = str(row["config"])
        if name not in by_name:
            raise ValueError("measured candidate coverage mismatch")
        by_name[name].append(row)

    sample_ids = None
    partial = []
    for candidate in candidates:
        selected = tuple(sorted(
            by_name[candidate.name], key=lambda row: int(row["sample_index"])))
        if any(isinstance(row["sample_index"], bool) or
               not isinstance(row["sample_index"], int)
               for row in selected):
            raise TypeError("measured sample identities must be integers")
        identities = tuple(int(row["sample_index"]) for row in selected)
        if len(selected) != int(expected_samples) or \
                len(identities) != len(set(identities)):
            raise ValueError("measured sample coverage mismatch")
        if sample_ids is None:
            sample_ids = identities
        elif identities != sample_ids:
            raise ValueError("paired measured sample identities differ")
        squared_error_sum = 0.0
        valid_pixels = 0
        sample_rmse = []
        sample_evidence = []
        flags = []
        finite = True
        for row in selected:
            if isinstance(row["squared_error_sum"], bool) or not isinstance(
                    row["squared_error_sum"], (int, float)):
                raise ValueError("measured squared error must be numeric")
            if isinstance(row["RMSE"], bool) or not isinstance(
                    row["RMSE"], (int, float)):
                raise ValueError("measured RMSE must be numeric")
            squared = float(row["squared_error_sum"])
            pixels = int(row["valid_pixels"])
            rmse = float(row["RMSE"])
            if isinstance(row["sample_index"], bool) or \
                    isinstance(row["valid_pixels"], bool) or not isinstance(
                        row["valid_pixels"], int) or pixels <= 0:
                raise ValueError("measured valid pixel count must be positive")
            validity = (
                row["prediction_finite"],
                row["propagation_valid"],
                row["reproducible"],
            )
            if not all(isinstance(value, bool) for value in validity):
                raise ValueError("measured validity flags must be booleans")
            if math.isfinite(squared) and squared < 0.0:
                raise ValueError("measured squared error must be nonnegative")
            if math.isfinite(squared) and math.isfinite(rmse) and not \
                    math.isclose(
                        rmse * rmse * pixels, squared,
                        rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("measured RMSE and squared error disagree")
            finite = finite and math.isfinite(squared) and math.isfinite(rmse)
            squared_error_sum += squared
            valid_pixels += pixels
            sample_rmse.append((int(row["sample_index"]), rmse))
            sample_evidence.append(P3T3SampleEvidence(
                sample_index=int(row["sample_index"]),
                squared_error_sum=squared,
                valid_pixels=pixels,
                prediction_finite=row["prediction_finite"],
                propagation_valid=row["propagation_valid"],
                reproducible=row["reproducible"],
            ))
            flags.append(all(validity))
        pooled_rmse = math.sqrt(squared_error_sum / float(valid_pixels)) \
            if finite and squared_error_sum >= 0.0 else float("inf")
        mean_sample_rmse = sum(value for index, value in sample_rmse) / \
            float(len(sample_rmse)) if finite else float("inf")
        weight_cost, activation_cost = _normalized_cost(
            candidate.assignment, costs,
            base_weight_bits, base_activation_bits)
        partial.append(P3T3CandidateResult(
            name=candidate.name,
            stage=candidate.stage,
            prefix=candidate.prefix,
            tail=candidate.tail,
            assignment=candidate.assignment,
            pooled_rmse=pooled_rmse,
            mean_sample_rmse=mean_sample_rmse,
            normalized_weight_cost=weight_cost,
            normalized_activation_cost=activation_cost,
            valid=finite and all(flags),
            sample_rmse=tuple(sample_rmse),
            sample_evidence=tuple(sample_evidence),
            paired_sample_differences=(),
        ))
    reference = partial[0].sample_rmse if reference_sample_rmse is None \
        else tuple(reference_sample_rmse)
    if tuple(index for index, value in reference) != sample_ids:
        raise ValueError("paired reference sample identities differ")
    baseline_samples = dict(reference)
    return tuple(replace(
        row,
        paired_sample_differences=tuple(
            value - baseline_samples[index]
            for index, value in row.sample_rmse),
    ) for row in partial)


def _dominates(left, right):
    no_worse = (
        left.pooled_rmse <= right.pooled_rmse and
        left.normalized_weight_cost <= right.normalized_weight_cost and
        left.normalized_activation_cost <= right.normalized_activation_cost)
    strictly_better = (
        left.pooled_rmse < right.pooled_rmse or
        left.normalized_weight_cost < right.normalized_weight_cost or
        left.normalized_activation_cost < right.normalized_activation_cost)
    return no_worse and strictly_better


def _prefix_knee(rows):
    stable = tuple(row for row in rows if row.stage == "prefix" and row.valid)
    if not stable:
        raise RuntimeError("P3/T3 search has no stable prefix candidate")
    frontier = tuple(
        row for row in stable
        if not any(_dominates(other, row) for other in stable if other != row))
    ordered = tuple(sorted(
        frontier,
        key=lambda row: (
            (row.normalized_weight_cost + row.normalized_activation_cost) / 2.0,
            row.pooled_rmse,
            row.prefix,
        )))
    if len(ordered) <= 2:
        return ordered[0]
    costs = tuple(
        (row.normalized_weight_cost + row.normalized_activation_cost) / 2.0
        for row in ordered)
    errors = tuple(row.pooled_rmse for row in ordered)
    cost_range = costs[-1] - costs[0]
    error_high = max(errors)
    error_low = min(errors)
    error_range = error_high - error_low
    if cost_range == 0.0 or error_range == 0.0:
        return ordered[0]
    points = tuple(
        ((cost - costs[0]) / cost_range,
         (error_high - error) / error_range)
        for cost, error in zip(costs, errors))
    x0, y0 = points[0]
    x1, y1 = points[-1]
    denominator = math.hypot(y1 - y0, x1 - x0)
    ranked = []
    for index, (x_value, y_value) in enumerate(points):
        distance = abs(
            (y1 - y0) * x_value - (x1 - x0) * y_value +
            x1 * y0 - y1 * x0) / denominator
        ranked.append((-distance, costs[index], ordered[index].prefix, ordered[index]))
    return min(ranked)[3]


def search_p3_t3(
        contract: QuantizationModelContract,
        costs: mixed_precision.CostBasis,
        evaluator: Callable[[Sequence[P3T3Candidate]], Sequence[Mapping[str, object]]],
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int,
        maximum_normalized_weight_cost: float,
        maximum_normalized_activation_cost: float,
        maximum_relative_rmse_loss: float,
        expected_samples: int) -> P3T3SearchResult:
    """Select P3 first, then measure T3 interactions for that prefix."""
    maximum_weight = float(maximum_normalized_weight_cost)
    maximum_activation = float(maximum_normalized_activation_cost)
    if not math.isfinite(maximum_weight) or maximum_weight <= 0.0 or \
            not math.isfinite(maximum_activation) or maximum_activation <= 0.0:
        raise ValueError("normalized precision budgets must be finite and positive")
    maximum_relative = float(maximum_relative_rmse_loss)
    if not math.isfinite(maximum_relative) or maximum_relative < 0.0:
        raise ValueError(
            "maximum relative RMSE loss must be finite and nonnegative")
    registry = mixed_precision.build_registry(contract, costs)
    p3_candidates = build_p3_candidates(
        contract, registry,
        base_weight_bits, base_activation_bits,
        promotion_weight_bits, promotion_activation_bits)
    p3_rows = tuple(evaluator(p3_candidates))
    p3_measured = _measured_rows(
        p3_candidates, p3_rows, costs,
        base_weight_bits, base_activation_bits, expected_samples)
    if not p3_measured[0].valid:
        raise RuntimeError("P3/T3 search requires a stable finite baseline")
    reference = tuple(sorted(
        evaluator.reference_sample_rmse(), key=lambda row: int(row[0])))
    reference_indices = tuple(index for index, value in reference)
    measured_indices = tuple(
        sample.sample_index for sample in p3_measured[0].sample_evidence)
    if reference_indices != measured_indices:
        raise ValueError("paired FP32 reference sample identities differ")
    reference_values = tuple(float(value) for index, value in reference)
    if not reference_values or any(
            not math.isfinite(value) or value < 0.0
            for value in reference_values):
        raise ValueError("paired FP32 reference RMSE values are invalid")
    reference_mean = sum(reference_values) / float(len(reference_values))
    if not math.isfinite(reference_mean) or reference_mean <= 0.0:
        raise ValueError("paired FP32 reference RMSE must be positive")
    prefix = _prefix_knee(p3_measured)
    t3_candidates = build_t3_candidates(
        contract, registry, prefix.prefix,
        base_weight_bits, base_activation_bits,
        promotion_weight_bits, promotion_activation_bits)
    t3_rows = tuple(evaluator(t3_candidates))
    t3_measured = _measured_rows(
        t3_candidates, t3_rows, costs,
        base_weight_bits, base_activation_bits, expected_samples,
        p3_measured[0].sample_rmse)
    measured = p3_measured + t3_measured
    tails = tuple(
        row for row in measured
        if row.stage == "interaction" and row.prefix == prefix.prefix and
        row.valid and row.normalized_weight_cost <= maximum_weight and
        row.normalized_activation_cost <= maximum_activation and
        row.mean_sample_rmse <= reference_mean * (1.0 + maximum_relative))
    if not tails:
        raise RuntimeError(
            "P3/T3 search has no candidate within relative RMSE gate")
    def protection_cost(row):
        return (row.normalized_weight_cost - 1.0) + \
            (row.normalized_activation_cost - 1.0)
    selected = min(tails, key=lambda row: (
        protection_cost(row),
        row.mean_sample_rmse,
        row.pooled_rmse,
        row.normalized_weight_cost,
        row.normalized_activation_cost,
        row.tail,
        row.name,
    ))
    return P3T3SearchResult(
        assignment=selected.assignment,
        prefix=prefix.prefix,
        tail=selected.tail,
        selected_candidate=selected.name,
        candidates=measured,
        cost_basis=costs,
        base_weight_bits=int(base_weight_bits),
        base_activation_bits=int(base_activation_bits),
        promotion_weight_bits=int(promotion_weight_bits),
        promotion_activation_bits=int(promotion_activation_bits),
        maximum_normalized_weight_cost=maximum_weight,
        maximum_normalized_activation_cost=maximum_activation,
        maximum_relative_rmse_loss=maximum_relative,
        reference_mean_sample_rmse=reference_mean,
        expected_samples=int(expected_samples),
    )


def _block_group(contract, name):
    owners = tuple(
        block.name for block in contract.blocks
        if name == block.name or name.startswith(block.name + "."))
    if len(owners) > 1:
        raise RuntimeError("hardware module has multiple contract blocks: %s" % name)
    return owners[0] if owners else None


def _calibration_indices(path, expected_count, evaluation_indices,
                         dataset_size):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = tuple(int(index) for index in payload["calibration_indices"])
    persisted_evaluation = tuple(
        int(index) for index in payload["evaluation_indices"])
    if len(indices) != int(expected_count):
        raise ValueError("calibration metadata count differs from model config")
    if len(indices) != len(set(indices)):
        raise ValueError("calibration metadata indices must be unique")
    if any(index < 0 or index >= int(dataset_size) for index in indices):
        raise ValueError("calibration metadata index is outside the train dataset")
    if persisted_evaluation != tuple(evaluation_indices):
        raise ValueError("calibration metadata evaluation identities changed")
    selection = payload["calibration_source"]["selection"]
    if selection != "32_tail_96_kmedoids":
        raise ValueError("P3/T3 requires stratified calibration metadata")
    return indices


def _site_boundary(site):
    parts = str(site.site).split("::")
    if site.owner_kind == "module_input" and len(parts) == 3 and \
            parts[0] == "activation" and parts[2] == "input":
        return parts[1], "input"
    if site.owner_kind == "module_output" and len(parts) == 3 and \
            parts[0] == "activation" and parts[2] == "output":
        return parts[1], "output"
    raise ValueError("contract activation site is not a module boundary: %s" %
                     site.site)


def _scale_aware_concat_consumers(model_name, model):
    if model_name != "nlspn":
        return ()
    modules = dict(model.named_modules())
    prefixes = ("id_dec1", "id_dec0", "gd_dec1", "gd_dec0")
    if "cf_dec0" in modules:
        prefixes += ("cf_dec1", "cf_dec0")
    names = tuple("%s.0" % prefix for prefix in prefixes)
    unknown = tuple(name for name in names if name not in modules)
    if unknown:
        raise ValueError("NLSPN concat consumer modules are missing: %s" %
                         (unknown,))
    if any(not isinstance(modules[name], nn.Conv2d) for name in names):
        raise TypeError("NLSPN concat consumers must be Conv2d: %s" %
                        (names,))
    return (None, None, None) + names


def _propagation_valid(model_name, preserve_input, rows):
    if model_name not in ("cspn", "dyspn", "nlspn", "completionformer"):
        raise ValueError("unknown propagation model: %s" % model_name)
    if model_name == "cspn":
        numeric_values = tuple(
            float(value) for row in rows for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool))
        return all(math.isfinite(value) for value in numeric_values)
    if not rows:
        return False
    states = tuple(row for row in rows if row["signal"] == "state")
    constraints = tuple(
        row for row in rows if row["signal"] == "affinity_constraints")
    anchors = tuple(
        row for row in rows
        if row["signal"] in ("anchor", "anchor_injection"))
    anchors_required = model_name == "dyspn" or bool(preserve_input)
    if not states or not constraints or bool(anchors) != anchors_required:
        return False
    state_mse = tuple(float(row["mse"]) for row in states)
    coefficient_errors = tuple(
        float(row["coefficient_sum_max_error"]) for row in constraints)
    contraction_rates = tuple(
        float(row["contraction_violation_rate"]) for row in constraints)
    anchor_errors = tuple(float(row["anchor_max_error"]) for row in anchors)
    values = state_mse + coefficient_errors + contraction_rates + anchor_errors
    return all(math.isfinite(value) for value in values) and \
        max(coefficient_errors) == 0.0 and \
        max(contraction_rates) == 0.0 and \
        (not anchor_errors or max(anchor_errors) == 0.0)


def _preserve_input_policy(model_name, propagation_adapter):
    if model_name == "cspn":
        return False
    if model_name == "dyspn":
        return True
    if model_name in ("nlspn", "completionformer"):
        return bool(propagation_adapter.module.args.preserve_input)
    raise ValueError("unknown propagation model: %s" % model_name)


class HardDeploymentP3T3Evaluator(object):
    """Measured RTN evaluator using existing hard QDQ and propagation APIs."""

    def __init__(self, runtime, model, contract, registry,
                 settings: HardDeploymentSettings,
                 propagation_projection_precision=None) -> None:
        self.runtime = runtime
        self.model = model
        self.contract = contract
        self.registry = registry
        self.settings = settings
        if propagation_projection_precision is not None:
            if len(propagation_projection_precision) != 2:
                raise ValueError(
                    "propagation projection precision requires W and A bits")
            self.propagation_projection_precision = tuple(
                int(bits) for bits in propagation_projection_precision)
        else:
            self.propagation_projection_precision = None
        self.device = torch.device(settings.device)
        if self.device != runtime.device:
            raise ValueError("evaluator device differs from selected runtime device")
        self.trainset = runtime.build_dataset("train")
        self.valset = runtime.build_dataset("val")
        self.calibration_indices = _calibration_indices(
            settings.calibration_metadata,
            settings.calibration_count,
            settings.evaluation_indices,
            len(self.trainset),
        )
        if any(index < 0 or index >= len(self.valset)
               for index in settings.evaluation_indices):
            raise ValueError("evaluation index is outside the validation dataset")
        self.seed = int(runtime.saved_args.seed)
        self.target_plan = resolve_qdrop_targets(runtime.model_name, model)
        contract_sites = set(
            owner for block in contract.blocks
            for owner, role in block.activation_owners)
        self.sites = dict(
            (site.site, site) for site in self.target_plan.activation_sites
            if site.site in contract_sites)
        if set(self.sites) != contract_sites:
            raise ValueError("contract activation sites differ from target plan")
        self.joint_adapter = None
        self.concat_adapter = None
        self.instrumentor = None
        self.propagation_projection_instrumentor = None
        self.propagation_adapter = None
        self._closed = False
        try:
            self._prepare_and_calibrate()
        except Exception:
            self.close()
            raise

    def _sample_batch(self, dataset, index):
        sample = rtn_runner.seeded_sample(dataset, index, self.seed)
        return rtn_runner.batch_from_sample(sample)

    def _prepare_and_calibrate(self):
        first_batch = self._sample_batch(
            self.trainset, self.calibration_indices[0])
        example_args, ground_truth = self.runtime.model_input(
            first_batch, self.device)
        del ground_truth
        preparation = prepare_hardware_model(
            self.model,
            example_args,
            fold=self.settings.fold_conv_bn,
        )
        if preparation["primary_max_abs_error"] > self.settings.fold_max_error:
            raise RuntimeError("Conv-BN fold changed FP32 output by %.8f" %
                               preparation["primary_max_abs_error"])
        self.graph_preparation = {
            "fold": int(self.settings.fold_conv_bn),
            "folded_pairs": list(preparation["folded_pairs"]),
            "unfolded_fanout_pairs": list(
                preparation["unfolded_fanout_pairs"]),
            "unfolded_conv_bn_pairs": list(
                preparation["unfolded_conv_bn_pairs"]),
        }

        attention_count = len(self.contract.attention_edges)
        concat_count = len(self.contract.concat_edges)
        if attention_count % 3 != 0 or concat_count % 2 != 0:
            raise ValueError("joint contract edge cardinality is invalid")
        if attention_count or concat_count:
            self.joint_adapter = CompletionFormerJointAdapter(
                model=self.model,
                expected_attention_modules=attention_count // 3,
                expected_concat_modules=concat_count // 2,
                weight_bits=self.settings.base_weight_bits,
                qkv_bits=self.settings.base_activation_bits,
                probability_bits=self.settings.promotion_activation_bits,
                concat_bits=self.settings.base_activation_bits,
                output_bits=self.settings.base_activation_bits,
                clip_factors=self.settings.joint_clip_factors,
                search_rounds=self.settings.joint_search_rounds,
                cache_sample_limit=self.settings.joint_cache_sample_limit,
                cache_byte_limit=self.settings.joint_cache_byte_limit,
            )

        propagation_outputs = set(propagation_projection_outputs(
            self.runtime.model_name, self.model))
        owned_outputs = set(propagation_outputs)
        owned_inputs = set()
        concat_call_consumers = _scale_aware_concat_consumers(
            self.runtime.model_name, self.model)
        concat_consumers = tuple(
            name for name in concat_call_consumers
            if name is not None and name in self.contract.weight_modules)
        if concat_consumers:
            self.concat_adapter = CallIndexedConcatConvAdapter(
                self.model, concat_consumers,
                weight_bits=self.settings.base_weight_bits,
                activation_bits=self.settings.base_activation_bits,
                output_bits=self.settings.base_activation_bits,
                cache_sample_limit=1,
                cache_byte_limit=self.settings.joint_cache_byte_limit,
                call_consumer_modules=tuple(
                    name if name is not None and name in concat_consumers else None
                    for name in concat_call_consumers))
            owned_inputs.update(self.concat_adapter.externally_owned_inputs())
            owned_outputs.update(self.concat_adapter.externally_owned_outputs())
        if self.joint_adapter is not None:
            owned_outputs.update(self.joint_adapter.externally_owned_outputs())
            owned_inputs.update(self.joint_adapter.externally_owned_inputs())

        def group_fn(name, module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) \
                    and name not in self.contract.weight_modules:
                return None
            return _block_group(self.contract, name)

        self.instrumentor = HardwareAlignedInstrumentor(
            self.model,
            group_fn,
            preparation["fused_relu_producers"],
            externally_owned_outputs=owned_outputs,
            externally_owned_inputs=owned_inputs,
        )
        if set(self.instrumentor.modules) != set(self.contract.weight_modules):
            raise ValueError("hardware weight ownership differs from contract")
        if self.propagation_projection_precision is not None:
            projection_names = set(propagation_outputs)

            def projection_group(name, module):
                if name in projection_names:
                    return "propagation_projection"
                return None

            self.propagation_projection_instrumentor = \
                HardwareAlignedInstrumentor(
                    self.model,
                    projection_group,
                    preparation["fused_relu_producers"],
                    fuse_layernorm=False,
                )
            if set(self.propagation_projection_instrumentor.modules) != \
                    projection_names:
                raise ValueError(
                    "propagation projection ownership differs from contract")
        self.propagation_adapter = install_propagation_adapter(
            self.runtime.model_name, self.model)
        self.preserve_input = _preserve_input_policy(
            self.runtime.model_name, self.propagation_adapter)
        self.instrumentor.observe()
        if self.propagation_projection_instrumentor is not None:
            self.propagation_projection_instrumentor.observe()
        self.propagation_adapter.observe()
        if self.concat_adapter is not None:
            self.concat_adapter.observe()
        if self.joint_adapter is not None:
            self.joint_adapter.observe_qdrop_ranges()
        with torch.no_grad():
            for index in self.calibration_indices:
                batch = self._sample_batch(self.trainset, index)
                model_args, ground_truth = self.runtime.model_input(
                    batch, self.device)
                del ground_truth
                self.model(*model_args)
        self.instrumentor.freeze()
        if self.propagation_projection_instrumentor is not None:
            self.propagation_projection_instrumentor.freeze()
        self.propagation_adapter.freeze()
        if self.concat_adapter is not None:
            self.concat_adapter.freeze()
        if self.joint_adapter is not None:
            self.joint_adapter.freeze_qdrop_ranges()
        self._cache_evaluation_batches()

    def _cache_evaluation_batches(self):
        self.evaluation_batches = tuple(
            (int(index), self._sample_batch(self.valset, index))
            for index in self.settings.evaluation_indices)
        if not self.evaluation_batches:
            raise ValueError("evaluation identities must be nonempty")
        first_batch = self.evaluation_batches[0][1]
        keys = tuple(first_batch)
        if any(tuple(batch) != keys
               for _, batch in self.evaluation_batches):
            raise ValueError("evaluation batch fields differ")
        if any(not torch.is_tensor(batch[key])
               for _, batch in self.evaluation_batches
               for key in keys):
            raise TypeError("evaluation batch fields must be tensors")
        self.evaluation_batch = dict(
            (key, torch.cat(tuple(batch[key]
                                 for index, batch in self.evaluation_batches),
                            dim=0))
            for key in keys)

    def _activation_configuration(self, candidate):
        assigned = dict(candidate.assignment.activation_bits)
        generic = {}
        joint = {}
        for owner in assigned:
            site_name, role = owner
            site = self.sites[site_name]
            if site.role != role:
                raise ValueError("assignment activation role differs from contract")
            bits = int(assigned[owner])
            if site.owner_kind in ("module_input", "module_output"):
                boundary = _site_boundary(site)
                if self.concat_adapter is not None and \
                        boundary[0] in self.concat_adapter.consumer_modules:
                    continue
                if boundary in generic and generic[boundary] != bits:
                    raise ValueError("one hardware boundary has multiple bit values")
                generic[boundary] = bits
            elif site.owner_kind in ("attention_qkv", "concat_input"):
                joint[site_name] = bits
            else:
                raise ValueError("unsupported hard-deployment activation owner")
        return generic, joint

    def _configure_joint(self, joint_bits):
        if self.joint_adapter is None:
            if joint_bits:
                raise ValueError("joint activation assignment lacks joint adapter")
            return
        expected = set(self.contract.attention_edges) | \
            set(self.contract.concat_edges)
        if set(joint_bits) != expected:
            raise ValueError("joint activation bit coverage differs from contract")
        self.joint_adapter.unbind_qdrop_sites()
        quantizers = {}
        joint_sites = []
        for site_name in sorted(joint_bits):
            site = self.sites[site_name]
            calibration = self.joint_adapter.qdrop_initialization_tensor(site)
            maximum = float(calibration.detach().abs().max().item())
            quantizers[site_name] = _SiteSymmetricActivationQuantizer(
                site_name, joint_bits[site_name], maximum)
            joint_sites.append(site)
        self.joint_adapter.bind_qdrop_sites(tuple(joint_sites), quantizers)

    def _configure_candidate(self, candidate):
        mixed_precision.validate_assignment_ownership(
            self.contract, candidate.assignment)
        generic_bits, joint_bits = self._activation_configuration(candidate)
        groups = set(self.registry.blocks)
        self.instrumentor.configure(
            self.settings.base_weight_bits,
            self.settings.base_activation_bits,
            groups,
            weight_bit_overrides=dict(candidate.assignment.weight_bits),
            activation_bit_overrides=generic_bits,
            external_output_ownership=True,
            quantize_bias=False,
        )
        self.instrumentor.set_runtime_statistics(False)
        if self.propagation_projection_instrumentor is not None:
            projection_weight_bits, projection_activation_bits = \
                self.propagation_projection_precision
            self.propagation_projection_instrumentor.configure(
                projection_weight_bits,
                projection_activation_bits,
                {"propagation_projection"},
                external_output_ownership=False,
                quantize_bias=False,
            )
        missing = set(generic_bits) - set(self.instrumentor.quantizers)
        if missing:
            raise RuntimeError(
                "contract activation boundaries were not calibrated: %s" %
                sorted(missing))
        self.instrumentor.quantizers = dict(
            (key, self.instrumentor.quantizers[key])
            for key in generic_bits)
        self.instrumentor.relu_quantizers = {}
        if self.concat_adapter is not None:
            assignment_weights = dict(candidate.assignment.weight_bits)
            assignment_activations = dict(candidate.assignment.activation_bits)
            concat_weights = {}
            concat_activations = {}
            for name in self.concat_adapter.consumer_modules:
                concat_weights[name] = assignment_weights[name]
                owner = ("activation::%s::input" % name, "module_input")
                if owner not in assignment_activations:
                    raise ValueError(
                        "concat consumer activation assignment is missing: %s" %
                        (owner,))
                concat_activations[name] = assignment_activations[owner]
            self.concat_adapter.configure(
                concat_weights, concat_activations, concat_activations)
        self.propagation_adapter.configure(PropagationQuantConfig(
            affinity_bits=self.settings.base_activation_bits,
            confidence_bits=self.settings.promotion_activation_bits,
            offset_bits=self.settings.base_activation_bits,
            state_bits=self.settings.base_activation_bits,
        ))
        self._configure_joint(joint_bits)

    def configure_uniform(self, weight_bits: int, activation_bits: int):
        """Materialize one uniform precision for every ordinary block."""
        weight_bits = int(weight_bits)
        activation_bits = int(activation_bits)
        promoted = tuple(self.registry.blocks)
        assignment = mixed_precision.promoted_assignment(
            self.registry,
            promoted,
            base_weight_bits=weight_bits,
            base_activation_bits=activation_bits,
            promotion_weight_bits=weight_bits,
            promotion_activation_bits=activation_bits,
        )
        self._configure_candidate(P3T3Candidate(
            name="UNIFORM_W%dA%d" % (weight_bits, activation_bits),
            stage="baseline",
            prefix=(),
            tail=(),
            promoted_blocks=promoted,
            assignment=assignment,
        ))

    def configure_uniform_w8a8(self):
        """Materialize W8A8 for every ordinary contract block."""
        self.configure_uniform(8, 8)

    def _forward(self, batch):
        model_args, ground_truth = self.runtime.model_input(batch, self.device)
        output = self.model(*model_args)
        prediction = self.runtime.prediction(output)
        return prediction.detach().cpu(), ground_truth.detach().cpu()

    def reference_sample_rmse(self):
        self.instrumentor.disable()
        if self.concat_adapter is not None:
            self.concat_adapter.disable()
        if self.propagation_projection_instrumentor is not None:
            self.propagation_projection_instrumentor.disable()
        self.propagation_adapter.disable()
        if self.joint_adapter is not None:
            self.joint_adapter.unbind_qdrop_sites()
        rows = []
        with torch.no_grad():
            prediction, ground_truth = self._forward(self.evaluation_batch)
            if prediction.shape[0] != len(self.evaluation_batches) or \
                    ground_truth.shape[0] != len(self.evaluation_batches):
                raise RuntimeError("evaluation batch output count differs")
            if not bool(torch.isfinite(prediction).all().item()):
                raise RuntimeError("FP32 reference prediction is non-finite")
            for position, (sample_index, batch) in enumerate(
                    self.evaluation_batches):
                sample_prediction = prediction[position]
                sample_ground_truth = ground_truth[position]
                valid = torch.isfinite(sample_ground_truth) & \
                    (sample_ground_truth > 1e-4)
                valid_pixels = int(valid.sum().item())
                if valid_pixels <= 0:
                    raise ValueError(
                        "FP32 reference sample has no valid depth pixels")
                if bool((sample_prediction[valid] <= 1e-4).any().item()):
                    raise RuntimeError(
                        "FP32 reference prediction is non-positive")
                difference = sample_prediction[valid].double() - \
                    sample_ground_truth[valid].double()
                squared_error_sum = float(difference.square().sum().item())
                rows.append((
                    int(sample_index),
                    math.sqrt(squared_error_sum / float(valid_pixels))))
        return tuple(rows)

    def _evaluate_candidate(self, candidate):
        self._configure_candidate(candidate)
        rows = []
        with torch.no_grad():
            first, ground_truth = self._forward(self.evaluation_batch)
            first_propagation = tuple(self.propagation_adapter.statistics())
            second, second_ground_truth = self._forward(self.evaluation_batch)
            second_propagation = tuple(self.propagation_adapter.statistics())
            if not torch.equal(ground_truth, second_ground_truth):
                raise RuntimeError("paired ground truth changed between forwards")
            if first.shape[0] != len(self.evaluation_batches) or \
                    second.shape[0] != len(self.evaluation_batches):
                raise RuntimeError("evaluation batch output count differs")
            finite = bool(torch.isfinite(first).all().item()) and \
                bool(torch.isfinite(second).all().item())
            reproducible = finite and torch.equal(first, second)
            propagation_valid = _propagation_valid(
                self.runtime.model_name, self.preserve_input,
                first_propagation) and _propagation_valid(
                    self.runtime.model_name, self.preserve_input,
                    second_propagation)
            for position, (sample_index, batch) in enumerate(
                    self.evaluation_batches):
                sample_first = first[position]
                sample_ground_truth = ground_truth[position]
                valid = torch.isfinite(sample_ground_truth) & \
                    (sample_ground_truth > 1e-4)
                valid_pixels = int(valid.sum().item())
                if valid_pixels <= 0:
                    raise ValueError("evaluation sample has no valid depth pixels")
                difference = sample_first[valid].double() - \
                    sample_ground_truth[valid].double()
                squared_error_sum = float(difference.square().sum().item())
                rmse = math.sqrt(squared_error_sum / float(valid_pixels)) \
                    if math.isfinite(squared_error_sum) else float("inf")
                rows.append({
                    "config": candidate.name,
                    "sample_index": int(sample_index),
                    "squared_error_sum": squared_error_sum,
                    "valid_pixels": valid_pixels,
                    "RMSE": rmse,
                    "prediction_finite": finite,
                    "propagation_valid": propagation_valid,
                    "reproducible": reproducible,
                })
        return rows

    def configure_hard_candidate(self, candidate):
        """Materialize one measured candidate for strict artifact export."""
        self._configure_candidate(candidate)

    def close(self):
        if self._closed:
            return
        if self.joint_adapter is not None:
            self.joint_adapter.unbind_qdrop_sites()
            self.joint_adapter.close()
        if self.concat_adapter is not None:
            self.concat_adapter.close()
        if self.propagation_adapter is not None:
            self.propagation_adapter.close()
        if self.propagation_projection_instrumentor is not None:
            self.propagation_projection_instrumentor.close()
        if self.instrumentor is not None:
            self.instrumentor.close()
        self._closed = True

    def __call__(self, candidates):
        rows = []
        for candidate in candidates:
            rows.extend(self._evaluate_candidate(candidate))
        return tuple(rows)


def _run_runtime_search(
        runtime: NYUModelRuntime,
        costs: mixed_precision.CostBasis,
        evaluator_factory,
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int,
        maximum_normalized_weight_cost: float,
        maximum_normalized_activation_cost: float,
        maximum_relative_rmse_loss: float,
        expected_samples: int,
        contract_builder) -> P3T3SearchResult:
    """Bind an official NYU runtime to the generic measured search."""
    try:
        model = runtime.build_model(runtime.device)
        contract = contract_builder(runtime.model_name, model)
        registry = mixed_precision.build_registry(contract, costs)
        evaluator = evaluator_factory(runtime, model, contract, registry)
        try:
            return search_p3_t3(
                contract, costs, evaluator,
                base_weight_bits, base_activation_bits,
                promotion_weight_bits, promotion_activation_bits,
                maximum_normalized_weight_cost,
                maximum_normalized_activation_cost,
                maximum_relative_rmse_loss,
                expected_samples,
            )
        finally:
            evaluator.close()
    finally:
        runtime.close()


def run_runtime_search(
        runtime: NYUModelRuntime,
        costs: mixed_precision.CostBasis,
        evaluator_factory,
        base_weight_bits: int,
        base_activation_bits: int,
        promotion_weight_bits: int,
        promotion_activation_bits: int,
        maximum_normalized_weight_cost: float,
        maximum_normalized_activation_cost: float,
        maximum_relative_rmse_loss: float,
        expected_samples: int) -> P3T3SearchResult:
    """Bind an official NYU runtime using the public contract builder."""
    return _run_runtime_search(
        runtime, costs, evaluator_factory,
        base_weight_bits, base_activation_bits,
        promotion_weight_bits, promotion_activation_bits,
        maximum_normalized_weight_cost,
        maximum_normalized_activation_cost,
        maximum_relative_rmse_loss,
        expected_samples,
        build_model_quantization_contract,
    )


def _assignment_payload(assignment):
    return {
        "model_name": assignment.model_name,
        "weight_bits": [[module, bits]
                        for module, bits in assignment.weight_bits],
        "activation_bits": [[[owner[0], owner[1]], bits]
                            for owner, bits in assignment.activation_bits],
    }


def _json_metric(value):
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _candidate_payload(row, reference_mean_sample_rmse):
    metric_values = (
        row.pooled_rmse,
        row.mean_sample_rmse,
    ) + tuple(value for index, value in row.sample_rmse) + \
        tuple(row.paired_sample_differences)
    return {
        "name": row.name,
        "stage": row.stage,
        "prefix": list(row.prefix),
        "tail": list(row.tail),
        "pooled_rmse": _json_metric(row.pooled_rmse),
        "mean_sample_rmse": _json_metric(row.mean_sample_rmse),
        "normalized_weight_cost": row.normalized_weight_cost,
        "normalized_activation_cost": row.normalized_activation_cost,
        "relative_rmse_loss": _json_metric(
            row.mean_sample_rmse / reference_mean_sample_rmse - 1.0),
        "valid": row.valid,
        "metrics_finite": all(math.isfinite(float(value))
                              for value in metric_values),
        "sample_rmse": [[index, _json_metric(value)]
                        for index, value in row.sample_rmse],
        "sample_evidence": [{
            "sample_index": sample.sample_index,
            "squared_error_sum": _json_metric(sample.squared_error_sum),
            "valid_pixels": sample.valid_pixels,
            "prediction_finite": sample.prediction_finite,
            "propagation_valid": sample.propagation_valid,
            "reproducible": sample.reproducible,
        } for sample in row.sample_evidence],
        "paired_sample_differences": [
            _json_metric(value) for value in row.paired_sample_differences],
        "assignment": _assignment_payload(row.assignment),
    }


def write_p3_t3_assignment(
        output: Path,
        result: P3T3SearchResult,
        source_checkpoint: CheckpointIdentity) -> Path:
    """Persist the selected tuple assignment and all measured search evidence."""
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("P3/T3 output directory is missing: %s" % root)
    path = root / "p3_t3_assignment.json"
    if path.exists():
        raise FileExistsError("P3/T3 assignment already exists: %s" % path)
    selected = tuple(
        row for row in result.candidates
        if row.name == result.selected_candidate)
    if len(selected) != 1:
        raise ValueError("selected P3/T3 candidate is missing or duplicated")
    selected_row = selected[0]
    selected_metrics = (
        selected_row.pooled_rmse,
        selected_row.mean_sample_rmse,
        selected_row.normalized_weight_cost,
        selected_row.normalized_activation_cost,
    ) + tuple(value for index, value in selected_row.sample_rmse) + \
        tuple(selected_row.paired_sample_differences)
    if not selected_row.valid or not all(
            math.isfinite(float(value)) for value in selected_metrics):
        raise ValueError("selected P3/T3 candidate must be stable and finite")
    if not isinstance(source_checkpoint, CheckpointIdentity):
        raise TypeError("P3/T3 source checkpoint identity is invalid")
    if capture_checkpoint_identity(source_checkpoint.path) != source_checkpoint:
        raise ValueError(
            "P3/T3 source checkpoint changed before publication")
    evaluation_indices = tuple(
        sample.sample_index for sample in result.candidates[0].sample_evidence)
    if len(evaluation_indices) != result.expected_samples or \
            len(evaluation_indices) != len(set(evaluation_indices)) or \
            any(tuple(sample.sample_index for sample in row.sample_evidence) !=
                evaluation_indices for row in result.candidates):
        raise ValueError("P3/T3 candidate sample identities differ")
    encoded_identity = json.dumps(
        [["val", index] for index in evaluation_indices],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    payload = {
        "format_version": P3_T3_ASSIGNMENT_VERSION,
        "artifact_kind": P3_T3_ASSIGNMENT_KIND,
        "model_name": result.assignment.model_name,
        "source_checkpoint": {
            "path": str(source_checkpoint.path),
            "size_bytes": source_checkpoint.size_bytes,
            "sha256": source_checkpoint.sha256,
        },
        "prefix": list(result.prefix),
        "tail": list(result.tail),
        "selected_candidate": result.selected_candidate,
        "precision": {
            "base_activation_bits": result.base_activation_bits,
            "base_weight_bits": result.base_weight_bits,
            "promotion_activation_bits": result.promotion_activation_bits,
            "promotion_weight_bits": result.promotion_weight_bits,
        },
        "budgets": {
            "maximum_normalized_activation_cost":
                result.maximum_normalized_activation_cost,
            "maximum_normalized_weight_cost":
                result.maximum_normalized_weight_cost,
        },
        "selection_policy": {
            "metric_aggregation": "mean_of_per_sample_rmse",
            "maximum_relative_rmse_loss":
                result.maximum_relative_rmse_loss,
            "reference_mean_sample_rmse":
                result.reference_mean_sample_rmse,
            "selected_relative_rmse_loss":
                selected_row.mean_sample_rmse /
                result.reference_mean_sample_rmse - 1.0,
        },
        "expected_samples": result.expected_samples,
        "evaluation": {
            "split": "val",
            "count": result.expected_samples,
            "indices": list(evaluation_indices),
            "identity_sha256": hashlib.sha256(encoded_identity).hexdigest(),
        },
        "cost_definition": {
            "activation_denominator": result.base_activation_bits * sum(
                elements for owner, elements
                in result.cost_basis.activation_elements),
            "activation_formula":
                "sum(activation_bits*elements)/activation_denominator",
            "weight_denominator": result.base_weight_bits * sum(
                macs for module, macs in result.cost_basis.weight_macs),
            "weight_formula": "sum(weight_bits*macs)/weight_denominator",
        },
        "cost_basis": {
            "activation_elements": [
                [[owner[0], owner[1]], elements]
                for owner, elements in result.cost_basis.activation_elements],
            "weight_macs": [[module, macs]
                            for module, macs in result.cost_basis.weight_macs],
        },
        "assignment": _assignment_payload(result.assignment),
        "candidates": [
            _candidate_payload(row, result.reference_mean_sample_rmse)
            for row in result.candidates],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _read_weight_cost_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames) != ("module", "macs"):
            raise ValueError("weight cost rows require module,macs columns")
        rows = tuple(
            (str(row["module"]), int(row["macs"])) for row in reader)
    if not rows or len(rows) != len(set(name for name, macs in rows)):
        raise ValueError("weight cost rows must be nonempty and unique")
    if any(not name or macs <= 0 for name, macs in rows):
        raise ValueError("weight cost rows require positive explicit MACs")
    return rows


def _read_activation_cost_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames) != ("site", "role", "elements"):
            raise ValueError(
                "activation cost rows require site,role,elements columns")
        rows = tuple(
            ((str(row["site"]), str(row["role"])), int(row["elements"]))
            for row in reader)
    owners = tuple(owner for owner, elements in rows)
    if not rows or len(rows) != len(set(owners)):
        raise ValueError("activation cost rows must be nonempty and unique")
    if any(not owner[0] or not owner[1] or elements <= 0
           for owner, elements in rows):
        raise ValueError(
            "activation cost rows require positive explicit element counts")
    return rows


def _production_evaluator_factory(runtime, model, contract, registry, settings):
    return HardDeploymentP3T3Evaluator(
        runtime, model, contract, registry, settings)


PRODUCTION_DEPENDENCIES = RunnerDependencies(
    runtime_factory=NYUModelRuntime.from_config,
    contract_builder=build_model_quantization_contract,
    evaluator_factory=_production_evaluator_factory,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run measured model-relative NYU P3/T3 selection")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--maximum-normalized-weight-cost", type=float, required=True)
    parser.add_argument(
        "--maximum-normalized-activation-cost", type=float, required=True)
    parser.add_argument(
        "--maximum-relative-rmse-loss", type=float, required=True)
    parser.add_argument("--weight-cost-rows", type=Path, required=True)
    parser.add_argument("--activation-cost-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    folding = parser.add_mutually_exclusive_group(required=True)
    folding.add_argument(
        "--fold-conv-bn", dest="fold_conv_bn", action="store_true")
    folding.add_argument(
        "--skip-conv-bn-fold", dest="fold_conv_bn", action="store_false")
    parser.add_argument("--fold-max-error", type=float, required=True)
    parser.add_argument(
        "--joint-clip-factors", type=float, nargs="+", required=True)
    parser.add_argument("--joint-search-rounds", type=int, required=True)
    parser.add_argument("--joint-cache-sample-limit", type=int, required=True)
    parser.add_argument("--joint-cache-byte-limit", type=int, required=True)
    return parser


def _settings(args, model_config, method):
    fold_max_error = float(args.fold_max_error)
    clip_factors = tuple(float(value) for value in args.joint_clip_factors)
    if not math.isfinite(fold_max_error) or fold_max_error < 0.0:
        raise ValueError("fold maximum error must be finite and nonnegative")
    if not clip_factors or any(
            not math.isfinite(value) or value <= 0.0
            for value in clip_factors):
        raise ValueError("joint clip factors must be finite and positive")
    if int(args.joint_search_rounds) <= 0 or \
            int(args.joint_cache_sample_limit) <= 0 or \
            int(args.joint_cache_byte_limit) <= 0:
        raise ValueError("joint calibration limits must be positive")
    return HardDeploymentSettings(
        device=str(args.device),
        calibration_metadata=model_config.calibration_metadata,
        calibration_count=int(model_config.calibration_count),
        evaluation_indices=tuple(model_config.evaluation_indices),
        base_weight_bits=int(method["base_weight_bits"]),
        base_activation_bits=int(method["base_activation_bits"]),
        promotion_weight_bits=int(method["promotion_weight_bits"]),
        promotion_activation_bits=int(method["promotion_activation_bits"]),
        fold_conv_bn=bool(args.fold_conv_bn),
        fold_max_error=fold_max_error,
        joint_clip_factors=clip_factors,
        joint_search_rounds=int(args.joint_search_rounds),
        joint_cache_sample_limit=int(args.joint_cache_sample_limit),
        joint_cache_byte_limit=int(args.joint_cache_byte_limit),
    )


def run_cli(argv, dependencies=PRODUCTION_DEPENDENCIES):
    args = build_parser().parse_args(tuple(argv))
    selected = load_selected_quantization_config(args.config)
    model_rows = tuple(
        model for model in selected.models if model.model == args.model)
    if len(model_rows) != 1:
        raise ValueError("selected experiment model entry is not unique")
    model_config = model_rows[0]
    if str(args.device) != model_config.device:
        raise ValueError("explicit device differs from selected model device")
    if not Path(args.output).is_dir():
        raise FileNotFoundError("P3/T3 output directory is missing: %s" %
                                args.output)
    costs = mixed_precision.CostBasis(
        weight_macs=_read_weight_cost_rows(args.weight_cost_rows),
        activation_elements=_read_activation_cost_rows(
            args.activation_cost_rows),
    )
    method = selected.method_hyperparameters["p3_t3_mixed_ptq"]
    settings = _settings(args, model_config, method)
    runtime = dependencies.runtime_factory(model_config)
    checkpoint_identity = capture_checkpoint_identity(
        model_config.checkpoint)

    def evaluator_factory(observed_runtime, model, contract, registry):
        return dependencies.evaluator_factory(
            observed_runtime, model, contract, registry, settings)

    result = _run_runtime_search(
        runtime=runtime,
        costs=costs,
        evaluator_factory=evaluator_factory,
        base_weight_bits=settings.base_weight_bits,
        base_activation_bits=settings.base_activation_bits,
        promotion_weight_bits=settings.promotion_weight_bits,
        promotion_activation_bits=settings.promotion_activation_bits,
        maximum_normalized_weight_cost=
            args.maximum_normalized_weight_cost,
        maximum_normalized_activation_cost=
            args.maximum_normalized_activation_cost,
        maximum_relative_rmse_loss=args.maximum_relative_rmse_loss,
        expected_samples=len(settings.evaluation_indices),
        contract_builder=dependencies.contract_builder,
    )
    if capture_checkpoint_identity(model_config.checkpoint) != \
            checkpoint_identity:
        raise ValueError("P3/T3 source checkpoint changed during search")
    return write_p3_t3_assignment(
        args.output, result, checkpoint_identity)


def main():
    path = run_cli(tuple(sys.argv[1:]))
    print(path)


if __name__ == "__main__":
    main()
