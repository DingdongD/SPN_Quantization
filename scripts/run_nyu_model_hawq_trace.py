#!/usr/bin/env python3
"""Run contract-driven HAWQ tracing for selected official NYU models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_qdrop_reconstruction import (  # noqa: E402
    ordered_sample_identity_sha256,
)
from scripts.run_nyu_rtn_quantization import seeded_sample  # noqa: E402
from spn_quant.experiment_config import (  # noqa: E402
    MODEL_ORDER,
    load_selected_quantization_config,
)
from spn_quant.hawq_trace import (  # noqa: E402
    BlockTraceEstimate,
    HutchinsonTraceConfig,
    estimate_parameter_block_trace_samples,
    estimate_parameter_block_trace_samples_finite_difference,
    masked_curvature_loss,
    weight_quantization_error,
)
from spn_quant.model_contracts import (  # noqa: E402
    QuantizationBlock,
    QuantizationModelContract,
    build_model_quantization_contract,
)


BITS = (4, 6, 8)
TRACE_SETTING_FIELDS = (
    "batch_size",
    "probes_per_batch",
    "seed",
    "depth_mse_weight",
    "boundary_mse_weight",
    "boundary_threshold_m",
)


def model_hessian_vector_settings(model_name: str):
    if model_name == "dyspn":
        return "central_finite_difference_block", 0.001
    if model_name in ("nlspn", "completionformer"):
        return "autograd_block", None
    raise ValueError("unsupported HAWQ model: %s" % model_name)


def _hessian_vector_payload(model_name: str):
    mode, epsilon = model_hessian_vector_settings(model_name)
    return {"mode": mode, "epsilon": epsilon}


def estimate_model_trace_samples(
        model_name: str, blocks, loss_fn, config: HutchinsonTraceConfig):
    mode, epsilon = model_hessian_vector_settings(model_name)
    if mode == "central_finite_difference_block":
        return estimate_parameter_block_trace_samples_finite_difference(
            blocks, loss_fn, config, epsilon)
    if mode == "autograd_block":
        return estimate_parameter_block_trace_samples(blocks, loss_fn, config)
    raise RuntimeError("validated HAWQ Hessian-vector mode changed")


@dataclass(frozen=True)
class HAWQTraceSettings:
    batch_size: int
    probes_per_batch: int
    seed: int
    depth_mse_weight: float
    boundary_mse_weight: float
    boundary_threshold_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(
            self, "probes_per_batch", int(self.probes_per_batch))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self, "depth_mse_weight", float(self.depth_mse_weight))
        object.__setattr__(
            self, "boundary_mse_weight", float(self.boundary_mse_weight))
        object.__setattr__(
            self, "boundary_threshold_m", float(self.boundary_threshold_m))
        if self.batch_size <= 0 or 128 % self.batch_size:
            raise ValueError("HAWQ batch size must be a positive divisor of 128")
        if self.probes_per_batch <= 0:
            raise ValueError("HAWQ probes per batch must be positive")
        if self.seed < 0:
            raise ValueError("HAWQ seed must be nonnegative")
        if not math.isfinite(self.depth_mse_weight) or \
                self.depth_mse_weight <= 0.0:
            raise ValueError("HAWQ depth MSE weight must be positive")
        if not math.isfinite(self.boundary_mse_weight) or \
                self.boundary_mse_weight < 0.0:
            raise ValueError("HAWQ boundary MSE weight must be nonnegative")
        if not math.isfinite(self.boundary_threshold_m) or \
                self.boundary_threshold_m <= 0.0:
            raise ValueError("HAWQ boundary threshold must be positive")


@dataclass(frozen=True)
class TraceParameterBlock:
    name: str
    module_names: Tuple[str, ...]
    parameters: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class CalibrationIdentity:
    indices: Tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class HAWQTraceRun:
    traces: Tuple[BlockTraceEstimate, ...]
    raw_rows: Tuple[dict, ...]
    calibration_indices: Tuple[int, ...]
    settings: HAWQTraceSettings
    checkpoint_identity: CheckpointIdentity


@dataclass(frozen=True)
class HAWQWeightObjective:
    block: str
    bits: int
    normalized_trace: float
    quantization_error: float
    cost: float


@dataclass(frozen=True)
class HAWQBlockCost:
    name: str
    weight_parameters: int
    weight_macs: int
    activation_traffic: int


@dataclass(frozen=True)
class ContractHAWQProblem:
    blocks: Tuple[HAWQBlockCost, ...]
    traces: Tuple[BlockTraceEstimate, ...]
    objective_components: Tuple[HAWQWeightObjective, ...]
    weight_macs: Tuple[Tuple[str, int], ...]
    activation_traffic: Tuple[Tuple[Tuple[str, str], int], ...]


@dataclass(frozen=True)
class ContractHAWQResult:
    blocks: Tuple[HAWQBlockCost, ...]
    traces: Tuple[BlockTraceEstimate, ...]
    objective_components: Tuple[HAWQWeightObjective, ...]
    assignment: object
    weight_macs: Tuple[Tuple[str, int], ...]
    activation_traffic: Tuple[Tuple[Tuple[str, str], int], ...]
    maximum_weight_bits: float
    maximum_activation_bits: float


@dataclass(frozen=True)
class RunnerDependencies:
    runtime_factory: Callable
    contract_builder: Callable


def _validate_indices(indices, dataset_size) -> Tuple[int, ...]:
    values = tuple(indices)
    if len(values) != 128:
        raise ValueError("HAWQ trace requires exactly 128 calibration identities")
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in values):
        raise TypeError("HAWQ calibration identities must be integers")
    normalized = tuple(int(index) for index in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("HAWQ calibration identities must be unique")
    if any(index < 0 or index >= int(dataset_size) for index in normalized):
        raise ValueError("HAWQ calibration identity is outside the train split")
    return normalized


def load_calibration_identity(
        path: Path,
        evaluation_indices: Sequence[int],
        dataset_size: int) -> CalibrationIdentity:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = _validate_indices(payload["calibration_indices"], dataset_size)
    persisted_evaluation = tuple(payload["evaluation_indices"])
    if persisted_evaluation != tuple(evaluation_indices):
        raise ValueError("HAWQ calibration metadata evaluation identities changed")
    if payload["calibration_source"]["selection"] != \
            "32_tail_96_kmedoids":
        raise ValueError("HAWQ requires stratified calibration metadata")
    return CalibrationIdentity(
        indices=indices,
        sha256=ordered_sample_identity_sha256("train", indices),
    )


def load_persisted_calibration_identity(
        path: Path,
        evaluation_indices: Sequence[int]) -> CalibrationIdentity:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_indices = tuple(payload["calibration_indices"])
    if len(raw_indices) != 128:
        raise ValueError("HAWQ trace requires exactly 128 calibration identities")
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in raw_indices):
        raise TypeError("HAWQ calibration identities must be integers")
    indices = _validate_indices(raw_indices, 1 + max(raw_indices))
    if tuple(payload["evaluation_indices"]) != tuple(evaluation_indices):
        raise ValueError("HAWQ calibration metadata evaluation identities changed")
    if payload["calibration_source"]["selection"] != \
            "32_tail_96_kmedoids":
        raise ValueError("HAWQ requires stratified calibration metadata")
    return CalibrationIdentity(
        indices=indices,
        sha256=ordered_sample_identity_sha256("train", indices),
    )


def _supported_weight(module: nn.Module) -> torch.Tensor:
    if not isinstance(
            module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        raise TypeError(
            "HAWQ contract module has unsupported weight type: %s" %
            type(module).__name__)
    if not isinstance(module.weight, torch.Tensor):
        raise TypeError("HAWQ contract module weight must be a tensor")
    return module.weight


def build_trace_parameter_blocks(
        model: nn.Module,
        contract: QuantizationModelContract
        ) -> Tuple[TraceParameterBlock, ...]:
    if not isinstance(contract, QuantizationModelContract):
        raise TypeError("HAWQ trace requires QuantizationModelContract")
    modules = dict(model.named_modules())
    protected = set(contract.protected_modules)
    protected_roles = set(contract.protected_roles)
    role_map = dict(contract.module_roles)
    if set(contract.weight_modules).intersection(protected):
        raise ValueError("HAWQ trace contract includes protected modules")
    output = []
    observed_parameters = []
    for block in contract.blocks:
        if any(name in protected for name in block.weight_modules):
            raise ValueError("HAWQ trace block includes protected modules")
        if any(name in role_map and role_map[name] in protected_roles
               for name in block.weight_modules):
            raise ValueError("HAWQ trace block includes protected roles")
        missing = tuple(name for name in block.weight_modules
                        if name not in modules)
        if missing:
            raise KeyError("HAWQ trace contract modules are missing: %s" %
                           (missing,))
        parameters = tuple(
            _supported_weight(modules[name]) for name in block.weight_modules)
        observed_parameters.extend(parameters)
        output.append(TraceParameterBlock(
            name=block.name,
            module_names=tuple(block.weight_modules),
            parameters=parameters,
        ))
    if len(observed_parameters) != len(
            set(id(parameter) for parameter in observed_parameters)):
        raise ValueError("HAWQ trace contract parameters contain duplicates")
    return tuple(output)


def _trace_summary(raw_rows, blocks) -> Tuple[BlockTraceEstimate, ...]:
    grouped = dict((block.name, []) for block in blocks)
    for row in raw_rows:
        if row["block"] not in grouped:
            raise ValueError("HAWQ raw trace block is outside the contract")
        grouped[row["block"]].append(float(row["estimate"]))
    if any(not values for values in grouped.values()):
        raise ValueError("HAWQ raw trace coverage is incomplete")
    output = []
    for block in blocks:
        values = torch.tensor(grouped[block.name], dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("HAWQ aggregate trace must be finite")
        mean = float(values.mean().item())
        if mean < 0.0:
            raise ValueError("HAWQ aggregate trace is negative: %s" % block.name)
        standard_error = float(
            values.std(unbiased=True).div(math.sqrt(values.numel())).item()) \
            if values.numel() > 1 else 0.0
        coefficient = 0.0 if mean == 0.0 else \
            float(values.std(unbiased=False).item()) / mean
        parameters = sum(int(parameter.numel())
                         for parameter in block.parameters)
        output.append(BlockTraceEstimate(
            block=block.name,
            estimates=tuple(float(value) for value in values.tolist()),
            mean=mean,
            standard_error=standard_error,
            normalized_mean=mean / float(parameters),
            coefficient_of_variation=coefficient,
            parameters=parameters,
        ))
    return tuple(output)


def trace_calibration_batches(
        runtime: NYUModelRuntime,
        model: nn.Module,
        contract: QuantizationModelContract,
        dataset,
        calibration_indices: Sequence[int],
        settings: HAWQTraceSettings,
        checkpoint_identity: CheckpointIdentity) -> HAWQTraceRun:
    if runtime.device != next(model.parameters()).device:
        raise ValueError("HAWQ runtime and model devices differ")
    if not isinstance(settings, HAWQTraceSettings):
        raise TypeError("HAWQ trace settings must be validated")
    if not isinstance(checkpoint_identity, CheckpointIdentity):
        raise TypeError("HAWQ checkpoint identity must be captured before tracing")
    indices = _validate_indices(calibration_indices, len(dataset))
    blocks = build_trace_parameter_blocks(model, contract)
    raw_rows = []
    for start in range(0, len(indices), settings.batch_size):
        batch_indices = indices[start:start + settings.batch_size]
        batch = default_collate(tuple(
            seeded_sample(dataset, index, settings.seed)
            for index in batch_indices))
        model_input, target = runtime.model_input(batch, runtime.device)

        def loss_fn():
            prediction = runtime.prediction(model(*model_input))
            return masked_curvature_loss(
                prediction,
                target,
                torch.isfinite(target) & (target > 0.0),
                settings.depth_mse_weight,
                settings.boundary_mse_weight,
                settings.boundary_threshold_m,
            )

        samples = estimate_model_trace_samples(
            runtime.model_name,
            tuple((block.name, block.parameters) for block in blocks),
            loss_fn,
            HutchinsonTraceConfig(
                settings.probes_per_batch,
                settings.seed + start // settings.batch_size,
            ),
        )
        for block_name, estimates in samples:
            for probe, estimate in enumerate(estimates):
                raw_rows.append({
                    "batch_start": start,
                    "block": block_name,
                    "probe": probe,
                    "estimate": estimate,
                })
    return HAWQTraceRun(
        traces=_trace_summary(tuple(raw_rows), blocks),
        raw_rows=tuple(raw_rows),
        calibration_indices=indices,
        settings=settings,
        checkpoint_identity=checkpoint_identity,
    )


def _cost_rows(rows, name):
    normalized = tuple(rows)
    keys = tuple(key for key, value in normalized)
    if not normalized:
        raise ValueError("HAWQ %s costs must be nonempty" % name)
    if len(keys) != len(set(keys)):
        raise ValueError("HAWQ %s costs contain duplicates" % name)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for key, value in normalized):
        raise ValueError("HAWQ %s costs must be explicit positive integers" % name)
    return normalized


def build_cost_blocks(
        model: nn.Module,
        contract: QuantizationModelContract,
        weight_macs: Sequence[Tuple[str, int]],
        activation_traffic: Sequence[Tuple[Tuple[str, str], int]]
        ) -> Tuple[HAWQBlockCost, ...]:
    trace_blocks = build_trace_parameter_blocks(model, contract)
    weight_rows = _cost_rows(weight_macs, "weight MAC")
    activation_rows = _cost_rows(activation_traffic, "activation traffic")
    weight_map = dict(weight_rows)
    activation_map = dict(activation_rows)
    expected_weights = set(contract.weight_modules)
    expected_activations = set(
        owner for block in contract.blocks for owner in block.activation_owners)
    if set(weight_map) != expected_weights:
        raise ValueError("HAWQ weight MAC cost coverage mismatch")
    if set(activation_map) != expected_activations:
        raise ValueError("HAWQ activation cost coverage mismatch")
    if not set(contract.attention_edges) <= set(
            owner[0] for owner in activation_map):
        raise ValueError("HAWQ attention traffic cost coverage mismatch")
    by_name = dict((block.name, block) for block in trace_blocks)
    return tuple(
        HAWQBlockCost(
            name=block.name,
            weight_parameters=sum(
                int(parameter.numel())
                for parameter in by_name[block.name].parameters),
            weight_macs=sum(weight_map[name]
                            for name in block.weight_modules),
            activation_traffic=sum(
                activation_map[owner] for owner in block.activation_owners),
        )
        for block in contract.blocks)


def _channel_dim(module: nn.Module) -> int:
    return 1 if isinstance(module, nn.ConvTranspose2d) else 0


def build_contract_hawq_problem(
        model: nn.Module,
        contract: QuantizationModelContract,
        traces: Sequence[BlockTraceEstimate],
        weight_macs: Sequence[Tuple[str, int]],
        activation_traffic: Sequence[Tuple[Tuple[str, str], int]],
        bits: Sequence[int]) -> ContractHAWQProblem:
    declared_bits = tuple(int(value) for value in bits)
    if declared_bits != BITS:
        raise ValueError("HAWQ allocation bits must be exactly (4, 6, 8)")
    trace_rows = tuple(traces)
    if tuple(row.block for row in trace_rows) != contract.block_names:
        raise ValueError("HAWQ trace rows differ from contract block order")
    if any(not math.isfinite(row.normalized_mean) or row.normalized_mean < 0.0
           for row in trace_rows):
        raise ValueError("HAWQ normalized traces must be finite and nonnegative")
    blocks = build_cost_blocks(
        model, contract, weight_macs, activation_traffic)
    modules = dict(model.named_modules())
    trace_map = dict((row.block, row) for row in trace_rows)
    components = []
    for block in contract.blocks:
        normalized_trace = trace_map[block.name].normalized_mean
        for current_bits in declared_bits:
            quantization_error = sum(
                weight_quantization_error(
                    _supported_weight(modules[name]),
                    current_bits,
                    _channel_dim(modules[name]),
                )
                for name in block.weight_modules)
            cost = normalized_trace * quantization_error
            components.append(HAWQWeightObjective(
                block=block.name,
                bits=current_bits,
                normalized_trace=normalized_trace,
                quantization_error=quantization_error,
                cost=cost,
            ))
    return ContractHAWQProblem(
        blocks=blocks,
        traces=trace_rows,
        objective_components=tuple(components),
        weight_macs=tuple(weight_macs),
        activation_traffic=tuple(activation_traffic),
    )


def _solve_contract_hawq_problem(
        problem: ContractHAWQProblem,
        maximum_weight_bits: float,
        maximum_activation_bits: float) -> ContractHAWQResult:
    from spn_quant.hawq_allocation import (
        HAWQCandidate,
        HAWQIndependentBlock,
        solve_independent_hawq_assignment,
    )

    blocks = tuple(
        HAWQIndependentBlock(
            block.name,
            block.weight_parameters,
            block.weight_macs,
            block.activation_traffic,
        )
        for block in problem.blocks)
    candidates = tuple(
        HAWQCandidate(row.block, row.bits, row.cost)
        for row in problem.objective_components)
    assignment = solve_independent_hawq_assignment(
        blocks, candidates, maximum_weight_bits, maximum_activation_bits)
    return ContractHAWQResult(
        blocks=problem.blocks,
        traces=problem.traces,
        objective_components=problem.objective_components,
        assignment=assignment,
        weight_macs=problem.weight_macs,
        activation_traffic=problem.activation_traffic,
        maximum_weight_bits=float(maximum_weight_bits),
        maximum_activation_bits=float(maximum_activation_bits),
    )


def allocate_contract_hawq(
        model: nn.Module,
        contract: QuantizationModelContract,
        traces: Sequence[BlockTraceEstimate],
        weight_macs: Sequence[Tuple[str, int]],
        activation_traffic: Sequence[Tuple[Tuple[str, str], int]],
        bits: Sequence[int],
        maximum_weight_bits: float,
        maximum_activation_bits: float) -> ContractHAWQResult:
    problem = build_contract_hawq_problem(
        model, contract, traces, weight_macs, activation_traffic, bits)
    return _solve_contract_hawq_problem(
        problem, maximum_weight_bits, maximum_activation_bits)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_checkpoint_identity(path: Path) -> CheckpointIdentity:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError("HAWQ checkpoint is missing: %s" % resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    signature_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current = resolved.stat()
    signature_current = (
        current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if signature_before != signature_after or \
            signature_after != signature_current:
        raise RuntimeError("HAWQ checkpoint changed during identity capture")
    return CheckpointIdentity(
        path=resolved,
        size_bytes=int(after.st_size),
        sha256=digest.hexdigest(),
    )


def _checkpoint_payload(identity: CheckpointIdentity):
    if not isinstance(identity, CheckpointIdentity):
        raise TypeError("HAWQ checkpoint identity must be captured before tracing")
    return {
        "path": str(identity.path),
        "size_bytes": identity.size_bytes,
        "sha256": identity.sha256,
    }


def _trace_settings_payload(settings: HAWQTraceSettings):
    if not isinstance(settings, HAWQTraceSettings):
        raise TypeError("HAWQ trace settings must be validated")
    return dict((field, getattr(settings, field))
                for field in TRACE_SETTING_FIELDS)


def _trace_settings_from_payload(payload) -> HAWQTraceSettings:
    if set(payload) != set(TRACE_SETTING_FIELDS):
        raise ValueError("HAWQ trace settings fields changed")
    for field in ("batch_size", "probes_per_batch", "seed"):
        if isinstance(payload[field], bool) or not isinstance(
                payload[field], int):
            raise ValueError("HAWQ trace setting %s must be an integer" % field)
    for field in (
            "depth_mse_weight", "boundary_mse_weight",
            "boundary_threshold_m"):
        if not isinstance(payload[field], float):
            raise ValueError("HAWQ trace setting %s must be a float" % field)
    return HAWQTraceSettings(**dict(
        (field, payload[field]) for field in TRACE_SETTING_FIELDS))


def _trace_settings_from_method(method) -> HAWQTraceSettings:
    return _trace_settings_from_payload(method["trace"])


def _contract_payload(contract: QuantizationModelContract):
    return {
        "model_name": contract.model_name,
        "blocks": [{
            "name": block.name,
            "weight_modules": list(block.weight_modules),
            "activation_owners": [
                {"site": owner[0], "role": owner[1]}
                for owner in block.activation_owners],
        } for block in contract.blocks],
        "prefix_groups": [list(group) for group in contract.prefix_groups],
        "tail_groups": [list(group) for group in contract.tail_groups],
        "protected_roles": list(contract.protected_roles),
        "attention_edges": list(contract.attention_edges),
        "concat_edges": list(contract.concat_edges),
        "protected_modules": list(contract.protected_modules),
        "module_roles": [
            {"module": name, "role": role}
            for name, role in contract.module_roles],
    }


def _contract_from_payload(payload) -> QuantizationModelContract:
    expected = {
        "model_name", "blocks", "prefix_groups", "tail_groups",
        "protected_roles", "attention_edges", "concat_edges",
        "protected_modules", "module_roles",
    }
    if set(payload) != expected:
        raise ValueError("HAWQ trace artifact contract fields changed")
    blocks = []
    for row in payload["blocks"]:
        if set(row) != {"name", "weight_modules", "activation_owners"}:
            raise ValueError("HAWQ trace artifact block fields changed")
        owners = []
        for owner in row["activation_owners"]:
            if set(owner) != {"site", "role"}:
                raise ValueError(
                    "HAWQ trace artifact activation owner fields changed")
            owners.append((str(owner["site"]), str(owner["role"])))
        blocks.append(QuantizationBlock(
            name=str(row["name"]),
            weight_modules=tuple(str(name) for name in row["weight_modules"]),
            activation_owners=tuple(owners),
        ))
    module_roles = []
    for row in payload["module_roles"]:
        if set(row) != {"module", "role"}:
            raise ValueError("HAWQ trace artifact module role fields changed")
        module_roles.append((str(row["module"]), str(row["role"])))
    return QuantizationModelContract(
        model_name=str(payload["model_name"]),
        blocks=tuple(blocks),
        prefix_groups=tuple(
            tuple(str(name) for name in group)
            for group in payload["prefix_groups"]),
        tail_groups=tuple(
            tuple(str(name) for name in group)
            for group in payload["tail_groups"]),
        protected_roles=tuple(
            str(role) for role in payload["protected_roles"]),
        attention_edges=tuple(
            str(site) for site in payload["attention_edges"]),
        concat_edges=tuple(str(site) for site in payload["concat_edges"]),
        protected_modules=tuple(
            str(name) for name in payload["protected_modules"]),
        module_roles=tuple(module_roles),
    )


def _trace_payload(row: BlockTraceEstimate):
    return {
        "block": row.block,
        "estimates": list(row.estimates),
        "mean": row.mean,
        "standard_error": row.standard_error,
        "normalized_mean": row.normalized_mean,
        "coefficient_of_variation": row.coefficient_of_variation,
        "parameters": row.parameters,
    }


def _finite_close(actual, expected, name) -> None:
    actual = float(actual)
    expected = float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected) or not \
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("HAWQ trace artifact %s is inconsistent" % name)


def _traces_from_payload(rows, contract):
    expected_fields = {
        "block", "estimates", "mean", "standard_error",
        "normalized_mean", "coefficient_of_variation", "parameters",
    }
    traces = []
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("HAWQ trace artifact trace fields changed")
        estimates = tuple(float(value) for value in row["estimates"])
        if not estimates or not all(math.isfinite(value) for value in estimates):
            raise ValueError("HAWQ trace artifact estimates must be finite")
        parameters = row["parameters"]
        if isinstance(parameters, bool) or not isinstance(parameters, int) or \
                parameters <= 0:
            raise ValueError("HAWQ trace artifact parameter count is invalid")
        values = torch.tensor(estimates, dtype=torch.float64)
        mean = float(values.mean().item())
        if mean < 0.0:
            raise ValueError("HAWQ trace artifact mean must be nonnegative")
        standard_error = float(
            values.std(unbiased=True).div(math.sqrt(values.numel())).item()) \
            if values.numel() > 1 else 0.0
        coefficient = 0.0 if mean == 0.0 else \
            float(values.std(unbiased=False).item()) / mean
        _finite_close(row["mean"], mean, "trace mean")
        _finite_close(
            row["standard_error"], standard_error, "trace standard error")
        _finite_close(
            row["normalized_mean"], mean / float(parameters),
            "normalized trace")
        _finite_close(
            row["coefficient_of_variation"], coefficient,
            "trace coefficient of variation")
        traces.append(BlockTraceEstimate(
            block=str(row["block"]),
            estimates=estimates,
            mean=mean,
            standard_error=standard_error,
            normalized_mean=mean / float(parameters),
            coefficient_of_variation=coefficient,
            parameters=parameters,
        ))
    if tuple(row.block for row in traces) != contract.block_names:
        raise ValueError("HAWQ trace artifact block trace order changed")
    return tuple(traces)


def _problem_from_payload(payload, contract, bits):
    if set(payload["cost_basis"]) != {
            "blocks", "weight_macs", "activation_traffic"}:
        raise ValueError("HAWQ trace artifact cost basis fields changed")
    traces = _traces_from_payload(payload["traces"], contract)
    block_rows = payload["cost_basis"]["blocks"]
    blocks = []
    for row in block_rows:
        if set(row) != {
                "name", "weight_parameters", "weight_macs",
                "activation_traffic"}:
            raise ValueError("HAWQ trace artifact block cost fields changed")
        values = (
            row["weight_parameters"], row["weight_macs"],
            row["activation_traffic"])
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values) or values[0] <= 0 or values[1] <= 0 or \
                values[2] < 0:
            raise ValueError("HAWQ trace artifact block costs are invalid")
        blocks.append(HAWQBlockCost(
            str(row["name"]), values[0], values[1], values[2]))
    blocks = tuple(blocks)
    if tuple(block.name for block in blocks) != contract.block_names:
        raise ValueError("HAWQ trace artifact block cost order changed")
    if sum(block.activation_traffic for block in blocks) <= 0:
        raise ValueError("HAWQ trace artifact activation denominator is invalid")
    if any(block.weight_parameters != trace.parameters
           for block, trace in zip(blocks, traces)):
        raise ValueError("HAWQ trace artifact parameter costs differ from traces")
    weight_rows = payload["cost_basis"]["weight_macs"]
    if any(set(row) != {"module", "macs"} for row in weight_rows):
        raise ValueError("HAWQ trace artifact weight MAC fields changed")
    activation_rows = payload["cost_basis"]["activation_traffic"]
    if any(set(row) != {"site", "role", "elements"}
           for row in activation_rows):
        raise ValueError("HAWQ trace artifact activation cost fields changed")
    weight_macs = _cost_rows(tuple(
        (str(row["module"]), row["macs"])
        for row in weight_rows), "weight MAC")
    activation_traffic = _cost_rows(tuple(
        ((str(row["site"]), str(row["role"])), row["elements"])
        for row in activation_rows), "activation traffic")
    weight_map = dict(weight_macs)
    activation_map = dict(activation_traffic)
    if set(weight_map) != set(contract.weight_modules):
        raise ValueError("HAWQ trace artifact weight MAC coverage mismatch")
    expected_owners = set(
        owner for block in contract.blocks for owner in block.activation_owners)
    if set(activation_map) != expected_owners:
        raise ValueError("HAWQ trace artifact activation cost coverage mismatch")
    for block, costs in zip(contract.blocks, blocks):
        if costs.weight_macs != sum(
                weight_map[name] for name in block.weight_modules):
            raise ValueError("HAWQ trace artifact block MAC cost is inconsistent")
        if costs.activation_traffic != sum(
                activation_map[owner] for owner in block.activation_owners):
            raise ValueError(
                "HAWQ trace artifact block activation cost is inconsistent")
    objective = payload["objective"]
    if set(objective) != {"kind", "activation_sensitivity", "components"}:
        raise ValueError("HAWQ trace artifact objective fields changed")
    if objective["kind"] != \
            "weight_hessian_times_squared_quantization_error" or \
            objective["activation_sensitivity"] != "not_estimated":
        raise ValueError("HAWQ trace artifact objective identity changed")
    components = []
    expected_keys = tuple(
        (block, current_bits)
        for block in contract.block_names for current_bits in bits)
    for row in objective["components"]:
        if set(row) != {
                "block", "bits", "normalized_trace", "quantization_error",
                "cost"}:
            raise ValueError("HAWQ trace artifact objective component fields changed")
        if isinstance(row["bits"], bool) or not isinstance(row["bits"], int):
            raise ValueError("HAWQ trace artifact objective bits are invalid")
        component = HAWQWeightObjective(
            block=str(row["block"]),
            bits=row["bits"],
            normalized_trace=float(row["normalized_trace"]),
            quantization_error=float(row["quantization_error"]),
            cost=float(row["cost"]),
        )
        values = (
            component.normalized_trace,
            component.quantization_error,
            component.cost,
        )
        if not all(math.isfinite(value) for value in values) or \
                component.normalized_trace < 0.0 or \
                component.quantization_error < 0.0 or component.cost < 0.0:
            raise ValueError("HAWQ trace artifact objective must be nonnegative")
        components.append(component)
    components = tuple(components)
    if tuple((row.block, row.bits) for row in components) != expected_keys:
        raise ValueError("HAWQ trace artifact objective component coverage mismatch")
    trace_map = dict((row.block, row) for row in traces)
    for row in components:
        _finite_close(
            row.normalized_trace, trace_map[row.block].normalized_mean,
            "objective normalized trace")
        _finite_close(
            row.cost, row.normalized_trace * row.quantization_error,
            "objective component")
    return ContractHAWQProblem(
        blocks=blocks,
        traces=traces,
        objective_components=components,
        weight_macs=weight_macs,
        activation_traffic=activation_traffic,
    )


def _expected_trace_row_identities(problem, settings):
    return tuple(
        (batch_start, trace.block, probe)
        for batch_start in range(0, 128, settings.batch_size)
        for trace in problem.traces
        for probe in range(settings.probes_per_batch)
    )


def _validate_trace_row_values(rows, problem, settings) -> None:
    identities = tuple(
        (row["batch_start"], row["block"], row["probe"])
        for row in rows)
    if identities != _expected_trace_row_identities(problem, settings):
        raise ValueError("HAWQ trace row coverage differs from trace settings")
    grouped = dict((trace.block, []) for trace in problem.traces)
    for row in rows:
        grouped[row["block"]].append(row["estimate"])
    if any(tuple(grouped[trace.block]) != trace.estimates
           for trace in problem.traces):
        raise ValueError("HAWQ trace rows differ from trace summary")


def _validate_producer_trace_rows(traced, problem) -> None:
    rows = []
    for row in traced.raw_rows:
        if set(row) != {"batch_start", "block", "probe", "estimate"}:
            raise ValueError("HAWQ trace row fields changed")
        batch_start = row["batch_start"]
        probe = row["probe"]
        if isinstance(batch_start, bool) or not isinstance(batch_start, int) or \
                isinstance(probe, bool) or not isinstance(probe, int):
            raise ValueError("HAWQ trace row identity is invalid")
        estimate = float(row["estimate"])
        if not math.isfinite(estimate):
            raise ValueError("HAWQ trace row estimate is non-finite")
        rows.append({
            "batch_start": batch_start,
            "block": str(row["block"]),
            "probe": probe,
            "estimate": estimate,
        })
    _validate_trace_row_values(tuple(rows), problem, traced.settings)


def _validate_trace_files(
        artifact_path, payload, problem, settings) -> None:
    expected_files = {
        "trace_rows": "hawq_trace_rows.csv",
        "trace_summary": "hawq_trace_summary.csv",
        "objective_components": "hawq_objective_components.csv",
    }
    if set(payload["files"]) != set(expected_files):
        raise ValueError("HAWQ trace artifact file manifest changed")
    for key, filename in expected_files.items():
        row = payload["files"][key]
        if set(row) != {"path", "sha256"} or row["path"] != filename:
            raise ValueError("HAWQ trace artifact file identity changed")
        path = Path(artifact_path).parent / filename
        if not path.is_file() or _file_sha256(path) != row["sha256"]:
            raise ValueError("HAWQ trace artifact file fingerprint changed")
    with (Path(artifact_path).parent / expected_files["trace_rows"]).open(
            "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != \
                ("batch_start", "block", "probe", "estimate"):
            raise ValueError("HAWQ trace row CSV fields changed")
        rows = tuple(reader)
    normalized_rows = []
    for row in rows:
        block = str(row["block"])
        batch_start = int(row["batch_start"])
        probe = int(row["probe"])
        if str(batch_start) != row["batch_start"] or \
                str(probe) != row["probe"] or batch_start < 0 or probe < 0:
            raise ValueError("HAWQ trace row CSV identity is invalid")
        estimate = float(row["estimate"])
        if not math.isfinite(estimate):
            raise ValueError("HAWQ trace row CSV estimate is non-finite")
        normalized_rows.append({
            "batch_start": batch_start,
            "block": block,
            "probe": probe,
            "estimate": estimate,
        })
    _validate_trace_row_values(tuple(normalized_rows), problem, settings)
    with (Path(artifact_path).parent / expected_files["trace_summary"]).open(
            "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        summary_fields = (
            "block", "mean", "standard_error", "normalized_mean",
            "coefficient_of_variation", "parameters")
        if tuple(reader.fieldnames or ()) != summary_fields:
            raise ValueError("HAWQ trace summary CSV fields changed")
        summaries = tuple(reader)
    if tuple(row["block"] for row in summaries) != tuple(
            trace.block for trace in problem.traces):
        raise ValueError("HAWQ trace summary CSV block order changed")
    for row, trace in zip(summaries, problem.traces):
        if int(row["parameters"]) != trace.parameters:
            raise ValueError("HAWQ trace summary CSV parameters changed")
        for field in (
                "mean", "standard_error", "normalized_mean",
                "coefficient_of_variation"):
            _finite_close(row[field], getattr(trace, field),
                          "trace summary CSV %s" % field)
    with (Path(artifact_path).parent /
          expected_files["objective_components"]).open(
            "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        objective_fields = (
            "block", "bits", "normalized_trace", "quantization_error",
            "cost")
        if tuple(reader.fieldnames or ()) != objective_fields:
            raise ValueError("HAWQ objective CSV fields changed")
        objectives = tuple(reader)
    if tuple((row["block"], int(row["bits"])) for row in objectives) != tuple(
            (component.block, component.bits)
            for component in problem.objective_components):
        raise ValueError("HAWQ objective CSV component order changed")
    for row, component in zip(objectives, problem.objective_components):
        for field in ("normalized_trace", "quantization_error", "cost"):
            _finite_close(row[field], getattr(component, field),
                          "objective CSV %s" % field)


def write_trace_artifact(
        output: Path,
        model: nn.Module,
        contract: QuantizationModelContract,
        traced: HAWQTraceRun,
        weight_macs: Sequence[Tuple[str, int]],
        activation_traffic: Sequence[Tuple[Tuple[str, str], int]],
        bits: Sequence[int],
        model_name: str,
        calibration_identity: str) -> Path:
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("HAWQ trace output directory is missing: %s" % root)
    filenames = {
        "artifact": "hawq_trace_artifact.json",
        "trace_rows": "hawq_trace_rows.csv",
        "trace_summary": "hawq_trace_summary.csv",
        "objective_components": "hawq_objective_components.csv",
    }
    existing = tuple(root / filename for filename in filenames.values()
                     if (root / filename).exists())
    if existing:
        raise FileExistsError("HAWQ trace artifact file already exists: %s" %
                              existing[0])
    artifact_path = root / filenames["artifact"]
    if str(model_name) != contract.model_name:
        raise ValueError("HAWQ trace model differs from contract")
    if not isinstance(traced.settings, HAWQTraceSettings):
        raise TypeError("HAWQ trace settings must be validated")
    captured_checkpoint = traced.checkpoint_identity
    if not isinstance(captured_checkpoint, CheckpointIdentity):
        raise TypeError("HAWQ checkpoint identity must be captured before tracing")
    raw_indices = tuple(traced.calibration_indices)
    if len(raw_indices) != 128:
        raise ValueError("HAWQ trace requires exactly 128 calibration identities")
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in raw_indices):
        raise TypeError("HAWQ calibration identities must be integers")
    indices = _validate_indices(raw_indices, 1 + max(raw_indices))
    expected_identity = ordered_sample_identity_sha256("train", indices)
    if str(calibration_identity) != expected_identity:
        raise ValueError("HAWQ trace calibration identity does not match indices")
    problem = build_contract_hawq_problem(
        model, contract, traced.traces, weight_macs, activation_traffic, bits)
    _validate_producer_trace_rows(traced, problem)
    current_checkpoint = capture_checkpoint_identity(
        captured_checkpoint.path)
    if current_checkpoint != captured_checkpoint:
        raise ValueError("HAWQ checkpoint identity changed during tracing")
    _write_trace_rows(root, traced, problem)
    payload = {
        "format_version": 3,
        "artifact_kind": "nyu_contract_hawq_trace",
        "model_name": contract.model_name,
        "checkpoint": _checkpoint_payload(captured_checkpoint),
        "trace_settings": _trace_settings_payload(traced.settings),
        "hessian_vector": _hessian_vector_payload(contract.model_name),
        "calibration": {
            "count": len(indices),
            "indices": list(indices),
            "identity_sha256": str(calibration_identity),
        },
        "bits": list(int(value) for value in bits),
        "contract": _contract_payload(contract),
        "traces": [_trace_payload(row) for row in problem.traces],
        "cost_basis": {
            "blocks": [vars(row) for row in problem.blocks],
            "weight_macs": [
                {"module": module, "macs": macs}
                for module, macs in problem.weight_macs],
            "activation_traffic": [
                {"site": owner[0], "role": owner[1], "elements": elements}
                for owner, elements in problem.activation_traffic],
        },
        "objective": {
            "kind": "weight_hessian_times_squared_quantization_error",
            "activation_sensitivity": "not_estimated",
            "components": [vars(row) for row in problem.objective_components],
        },
        "files": dict(
            (key, {"path": filename,
                   "sha256": _file_sha256(root / filename)})
            for key, filename in filenames.items() if key != "artifact"),
    }
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def load_trace_artifact(
        path: Path,
        expected_model_name: str,
        expected_checkpoint: Path,
        expected_calibration_indices: Sequence[int],
        expected_calibration_identity: str,
        expected_trace_settings: HAWQTraceSettings,
        bits: Sequence[int]):
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_fields = {
        "format_version", "artifact_kind", "model_name", "checkpoint",
        "trace_settings", "hessian_vector", "calibration", "bits",
        "contract", "traces", "cost_basis", "objective", "files",
    }
    if set(payload) != expected_fields or isinstance(
            payload["format_version"], bool) or not isinstance(
                payload["format_version"], int) or \
            payload["format_version"] != 3 or \
            payload["artifact_kind"] != "nyu_contract_hawq_trace":
        raise ValueError("HAWQ trace artifact identity fields changed")
    if payload["model_name"] != str(expected_model_name):
        raise ValueError("HAWQ trace artifact model identity changed")
    if payload["hessian_vector"] != _hessian_vector_payload(
            str(expected_model_name)):
        raise ValueError("HAWQ trace Hessian-vector settings changed")
    current_checkpoint = capture_checkpoint_identity(expected_checkpoint)
    if set(payload["checkpoint"]) != {"path", "size_bytes", "sha256"} or \
            payload["checkpoint"] != _checkpoint_payload(current_checkpoint):
        raise ValueError("HAWQ trace artifact checkpoint identity changed")
    if not isinstance(expected_trace_settings, HAWQTraceSettings):
        raise TypeError("expected HAWQ trace settings must be validated")
    settings = _trace_settings_from_payload(payload["trace_settings"])
    if settings != expected_trace_settings:
        raise ValueError("HAWQ trace settings differ from explicit config")
    indices = tuple(expected_calibration_indices)
    expected_identity = ordered_sample_identity_sha256("train", indices)
    if str(expected_calibration_identity) != expected_identity:
        raise ValueError("expected HAWQ calibration identity is invalid")
    calibration = payload["calibration"]
    if set(calibration) != {"count", "indices", "identity_sha256"} or \
            calibration["count"] != 128 or \
            tuple(calibration["indices"]) != indices or \
            calibration["identity_sha256"] != expected_identity:
        raise ValueError("HAWQ trace artifact calibration identity changed")
    declared_bits = tuple(int(value) for value in bits)
    if declared_bits != BITS or tuple(payload["bits"]) != declared_bits:
        raise ValueError("HAWQ trace artifact bit candidates changed")
    contract = _contract_from_payload(payload["contract"])
    if contract.model_name != str(expected_model_name):
        raise ValueError("HAWQ trace artifact contract model changed")
    problem = _problem_from_payload(payload, contract, declared_bits)
    _validate_trace_files(artifact_path, payload, problem, settings)
    return contract, problem, indices, expected_identity


def _validate_mixed_le6_budgets(
        maximum_weight_bits: float,
        maximum_activation_bits: float) -> Tuple[float, float]:
    weight = float(maximum_weight_bits)
    activation = float(maximum_activation_bits)
    if not math.isfinite(weight) or weight <= 0.0 or weight > 6.0:
        raise ValueError("mixed_le6 weight budget must be in (0, 6]")
    if not math.isfinite(activation) or activation <= 0.0 or activation > 6.0:
        raise ValueError("mixed_le6 activation budget must be in (0, 6]")
    return weight, activation


def allocate_trace_artifact(
        trace_artifact: Path,
        output: Path,
        expected_model_name: str,
        expected_checkpoint: Path,
        expected_calibration_indices: Sequence[int],
        expected_calibration_identity: str,
        expected_trace_settings: HAWQTraceSettings,
        bits: Sequence[int],
        maximum_weight_bits: float,
        maximum_activation_bits: float) -> Path:
    weight_budget, activation_budget = _validate_mixed_le6_budgets(
        maximum_weight_bits, maximum_activation_bits)
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("HAWQ allocation output is missing: %s" % root)
    if any(root.iterdir()):
        raise RuntimeError("HAWQ allocation output directory must be empty")
    checkpoint_identity = capture_checkpoint_identity(expected_checkpoint)
    trace_sha256 = _file_sha256(trace_artifact)
    contract, problem, indices, identity = load_trace_artifact(
        trace_artifact,
        expected_model_name,
        expected_checkpoint,
        expected_calibration_indices,
        expected_calibration_identity,
        expected_trace_settings,
        bits,
    )
    result = _solve_contract_hawq_problem(
        problem, weight_budget, activation_budget)
    if capture_checkpoint_identity(expected_checkpoint) != checkpoint_identity:
        raise ValueError("HAWQ checkpoint changed during allocation")
    if _file_sha256(trace_artifact) != trace_sha256:
        raise ValueError("HAWQ trace artifact changed during allocation")
    return write_hawq_assignment(
        root,
        contract,
        result,
        indices,
        identity,
        checkpoint_identity,
        expected_trace_settings,
        trace_artifact,
    )


def _assignment_payload(contract, assignment):
    weight_blocks = dict(assignment.weight_block_bits)
    activation_blocks = dict(assignment.activation_block_bits)
    return {
        "model_name": contract.model_name,
        "weight_block_bits": [
            {"block": name, "bits": bits}
            for name, bits in assignment.weight_block_bits],
        "activation_block_bits": [
            {"block": name, "bits": bits}
            for name, bits in assignment.activation_block_bits],
        "weight_bits": [
            {"module": module, "bits": weight_blocks[block.name]}
            for block in contract.blocks for module in block.weight_modules],
        "activation_bits": [
            {"site": owner[0], "role": owner[1],
             "bits": activation_blocks[block.name]}
            for block in contract.blocks for owner in block.activation_owners],
    }


def write_hawq_assignment(
        output: Path,
        contract: QuantizationModelContract,
        result: ContractHAWQResult,
        calibration_indices: Sequence[int],
        calibration_identity: str,
        checkpoint_identity: CheckpointIdentity,
        trace_settings: HAWQTraceSettings,
        trace_artifact: Path) -> Path:
    _validate_mixed_le6_budgets(
        result.maximum_weight_bits, result.maximum_activation_bits)
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("HAWQ output directory is missing: %s" % root)
    path = root / "hawq_mixed_le6_assignment.json"
    if path.exists():
        raise FileExistsError("HAWQ assignment already exists: %s" % path)
    raw_indices = tuple(calibration_indices)
    if len(raw_indices) != 128:
        raise ValueError("HAWQ trace requires exactly 128 calibration identities")
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in raw_indices):
        raise TypeError("HAWQ calibration identities must be integers")
    indices = _validate_indices(raw_indices, 1 + max(raw_indices))
    expected_identity = ordered_sample_identity_sha256("train", indices)
    if str(calibration_identity) != expected_identity:
        raise ValueError("HAWQ calibration identity does not match indices")
    checkpoint = _checkpoint_payload(checkpoint_identity)
    if capture_checkpoint_identity(checkpoint_identity.path) != \
            checkpoint_identity:
        raise ValueError("HAWQ checkpoint identity changed before publication")
    settings = _trace_settings_payload(trace_settings)
    trace_path = Path(trace_artifact)
    trace_sha256 = _file_sha256(trace_path)
    trace_contract, trace_problem, trace_indices, trace_identity = \
        load_trace_artifact(
            trace_path,
            expected_model_name=contract.model_name,
            expected_checkpoint=checkpoint_identity.path,
            expected_calibration_indices=indices,
            expected_calibration_identity=expected_identity,
            expected_trace_settings=trace_settings,
            bits=BITS,
        )
    if trace_contract != contract or trace_indices != indices or \
            trace_identity != expected_identity:
        raise ValueError("HAWQ trace linkage identity differs")
    if trace_problem.blocks != result.blocks or \
            trace_problem.traces != result.traces or \
            trace_problem.objective_components != \
            result.objective_components or \
            trace_problem.weight_macs != result.weight_macs or \
            trace_problem.activation_traffic != result.activation_traffic:
        raise ValueError("HAWQ assignment differs from trace artifact")
    if _file_sha256(trace_path) != trace_sha256:
        raise ValueError("HAWQ trace artifact changed during publication")
    selected = dict(
        ((term.block, term.bits), term)
        for term in result.objective_components)
    selected_components = tuple(
        selected[(block, bits)]
        for block, bits in result.assignment.weight_block_bits)
    payload = {
        "model_name": contract.model_name,
        "provenance": {
            "checkpoint": checkpoint,
            "trace_settings": settings,
            "trace_artifact_sha256": trace_sha256,
        },
        "calibration": {
            "count": len(indices),
            "indices": list(indices),
            "identity_sha256": str(calibration_identity),
        },
        "contract": {
            "blocks": list(contract.block_names),
            "protected_roles": list(contract.protected_roles),
            "protected_modules": list(contract.protected_modules),
            "attention_edges": list(contract.attention_edges),
            "concat_edges": list(contract.concat_edges),
        },
        "average_weight_bits": result.assignment.average_weight_bits,
        "average_weight_mac_bits":
            result.assignment.average_weight_mac_bits,
        "average_activation_bits":
            result.assignment.average_activation_bits,
        "assignment": _assignment_payload(contract, result.assignment),
        "objective": {
            "kind": "weight_hessian_times_squared_quantization_error",
            "activation_sensitivity": "not_estimated",
            "total": result.assignment.objective,
            "components": [vars(row)
                           for row in result.objective_components],
            "selected_components": [vars(row)
                                    for row in selected_components],
        },
        "constraints": {
            "maximum_average_weight_bits": result.maximum_weight_bits,
            "maximum_average_activation_bits":
                result.maximum_activation_bits,
            "average_weight_parameter_bits":
                result.assignment.average_weight_bits,
            "average_weight_mac_bits":
                result.assignment.average_weight_mac_bits,
            "average_activation_traffic_bits":
                result.assignment.average_activation_bits,
            "weight_parameter_residual":
                result.assignment.weight_parameter_budget_residual,
            "weight_mac_residual":
                result.assignment.weight_mac_budget_residual,
            "activation_traffic_residual":
                result.assignment.activation_budget_residual,
        },
        "cost_basis": {
            "weight_parameters": [
                {"block": block.name,
                 "parameters": block.weight_parameters}
                for block in result.blocks],
            "weight_macs": [
                {"module": module, "macs": macs}
                for module, macs in result.weight_macs],
            "activation_traffic": [
                {"site": owner[0], "role": owner[1], "elements": elements}
                for owner, elements in result.activation_traffic],
        },
        "solver_success": True,
        "solver_status": "optimal",
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _read_weight_cost_rows(path: Path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("module", "macs"):
            raise ValueError("weight costs require module,macs columns")
        rows = tuple(
            (str(row["module"]), int(row["macs"])) for row in reader)
    return _cost_rows(rows, "weight MAC")


def _read_activation_cost_rows(path: Path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("site", "role", "elements"):
            raise ValueError(
                "activation costs require site,role,elements columns")
        rows = tuple(
            ((str(row["site"]), str(row["role"])), int(row["elements"]))
            for row in reader)
    return _cost_rows(rows, "activation traffic")


def _write_trace_rows(output, traced, result):
    with (output / "hawq_trace_rows.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("batch_start", "block", "probe", "estimate"))
        writer.writeheader()
        writer.writerows(traced.raw_rows)
    with (output / "hawq_trace_summary.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = (
            "block", "mean", "standard_error", "normalized_mean",
            "coefficient_of_variation", "parameters")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in traced.traces:
            writer.writerow(dict((field, getattr(row, field))
                                 for field in fields))
    with (output / "hawq_objective_components.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = (
            "block", "bits", "normalized_trace", "quantization_error",
            "cost")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(vars(row) for row in result.objective_components)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run contract-driven NYU HAWQ trace or allocation phase")
    parser.add_argument(
        "--phase", choices=("trace", "allocate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device")
    parser.add_argument("--weight-cost-rows", type=Path)
    parser.add_argument("--activation-cost-rows", type=Path)
    parser.add_argument("--trace-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--probes-per-batch", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--depth-mse-weight", type=float)
    parser.add_argument("--boundary-mse-weight", type=float)
    parser.add_argument("--boundary-threshold-m", type=float)
    return parser


def _require_phase_arguments(args, phase, names) -> None:
    missing = tuple(name for name in names if getattr(args, name) is None)
    if missing:
        raise ValueError(
            "HAWQ %s phase arguments are missing: %s" % (phase, missing))


def _reject_phase_arguments(args, phase, names) -> None:
    present = tuple(name for name in names if getattr(args, name) is not None)
    if present:
        raise ValueError(
            "HAWQ %s phase arguments are not allowed: %s" % (phase, present))


def run_cli(argv, dependencies: RunnerDependencies):
    args = build_parser().parse_args(tuple(argv))
    selected = load_selected_quantization_config(args.config)
    model_rows = tuple(
        model for model in selected.models if model.model == args.model)
    if len(model_rows) != 1:
        raise ValueError("selected HAWQ model entry is not unique")
    model_config = model_rows[0]
    if not args.output.is_dir():
        raise FileNotFoundError("HAWQ output directory is missing: %s" %
                                args.output)
    if any(args.output.iterdir()):
        raise RuntimeError("HAWQ output directory must be empty")
    method = selected.method_hyperparameters["hawq_mixed_le6"]
    if int(method["calibration_count"]) != 128:
        raise ValueError("HAWQ calibration count must equal 128")
    configured_settings = _trace_settings_from_method(method)
    if args.phase == "allocate":
        _require_phase_arguments(args, "allocate", ("trace_artifact",))
        _reject_phase_arguments(args, "allocate", (
            "device", "weight_cost_rows", "activation_cost_rows",
            "batch_size", "probes_per_batch", "seed", "depth_mse_weight",
            "boundary_mse_weight", "boundary_threshold_m",
        ))
        identity = load_persisted_calibration_identity(
            model_config.calibration_metadata,
            model_config.evaluation_indices,
        )
        return allocate_trace_artifact(
            args.trace_artifact,
            args.output,
            expected_model_name=model_config.model,
            expected_checkpoint=model_config.checkpoint,
            expected_calibration_indices=identity.indices,
            expected_calibration_identity=identity.sha256,
            expected_trace_settings=configured_settings,
            bits=method["bits"],
            maximum_weight_bits=method["maximum_average_weight_bits"],
            maximum_activation_bits=
                method["maximum_average_activation_bits"],
        )
    _require_phase_arguments(args, "trace", (
        "device", "weight_cost_rows", "activation_cost_rows", "batch_size",
        "probes_per_batch", "seed", "depth_mse_weight",
        "boundary_mse_weight", "boundary_threshold_m",
    ))
    _reject_phase_arguments(args, "trace", ("trace_artifact",))
    if str(args.device) != model_config.device:
        raise ValueError("explicit HAWQ device differs from model config")
    explicit_settings = HAWQTraceSettings(
        batch_size=args.batch_size,
        probes_per_batch=args.probes_per_batch,
        seed=args.seed,
        depth_mse_weight=args.depth_mse_weight,
        boundary_mse_weight=args.boundary_mse_weight,
        boundary_threshold_m=args.boundary_threshold_m,
    )
    if explicit_settings != configured_settings:
        raise ValueError("explicit HAWQ trace settings differ from config")
    weight_macs = _read_weight_cost_rows(args.weight_cost_rows)
    activation_traffic = _read_activation_cost_rows(
        args.activation_cost_rows)
    checkpoint_identity = capture_checkpoint_identity(
        model_config.checkpoint)
    runtime = dependencies.runtime_factory(model_config)
    try:
        model = runtime.build_model(runtime.device)
        contract = dependencies.contract_builder(runtime.model_name, model)
        dataset = runtime.build_dataset("train")
        identity = load_calibration_identity(
            model_config.calibration_metadata,
            model_config.evaluation_indices,
            len(dataset),
        )
        traced = trace_calibration_batches(
            runtime, model, contract, dataset, identity.indices,
            configured_settings, checkpoint_identity)
        return write_trace_artifact(
            args.output,
            model=model,
            contract=contract,
            traced=traced,
            weight_macs=weight_macs,
            activation_traffic=activation_traffic,
            bits=method["bits"],
            model_name=model_config.model,
            calibration_identity=identity.sha256,
        )
    finally:
        runtime.close()


PRODUCTION_DEPENDENCIES = RunnerDependencies(
    runtime_factory=NYUModelRuntime.from_config,
    contract_builder=build_model_quantization_contract,
)


def main(argv=None) -> None:
    run_cli(sys.argv[1:] if argv is None else argv, PRODUCTION_DEPENDENCIES)


if __name__ == "__main__":
    main()
