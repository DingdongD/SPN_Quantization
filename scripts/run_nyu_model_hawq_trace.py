#!/usr/bin/env python3
"""Run contract-driven HAWQ tracing for selected official NYU models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
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
from spn_quant.hawq_allocation import (  # noqa: E402
    BITS,
    HAWQAssignment,
    HAWQCandidate,
    HAWQIndependentBlock,
    solve_independent_hawq_assignment,
    weight_quantization_error,
)
from spn_quant.hawq_trace import (  # noqa: E402
    BlockTraceEstimate,
    HutchinsonTraceConfig,
    estimate_parameter_block_trace_samples,
    masked_curvature_loss,
)
from spn_quant.model_contracts import (  # noqa: E402
    QuantizationModelContract,
    build_model_quantization_contract,
)


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
class HAWQTraceRun:
    traces: Tuple[BlockTraceEstimate, ...]
    raw_rows: Tuple[dict, ...]
    calibration_indices: Tuple[int, ...]


@dataclass(frozen=True)
class HAWQWeightObjective:
    block: str
    bits: int
    normalized_trace: float
    quantization_error: float
    cost: float


@dataclass(frozen=True)
class ContractHAWQResult:
    blocks: Tuple[HAWQIndependentBlock, ...]
    traces: Tuple[BlockTraceEstimate, ...]
    objective_components: Tuple[HAWQWeightObjective, ...]
    assignment: HAWQAssignment
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
        settings: HAWQTraceSettings) -> HAWQTraceRun:
    if runtime.device != next(model.parameters()).device:
        raise ValueError("HAWQ runtime and model devices differ")
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

        samples = estimate_parameter_block_trace_samples(
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
        ) -> Tuple[HAWQIndependentBlock, ...]:
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
        HAWQIndependentBlock(
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


def allocate_contract_hawq(
        model: nn.Module,
        contract: QuantizationModelContract,
        traces: Sequence[BlockTraceEstimate],
        weight_macs: Sequence[Tuple[str, int]],
        activation_traffic: Sequence[Tuple[Tuple[str, str], int]],
        bits: Sequence[int],
        maximum_weight_bits: float,
        maximum_activation_bits: float) -> ContractHAWQResult:
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
    candidates = []
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
            candidates.append(HAWQCandidate(
                block=block.name, bits=current_bits, cost=cost))
    assignment = solve_independent_hawq_assignment(
        blocks,
        tuple(candidates),
        maximum_weight_bits,
        maximum_activation_bits,
    )
    return ContractHAWQResult(
        blocks=blocks,
        traces=trace_rows,
        objective_components=tuple(components),
        assignment=assignment,
        weight_macs=tuple(weight_macs),
        activation_traffic=tuple(activation_traffic),
        maximum_weight_bits=float(maximum_weight_bits),
        maximum_activation_bits=float(maximum_activation_bits),
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
        calibration_identity: str) -> Path:
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
    selected = dict(
        ((term.block, term.bits), term)
        for term in result.objective_components)
    selected_components = tuple(
        selected[(block, bits)]
        for block, bits in result.assignment.weight_block_bits)
    payload = {
        "model_name": contract.model_name,
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
        "solver_status": result.assignment.solver_status,
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
        description="Run contract-driven NYU HAWQ tracing")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--weight-cost-rows", type=Path, required=True)
    parser.add_argument("--activation-cost-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--probes-per-batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth-mse-weight", type=float, required=True)
    parser.add_argument("--boundary-mse-weight", type=float, required=True)
    parser.add_argument("--boundary-threshold-m", type=float, required=True)
    return parser


def run_cli(argv, dependencies: RunnerDependencies):
    args = build_parser().parse_args(tuple(argv))
    selected = load_selected_quantization_config(args.config)
    model_rows = tuple(
        model for model in selected.models if model.model == args.model)
    if len(model_rows) != 1:
        raise ValueError("selected HAWQ model entry is not unique")
    model_config = model_rows[0]
    if str(args.device) != model_config.device:
        raise ValueError("explicit HAWQ device differs from model config")
    if not args.output.is_dir():
        raise FileNotFoundError("HAWQ output directory is missing: %s" %
                                args.output)
    if any(args.output.iterdir()):
        raise RuntimeError("HAWQ output directory must be empty")
    settings = HAWQTraceSettings(
        batch_size=args.batch_size,
        probes_per_batch=args.probes_per_batch,
        seed=args.seed,
        depth_mse_weight=args.depth_mse_weight,
        boundary_mse_weight=args.boundary_mse_weight,
        boundary_threshold_m=args.boundary_threshold_m,
    )
    method = selected.method_hyperparameters["hawq_mixed_le6"]
    if int(method["calibration_count"]) != 128:
        raise ValueError("HAWQ calibration count must equal 128")
    weight_macs = _read_weight_cost_rows(args.weight_cost_rows)
    activation_traffic = _read_activation_cost_rows(
        args.activation_cost_rows)
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
            runtime, model, contract, dataset, identity.indices, settings)
        result = allocate_contract_hawq(
            model=model,
            contract=contract,
            traces=traced.traces,
            weight_macs=weight_macs,
            activation_traffic=activation_traffic,
            bits=method["bits"],
            maximum_weight_bits=method["maximum_average_weight_bits"],
            maximum_activation_bits=
                method["maximum_average_activation_bits"],
        )
        _write_trace_rows(args.output, traced, result)
        return write_hawq_assignment(
            args.output, contract, result,
            identity.indices, identity.sha256)
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
