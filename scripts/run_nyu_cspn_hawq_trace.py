#!/usr/bin/env python3
"""Estimate official CSPN HAWQ traces and solve the mixed-bit assignment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts import run_nyu_cspn_task_sensitive_bits as task_bits  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    calibration_dataset,
    seeded_sample,
)
from spn_quant.activation_boundaries import (  # noqa: E402
    CSPNActivationBoundaryController,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.cspn_task_sensitive_bits import (  # noqa: E402
    AllocationRegistry,
    BitAssignment,
)
from spn_quant.hawq_allocation import (  # noqa: E402
    HAWQAssignment,
    HAWQBlock,
    HAWQCandidate,
    candidate_cost,
    solve_hawq_assignment,
)
from spn_quant.hawq_trace import (  # noqa: E402
    BlockTraceEstimate,
    HutchinsonTraceConfig,
    estimate_block_trace_samples,
    masked_curvature_loss,
)
from spn_quant.qat.method_config import load_method_config  # noqa: E402


BLOCK_ALIASES = {
    "stem": "encoder_stem",
    "encoder_layer1": "encoder_layer1",
    "encoder_layer2": "encoder_layer2",
    "encoder_layer3": "encoder_layer3",
    "encoder_layer4": "encoder_layer4",
    "decoder_layer1": "decoder_layer1",
    "decoder_layer2": "decoder_layer2",
    "decoder_layer3": "decoder_layer3",
    "decoder_layer4": "decoder_layer4",
    "initial_depth": "initial_depth",
}


@dataclass(frozen=True)
class HAWQTraceBlock:
    name: str
    weight_modules: Tuple[str, ...]
    activation_owners: Tuple[Tuple[str, str], ...]
    fixed_eight: bool


@dataclass(frozen=True)
class HAWQAllocationResult:
    blocks: Tuple[HAWQBlock, ...]
    traces: Tuple[BlockTraceEstimate, ...]
    candidates: Tuple[HAWQCandidate, ...]
    assignment: HAWQAssignment


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args(argv)


def expected_registry() -> AllocationRegistry:
    return task_bits.expected_registry()


def hawq_block_contract(
        registry: AllocationRegistry) -> Tuple[HAWQTraceBlock, ...]:
    fixed = {"encoder_stem", "initial_depth"}
    blocks = []
    for source_name in task_bits.allocation.BLOCK_ORDER:
        name = BLOCK_ALIASES[source_name]
        weights = tuple(registry.weights_by_block[source_name])
        activations = tuple(registry.activations_by_block[source_name])
        if any(module.startswith("gud_up_proj_layer6") for module in weights):
            raise RuntimeError("guidance entered the HAWQ block contract")
        blocks.append(HAWQTraceBlock(
            name, weights, activations, name in fixed))
    return tuple(blocks)


def validate_calibration_indices(indices: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(int(index) for index in indices)
    if len(values) != 128:
        raise ValueError("HAWQ trace requires 128 calibration indices")
    if len(values) != len(set(values)):
        raise ValueError("HAWQ trace calibration indices must be unique")
    if any(index < 0 for index in values):
        raise ValueError("HAWQ trace calibration indices must be nonnegative")
    return values


def expand_assignment(
        registry: AllocationRegistry,
        block_bits: Sequence[Tuple[str, int]]) -> BitAssignment:
    selected = dict((str(name), int(bits)) for name, bits in block_bits)
    if set(selected) != set(BLOCK_ALIASES.values()):
        raise ValueError("HAWQ block assignment coverage mismatch")
    weight_bits = []
    activation_bits = []
    for source_name in task_bits.allocation.BLOCK_ORDER:
        bits = selected[BLOCK_ALIASES[source_name]]
        weight_bits.extend(
            (module, bits)
            for module in registry.weights_by_block[source_name])
        activation_bits.extend(
            (owner, bits)
            for owner in registry.activations_by_block[source_name])
    assignment = BitAssignment(
        weight_bits=tuple(weight_bits),
        activation_bits=tuple(activation_bits))
    task_bits.validate_assignment_contract(assignment)
    return assignment


def _block_cost(block: HAWQTraceBlock, traces, modules, bits: int) -> float:
    return sum(
        candidate_cost(
            traces[name].normalized_mean,
            modules[name].weight,
            bits,
            1 if isinstance(modules[name], nn.ConvTranspose2d) else 0,
        )
        for name in block.weight_modules)


def allocate_from_trace_summary(
        model: nn.Module,
        blocks: Sequence[HAWQTraceBlock],
        traces: Sequence[BlockTraceEstimate],
        activation_elements: Sequence[Tuple[Tuple[str, str], int]],
        bits: Sequence[int],
        maximum_weight_bits: float,
        maximum_activation_bits: float) -> HAWQAllocationResult:
    declared_bits = tuple(int(value) for value in bits)
    if declared_bits != (4, 6, 8):
        raise ValueError("HAWQ allocation bits must be exactly (4, 6, 8)")
    blocks = tuple(blocks)
    trace_rows = tuple(traces)
    trace_map = dict((row.block, row) for row in trace_rows)
    modules = dict(model.named_modules())
    required_modules = set(
        name for block in blocks for name in block.weight_modules)
    if set(trace_map) != required_modules:
        raise ValueError("HAWQ trace coverage mismatch")
    activation_map = dict(
        ((str(owner[0]), str(owner[1])), int(elements))
        for owner, elements in activation_elements)
    required_owners = set(
        owner for block in blocks for owner in block.activation_owners)
    if set(activation_map) != required_owners:
        raise ValueError("HAWQ activation cost coverage mismatch")
    allocation_blocks = tuple(
        HAWQBlock(
            block.name,
            sum(int(modules[name].weight.numel())
                for name in block.weight_modules),
            sum(activation_map[owner] for owner in block.activation_owners),
            block.fixed_eight,
        )
        for block in blocks)
    candidates = tuple(
        HAWQCandidate(
            block.name, current_bits,
            _block_cost(block, trace_map, modules, current_bits))
        for block in blocks for current_bits in declared_bits)
    assignment = solve_hawq_assignment(
        allocation_blocks,
        candidates,
        maximum_weight_bits,
        maximum_activation_bits,
        (),
    )
    return HAWQAllocationResult(
        allocation_blocks, trace_rows, candidates, assignment)


def _write_csv(path: Path, rows, fields) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((field, row[field]) for field in fields))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_allocation_artifacts(
        output: Path, result: HAWQAllocationResult) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace_rows = tuple({
        "module": row.block,
        "trace_mean": row.mean,
        "trace_standard_error": row.standard_error,
        "normalized_trace": row.normalized_mean,
        "coefficient_of_variation": row.coefficient_of_variation,
        "parameters": row.parameters,
    } for row in result.traces)
    _write_csv(
        output / "trace_summary.csv",
        trace_rows,
        ("module", "trace_mean", "trace_standard_error",
         "normalized_trace", "coefficient_of_variation", "parameters"))
    candidate_rows = tuple({
        "block": row.block,
        "bits": row.bits,
        "cost": row.cost,
    } for row in result.candidates)
    _write_csv(
        output / "candidate_costs.csv", candidate_rows,
        ("block", "bits", "cost"))
    _write_json(output / "cost_basis.json", {
        "blocks": [{
            "block": block.name,
            "weight_elements": block.weight_elements,
            "activation_elements": block.activation_elements,
            "fixed_eight": block.fixed_eight,
        } for block in result.blocks],
    })
    _write_json(output / "selected_assignment.json", {
        "block_bits": [
            {"block": name, "bits": bits}
            for name, bits in result.assignment.block_bits],
        "average_weight_bits": result.assignment.average_weight_bits,
        "average_activation_bits": result.assignment.average_activation_bits,
        "objective": result.assignment.objective,
        "solver_status": result.assignment.solver_status,
    })


def _load_calibration_indices(path: Path) -> Tuple[int, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["calibration_source"]["selection"] != \
            "32_tail_96_kmedoids":
        raise ValueError("HAWQ trace requires stratified calibration indices")
    return validate_calibration_indices(payload["calibration_indices"])


def _saved_args(checkpoint: Path, args, config):
    payload = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False)
    saved = SimpleNamespace(**payload["args"])
    if saved.model != "cspn":
        raise ValueError("HAWQ trace checkpoint must contain official CSPN")
    saved.from_scratch = True
    saved.data_root = args.data_root
    saved.device = args.device
    saved.seed = config.hawq.trace.seed
    saved.batch_size = config.hawq.trace.batch_size
    saved.val_batch_size = 1
    saved.workers = config.training.workers
    return saved


def _activation_element_counts(
        instrumentor, boundary_controller,
        samples: int) -> Tuple[Tuple[Tuple[str, str], int], ...]:
    counts = {}
    for key in instrumentor.activation_site_keys(base.ORDINARY_GROUPS):
        owner = base.activation_owner(key)
        observer = instrumentor.channel_observers[key] \
            if not isinstance(key, str) else \
            instrumentor.relu_channel_observers[key]
        channels = int(observer.minimum.numel())
        total = int(observer.scalar_count) * channels
        if total % samples:
            raise RuntimeError("activation element count is not sample-aligned")
        if owner in counts:
            raise RuntimeError("activation element owner is duplicated")
        counts[owner] = total // samples
    for name in boundary_controller.channels:
        owner = "boundary_controller.%s" % name, "boundary"
        total = int(boundary_controller.observers[name].scalar_count)
        if total % samples:
            raise RuntimeError("boundary element count is not sample-aligned")
        counts[owner] = total // samples
    return tuple(sorted(counts.items()))


def _summarize_traces(rows, modules) -> Tuple[BlockTraceEstimate, ...]:
    observed = set(row.block for row in rows)
    if observed != set(modules):
        raise ValueError("HAWQ trace summary coverage mismatch")
    grouped = dict((name, []) for name in modules)
    for row in rows:
        grouped[row.block].extend(row.estimates)
    output = []
    for name in modules:
        values = torch.tensor(grouped[name], dtype=torch.float64)
        mean = float(values.mean().item())
        if mean < 0.0:
            raise ValueError("HAWQ aggregate trace is negative: %s" % name)
        standard_error = float(
            values.std(unbiased=True).div(math.sqrt(values.numel())).item()) \
            if values.numel() > 1 else 0.0
        coefficient = 0.0 if mean == 0.0 else \
            float(values.std(unbiased=False).item()) / mean
        parameters = int(modules[name].weight.numel())
        output.append(BlockTraceEstimate(
            name, tuple(float(value) for value in values.tolist()),
            mean, standard_error, mean / float(parameters), coefficient,
            parameters))
    return tuple(output)


def _trace_official(args, config, indices, blocks):
    device = torch.device(args.device)
    saved = _saved_args(Path(args.checkpoint), args, config)
    model, architecture, load_report = base._load_cspn(
        saved, Path(args.checkpoint), device)
    dataset = calibration_dataset(saved)
    if max(indices) >= len(dataset):
        raise ValueError("HAWQ calibration index exceeds NYU train split")
    sample = seeded_sample(dataset, indices[0], config.hawq.trace.seed)
    model_args = base._model_args(saved, sample, device)
    preparation = prepare_hardware_model(
        model, model_args, excluded_pairs=(("conv1_1", "bn1"),))
    if preparation["primary_max_abs_error"] > config.training.fold_max_error:
        raise RuntimeError("Conv-BN fold exceeds declared error threshold")
    semantic = install_model_semantic_adapter(model, "cspn", strict=True)
    boundaries = semantic.activation_boundaries()
    semantic.close()
    instrumentor = HardwareAlignedInstrumentor(
        model, base.cspn_quant_group,
        preparation["fused_relu_producers"],
        externally_owned_outputs=base.strict_owned_outputs(),
        externally_owned_inputs=base.strict_owned_inputs())
    boundary_controller = CSPNActivationBoundaryController(model, boundaries)
    named = dict(model.named_modules())
    module_names = tuple(
        name for block in blocks for name in block.weight_modules)
    trace_rows = []
    raw_rows = []
    batch_size = config.hawq.trace.batch_size
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = default_collate(tuple(
            seeded_sample(dataset, index, config.hawq.trace.seed)
            for index in batch_indices))
        model_input, target = sweep.batch_to_model_input(
            "cspn", batch, device)
        instrumentor.observe()
        boundary_controller.observe()
        with torch.no_grad():
            model(*model_input)
        instrumentor.mode = "bypass"
        boundary_controller.mode = "bypass"

        def loss_fn():
            prediction = sweep.extract_pred(model(*model_input))
            return masked_curvature_loss(
                prediction, target, target > 0.0,
                config.hawq.trace.depth_mse_weight,
                config.hawq.trace.boundary_mse_weight,
                config.hawq.trace.boundary_threshold_m)

        samples = estimate_block_trace_samples(
            tuple((name, named[name].weight) for name in module_names),
            loss_fn,
            HutchinsonTraceConfig(
                config.hawq.trace.probes_per_batch,
                config.hawq.trace.seed + start // batch_size),
        )
        estimates = []
        for name, values in samples:
            current = torch.tensor(values, dtype=torch.float64)
            mean = float(current.mean().item())
            standard_error = float(
                current.std(unbiased=True).div(
                    math.sqrt(current.numel())).item()) \
                if current.numel() > 1 else 0.0
            coefficient = 0.0 if mean == 0.0 else \
                float(current.std(unbiased=False).item()) / abs(mean)
            parameters = int(named[name].weight.numel())
            estimates.append(BlockTraceEstimate(
                name, values, mean, standard_error,
                mean / float(parameters), coefficient, parameters))
        estimates = tuple(estimates)
        trace_rows.extend(estimates)
        for row in estimates:
            for probe, value in enumerate(row.estimates):
                raw_rows.append({
                    "batch_start": start,
                    "module": row.block,
                    "probe": probe,
                    "estimate": value,
                })
    base.validate_strict_site_contract(instrumentor, boundary_controller)
    activation_elements = _activation_element_counts(
        instrumentor, boundary_controller, len(indices))
    traced_modules = dict((name, named[name]) for name in module_names)
    summary = _summarize_traces(trace_rows, traced_modules)
    boundary_controller.close()
    instrumentor.close()
    return (
        model, summary, activation_elements, tuple(raw_rows),
        architecture, load_report, preparation)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("HAWQ trace requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_method_config(Path(args.config))
    indices = _load_calibration_indices(Path(args.calibration_metadata))
    registry = expected_registry()
    blocks = hawq_block_contract(registry)
    configured_fixed = set(config.hawq.fixed_blocks)
    if configured_fixed != set(
            block.name for block in blocks if block.fixed_eight):
        raise ValueError("HAWQ fixed block config differs from CSPN contract")
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("HAWQ trace output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    traced = _trace_official(args, config, indices, blocks)
    model, traces, activation_elements, raw_rows = traced[:4]
    result = allocate_from_trace_summary(
        model, blocks, traces, activation_elements,
        config.hawq.bits,
        config.hawq.maximum_average_weight_bits,
        config.hawq.maximum_average_activation_bits)
    write_allocation_artifacts(output, result)
    _write_csv(
        output / "trace_rows.csv", raw_rows,
        ("batch_start", "module", "probe", "estimate"))
    expanded = expand_assignment(registry, result.assignment.block_bits)
    selected_path = output / "selected_assignment.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected["weight_bits"] = [
        {"module": name, "bits": bits}
        for name, bits in expanded.weight_bits]
    selected["activation_bits"] = [
        {"module": owner[0], "kind": owner[1], "bits": bits}
        for owner, bits in expanded.activation_bits]
    _write_json(selected_path, selected)
    _write_json(output / "run_manifest.json", {
        "model": "cspn",
        "architecture": traced[4],
        "load_report": traced[5],
        "preparation": traced[6],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "calibration_indices": list(indices),
        "device": args.device,
        "trace_probes_per_batch": config.hawq.trace.probes_per_batch,
    })


if __name__ == "__main__":
    main()
