#!/usr/bin/env python3
"""Evaluate selective CSPN W4A4/W4A8 activation boundaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_activation_resolution as base
from scripts import run_nyu_cspn_decoder_sensitivity as decoder_runner
from scripts import run_nyu_cspn_encoder_prefix_joint as prefix_runner
from scripts import run_nyu_cspn_stem_precision as stem_runner
from scripts.run_nyu_rtn_quantization import (
    aggregate_region_rows,
    calibration_dataset,
    evaluation_dataset,
    prediction_payload,
    prepare_prediction_dir,
    seeded_sample,
    write_csv,
    write_json,
    write_prediction_payload,
)
from spn_quant.cspn_encoder_prefix import (
    ACTIVATION_OWNERS_BY_UNIT,
    ALL_UNIT_ORDER,
    UnitRegistry,
    WEIGHT_MODULES_BY_UNIT,
    build_candidates as build_prefix_candidates,
    build_unit_registry,
    precision_cost,
)
from spn_quant.cspn_selective_w4a8 import (
    INITIAL_DEPTH_WEIGHT,
    build_stage1_candidates,
    build_single_demotions,
    build_cumulative_path,
    candidate_is_feasible,
    pareto_rows,
    rank_demotions,
    select_stage1_anchor,
    select_winner,
)


STEM_WEIGHT_MODULE = "conv1_1"
STEM_INPUT_OWNER = "conv1_1", "input"
METRIC_FIELDS = (
    "RMSE", "MAE", "ABS_REL", "IRMSE",
    "flat_RMSE", "boundary_RMSE",
    "nonfinite_ratio", "nonpositive_ratio",
)
BLOCK_ORDER = (
    "encoder_stem",
    "encoder_layer1",
    "encoder_layer2",
    "encoder_layer3",
    "encoder_layer4",
    "decoder_layer1",
    "decoder_layer2",
    "decoder_layer3",
    "decoder_layer4",
    "initial_depth",
    "propagation",
)
OWNER_BLOCK = {}
for _unit, _block in (
        ("stem", "encoder_stem"),
        ("encoder_layer1", "encoder_layer1"),
        ("encoder_layer2", "encoder_layer2"),
        ("decoder_layer4", "decoder_layer4")):
    for _owner in ACTIVATION_OWNERS_BY_UNIT[_unit]:
        OWNER_BLOCK[_owner] = _block


def _ordered_union(sequences):
    output = []
    for sequence in sequences:
        for value in sequence:
            if value not in output:
                output.append(value)
    return tuple(output)


EXPECTED_WEIGHT_MODULES = _ordered_union(
    WEIGHT_MODULES_BY_UNIT[unit] for unit in ALL_UNIT_ORDER)
EXPECTED_ACTIVATION_OWNERS = _ordered_union(
    ACTIVATION_OWNERS_BY_UNIT[unit] for unit in ALL_UNIT_ORDER)


@dataclass(frozen=True)
class RuntimeCandidate:
    name: str
    role: str
    activation_owners: tuple
    weight_modules: tuple
    stem_config: str


def stage1_candidates(registry: UnitRegistry):
    primary = build_stage1_candidates(registry)
    p3t3 = next(
        candidate for candidate in build_prefix_candidates(registry)
        if candidate.prefix_index == 3 and candidate.tail_index == 3)
    contexts = (
        RuntimeCandidate(
            name="CONTEXT_STRICT_W4A4",
            role="context",
            activation_owners=(),
            weight_modules=(),
            stem_config="STRICT_W4A4",
        ),
        RuntimeCandidate(
            name="CONTEXT_P3_T3_W8A8",
            role="context",
            activation_owners=p3t3.activation_owners,
            weight_modules=p3t3.weight_modules,
            stem_config="STEM_W8A8",
        ),
    )
    return primary, contexts


def hardware_configuration(candidate) -> Dict[str, object]:
    promoted = tuple(
        owner for owner in candidate.activation_owners
        if owner != STEM_INPUT_OWNER)
    return base._configuration(
        candidate.name,
        base.ORDINARY_GROUPS,
        base.ORDINARY_GROUPS,
        base.PROPAGATION_A8_Q13,
        granularity="hybrid_group",
        group_size=8,
        promoted_owners=promoted,
        weight_bit_overrides=((INITIAL_DEPTH_WEIGHT, 8),))


def stem_configuration(candidate) -> str:
    return "STEM_W4A8" if STEM_INPUT_OWNER in \
        candidate.activation_owners else "STRICT_W4A4"


def validate_configured_precision(
        candidate, weight_bits: Mapping[str, int], specs,
        rotation_specs, stem_contract: Mapping[str, object]) -> None:
    if any(int(weight_bits[name]) not in (4, 8) for name in weight_bits):
        raise RuntimeError("configured weight precision is invalid")
    actual_w8 = {
        str(name) for name in weight_bits if int(weight_bits[name]) == 8}
    if actual_w8 != {INITIAL_DEPTH_WEIGHT}:
        raise RuntimeError("configured W8 weight set does not match candidate")

    actual_a8 = set()
    for key in specs:
        bits = int(specs[key].bits)
        if bits == 8:
            actual_a8.add(base.activation_owner(key))
        elif bits != 4:
            raise RuntimeError("configured activation precision is invalid")
    for owner in rotation_specs:
        bits = int(rotation_specs[owner].bits)
        if bits == 8:
            actual_a8.add(tuple(owner))
        elif bits != 4:
            raise RuntimeError("configured activation precision is invalid")

    expected_stem = stem_configuration(candidate)
    expected_stem_bits = 8 if expected_stem == "STEM_W4A8" else 4
    if str(stem_contract["config"]) != expected_stem:
        raise RuntimeError("configured stem contract differs from candidate")
    if int(stem_contract["weight_bits"]) != 4 or \
            int(stem_contract["activation_bits"]) != expected_stem_bits:
        raise RuntimeError("configured stem precision differs from candidate")
    if expected_stem_bits == 8:
        actual_a8.add(STEM_INPUT_OWNER)
    if actual_a8 != set(candidate.activation_owners):
        raise RuntimeError(
            "configured A8 activation set does not match candidate")


def _block_rows(rows: Sequence[Mapping[str, object]]):
    output = {}
    for source in rows:
        block = str(source["block"])
        if block == "__all__":
            continue
        if block in output:
            raise ValueError("calibration block rows contain duplicates")
        row = dict(source)
        numeric = (
            float(row["block_output_mse"]),
            float(row["error_energy"]),
            int(row["elements"]),
        )
        if not math.isfinite(numeric[0]) or not math.isfinite(numeric[1]) or \
                numeric[0] < 0.0 or numeric[1] < 0.0 or numeric[2] <= 0:
            raise ValueError("calibration block values are invalid")
        output[block] = row
    if "propagation" not in output:
        raise ValueError("calibration rows lack propagation output")
    return output


def _downstream_mse(rows, start_block: str) -> float:
    start = BLOCK_ORDER.index(start_block)
    selected = [
        rows[block] for block in BLOCK_ORDER[start:] if block in rows]
    error = sum(float(row["error_energy"]) for row in selected)
    elements = sum(int(row["elements"]) for row in selected)
    if elements <= 0:
        raise ValueError("downstream calibration rows are empty")
    return error / float(elements)


def demotion_calibration_row(
        demotion, anchor_block_rows: Sequence[Mapping[str, object]],
        demotion_block_rows: Sequence[Mapping[str, object]],
        anchor_cost: float, demotion_cost: float) -> Dict[str, object]:
    if demotion.owner not in OWNER_BLOCK:
        raise ValueError("demotion owner has no block assignment")
    anchor = _block_rows(anchor_block_rows)
    current = _block_rows(demotion_block_rows)
    if set(anchor) != set(current):
        raise ValueError("calibration block coverage changed")
    saved_cost = float(anchor_cost) - float(demotion_cost)
    if not math.isfinite(saved_cost) or saved_cost <= 0.0:
        raise ValueError("demotion saved cost must be positive")
    block = OWNER_BLOCK[demotion.owner]
    return {
        "config": demotion.name,
        "module": demotion.owner[0],
        "kind": demotion.owner[1],
        "owner_block": block,
        "propagation_mse": float(current["propagation"][
            "block_output_mse"]),
        "downstream_mse": _downstream_mse(current, block),
        "anchor_downstream_mse": _downstream_mse(anchor, block),
        "saved_cost": saved_cost,
    }


def _maximum_present(rows, field: str) -> float:
    values = [
        float(row[field]) for row in rows
        if field in row and str(row[field]) != ""]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("propagation safety field is incomplete: %s" % field)
    return max(values)


def aggregate_candidate_metrics(
        sample_rows: Sequence[Mapping[str, object]],
        propagation_rows: Sequence[Mapping[str, object]],
        candidates) -> list[Dict[str, object]]:
    output = []
    for candidate in candidates:
        selected = [
            row for row in sample_rows
            if str(row["config"]) == candidate.name]
        if len(selected) != 64:
            raise ValueError("%s requires exactly 64 samples" % candidate.name)
        identities = tuple(int(row["sample_index"]) for row in selected)
        if len(set(identities)) != len(identities):
            raise ValueError("sample identities must be unique")
        propagation = [
            row for row in propagation_rows
            if str(row["config"]) == candidate.name]
        if not propagation:
            raise ValueError("candidate propagation rows are missing")
        aggregate = {
            "config": candidate.name,
            "samples": len(selected),
        }
        for field in METRIC_FIELDS:
            values = np.asarray(
                [float(row[field]) for row in selected], dtype=np.float64)
            if not bool(np.isfinite(values).all()):
                raise ValueError("candidate metrics must be finite")
            aggregate[field] = float(values.mean())
        for field in (
                "coefficient_sum_max_error",
                "contraction_violation_rate",
                "anchor_max_error"):
            aggregate[field] = _maximum_present(propagation, field)
        output.append(aggregate)
    return output


def runtime_primary(candidate) -> RuntimeCandidate:
    return RuntimeCandidate(
        name=candidate.name,
        role="primary",
        activation_owners=candidate.activation_owners,
        weight_modules=candidate.weight_modules,
        stem_config=stem_configuration(candidate),
    )


def runtime_path(candidate) -> RuntimeCandidate:
    stem_config = "STEM_W4A8" if STEM_INPUT_OWNER in \
        candidate.activation_owners else "STRICT_W4A4"
    return RuntimeCandidate(
        name=candidate.name,
        role="path",
        activation_owners=candidate.activation_owners,
        weight_modules=candidate.weight_modules,
        stem_config=stem_config,
    )


def runtime_configuration(candidate: RuntimeCandidate) -> Dict[str, object]:
    generic_weights = tuple(
        name for name in candidate.weight_modules
        if name != STEM_WEIGHT_MODULE)
    generic_owners = tuple(
        owner for owner in candidate.activation_owners
        if owner != STEM_INPUT_OWNER)
    return base._configuration(
        candidate.name,
        base.ORDINARY_GROUPS,
        base.ORDINARY_GROUPS,
        base.PROPAGATION_A8_Q13,
        granularity="hybrid_group",
        group_size=8,
        promoted_owners=generic_owners,
        weight_bit_overrides=tuple(
            (name, 8) for name in generic_weights))


def validate_runtime_precision(
        candidate: RuntimeCandidate,
        weight_bits: Mapping[str, int], specs, rotation_specs,
        stem_contract: Mapping[str, object]) -> None:
    expected_generic_weights = set(candidate.weight_modules) - {
        STEM_WEIGHT_MODULE}
    actual_generic_weights = {
        str(name) for name in weight_bits if int(weight_bits[name]) == 8}
    if any(int(weight_bits[name]) not in (4, 8) for name in weight_bits):
        raise RuntimeError("configured weight precision is invalid")
    if actual_generic_weights != expected_generic_weights:
        raise RuntimeError("configured W8 weight set does not match runtime")

    actual_a8 = set()
    for key in specs:
        bits = int(specs[key].bits)
        if bits == 8:
            actual_a8.add(base.activation_owner(key))
        elif bits != 4:
            raise RuntimeError("configured activation precision is invalid")
    for owner in rotation_specs:
        bits = int(rotation_specs[owner].bits)
        if bits == 8:
            actual_a8.add(tuple(owner))
        elif bits != 4:
            raise RuntimeError("configured activation precision is invalid")

    if str(stem_contract["config"]) != candidate.stem_config:
        raise RuntimeError("configured stem contract differs from runtime")
    expected_stem_weight = 8 if candidate.stem_config == "STEM_W8A8" else 4
    expected_stem_activation = 8 if candidate.stem_config in (
        "STEM_W4A8", "STEM_W8A8") else 4
    if int(stem_contract["weight_bits"]) != expected_stem_weight or \
            int(stem_contract["activation_bits"]) != expected_stem_activation:
        raise RuntimeError("configured stem precision differs from runtime")
    actual_weights = set(actual_generic_weights)
    if expected_stem_weight == 8:
        actual_weights.add(STEM_WEIGHT_MODULE)
    if expected_stem_activation == 8:
        actual_a8.add(STEM_INPUT_OWNER)
    if actual_weights != set(candidate.weight_modules):
        raise RuntimeError("configured weight union differs from runtime")
    if actual_a8 != set(candidate.activation_owners):
        raise RuntimeError("configured activation union differs from runtime")


def configure_runtime_context(
        candidate: RuntimeCandidate,
        instrumentor, rotation, propagation, stem):
    config = runtime_configuration(candidate)
    specs, rotation_specs, active_merge = base._configure_quantized(
        config, instrumentor, rotation, propagation, {})
    if active_merge is not None:
        raise RuntimeError("selective W4A8 evaluation forbids merge adapters")
    stem.configure(candidate.stem_config)
    validate_runtime_precision(
        candidate, instrumentor.weight_bits_by_module(),
        specs, rotation_specs, stem.contract())
    return config, specs, rotation_specs


def _prediction_nonpositive_ratio(gt, prediction) -> float:
    valid = np.isfinite(gt) & (gt > 1e-4)
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("depth sample has no valid pixels")
    return float(np.count_nonzero(prediction[valid] <= 0.0)) / float(count)


def _evaluate_candidate(
        candidate: RuntimeCandidate, reference_model, model, saved_args,
        dataset, indices, device, seed, instrumentor, propagation, stem,
        prediction_root=None):
    reference_capture = base.ModuleOutputCapture(
        reference_model, base.CSPN_BLOCK_SITES)
    quantized_capture = base.ModuleOutputCapture(
        model, base.CSPN_BLOCK_SITES)
    operation_counter = stem_runner.ConvOperationCounter(
        model, prefix_runner.operation_module_names(instrumentor))
    prediction_dir = None
    if prediction_root is not None:
        prediction_dir = prepare_prediction_dir(
            prediction_root, candidate.name)
    block_error = base.BlockErrorAccumulator(base.CSPN_BLOCK_SITES)
    sample_rows = []
    region_rows = []
    propagation_rows = []
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            reference_prediction, reference_blocks = base._forward(
                reference_model, saved_args, sample, device,
                reference_capture)
            prediction, blocks = base._forward(
                model, saved_args, sample, device, quantized_capture)
            block_error.update(reference_blocks, blocks)
            pred = prediction.numpy()
            if not bool(np.isfinite(pred).all()):
                raise RuntimeError(
                    "non-finite prediction: config=%s sample=%d" %
                    (candidate.name, index))
            gt = sample["depth"][0].numpy()
            sparse = sample["rgbd"][3].numpy()
            metrics, regions = base.depth_sample_metrics(gt, pred, sparse)
            metrics["nonpositive_ratio"] = _prediction_nonpositive_ratio(
                gt, pred)
            metrics.update({
                "model": "cspn",
                "config": candidate.name,
                "sample_index": int(index),
            })
            sample_rows.append(metrics)
            for source in regions:
                row = dict(source)
                row.update({
                    "model": "cspn",
                    "config": candidate.name,
                    "sample_index": int(index),
                })
                region_rows.append(row)
            for source in propagation.statistics():
                row = dict(source)
                row.update({
                    "model": "cspn",
                    "config": candidate.name,
                    "sample_index": int(index),
                })
                propagation_rows.append(row)
            if prediction_dir is not None:
                payload = prediction_payload(
                    gt, reference_prediction.numpy(), pred,
                    int(index), "cspn", candidate.name,
                    sparse=sparse,
                    rgb=sample["rgbd"][:3].permute(1, 2, 0).numpy())
                write_prediction_payload(prediction_dir, payload)
            if rank % 16 == 0 or rank == len(indices):
                print("%s evaluation %d/%d" %
                      (candidate.name, rank, len(indices)), flush=True)

    block_rows = []
    for source in block_error.rows():
        row = dict(source)
        row.update({"model": "cspn", "config": candidate.name})
        block_rows.append(row)
    aggregate = block_error.aggregate()
    aggregate.update({
        "model": "cspn",
        "config": candidate.name,
        "block": "__all__",
    })
    block_rows.append(aggregate)
    operation_rows = operation_counter.rows()
    for row in operation_rows:
        row["config"] = candidate.name
        row["weight_bits"] = 8 \
            if row["module"] in candidate.weight_modules else 4
    layer_rows = instrumentor.statistics()
    for row in layer_rows:
        row.update({"model": "cspn", "config": candidate.name})
    stem_rows = []
    for source in stem.statistics():
        row = dict(source)
        row["stem_config"] = str(row["config"])
        row["config"] = candidate.name
        row["model"] = "cspn"
        stem_rows.append(row)
    reference_capture.close()
    quantized_capture.close()
    operation_counter.close()
    return {
        "sample_rows": sample_rows,
        "region_rows": region_rows,
        "propagation_rows": propagation_rows,
        "block_rows": block_rows,
        "operation_rows": operation_rows,
        "layer_rows": layer_rows,
        "stem_rows": stem_rows,
    }


def _run_candidate(
        candidate: RuntimeCandidate, expected_registry,
        reference_model, architecture, reference_load,
        reference_preparation, saved_args, checkpoint,
        preparation_args, trainset, dataset, evaluation_indices,
        protocol, device, fold_max_error, prediction_root=None):
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
        stem_runner._build_quantization_context(
            model, preparation, protocol.seed)
    stem_runner._calibrate(
        model, saved_args, trainset, protocol.calibration_indices,
        device, protocol.seed, instrumentor, rotation, propagation,
        stem, candidate.name)
    stem_runner._validate_site_contract(instrumentor, rotation)
    registry = prefix_runner.candidate_registry_from_context(
        instrumentor, rotation)
    if expected_registry is not None and registry != expected_registry:
        raise RuntimeError("fresh CSPN selective registry changed")
    activation_rows = decoder_runner.activation_cost_rows(
        instrumentor, rotation, len(protocol.calibration_indices),
        int(preparation_args[0].numel()))
    config, specs, rotation_specs = configure_runtime_context(
        candidate, instrumentor, rotation, propagation, stem)
    result = _evaluate_candidate(
        candidate, reference_model, model, saved_args, dataset,
        evaluation_indices, device, protocol.seed,
        instrumentor, propagation, stem, prediction_root)
    result["activation_rows"] = activation_rows
    result["checkpoint_load"] = load_report
    result["site_counts"] = {
        "ordinary": len(specs),
        "rotation": len(rotation_specs),
    }
    result["hardware_configuration"] = config
    result["stem_contract"] = stem.contract()
    stem.close()
    propagation.close()
    rotation.close()
    instrumentor.close()
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return registry, result


def run_runtime_matrix(candidates, evaluator) -> Dict[str, object]:
    expected_registry = None
    operation_basis = None
    activation_basis = None
    results_by_name = {}
    result_rows = dict((key, []) for key in (
        "sample_rows", "region_rows", "propagation_rows",
        "block_rows", "operation_rows", "layer_rows", "stem_rows"))
    for candidate in candidates:
        registry, result = evaluator(candidate, expected_registry)
        if expected_registry is None:
            expected_registry = registry
            operation_basis = [{
                "module": str(row["module"]),
                "macs": int(row["macs"]),
                "weight_elements": int(row["weight_elements"]),
                "input_elements": int(row["input_elements"]),
            } for row in result["operation_rows"]]
            activation_basis = [
                dict(row) for row in result["activation_rows"]]
        else:
            if registry != expected_registry:
                raise RuntimeError("candidate registry changed across runs")
            prefix_runner.validate_operation_basis(
                operation_basis, result["operation_rows"])
            if result["activation_rows"] != activation_basis:
                raise RuntimeError(
                    "activation basis changed across configurations")
        if candidate.name in results_by_name:
            raise ValueError("runtime candidates contain duplicate names")
        results_by_name[candidate.name] = result
        for key in result_rows:
            result_rows[key].extend(result[key])
    if expected_registry is None or operation_basis is None or \
            activation_basis is None:
        raise ValueError("runtime matrix must not be empty")
    return {
        "registry": expected_registry,
        "results_by_name": results_by_name,
        "result_rows": result_rows,
        "operation_basis": operation_basis,
        "activation_basis": activation_basis,
    }


def enrich_metrics(
        aggregate_rows, runtime_candidates,
        operation_basis, activation_basis):
    by_name = dict(
        (candidate.name, candidate) for candidate in runtime_candidates)
    if len(by_name) != len(runtime_candidates):
        raise ValueError("runtime candidates contain duplicate names")
    output = []
    for source in aggregate_rows:
        row = dict(source)
        name = str(row["config"])
        if name not in by_name:
            raise ValueError("aggregate runtime candidate is unknown")
        row.update(precision_cost(
            operation_basis, activation_basis, by_name[name]))
        row["role"] = by_name[name].role
        output.append(row)
    return output


def _regional_aggregates(region_rows, candidates):
    output = []
    for candidate in candidates:
        selected = [
            row for row in region_rows
            if str(row["config"]) == candidate.name]
        for source in aggregate_region_rows(selected):
            row = dict(source)
            row.update({"model": "cspn", "config": candidate.name})
            output.append(row)
    return output


def _candidate_cost(candidate, operation_basis, activation_basis):
    return precision_cost(
        operation_basis, activation_basis, candidate)


def _runtime_contract(candidate: RuntimeCandidate) -> Dict[str, object]:
    return {
        "name": candidate.name,
        "role": candidate.role,
        "activation_owners": [
            list(owner) for owner in candidate.activation_owners],
        "weight_modules": list(candidate.weight_modules),
        "stem_config": candidate.stem_config,
    }


def _write_result_tables(
        root: Path, stage1_matrix, stage1_aggregate,
        stage1_regional, calibration_rows, ranking,
        path_matrix, path_aggregate, path_regional,
        feasible, cost_pareto, a8_pareto, manifest):
    stage1_rows = stage1_matrix["result_rows"]
    path_rows = path_matrix["result_rows"]
    write_csv(
        root / "stage1_sample_metrics_64.csv",
        stage1_rows["sample_rows"],
        ("model", "config", "sample_index") + METRIC_FIELDS)
    write_csv(
        root / "stage1_aggregate_metrics.csv", stage1_aggregate,
        ("config", "role", "samples") + METRIC_FIELDS)
    write_csv(
        root / "stage1_regional_metrics.csv", stage1_regional,
        ("model", "config", "region", "RMSE", "MAE", "ABS_REL"))
    write_csv(
        root / "stage1_block_metrics.csv", stage1_rows["block_rows"],
        ("model", "config", "block", "block_output_mse",
         "block_output_sqnr"))
    write_csv(
        root / "stage1_propagation_metrics.csv",
        stage1_rows["propagation_rows"],
        ("model", "config", "sample_index", "signal", "iteration"))
    write_csv(
        root / "stage1_operation_counts.csv",
        stage1_rows["operation_rows"],
        ("config", "module", "weight_bits", "macs",
         "weight_elements", "input_elements"))
    write_csv(
        root / "stage1_precision_coverage.csv", stage1_aggregate,
        ("config", "role", "RMSE", "normalized_added_bit_cost",
         "a8_activation_element_fraction", "w8_weight_mac_fraction",
         "w8_weight_element_fraction"))
    write_csv(
        root / "activation_cost_basis.csv",
        stage1_matrix["activation_basis"],
        ("module", "kind", "elements"))
    write_csv(
        root / "stage2_single_demotion_calibration.csv",
        calibration_rows,
        ("config", "module", "kind", "owner_block",
         "propagation_mse", "downstream_mse",
         "anchor_downstream_mse", "saved_cost"))
    write_csv(
        root / "stage2_boundary_ranking.csv", ranking,
        ("rank", "config", "module", "kind", "score",
         "propagation_mse_increase", "downstream_mse_increase",
         "saved_cost"))
    write_csv(
        root / "stage2_path_sample_metrics_64.csv",
        path_rows["sample_rows"],
        ("model", "config", "sample_index") + METRIC_FIELDS)
    write_csv(
        root / "stage2_path_aggregate_metrics.csv", path_aggregate,
        ("config", "role", "samples") + METRIC_FIELDS)
    write_csv(
        root / "stage2_path_regional_metrics.csv", path_regional,
        ("model", "config", "region", "RMSE", "MAE", "ABS_REL"))
    write_csv(
        root / "stage2_path_block_metrics.csv", path_rows["block_rows"],
        ("model", "config", "block", "block_output_mse",
         "block_output_sqnr"))
    write_csv(
        root / "stage2_path_propagation_metrics.csv",
        path_rows["propagation_rows"],
        ("model", "config", "sample_index", "signal", "iteration"))
    write_csv(
        root / "feasible_candidates.csv", feasible,
        ("config", "role", "RMSE", "normalized_added_bit_cost",
         "a8_activation_element_fraction"))
    write_csv(
        root / "pareto_normalized_cost.csv", cost_pareto,
        ("config", "role", "RMSE", "normalized_added_bit_cost"))
    write_csv(
        root / "pareto_a8_fraction.csv", a8_pareto,
        ("config", "role", "RMSE", "a8_activation_element_fraction"))
    manifest["artifacts"] = {}
    write_json(root / "manifest.json", manifest)
    manifest["artifacts"] = stem_runner._artifact_hashes(root)
    write_json(root / "manifest.json", manifest)


def write_stage1_diagnostics(root: Path, stage1_matrix, stage1_aggregate):
    root.mkdir(parents=True)
    rows = stage1_matrix["result_rows"]
    write_csv(
        root / "stage1_sample_metrics_64.csv", rows["sample_rows"],
        ("model", "config", "sample_index") + METRIC_FIELDS)
    write_csv(
        root / "stage1_aggregate_metrics.csv", stage1_aggregate,
        ("config", "role", "samples") + METRIC_FIELDS)
    write_csv(
        root / "stage1_propagation_metrics.csv", rows["propagation_rows"],
        ("model", "config", "sample_index", "signal", "iteration"))
    write_csv(
        root / "activation_cost_basis.csv",
        stage1_matrix["activation_basis"],
        ("module", "kind", "elements"))


def validate_output_directories(output: Path) -> Path:
    staging = Path(str(output) + ".incomplete")
    if output.exists():
        raise FileExistsError("output directory already exists: %s" % output)
    if staging.exists():
        raise FileExistsError("staging directory already exists: %s" % staging)
    return staging


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rmse-limit", type=float, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("selective W4A8 evaluation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not math.isfinite(args.fold_max_error) or args.fold_max_error <= 0.0:
        raise ValueError("fold error threshold must be finite and positive")
    if not math.isfinite(args.rmse_limit) or args.rmse_limit <= 0.0:
        raise ValueError("RMSE limit must be finite and positive")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    checkpoint = Path(args.checkpoint)
    calibration_indices_path = Path(args.calibration_indices)
    calibration_metadata_path = Path(args.calibration_metadata)
    evaluation_protocol_path = Path(args.evaluation_protocol)
    calibration_payload = json.loads(calibration_indices_path.read_text())
    calibration_metadata = json.loads(calibration_metadata_path.read_text())
    evaluation_metadata = json.loads(evaluation_protocol_path.read_text())
    protocol = stem_runner.index_protocol(
        calibration_payload, evaluation_metadata)
    if int(args.seed) != protocol.seed:
        raise ValueError("runner seed differs from evaluation protocol")
    checkpoint_sha256 = stem_runner._sha256(checkpoint)
    stem_runner._validate_source_metadata(
        args, calibration_metadata, evaluation_metadata,
        checkpoint_sha256)
    output = Path(args.out_dir)
    staging = validate_output_directories(output)
    saved_args = stem_runner._saved_args(args)
    trainset = calibration_dataset(saved_args)
    valset = evaluation_dataset(saved_args)
    if max(protocol.calibration_indices) >= len(trainset):
        raise ValueError("calibration index exceeds the train split")
    if max(protocol.evaluation_indices) >= len(valset):
        raise ValueError("evaluation index exceeds the validation split")
    preparation_sample = seeded_sample(
        trainset, protocol.calibration_indices[0], protocol.seed)
    preparation_args = base._model_args(
        saved_args, preparation_sample, device)
    reference_model, architecture, reference_load, reference_preparation = \
        stem_runner._prepare_model(
            saved_args, checkpoint, device, preparation_args,
            args.fold_max_error)

    declared_registry = build_unit_registry(
        EXPECTED_WEIGHT_MODULES, EXPECTED_ACTIVATION_OWNERS)
    primary_candidates, contexts = stage1_candidates(declared_registry)
    primary_runtime = tuple(
        runtime_primary(candidate) for candidate in primary_candidates)
    stage1_runtime = primary_runtime + contexts

    def validation_evaluator(candidate, expected_registry):
        expected = declared_registry \
            if expected_registry is None else expected_registry
        return _run_candidate(
            candidate, expected, reference_model, architecture,
            reference_load, reference_preparation, saved_args,
            checkpoint, preparation_args, trainset, valset,
            protocol.evaluation_indices, protocol, device,
            args.fold_max_error)

    stage1_matrix = run_runtime_matrix(
        stage1_runtime, validation_evaluator)
    stage1_aggregate = aggregate_candidate_metrics(
        stage1_matrix["result_rows"]["sample_rows"],
        stage1_matrix["result_rows"]["propagation_rows"],
        stage1_runtime)
    stage1_aggregate = enrich_metrics(
        stage1_aggregate, stage1_runtime,
        stage1_matrix["operation_basis"],
        stage1_matrix["activation_basis"])
    primary_by_name = dict(
        (candidate.name, candidate) for candidate in primary_candidates)
    for row in stage1_aggregate:
        name = str(row["config"])
        if name in primary_by_name:
            row["mask"] = primary_by_name[name].mask
            row["selected_units"] = "+".join(
                primary_by_name[name].selected_units)
        else:
            row["mask"] = ""
            row["selected_units"] = ""
    primary_rows = [
        row for row in stage1_aggregate
        if str(row["config"]) in primary_by_name]
    write_stage1_diagnostics(staging, stage1_matrix, stage1_aggregate)
    anchor_row = select_stage1_anchor(
        primary_rows, primary_candidates, args.rmse_limit)
    anchor_candidate = primary_by_name[str(anchor_row["config"])]
    anchor_runtime = runtime_primary(anchor_candidate)

    _, anchor_calibration = _run_candidate(
        anchor_runtime, declared_registry, reference_model, architecture,
        reference_load, reference_preparation, saved_args, checkpoint,
        preparation_args, trainset, trainset,
        protocol.calibration_indices, protocol, device,
        args.fold_max_error)
    prefix_runner.validate_operation_basis(
        stage1_matrix["operation_basis"],
        anchor_calibration["operation_rows"])
    if anchor_calibration["activation_rows"] != \
            stage1_matrix["activation_basis"]:
        raise RuntimeError("anchor calibration activation basis changed")
    anchor_cost = _candidate_cost(
        anchor_runtime, stage1_matrix["operation_basis"],
        stage1_matrix["activation_basis"])[
            "normalized_added_bit_cost"]
    anchor_propagation = next(
        row for row in anchor_calibration["block_rows"]
        if str(row["block"]) == "propagation")
    single_demotions = build_single_demotions(anchor_candidate)
    calibration_rows = []
    for demotion in single_demotions:
        current_runtime = runtime_path(demotion)
        _, result = _run_candidate(
            current_runtime, declared_registry, reference_model,
            architecture, reference_load, reference_preparation,
            saved_args, checkpoint, preparation_args, trainset,
            trainset, protocol.calibration_indices, protocol, device,
            args.fold_max_error)
        prefix_runner.validate_operation_basis(
            stage1_matrix["operation_basis"], result["operation_rows"])
        if result["activation_rows"] != stage1_matrix["activation_basis"]:
            raise RuntimeError("demotion calibration activation basis changed")
        current_cost = _candidate_cost(
            current_runtime, stage1_matrix["operation_basis"],
            stage1_matrix["activation_basis"])[
                "normalized_added_bit_cost"]
        calibration_rows.append(demotion_calibration_row(
            demotion, anchor_calibration["block_rows"],
            result["block_rows"], anchor_cost, current_cost))
    ranking = rank_demotions(
        {"propagation_mse": float(anchor_propagation[
            "block_output_mse"])},
        calibration_rows, single_demotions)
    path_candidates = build_cumulative_path(anchor_candidate, ranking)
    path_runtime = tuple(
        runtime_path(candidate) for candidate in path_candidates)

    path_matrix = run_runtime_matrix(path_runtime, validation_evaluator)
    path_aggregate = aggregate_candidate_metrics(
        path_matrix["result_rows"]["sample_rows"],
        path_matrix["result_rows"]["propagation_rows"],
        path_runtime)
    path_aggregate = enrich_metrics(
        path_aggregate, path_runtime,
        path_matrix["operation_basis"],
        path_matrix["activation_basis"])
    path_by_name = dict(
        (candidate.name, candidate) for candidate in path_candidates)
    for row in path_aggregate:
        candidate = path_by_name[str(row["config"])]
        row["step"] = candidate.step
        row["demoted_owners"] = "+".join(
            "%s::%s" % owner for owner in candidate.demoted_owners)

    selectable = list(primary_rows)
    selectable.extend(
        row for row in path_aggregate if int(row["step"]) > 0)
    feasible = [
        dict(row) for row in selectable
        if candidate_is_feasible(
            row, args.rmse_limit, require_rerun=False)]
    preliminary_winner = select_winner(
        selectable, args.rmse_limit, require_rerun=False)
    cost_pareto = pareto_rows(
        feasible, "normalized_added_bit_cost")
    a8_pareto = pareto_rows(
        feasible, "a8_activation_element_fraction")

    runtime_by_name = dict(
        (candidate.name, candidate)
        for candidate in stage1_runtime + path_runtime)
    prediction_names = []
    for name in (
            "CONTEXT_STRICT_W4A4", "ACT_MASK_00",
            anchor_candidate.name, str(preliminary_winner["config"]),
            "CONTEXT_P3_T3_W8A8"):
        if name not in prediction_names:
            prediction_names.append(name)
    prediction_root = staging / "predictions"
    prediction_root.mkdir()
    aggregate_by_name = dict(
        (str(row["config"]), row)
        for row in stage1_aggregate + path_aggregate)
    for name in prediction_names:
        selected = runtime_by_name[name]
        _, prediction_result = _run_candidate(
            selected, declared_registry, reference_model, architecture,
            reference_load, reference_preparation, saved_args, checkpoint,
            preparation_args, trainset, valset,
            protocol.evaluation_indices, protocol, device,
            args.fold_max_error, prediction_root=prediction_root)
        observed = aggregate_candidate_metrics(
            prediction_result["sample_rows"],
            prediction_result["propagation_rows"], (selected,))[0]
        expected = aggregate_by_name[name]
        if float(observed["RMSE"]) != float(expected["RMSE"]):
            raise RuntimeError("prediction rerun metric changed: %s" % name)
        expected["rerun_RMSE"] = float(observed["RMSE"])
        prefix_runner.validate_operation_basis(
            stage1_matrix["operation_basis"],
            prediction_result["operation_rows"])
        if prediction_result["activation_rows"] != \
                stage1_matrix["activation_basis"]:
            raise RuntimeError("prediction rerun activation basis changed")
    winner_row = aggregate_by_name[str(preliminary_winner["config"])]
    winner = select_winner(
        (winner_row,), args.rmse_limit, require_rerun=True)

    stage1_regional = _regional_aggregates(
        stage1_matrix["result_rows"]["region_rows"], stage1_runtime)
    path_regional = _regional_aggregates(
        path_matrix["result_rows"]["region_rows"], path_runtime)
    manifest = {
        "model": "cspn",
        "architecture": architecture,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_load": reference_load,
        "data_root": str(Path(args.data_root).resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "device": str(device),
        "seed": protocol.seed,
        "rmse_limit": float(args.rmse_limit),
        "calibration_selection": protocol.selection,
        "calibration_indices": list(protocol.calibration_indices),
        "evaluation_indices": list(protocol.evaluation_indices),
        "calibration_indices_sha256": stem_runner._sha256(
            calibration_indices_path),
        "calibration_metadata_sha256": stem_runner._sha256(
            calibration_metadata_path),
        "evaluation_protocol_sha256": stem_runner._sha256(
            evaluation_protocol_path),
        "stage1_configurations": [
            _runtime_contract(candidate) for candidate in stage1_runtime],
        "stage1_anchor": anchor_candidate.name,
        "boundary_ranking": [
            {"rank": int(row["rank"]),
             "owner": [str(row["module"]), str(row["kind"])]}
            for row in ranking],
        "path_configurations": [
            _runtime_contract(candidate) for candidate in path_runtime],
        "winner": str(winner["config"]),
        "prediction_configurations": prediction_names,
        "guidance": "fp32",
        "bias": "fp32",
        "propagation": dict(base.PROPAGATION_A8_Q13),
        "folded_pairs": reference_preparation["folded_pairs"],
        "source_hashes": {
            "runner": stem_runner._sha256(Path(__file__)),
            "candidate_module": stem_runner._sha256(
                REPO_ROOT / "spn_quant" / "cspn_selective_w4a8.py"),
        },
    }
    _write_result_tables(
        staging, stage1_matrix, stage1_aggregate,
        stage1_regional, calibration_rows, ranking,
        path_matrix, path_aggregate, path_regional,
        feasible, cost_pareto, a8_pareto, manifest)
    if stem_runner._artifact_hashes(staging) != manifest["artifacts"]:
        raise RuntimeError("artifact hashes changed before publication")
    staging.rename(output)
    reference_model.cpu()
    del reference_model
    torch.cuda.empty_cache()
    print("CSPN selective W4A8 evaluation complete: %s" % output,
          flush=True)


if __name__ == "__main__":
    main()
