#!/usr/bin/env python3
"""Evaluate CSPN encoder-prefix and sensitive-tail W8A8 combinations."""

from __future__ import annotations

import argparse
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
    WEIGHT_MODULES_BY_UNIT,
    PrefixTailCandidate,
    build_candidates,
    build_unit_registry,
    interaction_rows,
    pareto_rows,
    precision_cost,
    prediction_candidate_names,
)


STEM_WEIGHT_MODULE = "conv1_1"
STEM_INPUT_OWNER = "conv1_1", "input"


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


def hardware_configuration(
        candidate: PrefixTailCandidate) -> Dict[str, object]:
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


def stem_configuration(candidate: PrefixTailCandidate) -> str:
    return "STEM_W8A8" if candidate.stem_w8a8 else "STRICT_W4A4"


def validate_configured_precision(
        candidate: PrefixTailCandidate,
        weight_bits: Mapping[str, int], specs, rotation_specs,
        stem_contract: Mapping[str, object]) -> None:
    expected_generic_weights = set(candidate.weight_modules) - {
        STEM_WEIGHT_MODULE}
    actual_generic_weights = {
        str(name) for name in weight_bits if int(weight_bits[name]) == 8
    }
    if any(int(weight_bits[name]) not in (4, 8) for name in weight_bits):
        raise RuntimeError("configured weight precision is invalid")
    if actual_generic_weights != expected_generic_weights:
        raise RuntimeError("configured weight promotion differs from candidate")

    expected_generic_owners = set(candidate.activation_owners) - {
        STEM_INPUT_OWNER}
    actual_generic_owners = set()
    for key in specs:
        bits = int(specs[key].bits)
        if bits == 8:
            actual_generic_owners.add(base.activation_owner(key))
        elif bits != 4:
            raise RuntimeError("configured activation precision is invalid")
    for owner in rotation_specs:
        bits = int(rotation_specs[owner].bits)
        if bits == 8:
            actual_generic_owners.add(tuple(owner))
        elif bits != 4:
            raise RuntimeError("configured activation precision is invalid")
    if actual_generic_owners != expected_generic_owners:
        raise RuntimeError(
            "configured activation promotion differs from candidate")

    expected_stem = stem_configuration(candidate)
    if str(stem_contract["config"]) != expected_stem:
        raise RuntimeError("configured stem contract differs from candidate")
    expected_bits = 8 if candidate.stem_w8a8 else 4
    if int(stem_contract["weight_bits"]) != expected_bits or \
            int(stem_contract["activation_bits"]) != expected_bits:
        raise RuntimeError("configured stem precision differs from candidate")
    actual_weights = set(actual_generic_weights)
    actual_owners = set(actual_generic_owners)
    if candidate.stem_w8a8:
        actual_weights.add(STEM_WEIGHT_MODULE)
        actual_owners.add(STEM_INPUT_OWNER)
    if actual_weights != set(candidate.weight_modules):
        raise RuntimeError("configured weight promotion union is invalid")
    if actual_owners != set(candidate.activation_owners):
        raise RuntimeError("configured activation promotion union is invalid")


def candidate_registry_from_context(instrumentor, rotation):
    expected_generic_modules = set(EXPECTED_WEIGHT_MODULES) - {
        STEM_WEIGHT_MODULE}
    executed = set(stem_runner.executed_operation_modules(instrumentor))
    modules = [STEM_WEIGHT_MODULE]
    modules.extend(
        name for name in EXPECTED_WEIGHT_MODULES
        if name != STEM_WEIGHT_MODULE and name in executed)
    if set(modules[1:]) != expected_generic_modules:
        raise ValueError("executed candidate weight registry is incomplete")

    observed_owners = set()
    for key in instrumentor.activation_site_keys(base.ORDINARY_GROUPS):
        observed_owners.add(base.activation_owner(key))
    observed_owners.update(
        ("rotation.%s" % name, "boundary")
        for name in rotation.channels)
    owners = [STEM_INPUT_OWNER]
    owners.extend(
        owner for owner in EXPECTED_ACTIVATION_OWNERS
        if owner != STEM_INPUT_OWNER and owner in observed_owners)
    if set(owners[1:]) != set(EXPECTED_ACTIVATION_OWNERS) - {
            STEM_INPUT_OWNER}:
        raise ValueError("executed candidate activation registry is incomplete")
    return build_unit_registry(tuple(modules), tuple(owners))


def operation_module_names(instrumentor):
    generic = stem_runner.executed_operation_modules(instrumentor)
    if STEM_WEIGHT_MODULE in generic:
        raise RuntimeError("stem operation has duplicate ownership")
    return (STEM_WEIGHT_MODULE,) + tuple(generic)


def aggregate_candidate_metrics(
        rows: Sequence[Mapping[str, object]],
        candidates: Sequence[PrefixTailCandidate]) -> list[Dict[str, object]]:
    output = []
    for candidate in candidates:
        selected = [row for row in rows if row["config"] == candidate.name]
        if len(selected) != stem_runner.EVALUATION_SAMPLES:
            raise ValueError("%s requires exactly 64 samples" % candidate.name)
        identities = [int(row["sample_index"]) for row in selected]
        if len(set(identities)) != len(identities):
            raise ValueError("%s sample identities must be unique" %
                             candidate.name)
        aggregate = {
            "config": candidate.name,
            "prefix_index": candidate.prefix_index,
            "tail_index": candidate.tail_index,
            "encoder_units": "+".join(candidate.encoder_units),
            "tail_units": "+".join(candidate.tail_units),
            "samples": len(selected),
        }
        for field in stem_runner.METRIC_FIELDS:
            values = np.asarray(
                [float(row[field]) for row in selected], dtype=np.float64)
            if not bool(np.isfinite(values).all()):
                raise ValueError("%s %s must be finite" %
                                 (candidate.name, field))
            aggregate[field] = float(values.mean())
        output.append(aggregate)
    return output


def enrich_aggregate_metrics(
        aggregate_rows: Sequence[Mapping[str, object]],
        sample_rows: Sequence[Mapping[str, object]],
        candidates: Sequence[PrefixTailCandidate],
        operation_basis: Sequence[Mapping[str, object]],
        activation_basis: Sequence[Mapping[str, object]]
        ) -> list[Dict[str, object]]:
    by_name = dict((candidate.name, candidate) for candidate in candidates)
    if len(by_name) != len(candidates):
        raise ValueError("candidate names contain duplicates")
    strict_name = "PREFIX_P0__TAIL_T0"
    baseline_samples = [
        dict(row) for row in sample_rows if row["config"] == strict_name
    ]
    if len(baseline_samples) != stem_runner.EVALUATION_SAMPLES:
        raise ValueError("strict baseline requires exactly 64 samples")
    output = []
    for source in aggregate_rows:
        row = dict(source)
        name = str(row["config"])
        if name not in by_name:
            raise ValueError("aggregate candidate is unknown: %s" % name)
        row.update(precision_cost(
            operation_basis, activation_basis, by_name[name]))
        if name == strict_name:
            row.update({
                "baseline_rmse": float(row["RMSE"]),
                "candidate_rmse": float(row["RMSE"]),
                "rmse_delta": 0.0,
                "wins": 0,
                "accepted": True,
            })
        else:
            selected = [
                dict(sample) for sample in sample_rows
                if sample["config"] == name
            ]
            row.update(stem_runner.acceptance(
                baseline_samples, selected))
        output.append(row)
    return output


def _operation_contract(rows: Sequence[Mapping[str, object]]):
    output = []
    names = set()
    for row in rows:
        name = str(row["module"])
        if name in names:
            raise ValueError("operation basis contains duplicate modules")
        names.add(name)
        output.append((
            name,
            int(row["macs"]),
            int(row["weight_elements"]),
            int(row["input_elements"]),
        ))
    return tuple(output)


def validate_operation_basis(
        expected: Sequence[Mapping[str, object]],
        actual: Sequence[Mapping[str, object]]) -> None:
    if _operation_contract(expected) != _operation_contract(actual):
        raise RuntimeError("operation basis changed across configurations")


def candidate_contract(candidate: PrefixTailCandidate) -> Dict[str, object]:
    return {
        "name": candidate.name,
        "prefix_index": candidate.prefix_index,
        "tail_index": candidate.tail_index,
        "encoder_units": list(candidate.encoder_units),
        "tail_units": list(candidate.tail_units),
        "weight_modules": list(candidate.weight_modules),
        "activation_owners": [
            list(owner) for owner in candidate.activation_owners],
        "stem_w8a8": candidate.stem_w8a8,
    }


def run_candidate_matrix(candidates, evaluator) -> Dict[str, object]:
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
            validate_operation_basis(
                operation_basis, result["operation_rows"])
            if result["activation_rows"] != activation_basis:
                raise RuntimeError(
                    "activation basis changed across configurations")
        results_by_name[candidate.name] = result
        for key in result_rows:
            result_rows[key].extend(result[key])
    if expected_registry is None or operation_basis is None or \
            activation_basis is None:
        raise ValueError("candidate matrix must not be empty")
    return {
        "registry": expected_registry,
        "results_by_name": results_by_name,
        "result_rows": result_rows,
        "operation_basis": operation_basis,
        "activation_basis": activation_basis,
    }


def configure_candidate_context(
        candidate: PrefixTailCandidate,
        instrumentor, rotation, propagation, stem):
    config = hardware_configuration(candidate)
    specs, rotation_specs, active_merge = base._configure_quantized(
        config, instrumentor, rotation, propagation, {})
    if active_merge is not None:
        raise RuntimeError("encoder-prefix evaluation forbids merge adapters")
    stem.configure(stem_configuration(candidate))
    validate_configured_precision(
        candidate, instrumentor.weight_bits_by_module(),
        specs, rotation_specs, stem.contract())
    return config, specs, rotation_specs


def _evaluate_candidate(
        candidate, reference_model, model, saved_args, dataset, indices,
        device, seed, instrumentor, propagation, stem,
        prediction_root=None):
    reference_capture = base.ModuleOutputCapture(
        reference_model, base.CSPN_BLOCK_SITES)
    quantized_capture = base.ModuleOutputCapture(
        model, base.CSPN_BLOCK_SITES)
    operation_counter = stem_runner.ConvOperationCounter(
        model, operation_module_names(instrumentor))
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
        candidate, expected_registry, reference_model, architecture,
        reference_load, reference_preparation, saved_args, checkpoint,
        preparation_args, trainset, valset, protocol, device,
        fold_max_error, prediction_root=None):
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
    registry = candidate_registry_from_context(instrumentor, rotation)
    if expected_registry is not None and registry != expected_registry:
        raise RuntimeError("fresh CSPN prefix registry changed")
    activation_rows = decoder_runner.activation_cost_rows(
        instrumentor, rotation, len(protocol.calibration_indices),
        int(preparation_args[0].numel()))
    config, specs, rotation_specs = configure_candidate_context(
        candidate, instrumentor, rotation, propagation, stem)
    result = _evaluate_candidate(
        candidate, reference_model, model, saved_args, valset,
        protocol.evaluation_indices, device, protocol.seed,
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


def _regional_aggregates(region_rows, candidates):
    output = []
    for candidate in candidates:
        selected = [
            row for row in region_rows if row["config"] == candidate.name]
        for source in aggregate_region_rows(selected):
            row = dict(source)
            row.update({"model": "cspn", "config": candidate.name})
            output.append(row)
    return output


def _write_results(
        output, matrix, aggregate, interactions,
        normalized_pareto, mac_pareto, regional_rows,
        prediction_names, manifest):
    result_rows = matrix["result_rows"]
    write_csv(
        output / "sample_metrics_64.csv", result_rows["sample_rows"],
        base.SAMPLE_FIELDS)
    write_csv(
        output / "aggregate_metrics.csv", aggregate,
        ("config", "prefix_index", "tail_index", "encoder_units",
         "tail_units", "samples") + stem_runner.METRIC_FIELDS)
    write_csv(
        output / "regional_metrics.csv", regional_rows,
        ("model", "config", "region", "RMSE", "MAE", "ABS_REL"))
    write_csv(
        output / "block_metrics.csv", result_rows["block_rows"],
        ("model", "config", "block", "block_output_mse",
         "block_output_sqnr"))
    write_csv(
        output / "propagation_metrics.csv",
        result_rows["propagation_rows"],
        ("model", "config", "sample_index", "signal", "iteration"))
    write_csv(
        output / "operation_counts.csv", result_rows["operation_rows"],
        ("config", "module", "weight_bits", "macs",
         "weight_elements", "input_elements"))
    write_csv(
        output / "activation_cost_basis.csv", matrix["activation_basis"],
        ("module", "kind", "elements"))
    write_csv(
        output / "layer_quantization_metrics.csv",
        result_rows["layer_rows"],
        ("model", "config", "module", "group", "kind"))
    write_csv(
        output / "stem_quantization_metrics.csv",
        result_rows["stem_rows"],
        ("model", "config", "stem_config", "signal"))
    write_csv(
        output / "precision_coverage.csv", aggregate,
        ("config", "prefix_index", "tail_index",
         "normalized_added_bit_cost", "w8_weight_mac_fraction",
         "w8_weight_element_fraction", "a8_activation_element_fraction"))
    write_csv(
        output / "interaction_metrics.csv", interactions,
        ("config", "prefix_index", "tail_index", "RMSE",
         "interaction_rmse"))
    write_csv(
        output / "pareto_normalized_cost.csv", normalized_pareto,
        ("config", "prefix_index", "tail_index", "RMSE",
         "normalized_added_bit_cost"))
    write_csv(
        output / "pareto_w8_mac.csv", mac_pareto,
        ("config", "prefix_index", "tail_index", "RMSE",
         "w8_weight_mac_fraction"))
    manifest["prediction_configurations"] = list(prediction_names)
    manifest["artifacts"] = {}
    write_json(output / "manifest.json", manifest)
    manifest["artifacts"] = stem_runner._artifact_hashes(output)
    write_json(output / "manifest.json", manifest)


def validate_output_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError("output directory already exists: %s" % path)


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
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN encoder-prefix evaluation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not math.isfinite(args.fold_max_error) or args.fold_max_error <= 0.0:
        raise ValueError("fold error threshold must be finite and positive")
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
    validate_output_directory(output)
    output.mkdir(parents=True)
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
    candidates = build_candidates(declared_registry)

    def evaluator(candidate, expected_registry):
        expected = declared_registry \
            if expected_registry is None else expected_registry
        return _run_candidate(
            candidate, expected, reference_model, architecture,
            reference_load, reference_preparation, saved_args, checkpoint,
            preparation_args, trainset, valset, protocol, device,
            args.fold_max_error)

    matrix = run_candidate_matrix(candidates, evaluator)
    aggregate = aggregate_candidate_metrics(
        matrix["result_rows"]["sample_rows"], candidates)
    aggregate = enrich_aggregate_metrics(
        aggregate, matrix["result_rows"]["sample_rows"], candidates,
        matrix["operation_basis"], matrix["activation_basis"])
    interactions = interaction_rows(aggregate)
    normalized_pareto = pareto_rows(
        aggregate, "normalized_added_bit_cost")
    mac_pareto = pareto_rows(aggregate, "w8_weight_mac_fraction")
    prediction_names = prediction_candidate_names(
        aggregate, normalized_pareto)
    candidates_by_name = dict(
        (candidate.name, candidate) for candidate in candidates)
    prediction_root = output / "predictions"
    prediction_root.mkdir()
    for name in prediction_names:
        selected = candidates_by_name[name]
        _, prediction_result = _run_candidate(
            selected, declared_registry, reference_model, architecture,
            reference_load, reference_preparation, saved_args, checkpoint,
            preparation_args, trainset, valset, protocol, device,
            args.fold_max_error, prediction_root=prediction_root)
        expected = next(
            row for row in aggregate if row["config"] == name)
        observed = aggregate_candidate_metrics(
            prediction_result["sample_rows"], (selected,))[0]
        if float(observed["RMSE"]) != float(expected["RMSE"]):
            raise RuntimeError("prediction rerun metric changed: %s" % name)
        validate_operation_basis(
            matrix["operation_basis"], prediction_result["operation_rows"])
        if prediction_result["activation_rows"] != \
                matrix["activation_basis"]:
            raise RuntimeError("prediction rerun activation basis changed")

    regional_rows = _regional_aggregates(
        matrix["result_rows"]["region_rows"], candidates)
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
        "calibration_selection": protocol.selection,
        "calibration_indices": list(protocol.calibration_indices),
        "evaluation_indices": list(protocol.evaluation_indices),
        "calibration_indices_sha256": stem_runner._sha256(
            calibration_indices_path),
        "calibration_metadata_sha256": stem_runner._sha256(
            calibration_metadata_path),
        "evaluation_protocol_sha256": stem_runner._sha256(
            evaluation_protocol_path),
        "configurations": [
            candidate_contract(candidate) for candidate in candidates],
        "guidance": "fp32",
        "bias": "fp32",
        "propagation": dict(base.PROPAGATION_A8_Q13),
        "folded_pairs": reference_preparation["folded_pairs"],
        "source_hashes": {
            "runner": stem_runner._sha256(Path(__file__)),
            "candidate_module": stem_runner._sha256(
                REPO_ROOT / "spn_quant" / "cspn_encoder_prefix.py"),
        },
    }
    _write_results(
        output, matrix, aggregate, interactions,
        normalized_pareto, mac_pareto, regional_rows,
        prediction_names, manifest)
    reference_model.cpu()
    del reference_model
    torch.cuda.empty_cache()
    print("CSPN encoder-prefix W8A8 evaluation complete: %s" % output,
          flush=True)


if __name__ == "__main__":
    main()
