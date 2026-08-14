#!/usr/bin/env python3
"""Evaluate CSPN decoder and initial-depth precision sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from scripts import run_nyu_cspn_activation_resolution as base
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
from spn_quant.cspn_sensitivity import (
    BLOCK_ORDER,
    SensitivityCandidate,
    build_candidate_registry,
    build_cumulative_candidates,
    build_stage1_candidates,
    build_stage2_candidates,
    pareto_rows,
    precision_cost,
    select_sensitive_blocks,
)


def hardware_configuration(
        candidate: SensitivityCandidate) -> Dict[str, object]:
    return base._configuration(
        candidate.name,
        base.ORDINARY_GROUPS,
        base.ORDINARY_GROUPS,
        base.PROPAGATION_A8_Q13,
        granularity="hybrid_group",
        group_size=8,
        promoted_owners=candidate.activation_owners,
        weight_bit_overrides=tuple(
            (name, 8) for name in candidate.weight_modules))


def candidate_registry_from_context(instrumentor, rotation):
    modules = tuple(
        name for name in stem_runner.executed_operation_modules(instrumentor)
        if name.startswith("gud_up_proj_layer"))
    owners = []
    for key in instrumentor.activation_site_keys(base.ORDINARY_GROUPS):
        owner = base.activation_owner(key)
        if base.owner_block(owner) in BLOCK_ORDER:
            owners.append(owner)
    owners.extend(
        ("rotation.%s" % name, "boundary")
        for name in sorted(rotation.channels))
    return build_candidate_registry(modules, tuple(owners))


def validate_configured_precision(
        candidate: SensitivityCandidate,
        weight_bits: Mapping[str, int], specs, rotation_specs) -> None:
    actual_weights = {
        str(name) for name in weight_bits if int(weight_bits[name]) == 8
    }
    invalid_weight_bits = {
        str(name): int(weight_bits[name]) for name in weight_bits
        if int(weight_bits[name]) not in (4, 8)
    }
    if invalid_weight_bits:
        raise RuntimeError("configured weight precision is invalid")
    if actual_weights != set(candidate.weight_modules):
        raise RuntimeError("configured weight promotion differs from candidate")
    actual_owners = set()
    for key in specs:
        if int(specs[key].bits) == 8:
            actual_owners.add(base.activation_owner(key))
        elif int(specs[key].bits) != 4:
            raise RuntimeError("configured activation precision is invalid")
    for owner in rotation_specs:
        if int(rotation_specs[owner].bits) == 8:
            actual_owners.add(tuple(owner))
        elif int(rotation_specs[owner].bits) != 4:
            raise RuntimeError("configured activation precision is invalid")
    if actual_owners != set(candidate.activation_owners):
        raise RuntimeError(
            "configured activation promotion differs from candidate")


def configure_candidate_context(
        candidate: SensitivityCandidate,
        instrumentor, rotation, propagation, stem):
    config = hardware_configuration(candidate)
    specs, rotation_specs, active_merge = base._configure_quantized(
        config, instrumentor, rotation, propagation, {})
    if active_merge is not None:
        raise RuntimeError("sensitivity search forbids merge adapters")
    stem.configure("STRICT_W4A4")
    validate_configured_precision(
        candidate, instrumentor.weight_bits_by_module(),
        specs, rotation_specs)
    return config, specs, rotation_specs


def activation_cost_rows(
        instrumentor, rotation, calibration_samples: int,
        stem_input_elements: int) -> list[Dict[str, object]]:
    samples = int(calibration_samples)
    stem_elements = int(stem_input_elements)
    if samples <= 0 or stem_elements <= 0:
        raise ValueError("activation cost dimensions must be positive")
    rows = []
    owners = set()
    for key in instrumentor.activation_site_keys(base.ORDINARY_GROUPS):
        owner = base.activation_owner(key)
        observer = instrumentor.relu_channel_observers[key] \
            if isinstance(key, str) else instrumentor.channel_observers[key]
        channels = int(torch.as_tensor(observer.minimum).numel())
        total = int(observer.scalar_count) * channels
        if total <= 0 or total % samples:
            raise ValueError("activation observer count is not per-sample exact")
        if owner in owners:
            raise ValueError("activation cost owner is duplicated: %s" % (owner,))
        owners.add(owner)
        rows.append({
            "module": owner[0],
            "kind": owner[1],
            "elements": total // samples,
        })
    for name in sorted(rotation.observers):
        observer = rotation.observers[name]["identity"]
        total = int(observer.scalar_count)
        if total <= 0 or total % samples:
            raise ValueError("rotation observer count is not per-sample exact")
        owner = "rotation.%s" % name, "boundary"
        if owner in owners:
            raise ValueError("activation cost owner is duplicated: %s" % (owner,))
        owners.add(owner)
        rows.append({
            "module": owner[0],
            "kind": owner[1],
            "elements": total // samples,
        })
    stem_owner = "conv1_1", "input"
    if stem_owner in owners:
        raise ValueError("stem activation cost owner is duplicated")
    rows.append({
        "module": stem_owner[0],
        "kind": stem_owner[1],
        "elements": stem_elements,
    })
    return rows


def aggregate_candidate_metrics(
        rows: Sequence[Mapping[str, object]],
        candidates: Sequence[SensitivityCandidate]) -> list[Dict[str, object]]:
    output = []
    for candidate in candidates:
        selected = [row for row in rows if row["config"] == candidate.name]
        if len(selected) != stem_runner.EVALUATION_SAMPLES:
            raise ValueError("%s requires exactly 64 samples" % candidate.name)
        identities = [int(row["sample_index"]) for row in selected]
        if len(set(identities)) != len(identities):
            raise ValueError("%s sample identities must be unique" % candidate.name)
        aggregate = {
            "config": candidate.name,
            "stage": candidate.stage,
            "block": candidate.block,
            "mode": candidate.mode,
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


def prediction_candidate_names(
        aggregate_rows: Sequence[Mapping[str, object]],
        pareto: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
    by_stage = {}
    for row in aggregate_rows:
        stage = str(row["stage"])
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append(row)
    for stage in ("baseline", "block", "site"):
        if stage not in by_stage:
            raise ValueError("prediction selection lacks stage: %s" % stage)
    names = [min(
        by_stage[stage],
        key=lambda row: (float(row["RMSE"]), str(row["config"])))
        ["config"] for stage in ("baseline", "block", "site")]
    pareto_names = {str(row["config"]) for row in pareto}
    cumulative = []
    if "cumulative" in by_stage:
        cumulative = sorted(
            (row for row in by_stage["cumulative"]
             if str(row["config"]) in pareto_names),
            key=lambda row: (
                float(row["normalized_added_bit_cost"]),
                str(row["config"])))
    names.extend(str(row["config"]) for row in cumulative)
    output = []
    for name in names:
        value = str(name)
        if value not in output:
            output.append(value)
    return tuple(output)


def _candidate_contract(candidate: SensitivityCandidate):
    return {
        "name": candidate.name,
        "stage": candidate.stage,
        "block": candidate.block,
        "mode": candidate.mode,
        "weight_modules": list(candidate.weight_modules),
        "activation_owners": [list(owner)
                              for owner in candidate.activation_owners],
    }


def _evaluate_candidate(
        candidate, reference_model, model, saved_args, dataset, indices,
        device, seed, instrumentor, rotation, propagation,
        prediction_root=None):
    reference_capture = base.ModuleOutputCapture(
        reference_model, base.CSPN_BLOCK_SITES)
    quantized_capture = base.ModuleOutputCapture(
        model, base.CSPN_BLOCK_SITES)
    operation_modules = stem_runner.executed_operation_modules(instrumentor)
    operation_counter = stem_runner.ConvOperationCounter(
        model, operation_modules)
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
        row.update({
            "model": "cspn",
            "config": candidate.name,
            "block": source["block"],
        })
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
        raise RuntimeError("fresh CSPN sensitivity registry changed")
    activation_rows = activation_cost_rows(
        instrumentor, rotation, len(protocol.calibration_indices),
        int(preparation_args[0].numel()))
    config, specs, rotation_specs = configure_candidate_context(
        candidate, instrumentor, rotation, propagation, stem)
    result = _evaluate_candidate(
        candidate, reference_model, model, saved_args, valset,
        protocol.evaluation_indices, device, protocol.seed,
        instrumentor, rotation, propagation, prediction_root)
    result["activation_rows"] = activation_rows
    result["checkpoint_load"] = load_report
    result["site_counts"] = {
        "ordinary": len(specs),
        "rotation": len(rotation_specs),
    }
    result["hardware_configuration"] = config
    stem.close()
    propagation.close()
    rotation.close()
    instrumentor.close()
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return registry, result


def _extend_results(target, result) -> None:
    for key in (
            "sample_rows", "region_rows", "propagation_rows",
            "block_rows", "operation_rows", "layer_rows"):
        target[key].extend(result[key])


def _aggregate_with_cost(
        sample_rows, candidates, operation_basis, activation_basis):
    rows = aggregate_candidate_metrics(sample_rows, candidates)
    by_name = dict((candidate.name, candidate) for candidate in candidates)
    for row in rows:
        row.update(precision_cost(
            operation_basis, activation_basis, by_name[row["config"]]))
    return rows


def _regional_aggregates(region_rows, candidates):
    output = []
    for candidate in candidates:
        selected = [row for row in region_rows
                    if row["config"] == candidate.name]
        for source in aggregate_region_rows(selected):
            row = dict(source)
            row.update({"model": "cspn", "config": candidate.name})
            output.append(row)
    return output


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
        raise ValueError("CSPN sensitivity evaluation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.fold_max_error <= 0.0:
        raise ValueError("fold error threshold must be positive")
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
    if output.exists():
        raise FileExistsError("output directory already exists: %s" % output)
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
    result_rows = dict((key, []) for key in (
        "sample_rows", "region_rows", "propagation_rows",
        "block_rows", "operation_rows", "layer_rows"))
    results_by_name = {}
    candidates_by_name = {}

    strict = SensitivityCandidate(
        "STRICT_W4A4", "baseline", "all", "W4A4", (), ())
    registry, strict_result = _run_candidate(
        strict, None, reference_model, architecture, reference_load,
        reference_preparation, saved_args, checkpoint, preparation_args,
        trainset, valset, protocol, device, args.fold_max_error)
    _extend_results(result_rows, strict_result)
    results_by_name[strict.name] = strict_result
    candidates_by_name[strict.name] = strict
    operation_basis = [dict(row) for row in strict_result["operation_rows"]]
    activation_basis = [dict(row)
                        for row in strict_result["activation_rows"]]

    stage1_candidates = build_stage1_candidates(registry)
    if stage1_candidates[0] != strict:
        raise RuntimeError("Stage-1 strict candidate contract changed")
    for candidate in stage1_candidates[1:]:
        _, result = _run_candidate(
            candidate, registry, reference_model, architecture,
            reference_load, reference_preparation, saved_args, checkpoint,
            preparation_args, trainset, valset, protocol, device,
            args.fold_max_error)
        _extend_results(result_rows, result)
        results_by_name[candidate.name] = result
        candidates_by_name[candidate.name] = candidate
        if result["activation_rows"] != activation_basis:
            raise RuntimeError("calibrated activation element counts changed")
    stage1_metrics = _aggregate_with_cost(
        result_rows["sample_rows"], stage1_candidates,
        operation_basis, activation_basis)
    selected_blocks = select_sensitive_blocks(
        stage1_metrics, stage1_candidates)

    stage2_candidates = build_stage2_candidates(registry, selected_blocks)
    for candidate in stage2_candidates:
        _, result = _run_candidate(
            candidate, registry, reference_model, architecture,
            reference_load, reference_preparation, saved_args, checkpoint,
            preparation_args, trainset, valset, protocol, device,
            args.fold_max_error)
        _extend_results(result_rows, result)
        results_by_name[candidate.name] = result
        candidates_by_name[candidate.name] = candidate
        if result["activation_rows"] != activation_basis:
            raise RuntimeError("calibrated activation element counts changed")
    stage2_metrics = _aggregate_with_cost(
        result_rows["sample_rows"], stage2_candidates,
        operation_basis, activation_basis)
    strict_rmse = float(stage1_metrics[0]["RMSE"])
    cumulative_candidates = build_cumulative_candidates(
        stage2_metrics, stage2_candidates, strict_rmse)
    for candidate in cumulative_candidates:
        _, result = _run_candidate(
            candidate, registry, reference_model, architecture,
            reference_load, reference_preparation, saved_args, checkpoint,
            preparation_args, trainset, valset, protocol, device,
            args.fold_max_error)
        _extend_results(result_rows, result)
        results_by_name[candidate.name] = result
        candidates_by_name[candidate.name] = candidate
        if result["activation_rows"] != activation_basis:
            raise RuntimeError("calibrated activation element counts changed")
    cumulative_metrics = _aggregate_with_cost(
        result_rows["sample_rows"], cumulative_candidates,
        operation_basis, activation_basis) if cumulative_candidates else []
    all_candidates = stage1_candidates + stage2_candidates + \
        cumulative_candidates
    all_metrics = stage1_metrics + stage2_metrics + cumulative_metrics

    baseline_samples = [row for row in result_rows["sample_rows"]
                        if row["config"] == strict.name]
    for row in all_metrics:
        if row["config"] == strict.name:
            row.update({
                "baseline_rmse": strict_rmse,
                "rmse_delta": 0.0,
                "wins": 0,
                "accepted": True,
            })
            continue
        candidate_samples = [
            sample for sample in result_rows["sample_rows"]
            if sample["config"] == row["config"]
        ]
        row.update(stem_runner.acceptance(
            baseline_samples, candidate_samples))
    pareto = pareto_rows(all_metrics)
    prediction_names = prediction_candidate_names(all_metrics, pareto)
    prediction_root = output / "predictions"
    prediction_root.mkdir()
    for name in prediction_names:
        candidate = candidates_by_name[name]
        _, prediction_result = _run_candidate(
            candidate, registry, reference_model, architecture,
            reference_load, reference_preparation, saved_args, checkpoint,
            preparation_args, trainset, valset, protocol, device,
            args.fold_max_error, prediction_root=prediction_root)
        expected = next(row for row in all_metrics if row["config"] == name)
        observed = aggregate_candidate_metrics(
            prediction_result["sample_rows"], (candidate,))[0]
        if float(observed["RMSE"]) != float(expected["RMSE"]):
            raise RuntimeError("prediction rerun metric changed: %s" % name)

    regional_rows = _regional_aggregates(
        result_rows["region_rows"], all_candidates)
    write_csv(
        output / "sample_metrics_64.csv", result_rows["sample_rows"],
        base.SAMPLE_FIELDS)
    write_csv(
        output / "aggregate_metrics.csv", all_metrics,
        ("config", "stage", "block", "mode", "samples") +
        stem_runner.METRIC_FIELDS)
    write_csv(
        output / "stage1_block_metrics.csv", stage1_metrics,
        ("config", "stage", "block", "mode", "RMSE", "rmse_delta"))
    write_csv(
        output / "stage2_site_metrics.csv", stage2_metrics,
        ("config", "stage", "block", "mode", "RMSE", "rmse_delta"))
    write_csv(
        output / "cumulative_metrics.csv", cumulative_metrics,
        ("config", "stage", "block", "mode", "RMSE", "rmse_delta"))
    write_csv(
        output / "pareto_metrics.csv", pareto,
        ("config", "stage", "block", "mode", "RMSE",
         "normalized_added_bit_cost"))
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
        output / "activation_cost_basis.csv", activation_basis,
        ("module", "kind", "elements"))
    write_csv(
        output / "layer_quantization_metrics.csv", result_rows["layer_rows"],
        ("model", "config", "module", "group", "kind"))
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
        "selected_blocks": list(selected_blocks),
        "configurations": [_candidate_contract(candidate)
                           for candidate in all_candidates],
        "prediction_configurations": list(prediction_names),
        "guidance": "fp32",
        "bias": "fp32",
        "propagation": dict(base.PROPAGATION_A8_Q13),
        "artifacts": {},
    }
    write_json(output / "manifest.json", manifest)
    manifest["artifacts"] = stem_runner._artifact_hashes(output)
    write_json(output / "manifest.json", manifest)
    reference_model.cpu()
    del reference_model
    torch.cuda.empty_cache()
    print("CSPN decoder sensitivity evaluation complete: %s" % output,
          flush=True)


if __name__ == "__main__":
    main()
