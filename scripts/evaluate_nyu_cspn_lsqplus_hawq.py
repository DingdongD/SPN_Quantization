#!/usr/bin/env python3
"""Strict fixed-64 evaluation for CSPN LSQ+, HAWQ, and shared baselines."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts import train_nyu_cspn_group_a4_qat as qat_base  # noqa: E402
from scripts import train_nyu_cspn_lsqplus_hawq as trainer  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.evaluate_nyu_selected_quantization import (  # noqa: E402
    prediction_sample_metrics,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    calibration_dataset,
    evaluation_dataset,
    seeded_sample,
)
from spn_quant.activation_boundaries import (  # noqa: E402
    CSPNActivationBoundaryController,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.propagation import install_propagation_adapter  # noqa: E402
from spn_quant.qat.method_config import load_method_config  # noqa: E402


CONFIGURATIONS = (
    "FP32",
    "PA_RTN_W4A4",
    "PA_RTN_W6A6",
    "LSQPLUS_W4A4",
    "LSQPLUS_W6A6",
    "HAWQ_MIXED_LE6",
    "MIXED_TASK_AWARE_QAT",
)


@dataclass(frozen=True)
class AggregationResult:
    metrics: Tuple[dict, ...]
    sample_metrics: Tuple[dict, ...]
    relative_fp_loss: Tuple[dict, ...]


def _prediction_path(root: Path, configuration: str, index: int) -> Path:
    return Path(root) / "shards" / configuration / \
        ("sample_%05d.npz" % int(index))


def _sample_metrics(gt: np.ndarray, pred: np.ndarray):
    row = prediction_sample_metrics({
        "sample_index": 0,
        "gt": gt,
        "pred": pred,
    })
    return {
        "pixels": row["valid_pixel_count"],
        "nonpositive_pixels": row["nonpositive_pixel_count"],
        "sum_square": row["squared_error_sum"],
        "sum_absolute": row["absolute_error_sum"],
        "sum_abs_rel": row["abs_rel_sum"],
        "sum_inverse_square": row["inverse_squared_error_sum"],
        "RMSE": row["sample_rmse"],
        "MAE": row["sample_mae"],
        "ABS_REL": row["sample_abs_rel"],
        "IRMSE": row["sample_irmse"],
        "nonpositive_ratio": row["sample_nonpositive_ratio"],
    }


def aggregate_shards(
        root: Path, indices: Sequence[int]) -> AggregationResult:
    root = Path(root)
    indices = tuple(int(index) for index in indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("evaluation indices must be unique and nonempty")
    reference_gt = {}
    sample_rows = []
    metric_rows = []
    for configuration in CONFIGURATIONS:
        directory = root / "shards" / configuration
        actual = set(
            int(path.stem.split("_")[1])
            for path in directory.glob("sample_*.npz")) \
            if directory.is_dir() else set()
        if actual != set(indices):
            raise RuntimeError(
                "prediction coverage mismatch: %s" % configuration)
        totals = {
            "pixels": 0,
            "sum_square": 0.0,
            "sum_absolute": 0.0,
            "sum_abs_rel": 0.0,
            "sum_inverse_square": 0.0,
            "nonpositive_pixels": 0,
        }
        for index in indices:
            with np.load(
                    _prediction_path(root, configuration, index),
                    allow_pickle=False) as source:
                payload = dict((key, source[key]) for key in source.files)
            if int(payload["sample_index"].item()) != index:
                raise RuntimeError("prediction sample identity mismatch")
            gt = np.asarray(payload["gt"], dtype=np.float32)
            pred = np.asarray(payload["pred"], dtype=np.float32)
            if gt.shape != pred.shape:
                raise RuntimeError("prediction and GT shapes differ")
            if configuration == "FP32":
                reference_gt[index] = gt.copy()
            elif not np.array_equal(reference_gt[index], gt):
                raise RuntimeError("GT identity differs across configurations")
            current = _sample_metrics(gt, pred)
            row = {
                "configuration": configuration,
                "sample_index": index,
                "RMSE": current["RMSE"],
                "MAE": current["MAE"],
                "ABS_REL": current["ABS_REL"],
                "IRMSE": current["IRMSE"],
                "nonpositive_ratio": current["nonpositive_ratio"],
            }
            sample_rows.append(row)
            for key in totals:
                totals[key] += current[key]
        pixels = totals["pixels"]
        metric_rows.append({
            "configuration": configuration,
            "samples": len(indices),
            "pixels": pixels,
            "RMSE": math.sqrt(totals["sum_square"] / pixels),
            "MAE": totals["sum_absolute"] / pixels,
            "ABS_REL": totals["sum_abs_rel"] / pixels,
            "IRMSE": math.sqrt(totals["sum_inverse_square"] / pixels),
            "nonpositive_pixels": totals["nonpositive_pixels"],
            "nonpositive_ratio": totals["nonpositive_pixels"] / pixels,
        })
    fp = metric_rows[0]
    relative = tuple({
        "configuration": row["configuration"],
        "RMSE_delta_m": row["RMSE"] - fp["RMSE"],
        "RMSE_relative_percent": 0.0 if fp["RMSE"] == 0.0 else
        (row["RMSE"] / fp["RMSE"] - 1.0) * 100.0,
        "MAE_delta_m": row["MAE"] - fp["MAE"],
        "ABS_REL_delta": row["ABS_REL"] - fp["ABS_REL"],
        "IRMSE_delta": row["IRMSE"] - fp["IRMSE"],
    } for row in metric_rows)
    return AggregationResult(
        tuple(metric_rows), tuple(sample_rows), relative)


def _write_csv(path: Path, rows) -> None:
    rows = tuple(rows)
    if not rows:
        raise ValueError("cannot write an empty evaluation table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_aggregation(root: Path, result: AggregationResult) -> None:
    root = Path(root)
    _write_csv(root / "aggregate_metrics.csv", result.metrics)
    _write_csv(root / "sample_metrics.csv", result.sample_metrics)
    _write_csv(root / "relative_fp_loss.csv", result.relative_fp_loss)
    invalid = [
        row["configuration"] for row in result.metrics
        if int(row["nonpositive_pixels"]) > 0]
    _write_json(root / "strict_summary.json", {
        "model": "cspn",
        "configurations": list(CONFIGURATIONS),
        "samples": int(result.metrics[0]["samples"]),
        "strict_valid": not invalid,
        "nonpositive_configurations": invalid,
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", choices=CONFIGURATIONS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--fp32-checkpoint")
    parser.add_argument("--method-checkpoint")
    parser.add_argument("--assignment")
    parser.add_argument("--mixed-source-root")
    parser.add_argument("--data-root")
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device")
    return parser.parse_args(argv)


def _validate_paths(args) -> None:
    if args.aggregate:
        if args.configuration is not None:
            raise ValueError("aggregate mode does not accept a configuration")
        return
    required = (
        args.configuration, args.config, args.fp32_checkpoint,
        args.data_root, args.device)
    if any(value is None for value in required):
        raise ValueError("evaluation worker paths are incomplete")
    if args.configuration.startswith("LSQPLUS") or \
            args.configuration == "HAWQ_MIXED_LE6":
        if args.method_checkpoint is None:
            raise ValueError("method evaluation requires a checkpoint")
    if args.configuration == "HAWQ_MIXED_LE6" and args.assignment is None:
        raise ValueError("HAWQ evaluation requires an assignment")
    if args.configuration == "MIXED_TASK_AWARE_QAT" and \
            args.mixed_source_root is None:
        raise ValueError("mixed evaluation requires an explicit source root")


def _prepare_rtn(saved, checkpoint, metadata, bits, config, device):
    assignment = trainer.uniform_method_assignment(
        trainer.expected_registry(), bits)
    model, architecture, load_report = base._load_cspn(
        saved, checkpoint, device)
    trainset = calibration_dataset(saved)
    sample = seeded_sample(
        trainset, metadata.calibration_indices[0], config.training.seed)
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
    boundary = CSPNActivationBoundaryController(model, boundaries)
    propagation = install_propagation_adapter("cspn", model)
    base._calibrate(
        model, saved, trainset, metadata.calibration_indices,
        device, config.training.seed, instrumentor, boundary, propagation)
    base.validate_strict_site_contract(instrumentor, boundary)
    activation_specs = base.build_activation_specs(
        instrumentor, base.ORDINARY_GROUPS, bits, 8)
    boundary_specs = base.build_boundary_activation_specs(boundary, bits, 8)
    activation_specs, boundary_specs = base.apply_activation_bit_assignment(
        activation_specs, boundary_specs, assignment.activation_bits)
    instrumentor.configure_components_with_ranges(
        bits, bits, base.ORDINARY_GROUPS, base.ORDINARY_GROUPS,
        activation_specs, False, activation_maxima={},
        weight_bit_overrides=dict(assignment.weight_bits),
        weight_modules=tuple(
            name for name, current_bits in assignment.weight_bits))
    boundary.configure_specs(
        dict((name, boundary_specs[(
            "boundary_controller.%s" % name, "boundary")].bits)
             for name in boundary.channels),
        dict((name, int(boundary_specs[(
            "boundary_controller.%s" % name, "boundary")].group_size))
             for name in boundary.channels),
        dict((name, 1.0) for name in boundary.channels),
        quantize=True,
    )
    propagation.configure(trainer._propagation_config())
    model.eval()
    return model, (instrumentor, boundary, propagation), {
        "architecture": architecture,
        "load_report": load_report,
        "preparation": preparation,
    }


def _prepare_method(args, config, metadata, saved, device):
    mapping = {
        "LSQPLUS_W4A4": "lsqplus_w4a4",
        "LSQPLUS_W6A6": "lsqplus_w6a6",
        "HAWQ_MIXED_LE6": "hawq_mixed_le6",
    }
    method = mapping[args.configuration]
    registry = trainer.expected_registry()
    if method == "lsqplus_w4a4":
        assignment = trainer.uniform_method_assignment(registry, 4)
    elif method == "lsqplus_w6a6":
        assignment = trainer.uniform_method_assignment(registry, 6)
    elif method == "hawq_mixed_le6":
        assignment = trainer.load_hawq_assignment(
            Path(args.assignment), registry)
    else:
        raise ValueError("unsupported method evaluation")
    prepared = trainer.prepare_method_model(
        saved, Path(args.fp32_checkpoint), metadata, method,
        assignment, config, device)
    controller = prepared[1]
    payload = torch.load(
        args.method_checkpoint, map_location="cpu", weights_only=False)
    trainer.validate_checkpoint_payload(payload)
    if payload["assignment"] != trainer._assignment_payload(assignment):
        raise ValueError("method checkpoint assignment changed")
    controller.load_canonical_model_state_dict(payload["model_state"])
    controller.load_method_state_dict(payload["method_state"])
    if method == "hawq_mixed_le6":
        controller.freeze_activation_ranges()
    prepared[0].eval()
    controller.activation_modules.eval()
    return prepared[0], (
        controller, prepared[2], prepared[3], prepared[4]), {
            "architecture": prepared[5],
            "load_report": prepared[6],
            "preparation": prepared[7],
            "method_checkpoint": str(Path(args.method_checkpoint).resolve()),
        }


def _evaluate_model(
        model, saved, indices, configuration, output, device, seed):
    dataset = evaluation_dataset(saved)
    visualization = sweep.NyuHdf5Dataset(
        csv_file=saved.eval_list,
        root_dir=str(saved.data_root),
        split="val",
        n_sample=saved.n_sample,
        seed=seed,
    )
    rows = []
    with torch.no_grad():
        for index in indices:
            sample = seeded_sample(dataset, index, seed)
            batch = dict(
                (key, value.unsqueeze(0) if torch.is_tensor(value) else value)
                for key, value in sample.items())
            model_input, target = sweep.batch_to_model_input(
                "cspn", batch, device)
            prediction = sweep.extract_pred(model(*model_input))
            gt = target[0, 0].detach().cpu().numpy().astype(np.float32)
            pred = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
            sparse = sample["rgbd"][3].numpy().astype(np.float32)
            natural = visualization[index]["rgbd"][:3].permute(
                1, 2, 0).numpy().astype(np.float32)
            metrics = _sample_metrics(gt, pred)
            rows.append({
                "configuration": configuration,
                "sample_index": index,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "ABS_REL": metrics["ABS_REL"],
                "IRMSE": metrics["IRMSE"],
            })
            np.savez_compressed(
                output / ("sample_%05d.npz" % index),
                sample_index=np.int64(index),
                gt=gt,
                pred=pred,
                rgb=natural,
                sparse=sparse,
            )
    _write_csv(output / "sample_metrics.csv", rows)


def _copy_mixed_predictions(source_root, output, indices):
    source = Path(source_root) / "predictions" / "MIXED_TASK_AWARE_QAT"
    for index in indices:
        path = source / ("sample_%05d.npz" % index)
        if not path.is_file():
            raise RuntimeError("mixed source prediction coverage mismatch")
        shutil.copy2(path, output / path.name)


def _close_components(components) -> None:
    for component in components:
        component.close() if not isinstance(
            component, trainer.CSPNMethodQATController) else component.remove()


def main(argv=None) -> None:
    args = parse_args(argv)
    _validate_paths(args)
    metadata = qat_base.load_calibration_metadata(
        Path(args.calibration_metadata))
    root = Path(args.output_root)
    if args.aggregate:
        result = aggregate_shards(root, metadata.evaluation_indices)
        write_aggregation(root, result)
        return
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CSPN fixed-64 worker requires CUDA")
    output = root / "shards" / args.configuration
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("evaluation shard directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    if args.configuration == "MIXED_TASK_AWARE_QAT":
        _copy_mixed_predictions(
            args.mixed_source_root, output, metadata.evaluation_indices)
        return
    config = load_method_config(Path(args.config))
    saved = trainer._saved_args(Path(args.fp32_checkpoint), args, config)
    device = torch.device(args.device)
    sweep.seed_all(config.training.seed)
    if args.configuration == "FP32":
        model, architecture, load_report = base._load_cspn(
            saved, Path(args.fp32_checkpoint), device)
        components = ()
        manifest = {
            "architecture": architecture,
            "load_report": load_report,
        }
    elif args.configuration in ("PA_RTN_W4A4", "PA_RTN_W6A6"):
        bits = 4 if args.configuration == "PA_RTN_W4A4" else 6
        model, components, manifest = _prepare_rtn(
            saved, Path(args.fp32_checkpoint), metadata,
            bits, config, device)
    elif args.configuration in (
            "LSQPLUS_W4A4", "LSQPLUS_W6A6", "HAWQ_MIXED_LE6"):
        model, components, manifest = _prepare_method(
            args, config, metadata, saved, device)
    else:
        raise ValueError("unsupported evaluation configuration")
    _evaluate_model(
        model, saved, metadata.evaluation_indices,
        args.configuration, output, device, config.training.seed)
    _write_json(output / "manifest.json", manifest)
    _close_components(components)


if __name__ == "__main__":
    main()
