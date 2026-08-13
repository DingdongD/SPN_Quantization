#!/usr/bin/env python3
"""Evaluate RMS-ranked Group-8 activation quantization on official CSPN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    load_run_args,
    prepare_args,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    aggregate_region_rows,
    calibration_dataset,
    evaluation_dataset,
    load_sample_indices,
    seeded_sample,
    write_csv,
    write_json,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.propagation import install_propagation_adapter  # noqa: E402
from spn_quant.rotation import CSPNRotationController  # noqa: E402
from spn_quant.scale_aware_grouping import (  # noqa: E402
    build_scale_aware_grouping,
)


CALIBRATION_SAMPLES = 128
EVALUATION_SAMPLES = 64
GROUP_SIZE = 8
DISPERSION_EPSILON = 1e-12
EXPECTED_CONFIGURATIONS = (
    "W4A4_G8_MINMAX",
    "W4A4_G8_SCALE_AWARE",
)


def build_methods():
    return (
        {
            "config": "W4A4_G8_MINMAX",
            "calibration": "minmax",
            "group_size": 8,
            "scale_aware": False,
            "dynamic": False,
        },
        {
            "config": "W4A4_G8_SCALE_AWARE",
            "calibration": "minmax",
            "group_size": 8,
            "scale_aware": True,
            "dynamic": False,
        },
    )


def validate_runtime_contract(calibration_samples, evaluation_indices):
    if int(calibration_samples) != CALIBRATION_SAMPLES:
        raise ValueError("CSPN scale-aware calibration requires 128 samples")
    if len(evaluation_indices) != EVALUATION_SAMPLES:
        raise ValueError("CSPN scale-aware evaluation requires 64 samples")


def build_scale_aware_groupings(instrumentor, specs, group_size, epsilon):
    groupings = {}
    for key in sorted(specs, key=str):
        if isinstance(key, str):
            continue
        name, kind = key
        if kind != "input":
            continue
        module = instrumentor.modules[name]
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            continue
        if module.groups != 1:
            continue
        spec = specs[key]
        if spec.granularity != "group" or \
                int(spec.group_size) != int(group_size):
            continue
        observer = instrumentor.channel_observers[key]
        groupings[key] = build_scale_aware_grouping(
            observer.channel_rms(), group_size, epsilon)
    if not groupings:
        raise RuntimeError("official CSPN has no eligible scale-aware inputs")
    return groupings


def _weighted_dispersion(dispersion, group_rms):
    weights = group_rms.to(torch.float64).square().sum(dim=1)
    if float(weights.sum().item()) <= 0.0:
        raise ValueError("dispersion weights must contain activation energy")
    return float((
        dispersion.to(torch.float64) * weights / weights.sum()).sum().item())


def grouping_rows(groupings):
    channel_rows = []
    summary_rows = []
    for key in sorted(groupings, key=str):
        module, kind = key
        grouping = groupings[key]
        contiguous_groups = grouping.channel_rms.reshape(
            -1, grouping.group_size)
        for channel in range(int(grouping.channel_rms.numel())):
            position = int(grouping.inverse[channel].item())
            channel_rows.append({
                "module": module,
                "kind": kind,
                "channel": channel,
                "rms": float(grouping.channel_rms[channel].item()),
                "permuted_position": position,
                "scale_aware_group": position // grouping.group_size,
                "contiguous_group": channel // grouping.group_size,
                "group_size": grouping.group_size,
            })
        contiguous = grouping.contiguous_dispersion
        scale_aware = grouping.scale_aware_dispersion
        summary_rows.append({
            "module": module,
            "kind": kind,
            "channels": int(grouping.channel_rms.numel()),
            "group_size": grouping.group_size,
            "groups": int(scale_aware.numel()),
            "epsilon": grouping.epsilon,
            "contiguous_dispersion_mean": float(contiguous.mean().item()),
            "contiguous_dispersion_median": float(contiguous.median().item()),
            "contiguous_dispersion_max": float(contiguous.max().item()),
            "contiguous_dispersion_weighted_mean": _weighted_dispersion(
                contiguous, contiguous_groups),
            "scale_aware_dispersion_mean": float(scale_aware.mean().item()),
            "scale_aware_dispersion_median": float(scale_aware.median().item()),
            "scale_aware_dispersion_max": float(scale_aware.max().item()),
            "scale_aware_dispersion_weighted_mean": _weighted_dispersion(
                scale_aware, grouping.group_rms),
            "dispersion_sum_reduction": 1.0 - float(
                scale_aware.sum().item()) / float(contiguous.sum().item()),
            "permutation": " ".join(
                str(int(value)) for value in grouping.permutation.tolist()),
            "inverse_permutation": " ".join(
                str(int(value)) for value in grouping.inverse.tolist()),
        })
    return channel_rows, summary_rows


def build_configurations(groupings):
    permutations = tuple(
        (key, groupings[key].permutation.clone())
        for key in sorted(groupings, key=str))
    configs = []
    for method in build_methods():
        config = base._configuration(
            method["config"], base.ORDINARY_GROUPS, base.ORDINARY_GROUPS,
            base.PROPAGATION_A8_Q13,
            granularity="hybrid_group_tensor",
            group_size=int(method["group_size"]),
            activation_permutations=permutations
            if method["scale_aware"] else ())
        config["calibration"] = method["calibration"]
        config["scale_aware"] = bool(method["scale_aware"])
        config["dynamic"] = bool(method["dynamic"])
        configs.append(config)
    return tuple(configs)


def _collect(results, configs, field):
    rows = []
    for config in configs:
        rows.extend(results[config["name"]][field])
    return rows


def validate_prediction_coverage(model_output, indices):
    root = Path(model_output) / "predictions"
    actual_configs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_configs != set(EXPECTED_CONFIGURATIONS):
        raise ValueError("scale-aware prediction directories differ")
    expected_files = {
        "sample_%05d.npz" % int(index) for index in indices
    }
    for config in EXPECTED_CONFIGURATIONS:
        actual_files = {
            path.name for path in (root / config).glob("sample_*.npz")
        }
        if actual_files != expected_files:
            raise ValueError("prediction coverage mismatch: %s" % config)


def _weight_invariance_rows(layer_rows, grouping_keys):
    modules = set(key[0] for key in grouping_keys)
    rows = {}
    for source in layer_rows:
        if source["kind"] != "weight" or source["module"] not in modules:
            continue
        rows[(source["config"], source["module"])] = source
    output = []
    for module in sorted(modules):
        baseline = rows[(EXPECTED_CONFIGURATIONS[0], module)]
        scale_aware = rows[(EXPECTED_CONFIGURATIONS[1], module)]
        output.append({
            "module": module,
            "baseline_mse": float(baseline["mse"]),
            "scale_aware_mse": float(scale_aware["mse"]),
            "mse_abs_delta": abs(
                float(scale_aware["mse"]) - float(baseline["mse"])),
            "baseline_error_sq": float(baseline["error_sq"]),
            "scale_aware_error_sq": float(scale_aware["error_sq"]),
            "error_sq_abs_delta": abs(
                float(scale_aware["error_sq"]) -
                float(baseline["error_sq"])),
            "baseline_signal_sq": float(baseline["signal_sq"]),
            "scale_aware_signal_sq": float(scale_aware["signal_sq"]),
        })
    return output


def validate_weight_invariance(rows, expected_modules, tolerance):
    expected_modules = set(expected_modules)
    actual_modules = set(row["module"] for row in rows)
    if actual_modules != expected_modules:
        raise ValueError("weight invariance coverage changed")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("weight invariance tolerance must be finite")
    for row in rows:
        if float(row["mse_abs_delta"]) > tolerance or \
                float(row["error_sq_abs_delta"]) > tolerance:
            raise ValueError(
                "paired input permutation changed W4 reconstruction: %s" %
                row["module"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-metrics", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--calibration-samples", type=int,
        choices=(CALIBRATION_SAMPLES,), required=True)
    parser.add_argument(
        "--group-size", type=int, choices=(GROUP_SIZE,), required=True)
    parser.add_argument("--dispersion-epsilon", type=float, required=True)
    parser.add_argument("--sample-capacity", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN scale-aware evaluation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.dispersion_epsilon != DISPERSION_EPSILON:
        raise ValueError("dispersion epsilon must equal 1e-12")
    if args.sample_capacity <= 0:
        raise ValueError("sample capacity must be positive")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    saved_args = prepare_args(load_run_args(run_dir), args)
    saved_args.data_root = args.data_root
    reference_model, architecture, reference_load = base._load_cspn(
        saved_args, checkpoint, device)
    quantized_model, quantized_architecture, quantized_load = base._load_cspn(
        saved_args, checkpoint, device)
    if architecture != quantized_architecture or \
            reference_load != quantized_load:
        raise RuntimeError("paired CSPN model construction is inconsistent")

    trainset = calibration_dataset(saved_args)
    if len(trainset) < CALIBRATION_SAMPLES:
        raise ValueError("NYU training split has fewer than 128 samples")
    calibration_indices = np.random.RandomState(args.seed).choice(
        len(trainset), CALIBRATION_SAMPLES, replace=False).tolist()
    preparation_sample = seeded_sample(
        trainset, calibration_indices[0], args.seed)
    preparation_args = base._model_args(
        saved_args, preparation_sample, device)
    reference_preparation = prepare_hardware_model(
        reference_model, preparation_args,
        excluded_pairs=(("conv1_1", "bn1"),))
    quantized_preparation = prepare_hardware_model(
        quantized_model, preparation_args,
        excluded_pairs=(("conv1_1", "bn1"),))
    for preparation in (reference_preparation, quantized_preparation):
        if float(preparation["primary_max_abs_error"]) > args.fold_max_error:
            raise RuntimeError("Conv-BN fold exceeds declared error threshold")
    if reference_preparation["folded_pairs"] != \
            quantized_preparation["folded_pairs"]:
        raise RuntimeError("paired CSPN fold manifests differ")

    semantic = install_model_semantic_adapter(
        quantized_model, "cspn", strict=True)
    boundaries = semantic.rotation_boundaries()
    semantic.close()
    instrumentor = HardwareAlignedInstrumentor(
        quantized_model, base.cspn_quant_group,
        quantized_preparation["fused_relu_producers"],
        externally_owned_outputs=base.strict_owned_outputs(),
        externally_owned_inputs=base.strict_owned_inputs())
    rotation = CSPNRotationController(
        quantized_model, boundaries, seed=args.seed)
    propagation = install_propagation_adapter("cspn", quantized_model)
    reference_capture = base.ModuleOutputCapture(
        reference_model, base.CSPN_BLOCK_SITES)
    quantized_capture = base.ModuleOutputCapture(
        quantized_model, base.CSPN_BLOCK_SITES)
    merge_adapters = {}

    started = time.time()
    base._calibrate(
        quantized_model, saved_args, trainset, calibration_indices,
        device, args.seed, instrumentor, rotation, propagation)
    base.validate_strict_site_contract(instrumentor, rotation)
    specs = base.build_activation_specs(
        instrumentor, base.ORDINARY_GROUPS, 4, args.group_size)
    groupings = build_scale_aware_groupings(
        instrumentor, specs, args.group_size, args.dispersion_epsilon)
    configs = build_configurations(groupings)

    evalset = evaluation_dataset(saved_args)
    evaluation_indices = load_sample_indices(args.sample_metrics)
    validate_runtime_contract(args.calibration_samples, evaluation_indices)
    if max(evaluation_indices) >= len(evalset):
        raise ValueError("evaluation sample index exceeds NYU validation split")
    model_output = Path(args.out_dir) / "cspn"
    analysis_output = Path(args.out_dir) / "analysis"
    model_output.mkdir(parents=True, exist_ok=True)
    analysis_output.mkdir(parents=True, exist_ok=True)

    evaluation_results = {}
    for config in configs:
        evaluation_results[config["name"]] = base.run_configuration(
            reference_model, quantized_model, saved_args, evalset,
            evaluation_indices, "evaluation", device, args.seed, config,
            instrumentor, rotation, propagation, merge_adapters,
            reference_capture, quantized_capture, args.sample_capacity,
            prediction_root=model_output)
    sample_rows = _collect(evaluation_results, configs, "sample_rows")
    base.validate_sample_coverage(
        sample_rows, EXPECTED_CONFIGURATIONS, evaluation_indices)
    validate_prediction_coverage(model_output, evaluation_indices)
    mean_rows = base._mean_metric_rows(sample_rows, configs)

    region_rows = _collect(evaluation_results, configs, "region_rows")
    regional_rows = []
    for config in configs:
        selected = [
            row for row in region_rows if row["config"] == config["name"]
        ]
        for source in aggregate_region_rows(selected):
            regional_rows.append(dict(
                source, model="cspn", config=config["name"]))
    block_rows = _collect(evaluation_results, configs, "block_rows")
    tensor_rows = _collect(evaluation_results, configs, "tensor_rows")
    channel_metric_rows = _collect(
        evaluation_results, configs, "channel_rows")
    layer_rows = _collect(evaluation_results, configs, "layer_rows")
    propagation_rows = _collect(
        evaluation_results, configs, "propagation_rows")
    channel_rows, summary_rows = grouping_rows(groupings)
    weight_rows = _weight_invariance_rows(layer_rows, groupings)
    validate_weight_invariance(
        weight_rows, set(key[0] for key in groupings), tolerance=1e-12)

    manifests = []
    for config in configs:
        manifests.append({
            "config": config["name"],
            "calibration": config["calibration"],
            "weight_bits": 4,
            "activation_bits": 4,
            "activation_granularity": "hybrid_group_tensor",
            "group_size": args.group_size,
            "scale_aware": int(config["scale_aware"]),
            "permuted_input_sites": len(groupings)
            if config["scale_aware"] else 0,
            "dynamic": 0,
            "guidance_head": "fp32",
            "bias_format": "fp32",
            "propagation": "a8_int16_q13_int32",
        })

    write_csv(model_output / "config_manifest.csv", manifests)
    write_csv(model_output / "sample_metrics.csv", sample_rows,
              base.SAMPLE_FIELDS)
    write_csv(model_output / "aggregate_metrics.csv", mean_rows)
    write_csv(model_output / "regional_metrics.csv", regional_rows)
    write_csv(model_output / "block_metrics.csv", block_rows)
    write_csv(model_output / "activation_resolution_metrics.csv", tensor_rows)
    write_csv(model_output / "activation_channel_metrics.csv",
              channel_metric_rows)
    write_csv(model_output / "layer_quantization_metrics.csv", layer_rows)
    write_csv(model_output / "propagation_metrics.csv", propagation_rows)
    write_csv(analysis_output / "grouping_manifest.csv", channel_rows)
    write_csv(analysis_output / "grouping_summary.csv", summary_rows)
    write_csv(analysis_output / "weight_invariance.csv", weight_rows)
    write_json(model_output / "metadata.json", {
        "model": "cspn",
        "architecture": architecture,
        "model_class": type(reference_model).__name__,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_load": reference_load,
        "data_root": str(Path(args.data_root).resolve()),
        "seed": args.seed,
        "calibration_samples": CALIBRATION_SAMPLES,
        "calibration_indices": calibration_indices,
        "calibration_passes": 1,
        "calibration": "minmax_with_channel_rms",
        "evaluation_samples": EVALUATION_SAMPLES,
        "evaluation_indices": evaluation_indices,
        "group_size": args.group_size,
        "dispersion_epsilon": args.dispersion_epsilon,
        "eligible_input_sites": len(groupings),
        "prediction_configs": list(EXPECTED_CONFIGURATIONS),
        "guidance_head": "fp32",
        "bias_format": "fp32",
        "propagation": dict(base.PROPAGATION_A8_Q13),
        "coefficient_format": "signed_int16_q13",
        "propagation_accumulator": "int32",
        "fold_max_abs_error": max(
            float(reference_preparation["primary_max_abs_error"]),
            float(quantized_preparation["primary_max_abs_error"])),
        "folded_pairs": reference_preparation["folded_pairs"],
        "elapsed_seconds": time.time() - started,
    })

    reference_capture.close()
    quantized_capture.close()
    propagation.close()
    rotation.close()
    instrumentor.close()


if __name__ == "__main__":
    main()
