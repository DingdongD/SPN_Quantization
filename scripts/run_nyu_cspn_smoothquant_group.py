#!/usr/bin/env python3
"""Evaluate SmoothQuant with Group-A4 on the official CSPN model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_activation_resolution as base
from scripts.export_nyu_predictions import load_run_args, prepare_args
from scripts.hardware_aligned_quantization import (
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
    smoothquant_scale,
)
from scripts.run_nyu_rtn_quantization import (
    aggregate_region_rows,
    calibration_dataset,
    evaluation_dataset,
    load_sample_indices,
    seeded_sample,
    write_csv,
    write_json,
)
from spn_quant.adapters import install_model_semantic_adapter
from spn_quant.propagation import install_propagation_adapter
from spn_quant.rotation import CSPNRotationController


CALIBRATION_SAMPLES = 128
EVALUATION_SAMPLES = 64
ORDINARY_GROUPS = base.ORDINARY_GROUPS
PROPAGATION_A8_Q13 = base.PROPAGATION_A8_Q13
SMOOTH_ALPHAS = (0.25, 0.5, 0.75)
EXPECTED_CONFIGURATIONS = (
    "FP32", "PA_ONLY", "W4_ONLY", "SQ_W4_ONLY_A050",
    "A4_ONLY_GROUP16", "SQ_A4_ONLY_GROUP16_A050",
    "A4_ONLY_GROUP8", "SQ_A4_ONLY_GROUP8_A050",
    "W4A4_GROUP16", "SQ_W4A4_GROUP16_A025",
    "SQ_W4A4_GROUP16_A050", "SQ_W4A4_GROUP16_A075",
    "W4A4_GROUP8", "SQ_W4A4_GROUP8_A025",
    "SQ_W4A4_GROUP8_A050", "SQ_W4A4_GROUP8_A075",
)


def _config(name, weight_groups, activation_groups, group_size=None,
            smooth_alpha=None):
    smooth_groups = ORDINARY_GROUPS if smooth_alpha is not None else set()
    return base._configuration(
        name, weight_groups, activation_groups,
        None if name == "FP32" else PROPAGATION_A8_Q13,
        granularity="tensor" if group_size is None else
        "hybrid_group_tensor",
        group_size=group_size,
        smooth_groups=smooth_groups,
        smooth_alpha=smooth_alpha)


def build_configurations():
    configs = [
        _config("FP32", set(), set()),
        _config("PA_ONLY", set(), set()),
        _config("W4_ONLY", ORDINARY_GROUPS, set()),
        _config("SQ_W4_ONLY_A050", ORDINARY_GROUPS, set(),
                smooth_alpha=0.5),
    ]
    for group_size in (16, 8):
        configs.extend((
            _config("A4_ONLY_GROUP%d" % group_size, set(),
                    ORDINARY_GROUPS, group_size),
            _config("SQ_A4_ONLY_GROUP%d_A050" % group_size, set(),
                    ORDINARY_GROUPS, group_size, 0.5),
            _config("W4A4_GROUP%d" % group_size, ORDINARY_GROUPS,
                    ORDINARY_GROUPS, group_size),
        ))
        for alpha in SMOOTH_ALPHAS:
            configs.append(_config(
                "SQ_W4A4_GROUP%d_A%03d" %
                (group_size, int(round(alpha * 100.0))),
                ORDINARY_GROUPS, ORDINARY_GROUPS, group_size, alpha))
    by_name = dict((config["name"], config) for config in configs)
    return tuple(by_name[name] for name in EXPECTED_CONFIGURATIONS)


def calibration_configurations(configs):
    return tuple(
        config for config in configs
        if config["weight_groups"] and config["activation_groups"])


def build_smooth_channel_maxima(instrumentor, groups):
    return base.build_smooth_channel_maxima(instrumentor, groups)


def transformed_group_maxima(channel_maxima, scales, group_size):
    channel_maxima = torch.as_tensor(channel_maxima, dtype=torch.float32)
    scales = torch.as_tensor(scales, dtype=torch.float32)
    if channel_maxima.ndim != 1 or scales.shape != channel_maxima.shape:
        raise ValueError("SmoothQuant channel maxima and scales must align")
    if channel_maxima.numel() % int(group_size) != 0:
        raise ValueError("SmoothQuant channels must divide the group size")
    return (channel_maxima / scales).reshape(
        -1, int(group_size)).amax(dim=1)


def select_smooth_configuration(rows, group_size):
    prefix = "SQ_W4A4_GROUP%d_" % int(group_size)
    selected = [
        row for row in rows
        if row["split"] == "calibration"
        and str(row["config"]).startswith(prefix)
    ]
    if not selected:
        raise ValueError(
            "missing SmoothQuant calibration rows for group %d" % group_size)
    return min(selected, key=lambda row: (
        float(row["block_output_mse"]),
        -float(row["block_output_sqnr"]),
        str(row["config"])))


def smooth_diagnostic_rows(instrumentor, configs):
    channel_maxima = build_smooth_channel_maxima(
        instrumentor, ORDINARY_GROUPS)
    rows = []
    for config in configs:
        if config["smooth_alpha"] is None or \
                not config["activation_groups"]:
            continue
        group_size = int(config["group_size"])
        alpha = float(config["smooth_alpha"])
        for name in sorted(channel_maxima):
            module = instrumentor.modules[name]
            maximum = channel_maxima[name]
            channels = int(maximum.numel())
            if channels % group_size != 0:
                continue
            input_channel_dim = 0 if isinstance(
                module, torch.nn.ConvTranspose2d) else 1
            scales = smoothquant_scale(
                instrumentor.original_weights[name], maximum, alpha,
                input_channel_dim=input_channel_dim)
            transformed = maximum / scales
            before_groups = maximum.reshape(
                -1, group_size).amax(dim=1)
            after_groups = transformed.reshape(
                -1, group_size).amax(dim=1)
            rows.append({
                "config": config["name"],
                "module": name,
                "group": instrumentor.groups[name],
                "channels": channels,
                "group_size": group_size,
                "alpha": alpha,
                "activation_max_before": float(maximum.max().item()),
                "activation_max_after": float(transformed.max().item()),
                "channel_imbalance_before": float(
                    maximum.max().item() / maximum.mean().clamp_min(1e-12).item()),
                "channel_imbalance_after": float(
                    transformed.max().item() /
                    transformed.mean().clamp_min(1e-12).item()),
                "group_range_max_before": float(before_groups.max().item()),
                "group_range_max_after": float(after_groups.max().item()),
                "scale_min": float(scales.min().item()),
                "scale_max": float(scales.max().item()),
            })
    return rows


def activation_scale_count(instrumentor, rotation, config):
    if not config["activation_groups"]:
        return 0, 0
    specs = base.build_activation_specs(
        instrumentor, config["activation_groups"],
        int(config["a_bits"]), config["group_size"])
    rotation_specs = base.build_rotation_activation_specs(
        rotation, int(config["a_bits"]), config["group_size"])
    scale_count = 0
    for key in specs:
        spec = specs[key]
        observer = instrumentor.channel_observers[key] \
            if not isinstance(key, str) else \
            instrumentor.relu_channel_observers[key]
        channels = int(observer.minimum.numel())
        scale_count += 1 if spec.granularity == "tensor" else \
            channels if spec.granularity == "channel" else \
            channels // int(spec.group_size)
    for owner in rotation_specs:
        spec = rotation_specs[owner]
        name = owner[0].split(".", 1)[1]
        channels = int(rotation.channels[name])
        scale_count += 1 if spec.granularity == "tensor" else \
            channels if spec.granularity == "channel" else \
            channels // int(spec.group_size)
    return len(specs) + len(rotation_specs), scale_count


def validate_prediction_coverage(model_output, configs, indices):
    expected_configs = set(configs)
    root = Path(model_output) / "predictions"
    actual_configs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_configs != expected_configs:
        raise ValueError("SmoothQuant prediction directories differ")
    expected_files = {
        "sample_%05d.npz" % int(index) for index in indices
    }
    for config in expected_configs:
        actual_files = {
            path.name for path in (root / config).glob("sample_*.npz")
        }
        if actual_files != expected_files:
            raise ValueError("prediction coverage mismatch: %s" % config)


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
    parser.add_argument("--sample-capacity", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def _collect(results, configs, field):
    rows = []
    for config in configs:
        rows.extend(results[config["name"]][field])
    return rows


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN SmoothQuant evaluation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
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
    configs = build_configurations()
    diagnostics = smooth_diagnostic_rows(instrumentor, configs)

    calibration_configs = calibration_configurations(configs)
    calibration_results = {}
    selection_rows = []
    for config in calibration_configs:
        result = base.run_configuration(
            reference_model, quantized_model, saved_args, trainset,
            calibration_indices, "calibration", device, args.seed, config,
            instrumentor, rotation, propagation, merge_adapters,
            reference_capture, quantized_capture, args.sample_capacity)
        calibration_results[config["name"]] = result
        aggregate = next(
            row for row in result["block_rows"]
            if row["block"] == "__all__")
        selection_rows.append(dict(aggregate, config=config["name"]))

    selected_g16 = select_smooth_configuration(selection_rows, 16)
    selected_g8 = select_smooth_configuration(selection_rows, 8)
    prediction_configs = {
        "FP32", "W4A4_GROUP16", str(selected_g16["config"]),
        "W4A4_GROUP8", str(selected_g8["config"]),
    }

    evalset = evaluation_dataset(saved_args)
    evaluation_indices = load_sample_indices(args.sample_metrics)
    if len(evaluation_indices) != EVALUATION_SAMPLES:
        raise ValueError("CSPN evaluation requires exactly 64 samples")
    if max(evaluation_indices) >= len(evalset):
        raise ValueError("evaluation sample index exceeds NYU validation split")

    model_output = Path(args.out_dir) / "cspn"
    analysis_output = Path(args.out_dir) / "analysis"
    model_output.mkdir(parents=True, exist_ok=True)
    analysis_output.mkdir(parents=True, exist_ok=True)
    evaluation_results = {}
    for config in configs:
        prediction_root = model_output \
            if config["name"] in prediction_configs else None
        evaluation_results[config["name"]] = base.run_configuration(
            reference_model, quantized_model, saved_args, evalset,
            evaluation_indices, "evaluation", device, args.seed, config,
            instrumentor, rotation, propagation, merge_adapters,
            reference_capture, quantized_capture, args.sample_capacity,
            prediction_root=prediction_root)

    sample_rows = _collect(evaluation_results, configs, "sample_rows")
    base.validate_sample_coverage(
        sample_rows, EXPECTED_CONFIGURATIONS, evaluation_indices)
    validate_prediction_coverage(
        model_output, prediction_configs, evaluation_indices)
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

    block_rows = _collect(
        calibration_results, calibration_configs, "block_rows") + \
        _collect(evaluation_results, configs, "block_rows")
    tensor_rows = _collect(
        calibration_results, calibration_configs, "tensor_rows") + \
        _collect(evaluation_results, configs, "tensor_rows")
    channel_rows = _collect(
        calibration_results, calibration_configs, "channel_rows") + \
        _collect(evaluation_results, configs, "channel_rows")
    layer_rows = _collect(evaluation_results, configs, "layer_rows")
    propagation_rows = _collect(
        evaluation_results, configs, "propagation_rows")

    manifest = []
    smooth_modules = build_smooth_channel_maxima(
        instrumentor, ORDINARY_GROUPS)
    for config in configs:
        activation_sites, activation_scales = activation_scale_count(
            instrumentor, rotation, config)
        manifest.append({
            "config": config["name"],
            "weight_bits": "" if not config["weight_groups"] else 4,
            "activation_bits": "" if not config["activation_groups"] else 4,
            "group_size": "" if config["group_size"] is None else
            config["group_size"],
            "smooth_alpha": "" if config["smooth_alpha"] is None else
            config["smooth_alpha"],
            "smooth_modules": 0 if config["smooth_alpha"] is None else
            len(smooth_modules),
            "activation_sites": activation_sites,
            "activation_scales": activation_scales,
            "guidance_head": "fp32",
            "propagation": "fp32" if config["name"] == "FP32" else
            "a8_int16_q13_int32",
        })

    write_csv(model_output / "config_manifest.csv", manifest)
    write_csv(model_output / "sample_metrics.csv", sample_rows,
              base.SAMPLE_FIELDS)
    write_csv(model_output / "aggregate_metrics.csv", mean_rows)
    write_csv(model_output / "regional_metrics.csv", regional_rows)
    write_csv(model_output / "block_metrics.csv", block_rows)
    write_csv(model_output / "activation_resolution_metrics.csv", tensor_rows)
    write_csv(model_output / "activation_channel_metrics.csv", channel_rows)
    write_csv(model_output / "layer_quantization_metrics.csv", layer_rows)
    write_csv(model_output / "propagation_metrics.csv", propagation_rows)
    write_csv(analysis_output / "calibration_selection.csv", selection_rows)
    write_csv(analysis_output / "smoothquant_channel_diagnostics.csv",
              diagnostics)
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
        "evaluation_samples": EVALUATION_SAMPLES,
        "evaluation_indices": evaluation_indices,
        "selected_group16": selected_g16["config"],
        "selected_group8": selected_g8["config"],
        "prediction_configs": sorted(prediction_configs),
        "smoothquant_modules": sorted(smooth_modules),
        "guidance_head": "fp32",
        "bias_format": "fp32",
        "propagation": dict(PROPAGATION_A8_Q13),
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
