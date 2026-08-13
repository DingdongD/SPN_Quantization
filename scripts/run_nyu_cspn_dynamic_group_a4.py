#!/usr/bin/env python3
"""Evaluate per-sample Dynamic Group-A4 on the official CSPN model."""

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
GROUP_SIZE = 8
ORDINARY_GROUPS = base.ORDINARY_GROUPS
PROPAGATION_A8_Q13 = base.PROPAGATION_A8_Q13
EXPECTED_CONFIGURATIONS = (
    "FP32",
    "W4_ONLY",
    "A4_ONLY_G8_STATIC",
    "A4_ONLY_G8_DYNAMIC",
    "W4A4_G8_STATIC",
    "W4A4_G8_DYNAMIC",
)
PREDICTION_CONFIGURATIONS = (
    "FP32",
    "A4_ONLY_G8_STATIC",
    "A4_ONLY_G8_DYNAMIC",
    "W4A4_G8_STATIC",
    "W4A4_G8_DYNAMIC",
)


def _config(name, weight_groups, activation_groups, dynamic=False):
    return base._configuration(
        name, weight_groups, activation_groups,
        None if name == "FP32" else PROPAGATION_A8_Q13,
        granularity="hybrid_group_tensor"
        if activation_groups else "tensor",
        group_size=GROUP_SIZE if activation_groups else None,
        dynamic=dynamic)


def build_configurations():
    return (
        _config("FP32", set(), set()),
        _config("W4_ONLY", ORDINARY_GROUPS, set()),
        _config("A4_ONLY_G8_STATIC", set(), ORDINARY_GROUPS),
        _config(
            "A4_ONLY_G8_DYNAMIC", set(), ORDINARY_GROUPS,
            dynamic=True),
        _config(
            "W4A4_G8_STATIC", ORDINARY_GROUPS, ORDINARY_GROUPS),
        _config(
            "W4A4_G8_DYNAMIC", ORDINARY_GROUPS, ORDINARY_GROUPS,
            dynamic=True),
    )


def validate_runtime_contract(calibration_samples, evaluation_indices):
    if int(calibration_samples) != CALIBRATION_SAMPLES:
        raise ValueError("CSPN dynamic Group-A4 requires 128 samples")
    if len(evaluation_indices) != EVALUATION_SAMPLES:
        raise ValueError("CSPN dynamic Group-A4 requires 64 evaluation samples")


def validate_prediction_coverage(model_output, indices):
    root = Path(model_output) / "predictions"
    actual_configs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_configs != set(PREDICTION_CONFIGURATIONS):
        raise ValueError("dynamic Group-A4 prediction directories differ")
    expected_files = {
        "sample_%05d.npz" % int(index) for index in indices
    }
    for config in PREDICTION_CONFIGURATIONS:
        actual_files = {
            path.name for path in (root / config).glob("sample_*.npz")
        }
        if actual_files != expected_files:
            raise ValueError("prediction coverage mismatch: %s" % config)


def calibration_indices(dataset_size, seed):
    if dataset_size < CALIBRATION_SAMPLES:
        raise ValueError("NYU training split has fewer than 128 samples")
    return np.random.RandomState(int(seed)).choice(
        dataset_size, CALIBRATION_SAMPLES, replace=False).tolist()


def _collect(results, configs, field):
    rows = []
    for config in configs:
        rows.extend(results[config["name"]][field])
    return rows


def activation_scale_count(instrumentor, rotation, config):
    if not config["activation_groups"]:
        return 0, 0
    specs = base.build_activation_specs(
        instrumentor, config["activation_groups"],
        int(config["a_bits"]), config["group_size"],
        dynamic=config["dynamic"])
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


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN dynamic Group-A4 evaluation requires CUDA")
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
    calibration_set_indices = calibration_indices(len(trainset), args.seed)
    preparation_sample = seeded_sample(
        trainset, calibration_set_indices[0], args.seed)
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
        quantized_model, saved_args, trainset, calibration_set_indices,
        device, args.seed, instrumentor, rotation, propagation)
    base.validate_strict_site_contract(instrumentor, rotation)

    evalset = evaluation_dataset(saved_args)
    evaluation_indices = load_sample_indices(args.sample_metrics)
    validate_runtime_contract(args.calibration_samples, evaluation_indices)
    if max(evaluation_indices) >= len(evalset):
        raise ValueError("evaluation sample index exceeds NYU validation split")

    configs = build_configurations()
    model_output = Path(args.out_dir) / "cspn"
    model_output.mkdir(parents=True, exist_ok=True)
    results = {}
    overhead_rows = []
    for config in configs:
        prediction_root = model_output \
            if config["name"] in PREDICTION_CONFIGURATIONS else None
        results[config["name"]] = base.run_configuration(
            reference_model, quantized_model, saved_args, evalset,
            evaluation_indices, "evaluation", device, args.seed, config,
            instrumentor, rotation, propagation, merge_adapters,
            reference_capture, quantized_capture, args.sample_capacity,
            prediction_root=prediction_root)
        for source in instrumentor.dynamic_activation_rows():
            row = dict(source)
            row.update({"model": "cspn", "config": config["name"]})
            overhead_rows.append(row)

    sample_rows = _collect(results, configs, "sample_rows")
    base.validate_sample_coverage(
        sample_rows, EXPECTED_CONFIGURATIONS, evaluation_indices)
    validate_prediction_coverage(model_output, evaluation_indices)
    mean_rows = base._mean_metric_rows(sample_rows, configs)
    region_rows = _collect(results, configs, "region_rows")
    regional_rows = []
    for config in configs:
        selected = [
            row for row in region_rows if row["config"] == config["name"]
        ]
        for source in aggregate_region_rows(selected):
            regional_rows.append(dict(
                source, model="cspn", config=config["name"]))

    block_rows = _collect(results, configs, "block_rows")
    tensor_rows = _collect(results, configs, "tensor_rows")
    channel_rows = _collect(results, configs, "channel_rows")
    layer_rows = _collect(results, configs, "layer_rows")
    propagation_rows = _collect(results, configs, "propagation_rows")
    manifest = []
    for config in configs:
        activation_sites, activation_scales = activation_scale_count(
            instrumentor, rotation, config)
        manifest.append({
            "config": config["name"],
            "weight_bits": "" if not config["weight_groups"] else 4,
            "activation_bits": "" if not config["activation_groups"] else 4,
            "group_size": "" if config["group_size"] is None else
            config["group_size"],
            "dynamic": int(config["dynamic"]),
            "activation_sites": activation_sites,
            "activation_scales_per_sample": activation_scales,
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
    write_csv(model_output / "dynamic_overhead.csv", overhead_rows)
    write_json(model_output / "metadata.json", {
        "model": "cspn",
        "architecture": architecture,
        "model_class": type(reference_model).__name__,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_load": reference_load,
        "data_root": str(Path(args.data_root).resolve()),
        "seed": args.seed,
        "calibration_samples": CALIBRATION_SAMPLES,
        "calibration_indices": calibration_set_indices,
        "calibration_source": {
            "type": "random_seed",
            "seed": args.seed,
        },
        "evaluation_samples": EVALUATION_SAMPLES,
        "evaluation_indices": evaluation_indices,
        "prediction_configs": list(PREDICTION_CONFIGURATIONS),
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
