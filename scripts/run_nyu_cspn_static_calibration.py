#!/usr/bin/env python3
"""Compare static Group-A4 calibration methods on official CSPN."""

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

from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    load_run_args,
    prepare_args,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    activation_maximum_for_spec,
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
from spn_quant.static_calibration import (  # noqa: E402
    HistogramSite,
    StaticCalibrationRecorder,
)


CALIBRATION_SAMPLES = 128
EVALUATION_SAMPLES = 64
HISTOGRAM_BINS = 2048
EXPECTED_ACTIVATION_SITES = 71
EXPECTED_ACTIVATION_SCALES = 1673
ORDINARY_GROUPS = base.ORDINARY_GROUPS
PROPAGATION_A8_Q13 = base.PROPAGATION_A8_Q13
EXPECTED_CONFIGURATIONS = (
    "W4A4_G8_MINMAX",
    "W4A4_G8_PERCENTILE_P999",
    "W4A4_G8_PERCENTILE_P9999",
    "W4A4_G8_HIST_MSE",
)


def build_methods():
    return (
        {
            "config": "W4A4_G8_MINMAX",
            "calibration": "minmax",
            "group_size": 8,
            "dynamic": False,
        },
        {
            "config": "W4A4_G8_PERCENTILE_P999",
            "calibration": "percentile_p999",
            "group_size": 8,
            "dynamic": False,
        },
        {
            "config": "W4A4_G8_PERCENTILE_P9999",
            "calibration": "percentile_p9999",
            "group_size": 8,
            "dynamic": False,
        },
        {
            "config": "W4A4_G8_HIST_MSE",
            "calibration": "hist_mse",
            "group_size": 8,
            "dynamic": False,
        },
    )


def validate_activation_contract(site_count, scale_count):
    if int(site_count) != EXPECTED_ACTIVATION_SITES:
        raise ValueError("CSPN static calibration site contract changed")
    if int(scale_count) != EXPECTED_ACTIVATION_SCALES:
        raise ValueError("CSPN static calibration scale contract changed")


def split_thresholds(thresholds, ordinary_keys, rotation_names):
    rotation_owners = {
        ("rotation.%s" % name, "boundary"): name
        for name in rotation_names
    }
    expected = set(ordinary_keys) | set(rotation_owners)
    if set(thresholds) != expected:
        raise ValueError("threshold owners do not match strict CSPN sites")
    ordinary = dict(
        (ordinary_keys[owner], thresholds[owner]) for owner in ordinary_keys)
    rotation = dict(
        (rotation_owners[owner], thresholds[owner])
        for owner in rotation_owners)
    return ordinary, rotation


def _site_maximum(observer, spec):
    extent = observer.maximum if not spec.signed else torch.maximum(
        observer.minimum.abs(), observer.maximum.abs())
    return torch.as_tensor(
        activation_maximum_for_spec(spec, extent),
        dtype=torch.float32).reshape(-1)


def _site_group_size(spec, channels):
    if spec.granularity == "tensor":
        return int(channels)
    if spec.granularity == "channel":
        return 1
    return int(spec.group_size)


def build_histogram_sites(instrumentor, rotation, specs, rotation_specs):
    sites = []
    ordinary_keys = {}
    for key in specs:
        owner = base.activation_owner(key)
        ordinary_keys[owner] = key
        observer = instrumentor.channel_observers[key] \
            if not isinstance(key, str) else \
            instrumentor.relu_channel_observers[key]
        channels = int(observer.minimum.numel())
        group = instrumentor.groups[key[0]] \
            if not isinstance(key, str) else \
            instrumentor._relu_owner(key)[1]
        sites.append(HistogramSite(
            owner[0], owner[1], group, channels,
            int(observer.channel_dim), _site_group_size(specs[key], channels),
            _site_maximum(observer, specs[key]), bool(specs[key].signed)))

    for owner in rotation_specs:
        name = owner[0].split(".", 1)[1]
        spec = rotation_specs[owner]
        observer = rotation.observers[name]["identity"]
        channels = int(rotation.channels[name])
        maximum = activation_maximum_for_spec(spec, observer.channel_absmax)
        sites.append(HistogramSite(
            owner[0], owner[1], "decoder", channels, 1,
            _site_group_size(spec, channels),
            torch.as_tensor(maximum, dtype=torch.float32).reshape(-1), True))
    return tuple(sites), ordinary_keys


def _collect_histograms(model, saved_args, dataset, indices, device, seed,
                        instrumentor, rotation, propagation, recorder,
                        ordinary_owners):
    methods = {
        "decoder_entry": "identity",
        "layer4_signed_skip": "identity",
    }
    instrumentor.set_calibration_recorder(recorder, ordinary_owners)
    rotation.set_calibration_recorder(recorder, methods)
    instrumentor.observe()
    rotation.observe()
    propagation.disable()
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            model(*base._model_args(saved_args, sample, device))
            if rank % 16 == 0 or rank == len(indices):
                print("CSPN histogram calibration %d/%d" %
                      (rank, len(indices)), flush=True)
    instrumentor.freeze()
    rotation.freeze()
    instrumentor.clear_calibration_recorder()
    rotation.clear_calibration_recorder()
    recorder.validate_coverage()


def _configuration(method, thresholds, ordinary_keys, rotation_names):
    ordinary, rotation = split_thresholds(
        thresholds, ordinary_keys, rotation_names)
    config = base._configuration(
        method["config"], ORDINARY_GROUPS, ORDINARY_GROUPS,
        PROPAGATION_A8_Q13, granularity="hybrid_group_tensor",
        group_size=int(method["group_size"]),
        activation_range_overrides=tuple(ordinary.items()),
        rotation_range_overrides=tuple(rotation.items()))
    config["calibration"] = method["calibration"]
    config["dynamic"] = bool(method["dynamic"])
    return config


def _collect(results, configs, field):
    rows = []
    for config in configs:
        rows.extend(results[config["name"]][field])
    return rows


def validate_prediction_coverage(model_output, configs, indices):
    expected_configs = set(configs)
    root = Path(model_output) / "predictions"
    actual_configs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_configs != expected_configs:
        raise ValueError("static calibration prediction directories differ")
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
    parser.add_argument(
        "--histogram-bins", type=int,
        choices=(HISTOGRAM_BINS,), required=True)
    parser.add_argument("--sample-capacity", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN static calibration requires CUDA")
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
    specs = base.build_activation_specs(
        instrumentor, ORDINARY_GROUPS, 4, 8)
    rotation_specs = base.build_rotation_activation_specs(rotation, 4, 8)
    sites, ordinary_keys = build_histogram_sites(
        instrumentor, rotation, specs, rotation_specs)
    scale_count = sum(
        int(torch.as_tensor(site.maximum).numel()) for site in sites)
    validate_activation_contract(len(sites), scale_count)
    recorder = StaticCalibrationRecorder(sites, args.histogram_bins)
    _collect_histograms(
        quantized_model, saved_args, trainset, calibration_indices,
        device, args.seed, instrumentor, rotation, propagation, recorder,
        set(ordinary_keys))

    methods = build_methods()
    thresholds_by_method = {}
    threshold_site_rows = []
    threshold_scale_rows = []
    for method in methods:
        calibration = method["calibration"]
        thresholds = recorder.thresholds(
            calibration, bits=4, device=device)
        thresholds_by_method[calibration] = thresholds
        rows = recorder.rows(calibration, thresholds)
        for row in rows:
            row["config"] = method["config"]
        threshold_site_rows.extend(rows)
        rows = recorder.scale_rows(calibration, thresholds)
        for row in rows:
            row["config"] = method["config"]
        threshold_scale_rows.extend(rows)
    configs = tuple(
        _configuration(
            method, thresholds_by_method[method["calibration"]],
            ordinary_keys, tuple(rotation.channels))
        for method in methods)

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
        evaluation_results[config["name"]] = base.run_configuration(
            reference_model, quantized_model, saved_args, evalset,
            evaluation_indices, "evaluation", device, args.seed, config,
            instrumentor, rotation, propagation, merge_adapters,
            reference_capture, quantized_capture, args.sample_capacity)
    sample_rows = _collect(evaluation_results, configs, "sample_rows")
    base.validate_sample_coverage(
        sample_rows, EXPECTED_CONFIGURATIONS, evaluation_indices)
    mean_rows = base._mean_metric_rows(sample_rows, configs)
    mean_by_name = dict((row["config"], row) for row in mean_rows)
    non_minmax = tuple(
        config for config in configs if config["calibration"] != "minmax")
    selected = min(
        non_minmax,
        key=lambda config: (
            float(mean_by_name[config["name"]]["RMSE"]),
            config["name"]))
    prediction_configs = (EXPECTED_CONFIGURATIONS[0], selected["name"])
    config_by_name = dict((config["name"], config) for config in configs)
    for name in prediction_configs:
        base.run_configuration(
            reference_model, quantized_model, saved_args, evalset,
            evaluation_indices, "evaluation", device, args.seed,
            config_by_name[name], instrumentor, rotation, propagation,
            merge_adapters, reference_capture, quantized_capture,
            args.sample_capacity, prediction_root=model_output)
    validate_prediction_coverage(
        model_output, prediction_configs, evaluation_indices)

    region_rows = _collect(evaluation_results, configs, "region_rows")
    regional_rows = []
    for config in configs:
        selected_regions = [
            row for row in region_rows if row["config"] == config["name"]
        ]
        for source in aggregate_region_rows(selected_regions):
            regional_rows.append(dict(
                source, model="cspn", config=config["name"]))
    block_rows = _collect(evaluation_results, configs, "block_rows")
    tensor_rows = _collect(evaluation_results, configs, "tensor_rows")
    channel_rows = _collect(evaluation_results, configs, "channel_rows")
    layer_rows = _collect(evaluation_results, configs, "layer_rows")
    propagation_rows = _collect(
        evaluation_results, configs, "propagation_rows")
    manifests = []
    for config in configs:
        calibration = config["calibration"]
        selected_rows = [
            row for row in threshold_scale_rows
            if row["config"] == config["name"]
        ]
        manifests.append({
            "config": config["name"],
            "calibration": calibration,
            "weight_bits": 4,
            "activation_bits": 4,
            "activation_granularity": "hybrid_group_tensor",
            "group_size": 8,
            "activation_sites": len(sites),
            "activation_scales": len(selected_rows),
            "histogram_bins": args.histogram_bins,
            "dynamic": 0,
            "threshold_ratio_min": min(
                float(row["threshold_ratio"]) for row in selected_rows),
            "threshold_ratio_mean": float(np.mean(np.asarray(
                [float(row["threshold_ratio"]) for row in selected_rows],
                dtype=np.float64))),
            "guidance_head": "fp32",
            "propagation": "a8_int16_q13_int32",
        })

    write_csv(model_output / "config_manifest.csv", manifests)
    write_csv(model_output / "sample_metrics.csv", sample_rows,
              base.SAMPLE_FIELDS)
    write_csv(model_output / "aggregate_metrics.csv", mean_rows)
    write_csv(model_output / "regional_metrics.csv", regional_rows)
    write_csv(model_output / "block_metrics.csv", block_rows)
    write_csv(model_output / "activation_resolution_metrics.csv", tensor_rows)
    write_csv(model_output / "activation_channel_metrics.csv", channel_rows)
    write_csv(model_output / "layer_quantization_metrics.csv", layer_rows)
    write_csv(model_output / "propagation_metrics.csv", propagation_rows)
    write_csv(analysis_output / "threshold_site_summary.csv",
              threshold_site_rows)
    write_csv(analysis_output / "threshold_scale_manifest.csv",
              threshold_scale_rows)
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
        "calibration_passes": 2,
        "histogram_bins": args.histogram_bins,
        "evaluation_samples": EVALUATION_SAMPLES,
        "evaluation_indices": evaluation_indices,
        "activation_sites": len(sites),
        "activation_scales": scale_count,
        "selected_non_minmax_prediction": selected["name"],
        "prediction_configs": list(prediction_configs),
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
