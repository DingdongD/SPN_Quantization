#!/usr/bin/env python3
"""Collect strict W4A4 activation histograms on fixed NYU samples."""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.activation_histograms import (  # noqa: E402
    ActivationHistogramRecorder,
)
from scripts.activation_outlier_analysis import per_update_budget  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    build_model,
    load_run_args,
    prepare_args,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.nyu_quantization_analysis import (  # noqa: E402
    MODULE_GROUP_ORDER,
    classify_module,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    batch_from_sample,
    build_propagation_configurations,
    calibration_dataset,
    configure_runtime_adapter,
    instrumentor_options,
    seeded_sample,
)
from spn_quant.propagation import (  # noqa: E402
    install_propagation_adapter,
    propagation_projection_outputs,
)


PROFILE_CONFIG = "PA_W4A4_PROP_A8"


PER_CHANNEL_ACTIVATION_INPUT_RULES = {
    "cspn": (
        (r"^gud_up_proj_layer2\.conv1_1$", 1),
        (r"^gud_up_proj_layer3\.conv1_1$", 1),
        (r"^gud_up_proj_layer4\.conv1_1$", 1),
    ),
    "dyspn": (
        (r"^base\.dec4\.0$", 1),
        (r"^base\.dec3\.0$", 1),
        (r"^base\.dec2\.0$", 1),
        (r"^base\.gd_dec1_\.0$", 1),
        (r"^base\.gd_dec0_dyspn_\d+_\d+\.0$", 1),
    ),
    "nlspn": (
        (r"^dec4\.0$", 1),
        (r"^dec3\.0$", 1),
        (r"^dec2\.0$", 1),
        (r"^id_dec1\.0$", 1),
        (r"^id_dec0\.0$", 1),
        (r"^gd_dec1\.0$", 1),
        (r"^gd_dec0\.0$", 1),
        (r"^cf_dec1\.0$", 1),
        (r"^cf_dec0\.0$", 1),
    ),
    "completionformer": (
        (r"^backbone\.dec5\.0\.0$", 1),
        (r"^backbone\.dec4\.0\.0$", 1),
        (r"^backbone\.dec3\.0\.0$", 1),
        (r"^backbone\.dec2\.0\.0$", 1),
        (r"^backbone\.dep_dec1\.0$", 1),
        (r"^backbone\.dep_dec0\.0$", 1),
        (r"^backbone\.gd_dec1\.0$", 1),
        (r"^backbone\.gd_dec0\.0$", 1),
        (r"^backbone\.cf_dec1\.0$", 1),
        (r"^backbone\.cf_dec0\.0$", 1),
        (r"^backbone\.former\.block\d+\.\d+\.concat_conv$", 16),
    ),
}


def resolve_per_channel_activation_inputs(model_name, module_names):
    if model_name not in PER_CHANNEL_ACTIVATION_INPUT_RULES:
        raise ValueError("unknown histogram model: %s" % model_name)
    names = sorted(set(module_names))
    resolved = set()
    for pattern, expected_count in \
            PER_CHANNEL_ACTIVATION_INPUT_RULES[model_name]:
        matches = [name for name in names if re.search(pattern, name)]
        if len(matches) != expected_count:
            raise RuntimeError(
                "%s concat boundary matched %d modules, expected %d: %s" %
                (model_name, len(matches), expected_count, matches))
        resolved.update(matches)
    return resolved


def select_w4a4_config(configs):
    matches = [config for config in configs
               if config["name"] == PROFILE_CONFIG]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one %s configuration" % PROFILE_CONFIG)
    config = matches[0]
    if int(config["w_bits"]) != 4:
        raise ValueError("W4A4 configuration has incorrect weight bits")
    if int(config["a_bits"]) != 4:
        raise ValueError("W4A4 configuration has incorrect activation bits")
    if config["propagation"] is None:
        raise ValueError("W4A4 configuration lacks propagation policy")
    return config


def expected_site_names(manifest_rows, site_metadata):
    manifest_sites = {
        (row["module"], row["kind"]) for row in manifest_rows
    }
    observed_sites = {
        (row["module"], row["kind"])
        for row in site_metadata.values()
        if int(row["synthetic_slice"]) == 0
    }
    if manifest_sites != observed_sites:
        missing = sorted(manifest_sites - observed_sites)
        unexpected = sorted(observed_sites - manifest_sites)
        raise ValueError(
            "manifest site mismatch: missing=%s unexpected=%s" %
            (missing, unexpected))
    return {
        name for name, row in site_metadata.items()
        if int(row["synthetic_slice"]) == 0
    }


def profile_indices(calibration_indices, profile_samples):
    count = int(profile_samples)
    if count <= 0 or count > len(calibration_indices):
        raise ValueError("profile sample count is out of range")
    return list(calibration_indices[:count])


def validate_hardware_preparation(preparation, fold_max_error):
    if preparation["primary_max_abs_error"] > float(fold_max_error):
        raise RuntimeError("Conv-BN fold changed FP32 output by %.8f" %
                           preparation["primary_max_abs_error"])


def _ordered_groups(instrumentor):
    groups = set(instrumentor.module_groups().values())
    ordered = [group for group in MODULE_GROUP_ORDER if group in groups]
    ordered.extend(sorted(groups - set(ordered)))
    return ordered


def _run_indices(model, saved_args, dataset, indices, seed, device, label):
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            batch = batch_from_sample(sample)
            model_args, _ = sweep.batch_to_model_input(
                saved_args.model, batch, device)
            model(*model_args)
            if rank % 8 == 0 or rank == len(indices):
                print("%s %d/%d" % (label, rank, len(indices)), flush=True)


def _configure_w4a4(instrumentor, adapter, config, model_name):
    instrumentor.configure(
        config["w_bits"], config["a_bits"], config["groups"],
        **instrumentor_options(config))
    configure_runtime_adapter(
        config, adapter, propagation_backend=True,
        model_name=model_name)


def _write_metadata(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--calibration-samples", required=True, type=int)
    parser.add_argument("--profile-samples", required=True, type=int)
    parser.add_argument("--histogram-bins", type=int, default=128)
    parser.add_argument("--sample-capacity", type=int, default=1000000)
    parser.add_argument("--fold-max-error", required=True, type=float)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    if args.calibration_samples <= 0:
        raise ValueError("calibration sample count must be positive")

    saved_args = prepare_args(load_run_args(run_dir), args)
    saved_args.data_root = args.data_root
    if not saved_args.device.startswith("cuda"):
        raise ValueError("activation histogram profiling requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(saved_args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = saved_args.allow_tf32
    torch.backends.cudnn.allow_tf32 = saved_args.allow_tf32

    model, architecture = build_model(saved_args, checkpoint, device)
    provenance = architecture["model_provenance"]

    dataset = calibration_dataset(saved_args)
    if args.calibration_samples > len(dataset):
        raise ValueError("calibration sample count exceeds dataset size")
    calibration_indices = np.random.RandomState(args.seed).choice(
        len(dataset), args.calibration_samples, replace=False).tolist()
    selected_indices = profile_indices(
        calibration_indices, args.profile_samples)

    preparation_sample = seeded_sample(
        dataset, calibration_indices[0], args.seed)
    preparation_batch = batch_from_sample(preparation_sample)
    preparation_args, _ = sweep.batch_to_model_input(
        saved_args.model, preparation_batch, device)
    excluded_pairs = [("conv1_1", "bn1")] \
        if saved_args.model == "cspn" else []
    preparation = prepare_hardware_model(
        model, preparation_args, excluded_pairs=excluded_pairs, fold=True)
    group_fn = lambda name, module: classify_module(
        saved_args.model, name, module)
    owned_outputs = propagation_projection_outputs(saved_args.model, model)
    quantized_modules = {
        name for name, module in model.named_modules()
        if isinstance(module, (
            torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear))
    }
    per_channel_inputs = resolve_per_channel_activation_inputs(
        saved_args.model, quantized_modules)
    instrumentor = HardwareAlignedInstrumentor(
        model, group_fn, preparation["fused_relu_producers"],
        externally_owned_outputs=owned_outputs,
        per_channel_activation_inputs=per_channel_inputs)
    validate_hardware_preparation(preparation, args.fold_max_error)
    adapter = install_propagation_adapter(saved_args.model, model)

    groups = _ordered_groups(instrumentor)
    configs = build_propagation_configurations(groups)
    config = select_w4a4_config(configs)

    instrumentor.observe()
    adapter.observe()
    started = time.time()
    _run_indices(
        model, saved_args, dataset, calibration_indices,
        args.seed, device, "calibration")
    instrumentor.freeze()
    adapter.freeze()

    per_update = per_update_budget(
        args.sample_capacity, len(selected_indices))
    recorder = ActivationHistogramRecorder(
        saved_args.model, phase="range",
        capacity=args.sample_capacity, per_update=per_update)
    _configure_w4a4(instrumentor, adapter, config, saved_args.model)
    instrumentor.set_activation_recorder(recorder)
    _run_indices(
        model, saved_args, dataset, selected_indices,
        args.seed, device, "range")
    manifest_rows = instrumentor.manifest()
    expected_sites = expected_site_names(
        manifest_rows, recorder.site_metadata)
    recorder.freeze_ranges(args.histogram_bins)

    _configure_w4a4(instrumentor, adapter, config, saved_args.model)
    recorder.begin_histogram_pass()
    _run_indices(
        model, saved_args, dataset, selected_indices,
        args.seed, device, "histogram")
    recorder.validate(expected_sites, len(selected_indices))

    output = Path(args.out_dir) / saved_args.model
    recorder.write(output)
    metadata = {
        "model": saved_args.model,
        "iteration": int(saved_args.iteration),
        "architecture": dict(
            (key, value) for key, value in architecture.items()
            if key != "model_provenance"),
        "checkpoint": str(checkpoint.resolve()),
        "model_provenance": provenance,
        "seed": int(args.seed),
        "calibration_samples": len(calibration_indices),
        "calibration_indices": calibration_indices,
        "profile_samples": len(selected_indices),
        "profile_indices": selected_indices,
        "dataset_root": str(Path(args.data_root).resolve()),
        "ground_truth_quantized": False,
        "training_performed": False,
        "configuration": PROFILE_CONFIG,
        "weight_bits": int(config["w_bits"]),
        "activation_bits": int(config["a_bits"]),
        "activation_contract": "uniform_integer_qdq",
        "groups": sorted(config["groups"]),
        "propagation": config["propagation"],
        "per_channel_activation_modules":
            instrumentor.per_channel_activation_modules(),
        "histogram_bins": int(args.histogram_bins),
        "sample_capacity_per_site": int(args.sample_capacity),
        "samples_per_site_update": int(per_update),
        "manifest_sites": len(expected_sites),
        "synthetic_input_slices": len(recorder.site_names()) -
            len(expected_sites),
        "total_sites": len(recorder.site_names()),
        "elapsed_seconds": time.time() - started,
    }
    _write_metadata(output / "metadata.json", metadata)
    instrumentor.clear_activation_recorder()
    instrumentor.close()
    adapter.close()
    print("saved %d activation sites to %s" %
          (len(recorder.site_names()), output), flush=True)


if __name__ == "__main__":
    main()
