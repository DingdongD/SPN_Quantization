#!/usr/bin/env python3
"""Evaluate contiguous Group-8 outlier channel isolation on official CSPN."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from spn_quant.outlier_channel_isolation import (  # noqa: E402
    OutlierHarmAccumulator,
    build_outlier_candidates,
)
from spn_quant.propagation import install_propagation_adapter  # noqa: E402
from spn_quant.rotation import CSPNRotationController  # noqa: E402


CALIBRATION_SAMPLES = 128
EVALUATION_SAMPLES = 64
GROUP_SIZE = 8
PREFIX_LIMITS = (1, 2, 4, 8, 16, 32, 64)


def rank_candidates(rows):
    positive = [
        row for row in rows
        if float(row["rescued_energy"]) > 0.0 and
        int(row["rescued_elements"]) > 0
    ]
    return tuple(sorted(
        positive,
        key=lambda row: (
            -float(row["rescued_energy"]),
            -int(row["rescued_elements"]),
            str(row["module"]),
            str(row["kind"]),
            int(row["group_index"]),
            int(row["outlier_channel"]),
        )))


def isolation_mapping(selected):
    by_site = defaultdict(list)
    for row in selected:
        key = str(row["module"]) if row["kind"] == "relu_output" else \
            (str(row["module"]), str(row["kind"]))
        by_site[key].append(int(row["outlier_channel"]))
    return tuple(
        (key, tuple(sorted(by_site[key])))
        for key in sorted(by_site, key=str))


def build_budgets(rows, prefix_limits):
    ranked = rank_candidates(rows)
    lengths = [0]
    for limit in prefix_limits:
        current = min(int(limit), len(ranked))
        if current not in lengths:
            lengths.append(current)
    if len(ranked) not in lengths:
        lengths.append(len(ranked))
    budgets = []
    for length in lengths:
        name = "W4A4_G8_CONTIGUOUS" if length == 0 else \
            "W4A4_G8_OCI_ALL" if length == len(ranked) else \
            "W4A4_G8_OCI_%d" % length
        budgets.append({
            "name": name,
            "selected": ranked[:length],
            "activation_isolations": isolation_mapping(ranked[:length]),
        })
    return tuple(budgets)


def build_site_candidates(instrumentor, specs):
    sites = {}
    for key in sorted(specs, key=str):
        spec = specs[key]
        if spec.granularity != "group" or int(spec.group_size) != GROUP_SIZE:
            continue
        if spec.signed:
            continue
        observer = instrumentor.relu_channel_observers[key] \
            if isinstance(key, str) else instrumentor.channel_observers[key]
        sites[key] = build_outlier_candidates(
            observer.maximum, GROUP_SIZE)
    if not sites:
        raise RuntimeError("official CSPN has no eligible OCI activation sites")
    return sites


def _site_identity(instrumentor, key):
    if isinstance(key, str):
        return key, "relu_output", instrumentor._relu_owner(key)[1]
    name, kind = key
    return name, kind, instrumentor.groups[name]


def collect_harm(
        model, saved_args, dataset, indices, device, seed,
        instrumentor, site_candidates):
    accumulators = {}
    modules = dict(model.named_modules())
    handles = []
    for key in sorted(site_candidates, key=str):
        observer = instrumentor.relu_channel_observers[key] \
            if isinstance(key, str) else instrumentor.channel_observers[key]
        accumulator = OutlierHarmAccumulator(
            site_candidates[key], observer.channel_dim)
        accumulators[key] = accumulator

        if isinstance(key, str):
            continue
        name, kind = key
        if kind == "input":
            def input_hook(module, inputs, current=accumulator):
                del module
                current.update(inputs[0])
                return None

            handles.append(modules[name].register_forward_pre_hook(
                input_hook, prepend=True))
        elif kind == "output":
            def output_hook(module, inputs, output, current=accumulator):
                del module, inputs
                current.update(output)
                return None

            handles.append(modules[name].register_forward_hook(
                output_hook, prepend=True))
        else:
            raise ValueError("unsupported activation site kind: %s" % kind)

    relu_keys = set(
        key for key in site_candidates if isinstance(key, str))
    for module, name in instrumentor.relu_names.items():
        if not any(key.rpartition("#")[0] == name for key in relu_keys):
            continue

        def relu_hook(current_module, inputs, output,
                      current_name=name):
            del inputs
            index = instrumentor.relu_call_counts[current_name] \
                if current_name in instrumentor.relu_call_counts else 0
            key = "%s#%d" % (current_name, index)
            if key in accumulators:
                accumulators[key].update(output)
            return None

        handles.append(module.register_forward_hook(relu_hook, prepend=True))
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            model(*base._model_args(saved_args, sample, device))
            if rank % 16 == 0 or rank == len(indices):
                print("CSPN OCI harm calibration %d/%d" %
                      (rank, len(indices)), flush=True)
    for handle in handles:
        handle.remove()

    candidate_rows = []
    victim_rows = []
    for key in sorted(accumulators, key=str):
        name, kind, group = _site_identity(instrumentor, key)
        candidates, victims = accumulators[key].rows(
            name, kind, group)
        candidate_rows.extend(candidates)
        victim_rows.extend(victims)
    return candidate_rows, victim_rows


def build_configurations(budgets):
    configurations = []
    for budget in budgets:
        config = base._configuration(
            budget["name"], base.ORDINARY_GROUPS, base.ORDINARY_GROUPS,
            base.PROPAGATION_A8_Q13,
            granularity="hybrid_group_tensor", group_size=GROUP_SIZE,
            activation_isolations=budget["activation_isolations"])
        selected = budget["selected"]
        config["isolated_channels"] = len(selected)
        config["affected_sites"] = len(set(
            (str(row["module"]), str(row["kind"])) for row in selected))
        config["calibration_rescued_elements"] = sum(
            int(row["rescued_elements"]) for row in selected)
        config["calibration_rescued_energy"] = sum(
            float(row["rescued_energy"]) for row in selected)
        configurations.append(config)
    return tuple(configurations)


def _collect(results, configurations, field):
    rows = []
    for config in configurations:
        rows.extend(results[config["name"]][field])
    return rows


def validate_runtime_contract(calibration_indices, evaluation_indices):
    if len(calibration_indices) != CALIBRATION_SAMPLES:
        raise ValueError("CSPN OCI calibration requires 128 samples")
    if len(evaluation_indices) != EVALUATION_SAMPLES:
        raise ValueError("CSPN OCI evaluation requires 64 samples")


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
    parser.add_argument("--sample-capacity", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN OCI evaluation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.sample_capacity <= 0:
        raise ValueError("sample capacity must be positive")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    started = time.time()

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
    evalset = evaluation_dataset(saved_args)
    evaluation_indices = load_sample_indices(args.sample_metrics)
    validate_runtime_contract(calibration_indices, evaluation_indices)
    if max(evaluation_indices) >= len(evalset):
        raise ValueError("evaluation sample index exceeds NYU validation split")

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

    base._calibrate(
        quantized_model, saved_args, trainset, calibration_indices,
        device, args.seed, instrumentor, rotation, propagation)
    base.validate_strict_site_contract(instrumentor, rotation)
    specs = base.build_activation_specs(
        instrumentor, base.ORDINARY_GROUPS, 4, GROUP_SIZE)
    site_candidates = build_site_candidates(instrumentor, specs)
    harm_configuration = base._configuration(
        "W4A4_G8_CONTIGUOUS_HARM_CALIBRATION",
        base.ORDINARY_GROUPS, base.ORDINARY_GROUPS,
        base.PROPAGATION_A8_Q13,
        granularity="hybrid_group_tensor", group_size=GROUP_SIZE,
        activation_isolations=())
    base._configure_quantized(
        harm_configuration, instrumentor, rotation, propagation, {})
    candidate_rows, victim_rows = collect_harm(
        quantized_model, saved_args, trainset, calibration_indices,
        device, args.seed, instrumentor, site_candidates)
    ranked = rank_candidates(candidate_rows)
    rank_by_key = dict(
        ((str(row["module"]), str(row["kind"]),
          int(row["group_index"]),
          int(row["outlier_channel"])), rank)
        for rank, row in enumerate(ranked, 1))
    for row in candidate_rows:
        key = (str(row["module"]), str(row["kind"]),
               int(row["group_index"]),
               int(row["outlier_channel"]))
        row["selection_rank"] = rank_by_key[key] \
            if key in rank_by_key else ""

    budgets = build_budgets(candidate_rows, PREFIX_LIMITS)
    configurations = build_configurations(budgets)
    model_output = Path(args.out_dir) / "cspn"
    analysis_output = Path(args.out_dir) / "analysis"
    model_output.mkdir(parents=True, exist_ok=True)
    analysis_output.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_output / "outlier_harm_candidates.csv", candidate_rows)
    write_csv(analysis_output / "outlier_harm_victims.csv", victim_rows)

    reference_capture = base.ModuleOutputCapture(
        reference_model, base.CSPN_BLOCK_SITES)
    quantized_capture = base.ModuleOutputCapture(
        quantized_model, base.CSPN_BLOCK_SITES)
    evaluation_results = {}
    for config in configurations:
        evaluation_results[config["name"]] = base.run_configuration(
            reference_model, quantized_model, saved_args, evalset,
            evaluation_indices, "evaluation", device, args.seed, config,
            instrumentor, rotation, propagation, {}, reference_capture,
            quantized_capture, args.sample_capacity)

    sample_rows = _collect(
        evaluation_results, configurations, "sample_rows")
    base.validate_sample_coverage(
        sample_rows,
        tuple(config["name"] for config in configurations),
        evaluation_indices)
    mean_rows = base._mean_metric_rows(sample_rows, configurations)
    region_rows = _collect(
        evaluation_results, configurations, "region_rows")
    regional_rows = []
    for config in configurations:
        selected = [
            row for row in region_rows
            if row["config"] == config["name"]
        ]
        for source in aggregate_region_rows(selected):
            regional_rows.append(dict(
                source, model="cspn", config=config["name"]))
    block_rows = _collect(
        evaluation_results, configurations, "block_rows")
    tensor_rows = _collect(
        evaluation_results, configurations, "tensor_rows")
    channel_rows = _collect(
        evaluation_results, configurations, "channel_rows")
    layer_rows = _collect(
        evaluation_results, configurations, "layer_rows")
    propagation_rows = _collect(
        evaluation_results, configurations, "propagation_rows")

    manifest_rows = []
    for config in configurations:
        manifest_rows.append({
            "config": config["name"],
            "weight_bits": 4,
            "activation_bits": 4,
            "group_size": GROUP_SIZE,
            "isolated_channels": config["isolated_channels"],
            "isolated_channel_fraction": config["isolated_channels"] /
            float(sum(len(candidates) * GROUP_SIZE
                      for candidates in site_candidates.values())),
            "additional_scales": config["isolated_channels"],
            "affected_sites": config["affected_sites"],
            "calibration_rescued_elements":
                config["calibration_rescued_elements"],
            "calibration_rescued_energy":
                config["calibration_rescued_energy"],
            "selection": "calibration_rescued_energy",
            "guidance_head": "fp32",
            "bias_format": "fp32",
            "propagation": "a8_int16_q13_int32",
        })

    write_csv(model_output / "config_manifest.csv", manifest_rows)
    write_csv(model_output / "sample_metrics.csv", sample_rows,
              base.SAMPLE_FIELDS)
    write_csv(model_output / "aggregate_metrics.csv", mean_rows)
    write_csv(model_output / "regional_metrics.csv", regional_rows)
    write_csv(model_output / "block_metrics.csv", block_rows)
    write_csv(model_output / "activation_resolution_metrics.csv", tensor_rows)
    write_csv(model_output / "activation_channel_metrics.csv", channel_rows)
    write_csv(model_output / "layer_quantization_metrics.csv", layer_rows)
    write_csv(model_output / "propagation_metrics.csv", propagation_rows)
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
        "evaluation_samples": EVALUATION_SAMPLES,
        "evaluation_indices": evaluation_indices,
        "group_size": GROUP_SIZE,
        "eligible_activation_sites": len(site_candidates),
        "candidate_groups": len(candidate_rows),
        "positive_harm_candidates": len(ranked),
        "prefix_limits": list(PREFIX_LIMITS),
        "configurations": [
            config["name"] for config in configurations],
        "method": "outlier_channel_isolation",
        "channel_splitting": False,
        "channel_permutation": False,
        "isolation_format": "independent_outlier_A4_shared_victim_A4",
        "guidance_head": "fp32",
        "bias_format": "fp32",
        "propagation": dict(base.PROPAGATION_A8_Q13),
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
