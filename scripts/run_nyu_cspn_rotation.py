#!/usr/bin/env python3
"""Evaluate selective CSPN channel rotation under W4A4 quantization."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_quantization_analysis import (  # noqa: E402
    classify_module,
    regional_depth_metrics,
    tensor_metrics,
)
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    load_model_state,
    load_run_args,
    prepare_args,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    aggregate_region_rows,
    batch_from_sample,
    calibration_dataset,
    evaluation_dataset,
    load_sample_indices,
    prediction_payload,
    prepare_prediction_dir,
    seeded_sample,
    write_csv,
    write_json,
    write_prediction_payload,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.propagation import (  # noqa: E402
    PropagationQuantConfig,
    install_propagation_adapter,
)
from spn_quant.rotation import CSPNRotationController  # noqa: E402


PROPAGATION_A8_Q13 = {
    "affinity_bits": 8,
    "confidence_bits": 8,
    "offset_bits": 8,
    "state_bits": 8,
    "coefficient_fraction_bits": 13,
}

END_TO_END_FIELDS = (
    "model", "config", "sample_index", "RMSE", "MAE", "ABS_REL",
    "IRMSE", "flat_RMSE", "boundary_RMSE", "nonfinite_ratio",
)

BOUNDARY_FIELDS = (
    "model", "config", "boundary", "method", "bits", "group_size",
    "minimum", "maximum", "p75", "p99", "p99_9", "p99_99",
    "kurtosis", "channel_imbalance", "sqnr", "zero_code_ratio",
    "saturation_ratio", "block_output_mse", "block_output_sqnr",
)


def _quantized_configuration(
        name: str, decoder_entry: str, layer4_signed_skip: str,
        group_size: Optional[int]) -> Dict[str, object]:
    return {
        "name": name,
        "w_bits": 4,
        "a_bits": 4,
        "enabled_groups": {"encoder", "decoder", "depth_head"},
        "rotation_methods": {
            "decoder_entry": decoder_entry,
            "layer4_signed_skip": layer4_signed_skip,
        },
        "group_size": group_size,
        "quantize_bias": False,
        "propagation": dict(PROPAGATION_A8_Q13),
    }


def build_configurations(group_size: int) -> Sequence[Dict[str, object]]:
    group_size = int(group_size)
    return (
        {
            "name": "FP32",
            "w_bits": None,
            "a_bits": None,
            "enabled_groups": set(),
            "rotation_methods": {
                "decoder_entry": "identity",
                "layer4_signed_skip": "identity",
            },
            "group_size": None,
            "quantize_bias": False,
            "propagation": None,
        },
        _quantized_configuration(
            "RTN_W4A4", "identity", "identity", None),
        _quantized_configuration(
            "GROUP_W4A4", "identity", "identity", group_size),
        _quantized_configuration(
            "RANDOM_decoder_entry", "random", "identity", None),
        _quantized_configuration(
            "RANDOM_layer4_signed_skip", "identity", "random", None),
        _quantized_configuration(
            "RANDOM_both", "random", "random", None),
        _quantized_configuration(
            "HADAMARD_decoder_entry", "hadamard", "identity", None),
        _quantized_configuration(
            "HADAMARD_layer4_signed_skip", "identity", "hadamard", None),
        _quantized_configuration(
            "HADAMARD_both", "hadamard", "hadamard", None),
        _quantized_configuration(
            "HADAMARD_GROUP_both", "hadamard", "hadamard", group_size),
    )


def cspn_quant_group(name: str, module: nn.Module) -> Optional[str]:
    if name.startswith("gud_up_proj_layer6"):
        return None
    return classify_module("cspn", name, module)


def rotation_owned_inputs():
    return {
        "gud_up_proj_layer1.conv1",
        "gud_up_proj_layer1.sc_conv1",
        "gud_up_proj_layer4.conv1_1",
    }


def rotation_owned_outputs():
    return {"conv1_1", "conv2", "gud_up_proj_layer5.conv1"}


def validate_fp_equivalence(reference: torch.Tensor,
                            candidate: torch.Tensor, site: str) -> None:
    if reference.shape != candidate.shape or not torch.allclose(
            reference, candidate, rtol=1e-4, atol=1e-5):
        maximum = float((candidate - reference).abs().max().item()) \
            if reference.shape == candidate.shape else float("inf")
        raise RuntimeError(
            "FP equivalence failed at %s: max_abs_error=%.8f" %
            (site, maximum))


def depth_sample_metrics(gt, pred, sparse):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    sparse = np.asarray(sparse)
    regions = regional_depth_metrics(gt, pred, sparse)
    by_region = dict((row["region"], row) for row in regions)
    valid = np.isfinite(gt) & (gt > 1e-4)
    inverse_error = (
        1.0 / np.maximum(pred[valid], 1e-6)
        - 1.0 / np.maximum(gt[valid], 1e-6))
    nonfinite = valid & ~np.isfinite(pred)
    return {
        "RMSE": by_region["all"]["RMSE"],
        "MAE": by_region["all"]["MAE"],
        "ABS_REL": by_region["all"]["ABS_REL"],
        "IRMSE": float(np.sqrt(np.mean(inverse_error ** 2))),
        "flat_RMSE": by_region["smooth"]["RMSE"],
        "boundary_RMSE": by_region["boundary"]["RMSE"],
        "nonfinite_ratio": float(np.count_nonzero(nonfinite)) /
        float(np.count_nonzero(valid)),
    }, regions


class BlockOutputCapture:
    def __init__(self, model: nn.Module, boundaries) -> None:
        modules = dict(model.named_modules())
        self.current = {}
        self.handles = []
        for boundary in boundaries:
            self.handles.append(modules[boundary.module].register_forward_hook(
                self._make_hook(boundary.name)))

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            del module, inputs
            self.current[name] = output.detach().cpu().clone()
        return hook

    def reset(self) -> None:
        self.current = {}

    def values(self):
        return dict(self.current)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class BlockErrorAccumulator:
    def __init__(self, boundary_names) -> None:
        self.totals = dict((name, {
            "signal": 0.0, "error": 0.0, "numel": 0,
        }) for name in boundary_names)

    def update(self, reference, candidate) -> None:
        for name in self.totals:
            ref = reference[name].to(torch.float64)
            current = candidate[name].to(torch.float64)
            difference = current - ref
            self.totals[name]["signal"] += float(ref.square().sum().item())
            self.totals[name]["error"] += float(
                difference.square().sum().item())
            self.totals[name]["numel"] += int(ref.numel())

    def rows(self):
        output = {}
        for name in self.totals:
            total = self.totals[name]
            error = total["error"]
            signal = total["signal"]
            output[name] = {
                "block_output_mse": error / float(total["numel"]),
                "block_output_sqnr": float("inf") if error == 0.0 else
                10.0 * math.log10(signal / error),
            }
        return output

    def aggregate(self):
        signal = sum(row["signal"] for row in self.totals.values())
        error = sum(row["error"] for row in self.totals.values())
        numel = sum(row["numel"] for row in self.totals.values())
        return {
            "block_output_mse": error / float(numel),
            "block_output_sqnr": float("inf") if error == 0.0 else
            10.0 * math.log10(signal / error),
        }


def select_group_size(rows) -> int:
    if not rows:
        raise ValueError("group-size search rows must not be empty")
    selected = min(
        rows,
        key=lambda row: (
            float(row["block_output_mse"]),
            -float(row["block_output_sqnr"]),
            -int(row["group_size"]),
        ))
    return int(selected["group_size"])


def _load_cspn(saved_args, checkpoint: Path, device: torch.device):
    if saved_args.model != "cspn":
        raise ValueError("CSPN rotation runner requires model=cspn")
    model, architecture = sweep.BUILDERS["cspn"](saved_args, device)
    state = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = state["net"] \
        if isinstance(state, dict) and "net" in state else state
    load_report = load_model_state(model, state_dict, "cspn")
    model.eval()
    return model, architecture, load_report


def _model_input(saved_args, sample, device):
    batch = batch_from_sample(sample)
    model_args, _ = sweep.batch_to_model_input(
        "cspn", batch, device)
    return model_args


def _forward(model, saved_args, sample, device, block_capture):
    block_capture.reset()
    output = model(*_model_input(saved_args, sample, device))
    prediction = sweep.extract_pred(output).detach().cpu()[0, 0]
    return prediction, block_capture.values()


def _calibrate(model, saved_args, dataset, indices, device, seed,
               instrumentor, rotation, propagation, block_capture):
    instrumentor.observe()
    rotation.observe()
    propagation.observe()
    reference_blocks = []
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            _, blocks = _forward(
                model, saved_args, sample, device, block_capture)
            reference_blocks.append({
                "sample_index": int(index),
                "blocks": blocks,
            })
            if rank % 16 == 0 or rank == len(indices):
                print("Rotation calibration %d/%d" %
                      (rank, len(indices)), flush=True)
    instrumentor.freeze()
    rotation.freeze()
    propagation.freeze()
    return reference_blocks


def _search_group_sizes(
        model, saved_args, dataset, reference_blocks, device, seed,
        group_sizes, instrumentor, rotation, propagation, block_capture):
    rows = []
    boundary_names = [boundary.name for boundary in rotation.boundaries]
    for group_size in group_sizes:
        config = _quantized_configuration(
            "GROUP_SEARCH_%d" % int(group_size),
            "identity", "identity", int(group_size))
        _configure(config, instrumentor, rotation, propagation)
        accumulator = BlockErrorAccumulator(boundary_names)
        with torch.no_grad():
            for rank, reference in enumerate(reference_blocks, 1):
                sample = seeded_sample(
                    dataset, reference["sample_index"], seed)
                _, candidate = _forward(
                    model, saved_args, sample, device, block_capture)
                accumulator.update(reference["blocks"], candidate)
                if rank % 16 == 0 or rank == len(reference_blocks):
                    print("Group-A4 search g=%d %d/%d" % (
                        group_size, rank, len(reference_blocks)), flush=True)
        row = accumulator.aggregate()
        row["group_size"] = int(group_size)
        rows.append(row)
    selected = select_group_size(rows)
    for row in rows:
        row["selected"] = int(row["group_size"] == selected)
    rotation.disable()
    instrumentor.disable()
    propagation.disable()
    return selected, rows


def _validate_all_fp_equivalence(
        model, saved_args, sample, device, configurations,
        rotation, propagation, block_capture) -> None:
    rotation.disable()
    propagation.disable()
    with torch.no_grad():
        reference, reference_blocks = _forward(
            model, saved_args, sample, device, block_capture)
        observed = set()
        for config in configurations[1:]:
            methods = config["rotation_methods"]
            identity = (
                methods["decoder_entry"],
                methods["layer4_signed_skip"],
            )
            if identity in observed:
                continue
            observed.add(identity)
            rotation.configure(
                methods, bits=4, group_size=None, quantize=False)
            candidate, candidate_blocks = _forward(
                model, saved_args, sample, device, block_capture)
            for name in reference_blocks:
                validate_fp_equivalence(
                    reference_blocks[name], candidate_blocks[name], name)
            validate_fp_equivalence(
                reference, candidate, "model_output:%s" % config["name"])
            rotation.disable()


def _configure(config, instrumentor, rotation, propagation) -> None:
    rotation.disable()
    if config["name"] == "FP32":
        instrumentor.disable()
        propagation.disable()
        return
    instrumentor.configure(
        config["w_bits"], config["a_bits"], config["enabled_groups"],
        quantize_bias=config["quantize_bias"])
    rotation.configure(
        config["rotation_methods"], bits=config["a_bits"],
        group_size=config["group_size"], quantize=True)
    propagation.configure(PropagationQuantConfig(**config["propagation"]))


def _capture_fp_records(model, saved_args, dataset, indices, device, seed,
                        instrumentor, rotation, propagation, block_capture):
    config = build_configurations(32)[0]
    _configure(config, instrumentor, rotation, propagation)
    records = []
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            prediction, blocks = _forward(
                model, saved_args, sample, device, block_capture)
            records.append({
                "sample_index": int(index),
                "sample": sample,
                "gt": sample["depth"][0].clone(),
                "sparse": sample["rgbd"][3].clone(),
                "rgb": sample["rgbd"][:3].permute(1, 2, 0).clone(),
                "prediction": prediction,
                "blocks": blocks,
            })
            print("FP32 %d/%d sample=%05d" %
                  (rank, len(indices), index), flush=True)
    return records


def _evaluate_configuration(
        model, saved_args, records, device, config, instrumentor,
        rotation, propagation, block_capture, output):
    _configure(config, instrumentor, rotation, propagation)
    prediction_dir = prepare_prediction_dir(
        output, config["name"])
    sample_rows = []
    region_rows = []
    propagation_rows = []
    block_error = BlockErrorAccumulator(
        boundary.name for boundary in rotation.boundaries)
    with torch.no_grad():
        for rank, record in enumerate(records, 1):
            prediction, blocks = _forward(
                model, saved_args, record["sample"], device, block_capture)
            pred = prediction.numpy()
            if not np.isfinite(pred).all():
                raise RuntimeError(
                    "non-finite prediction: config=%s sample=%d" %
                    (config["name"], record["sample_index"]))
            gt = record["gt"].numpy()
            sparse = record["sparse"].numpy()
            metrics, regions = depth_sample_metrics(gt, pred, sparse)
            metrics.update({
                "model": "cspn",
                "config": config["name"],
                "sample_index": record["sample_index"],
            })
            sample_rows.append(metrics)
            for row in regions:
                row.update({
                    "model": "cspn",
                    "config": config["name"],
                    "sample_index": record["sample_index"],
                })
                region_rows.append(row)
            block_error.update(record["blocks"], blocks)
            if config["propagation"] is not None:
                for row in propagation.statistics():
                    row.update({
                        "model": "cspn",
                        "config": config["name"],
                        "sample_index": record["sample_index"],
                    })
                    propagation_rows.append(row)
            payload = prediction_payload(
                gt, record["prediction"].numpy(), pred,
                record["sample_index"], "cspn", config["name"],
                sparse=sparse, rgb=record["rgb"].numpy())
            write_prediction_payload(prediction_dir, payload)
            print("%s %d/%d sample=%05d RMSE=%.5f" % (
                config["name"], rank, len(records),
                record["sample_index"], metrics["RMSE"]), flush=True)

    boundary_rows = []
    if config["name"] != "FP32":
        block_rows = block_error.rows()
        for row in rotation.statistics(
                config["rotation_methods"], config["a_bits"],
                config["group_size"]):
            row.update(block_rows[row["boundary"]])
            row.update({"model": "cspn", "config": config["name"]})
            boundary_rows.append(row)
    layer_rows = instrumentor.statistics() \
        if config["name"] != "FP32" else []
    for row in layer_rows:
        row.update({"model": "cspn", "config": config["name"]})
    return sample_rows, region_rows, boundary_rows, layer_rows, propagation_rows


def _config_manifest(configurations):
    rows = []
    for config in configurations:
        rows.append({
            "config": config["name"],
            "weight_bits": "" if config["w_bits"] is None else
            config["w_bits"],
            "activation_bits": "" if config["a_bits"] is None else
            config["a_bits"],
            "decoder_entry": config["rotation_methods"]["decoder_entry"],
            "layer4_signed_skip":
            config["rotation_methods"]["layer4_signed_skip"],
            "group_size": "" if config["group_size"] is None else
            config["group_size"],
            "bias_format": "fp32",
            "guidance_head": "fp32",
            "affinity_bits": "" if config["propagation"] is None else
            config["propagation"]["affinity_bits"],
            "state_bits": "" if config["propagation"] is None else
            config["propagation"]["state_bits"],
            "coefficient_format": "" if config["propagation"] is None else
            "signed_int16_q13",
        })
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--sample-metrics", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--out-dir",
        default="profile_logs/nyu_cspn_rotation_w4a4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--group-sizes", type=int, nargs="+",
                        choices=(16, 32, 64), default=(16, 32, 64))
    parser.add_argument("--fold-max-error", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CSPN rotation evaluation")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    saved_args = prepare_args(load_run_args(run_dir), args)
    saved_args.data_root = args.data_root
    model, architecture, load_report = _load_cspn(
        saved_args, checkpoint, device)

    trainset = calibration_dataset(saved_args)
    if len(trainset) < args.calibration_samples:
        raise ValueError("NYU training split has fewer than 128 samples")
    calibration_indices = np.random.RandomState(args.seed).choice(
        len(trainset), args.calibration_samples, replace=False).tolist()
    preparation_sample = seeded_sample(
        trainset, calibration_indices[0], args.seed)
    preparation_args = _model_input(
        saved_args, preparation_sample, device)
    hardware_preparation = prepare_hardware_model(
        model, preparation_args, excluded_pairs=(("conv1_1", "bn1"),))
    fold_error = float(hardware_preparation["primary_max_abs_error"])
    if fold_error > args.fold_max_error:
        raise RuntimeError(
            "Conv-BN fold changed FP32 output by %.8f" % fold_error)

    semantic = install_model_semantic_adapter(
        model, "cspn", strict=True)
    boundaries = semantic.rotation_boundaries()
    semantic.close()
    instrumentor = HardwareAlignedInstrumentor(
        model, cspn_quant_group,
        hardware_preparation["fused_relu_producers"],
        externally_owned_outputs=rotation_owned_outputs(),
        externally_owned_inputs=rotation_owned_inputs())
    rotation = CSPNRotationController(model, boundaries, seed=args.seed)
    propagation = install_propagation_adapter("cspn", model)
    block_capture = BlockOutputCapture(model, boundaries)

    started = time.time()
    calibration_reference_blocks = _calibrate(
        model, saved_args, trainset, calibration_indices,
        device, args.seed, instrumentor, rotation, propagation,
        block_capture)
    selected_group_size, group_search_rows = _search_group_sizes(
        model, saved_args, trainset, calibration_reference_blocks,
        device, args.seed, args.group_sizes, instrumentor,
        rotation, propagation, block_capture)
    del calibration_reference_blocks
    configurations = build_configurations(selected_group_size)
    _validate_all_fp_equivalence(
        model, saved_args, preparation_sample, device,
        configurations, rotation, propagation, block_capture)

    evalset = evaluation_dataset(saved_args)
    evaluation_indices = load_sample_indices(args.sample_metrics)
    if len(evaluation_indices) != 64:
        raise ValueError(
            "CSPN rotation evaluation requires exactly 64 sample indices")
    if max(evaluation_indices) >= len(evalset):
        raise ValueError("evaluation sample index exceeds the NYU split")
    records = _capture_fp_records(
        model, saved_args, evalset, evaluation_indices,
        device, args.seed, instrumentor, rotation,
        propagation, block_capture)

    model_output = Path(args.out_dir) / "cspn"
    model_output.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    raw_region_rows = []
    boundary_rows = []
    layer_rows = []
    propagation_rows = []
    for config in configurations:
        current = _evaluate_configuration(
            model, saved_args, records, device, config,
            instrumentor, rotation, propagation,
            block_capture, model_output)
        sample_rows.extend(current[0])
        raw_region_rows.extend(current[1])
        boundary_rows.extend(current[2])
        layer_rows.extend(current[3])
        propagation_rows.extend(current[4])

    regional_rows = []
    for config in configurations:
        selected = [
            row for row in raw_region_rows
            if row["config"] == config["name"]]
        aggregate = aggregate_region_rows(selected)
        for row in aggregate:
            row.update({"model": "cspn", "config": config["name"]})
            regional_rows.append(row)

    write_csv(
        model_output / "config_manifest.csv",
        _config_manifest(configurations),
        ("config", "weight_bits", "activation_bits", "decoder_entry",
         "layer4_signed_skip", "group_size", "bias_format",
         "guidance_head", "affinity_bits", "state_bits",
         "coefficient_format"))
    write_csv(
        model_output / "group_size_search.csv", group_search_rows,
        ("group_size", "block_output_mse", "block_output_sqnr",
         "selected"))
    write_csv(
        model_output / "sample_metrics.csv", sample_rows,
        END_TO_END_FIELDS)
    write_csv(
        model_output / "regional_metrics.csv", regional_rows,
        ("model", "config", "region", "RMSE", "MAE", "ABS_REL"))
    write_csv(
        model_output / "boundary_metrics.csv", boundary_rows,
        BOUNDARY_FIELDS)
    write_csv(
        model_output / "layer_quantization_metrics.csv", layer_rows,
        ("model", "config", "module", "group", "kind"))
    write_csv(
        model_output / "propagation_metrics.csv", propagation_rows,
        ("model", "config", "sample_index", "signal", "iteration"))
    write_json(model_output / "metadata.json", {
        "model": "cspn",
        "architecture": architecture,
        "model_class": type(model).__name__,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_load": load_report,
        "seed": args.seed,
        "calibration_samples": args.calibration_samples,
        "calibration_indices": calibration_indices,
        "evaluation_samples": len(evaluation_indices),
        "evaluation_indices": evaluation_indices,
        "selected_group_size": selected_group_size,
        "bias_format": "fp32",
        "guidance_head": "fp32",
        "propagation": dict(PROPAGATION_A8_Q13),
        "coefficient_format": "signed_int16_q13",
        "fold_max_abs_error": fold_error,
        "folded_pairs": hardware_preparation["folded_pairs"],
        "elapsed_seconds": time.time() - started,
    })

    block_capture.close()
    propagation.close()
    rotation.close()
    instrumentor.close()


if __name__ == "__main__":
    main()
