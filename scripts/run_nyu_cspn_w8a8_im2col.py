#!/usr/bin/env python3
"""Profile official CSPN PA-W8A8 Conv structure in Im2Col space."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_activation_resolution as cspn  # noqa: E402
from scripts import run_nyu_rtn_quantization as rtn  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.export_nyu_predictions import build_model, file_sha256  # noqa: E402
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.hardware_merge_adapters import CallIndexedConcatAdapter  # noqa: E402
from scripts.nyu_quantization_analysis import regional_depth_metrics  # noqa: E402
from spn_quant.im2col_diagnostics import (  # noqa: E402
    CSPNW8A8Im2ColRecorder,
)
from spn_quant.propagation import (  # noqa: E402
    install_propagation_adapter,
    propagation_projection_outputs,
)


PROPAGATION_W8A8_Q13 = {
    "affinity_bits": 8,
    "confidence_bits": 8,
    "offset_bits": 8,
    "state_bits": 8,
    "coefficient_fraction_bits": 13,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stratified-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--percentile-capacity", type=int, required=True)
    parser.add_argument("--token-topk", type=int, required=True)
    parser.add_argument("--token-chunk", type=int, required=True)
    parser.add_argument("--plot-layer-count", type=int, required=True)
    parser.add_argument("--plot-sample-count", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def load_protocol(checkpoint: Path, metadata_path: Path) -> Dict[str, object]:
    checkpoint = Path(checkpoint).resolve()
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model"] != "cspn":
        raise ValueError("Im2Col diagnostics require model=cspn")
    if Path(metadata["checkpoint"]).resolve() != checkpoint:
        raise ValueError("stratified metadata checkpoint changed")
    calibration_indices = tuple(
        int(index) for index in metadata["calibration_indices"])
    evaluation_indices = tuple(
        int(index) for index in metadata["evaluation_indices"])
    if len(calibration_indices) != 128 or \
            len(set(calibration_indices)) != 128:
        raise ValueError("protocol requires 128 unique calibration indices")
    if len(evaluation_indices) != 64 or len(set(evaluation_indices)) != 64:
        raise ValueError("protocol requires 64 unique evaluation indices")
    return {
        "model": metadata["model"],
        "checkpoint": str(checkpoint),
        "seed": int(metadata["seed"]),
        "calibration_indices": calibration_indices,
        "evaluation_indices": evaluation_indices,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": file_sha256(metadata_path),
    }


def validate_w8a8_configuration(config: Dict[str, object]) -> None:
    if config["name"] != "PA_W8A8":
        raise ValueError("configuration must be PA_W8A8")
    if int(config["w_bits"]) != 8 or int(config["a_bits"]) != 8:
        raise ValueError("ordinary Conv contract must be W8A8")
    if config["propagation"] != PROPAGATION_W8A8_Q13:
        raise ValueError("propagation contract must be A8/Q13")
    if not bool(config["external_output_ownership"]):
        raise ValueError("PA_W8A8 requires external propagation ownership")
    if bool(config["activation_overrides"]) or \
            bool(config["activation_bit_overrides"]) or \
            bool(config["activation_format_overrides"]) or \
            bool(config["smooth_channel_maxima"]):
        raise ValueError("PA_W8A8 diagnostics prohibit activation overrides")
    if config["activation_mode"] != "uniform":
        raise ValueError("PA_W8A8 diagnostics require uniform activations")


def select_w8a8_configuration(groups: Sequence[str]) -> Dict[str, object]:
    selected = [
        config for config in rtn.build_propagation_configurations(groups)
        if config["name"] == "PA_W8A8"]
    if len(selected) != 1:
        raise RuntimeError("PA_W8A8 configuration coverage changed")
    config = selected[0]
    validate_w8a8_configuration(config)
    return config


def _saved_args(checkpoint: Path, data_root: Path, seed: int) -> Namespace:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = Namespace(**payload["args"])
    if saved_args.model != "cspn":
        raise ValueError("checkpoint must contain model=cspn")
    saved_args.data_root = str(Path(data_root).resolve())
    saved_args.seed = int(seed)
    return saved_args


def enter_official_cspn_root(data_root: Path) -> None:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    pretrained = root / "pretrained" / "resnet18.pth"
    models = root / "models"
    if not pretrained.is_file():
        raise FileNotFoundError(str(pretrained))
    if not models.is_dir():
        raise FileNotFoundError(str(models))
    os.chdir(root)


def _model_args(saved_args: Namespace, sample, device: torch.device):
    batch = rtn.batch_from_sample(sample)
    model_args, _ = sweep.batch_to_model_input("cspn", batch, device)
    return model_args


def _sample_metrics(config: str, sample_index: int,
                    sample, prediction: torch.Tensor) -> Dict[str, object]:
    gt = sample["depth"][0].numpy()
    sparse = sample["rgbd"][3].numpy()
    pred = prediction.detach().cpu()[0, 0].numpy()
    regions = regional_depth_metrics(gt, pred, sparse)
    row = next(row for row in regions if row["region"] == "all")
    return {
        "model": "cspn",
        "config": config,
        "sample_index": int(sample_index),
        "RMSE": row["RMSE"],
        "MAE": row["MAE"],
        "ABS_REL": row["ABS_REL"],
        "num_pixels": row["num_pixels"],
        "nonfinite_pixels": int(np.count_nonzero(~np.isfinite(pred))),
    }


def _evaluate_fp32(model, saved_args, dataset, indices, device,
                   instrumentor, propagation, merge_adapter):
    instrumentor.disable()
    propagation.disable()
    merge_adapter.disable()
    rows = []
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = rtn.seeded_sample(dataset, index, saved_args.seed)
            output = model(*_model_args(saved_args, sample, device))
            prediction = sweep.extract_pred(output)
            rows.append(_sample_metrics("FP32", index, sample, prediction))
            if rank % 16 == 0 or rank == len(indices):
                print("FP32 %d/%d" % (rank, len(indices)), flush=True)
    return rows


def _write_spatial(root: Path, module: str, sample_index: int,
                   arrays: Dict[str, np.ndarray]) -> str:
    directory = root.joinpath("spatial_tokens", *module.split("."))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("sample_%05d.npz" % int(sample_index))
    pending = path.with_name(path.name + ".pending")
    payload = dict(arrays)
    payload["module"] = np.array(module)
    payload["sample_index"] = np.array(int(sample_index))
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    pending.replace(path)
    return str(path.relative_to(root))


def _mean_metrics(rows, config: str) -> Dict[str, float]:
    selected = [row for row in rows if row["config"] == config]
    if len(selected) != 64:
        raise RuntimeError("metric aggregation requires 64 samples")
    return dict(
        (metric, float(np.mean([
            float(row[metric]) for row in selected], dtype=np.float64)))
        for metric in ("RMSE", "MAE", "ABS_REL"))


def _calibrate(model, saved_args, dataset, indices, device,
               instrumentor, propagation, merge_adapter) -> None:
    instrumentor.observe()
    propagation.observe()
    merge_adapter.observe()
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = rtn.seeded_sample(dataset, index, saved_args.seed)
            model(*_model_args(saved_args, sample, device))
            if rank % 16 == 0 or rank == len(indices):
                print("calibration %d/%d" % (rank, len(indices)), flush=True)
    instrumentor.freeze()
    propagation.freeze()
    merge_adapter.disable()


def _evaluate_w8a8(model, saved_args, dataset, indices, device,
                    config, instrumentor, propagation, merge_adapter,
                    recorder, output_root: Path):
    propagation_outputs = set(
        propagation_projection_outputs("cspn", model))
    rtn.configure_quantized_model(
        config, instrumentor, propagation, None,
        propagation_outputs, "cspn")
    merge_adapter.freeze(8)
    merge_adapter.quantize()
    instrumentor.set_runtime_statistics(False)
    first_index = int(indices[0])
    first_sample = rtn.seeded_sample(
        dataset, first_index, saved_args.seed)
    with torch.no_grad():
        baseline = sweep.extract_pred(
            model(*_model_args(saved_args, first_sample, device))
        ).detach().cpu()
    instrumentor.set_activation_recorder(recorder)
    rows = []
    spatial_rows = []
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = rtn.seeded_sample(dataset, index, saved_args.seed)
            recorder.begin_sample(index)
            output = model(*_model_args(saved_args, sample, device))
            prediction = sweep.extract_pred(output)
            arrays = recorder.end_sample()
            if int(index) == first_index and not torch.equal(
                    prediction.detach().cpu(), baseline):
                raise RuntimeError("diagnostic recorder changed W8A8 prediction")
            for module in sorted(arrays):
                spatial_rows.append({
                    "module": module,
                    "sample_index": int(index),
                    "path": _write_spatial(
                        output_root, module, index, arrays[module]),
                })
            recorder.clear_sample()
            rows.append(_sample_metrics(
                "PA_W8A8", index, sample, prediction))
            if rank % 8 == 0 or rank == len(indices):
                print("PA_W8A8 Im2Col %d/%d" %
                      (rank, len(indices)), flush=True)
    instrumentor.clear_activation_recorder()
    return rows, spatial_rows


def _selected_plot_items(layer_rows, token_rows,
                         layer_count: int, sample_count: int):
    ranked_layers = sorted(
        layer_rows,
        key=lambda row: (
            float(row["local_output_sqnr_db"]),
            -float(row["local_output_error_p99"]), str(row["module"])))
    modules = [str(row["module"]) for row in ranked_layers[:layer_count]]
    selected_tokens = sorted(
        [row for row in token_rows if row["module"] in set(modules)],
        key=lambda row: (
            -float(row["local_output_error"]), int(row["sample_index"]),
            str(row["module"])))
    samples = []
    for row in selected_tokens:
        index = int(row["sample_index"])
        if index not in samples:
            samples.append(index)
        if len(samples) == sample_count:
            break
    if len(modules) != layer_count or len(samples) != sample_count:
        raise RuntimeError("plot selection coverage is incomplete")
    return modules, samples


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN W8A8 Im2Col diagnostics require CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    limits = (
        args.percentile_capacity, args.token_topk, args.token_chunk,
        args.plot_layer_count, args.plot_sample_count)
    if any(int(value) <= 0 for value in limits):
        raise ValueError("diagnostic capacities and counts must be positive")
    if float(args.fold_max_error) <= 0.0:
        raise ValueError("fold max error must be positive")
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(str(output_root))
    output_root.mkdir(parents=True)
    checkpoint = Path(args.checkpoint).resolve()
    protocol = load_protocol(
        checkpoint, Path(args.stratified_metadata))
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    saved_args = _saved_args(
        checkpoint, Path(args.data_root), int(protocol["seed"]))
    enter_official_cspn_root(Path(args.data_root))
    model, architecture = build_model(saved_args, checkpoint, device)
    trainset = rtn.calibration_dataset(saved_args)
    valset = rtn.evaluation_dataset(saved_args)
    calibration_indices = protocol["calibration_indices"]
    evaluation_indices = protocol["evaluation_indices"]
    if max(calibration_indices) >= len(trainset) or \
            max(evaluation_indices) >= len(valset):
        raise ValueError("protocol sample index exceeds dataset length")
    preparation_sample = rtn.seeded_sample(
        trainset, calibration_indices[0], saved_args.seed)
    preparation_args = _model_args(
        saved_args, preparation_sample, device)
    preparation = prepare_hardware_model(
        model, preparation_args, excluded_pairs=(("conv1_1", "bn1"),))
    if float(preparation["primary_max_abs_error"]) > \
            float(args.fold_max_error):
        raise RuntimeError("Conv-BN fold exceeds declared error threshold")
    propagation_outputs = set(
        propagation_projection_outputs("cspn", model))
    instrumentor = HardwareAlignedInstrumentor(
        model, cspn.cspn_quant_group,
        preparation["fused_relu_producers"],
        externally_owned_outputs=propagation_outputs)
    propagation = install_propagation_adapter("cspn", model)
    merge_adapter = CallIndexedConcatAdapter(model)
    groups = sorted(set(instrumentor.module_groups().values()))
    config = select_w8a8_configuration(groups)
    started = time.time()
    _calibrate(
        model, saved_args, trainset, calibration_indices, device,
        instrumentor, propagation, merge_adapter)
    fp32_rows = _evaluate_fp32(
        model, saved_args, valset, evaluation_indices, device,
        instrumentor, propagation, merge_adapter)
    recorder = CSPNW8A8Im2ColRecorder(
        instrumentor.modules, instrumentor.original_weights,
        args.percentile_capacity, args.token_topk, args.token_chunk)
    w8a8_rows, spatial_rows = _evaluate_w8a8(
        model, saved_args, valset, evaluation_indices, device, config,
        instrumentor, propagation, merge_adapter, recorder, output_root)
    sample_rows = fp32_rows + w8a8_rows
    layer_rows = recorder.layer_rows()
    token_rows = recorder.top_token_rows()
    selected_modules, selected_samples = _selected_plot_items(
        layer_rows, token_rows,
        int(args.plot_layer_count), int(args.plot_sample_count))
    module_rows = recorder.module_manifest_rows()
    groups_by_module = instrumentor.module_groups()
    for row in module_rows:
        row["group"] = groups_by_module[row["module"]]
        row["weight_bits"] = 8 if row["status"] == "collected_conv2d" else ""
        row["activation_bits"] = 8 if \
            row["status"] == "collected_conv2d" else ""
    rtn.write_csv(output_root / "module_manifest.csv", module_rows)
    rtn.write_csv(
        output_root / "channel_offset_metrics.csv",
        recorder.channel_offset_rows())
    rtn.write_csv(output_root / "channel_metrics.csv", recorder.channel_rows())
    rtn.write_csv(
        output_root / "kernel_offset_metrics.csv", recorder.offset_rows())
    rtn.write_csv(output_root / "layer_metrics.csv", layer_rows)
    rtn.write_csv(output_root / "top_spatial_tokens.csv", token_rows)
    rtn.write_csv(output_root / "spatial_manifest.csv", spatial_rows)
    rtn.write_csv(output_root / "sample_metrics.csv", sample_rows)
    rtn.write_json(output_root / "run_manifest.json", {
        "model": "cspn",
        "configuration": "PA_W8A8",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "protocol": protocol,
        "data_root": str(Path(args.data_root).resolve()),
        "architecture": architecture,
        "hardware_preparation": preparation,
        "quantization": {
            "weight_bits": 8,
            "activation_bits": 8,
            "activation_mode": "uniform",
            "propagation": PROPAGATION_W8A8_Q13,
        },
        "percentile_capacity": int(args.percentile_capacity),
        "token_topk": int(args.token_topk),
        "token_chunk": int(args.token_chunk),
        "selected_plot_modules": selected_modules,
        "selected_plot_samples": selected_samples,
        "mean_metrics": {
            "FP32": _mean_metrics(sample_rows, "FP32"),
            "PA_W8A8": _mean_metrics(sample_rows, "PA_W8A8"),
        },
        "elapsed_seconds": time.time() - started,
    })
    instrumentor.close()
    propagation.close()
    merge_adapter.close()
    print("CSPN PA_W8A8 Im2Col diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
