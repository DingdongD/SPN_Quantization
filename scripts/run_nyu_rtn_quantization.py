#!/usr/bin/env python3
"""Run reproducible RTN W8A8/W4A4 analysis for one converged NYU model."""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.activation_bit_allocation import (  # noqa: E402
    build_mixed_configurations,
    config_manifest_rows,
    select_sensitive_modules,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.hardware_merge_adapters import CallIndexedConcatAdapter  # noqa: E402
from scripts.outlier_mitigation_quantization import (  # noqa: E402
    load_input_channel_maxima,
    load_percentile_overrides,
)
from scripts.export_nyu_predictions import (  # noqa: E402
    build_model,
    load_run_args,
    prepare_args,
)
from scripts.nyu_quantization_analysis import (  # noqa: E402
    MODULE_GROUP_ORDER,
    classify_module,
    compare_output_signals,
    extract_output_signals,
    regional_depth_metrics,
)
from scripts.propagation_quantization import install_state_adapter  # noqa: E402
from scripts.rtn_quantization import RTNInstrumentor  # noqa: E402


def build_configurations(groups):
    ordered = [group for group in MODULE_GROUP_ORDER if group in set(groups)]
    ordered.extend(sorted(set(groups) - set(ordered)))
    all_groups = set(ordered)
    configs = [
        {"name": "FP32", "w_bits": None, "a_bits": None,
         "groups": set(), "state_bits": None},
        {"name": "W8A8_full", "w_bits": 8, "a_bits": 8,
         "groups": all_groups, "state_bits": None},
        {"name": "W4A8_full", "w_bits": 4, "a_bits": 8,
         "groups": all_groups, "state_bits": None},
        {"name": "W4A4_full", "w_bits": 4, "a_bits": 4,
         "groups": all_groups, "state_bits": None},
    ]
    for group in ordered:
        configs.extend([
            {"name": "W8A8_%s_only" % group, "w_bits": 8, "a_bits": 8,
             "groups": {group}, "state_bits": None},
            {"name": "W4A4_%s_only" % group, "w_bits": 4, "a_bits": 4,
             "groups": {group}, "state_bits": None},
        ])
    configs.extend([
        {"name": "W8A8_full_stateA8", "w_bits": 8, "a_bits": 8,
         "groups": all_groups, "state_bits": 8},
        {"name": "W4A4_full_stateA4", "w_bits": 4, "a_bits": 4,
         "groups": all_groups, "state_bits": 4},
    ])
    return configs


def build_hardware_configurations(groups):
    all_groups = set(groups)
    return [
        {"name": "FP32", "w_bits": None, "a_bits": None,
         "groups": set(), "state_bits": None},
        {"name": "HW_W4A8_full", "w_bits": 4, "a_bits": 8,
         "groups": all_groups, "state_bits": None},
        {"name": "HW_W4A4_full", "w_bits": 4, "a_bits": 4,
         "groups": all_groups, "state_bits": None},
    ]


def build_outlier_configurations(groups):
    all_groups = set(groups)
    base = {"groups": all_groups, "state_bits": None}
    configs = [
        {"name": "FP32", "w_bits": None, "a_bits": None,
         "groups": set(), "state_bits": None},
        dict(base, name="HW_W4A4_MinMax", w_bits=4, a_bits=4),
        dict(base, name="HW_W8A4_full", w_bits=8, a_bits=4),
    ]
    for name, percentile in (
            ("HW_W4A4_P99", "p99"),
            ("HW_W4A4_P999", "p99_9"),
            ("HW_W4A4_P9999", "p99_99")):
        configs.append(dict(
            base, name=name, w_bits=4, a_bits=4,
            percentile=percentile))
    for name, alpha in (
            ("HW_W4A4_SQ_A25", 0.25),
            ("HW_W4A4_SQ_A50", 0.50),
            ("HW_W4A4_SQ_A75", 0.75)):
        configs.append(dict(
            base, name=name, w_bits=4, a_bits=4,
            smooth_alpha=alpha))
    for name, ratio in (
            ("HW_W4A4_AWQ_C90", 0.90),
            ("HW_W4A4_AWQ_C80", 0.80)):
        configs.append(dict(
            base, name=name, w_bits=4, a_bits=4,
            weight_clip_ratio=ratio))
    return configs


def build_lognp_configurations(groups):
    all_groups = set(groups)
    base = {
        "groups": all_groups,
        "state_bits": None,
        "w_bits": 8,
        "a_bits": 4,
        "activation_mode": "lognp",
        "alpha_factor": 1.0,
        "max_z": 24.0,
    }
    return [
        {"name": "FP32", "w_bits": None, "a_bits": None,
         "groups": set(), "state_bits": None},
        dict(base, name="LOGNP_W8A4_tensor", lognp_per_channel=False),
        dict(base, name="LOGNP_W8A4_channel", lognp_per_channel=True),
        dict(base, name="LOGNP_W8A4_channel_bias",
             lognp_per_channel=True, compensation_method="bias"),
        dict(base, name="LOGNP_W8A4_channel_weight",
             lognp_per_channel=True, compensation_method="weight"),
        dict(base, name="LOGNP_W4A4_channel_weight", w_bits=4,
             lognp_per_channel=True, compensation_method="weight"),
        {"name": "W4A8_full", "w_bits": 4, "a_bits": 8,
         "groups": all_groups, "state_bits": None},
    ]


def instrumentor_options(config):
    keys = ("activation_overrides", "smooth_channel_maxima",
            "smooth_alpha", "weight_clip_ratio",
            "activation_bit_overrides", "activation_mode",
            "alpha_factor", "max_z", "lognp_per_channel")
    return dict((key, config[key]) for key in keys if key in config)


def load_sample_indices(path):
    indices = []
    seen = set()
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            index = int(row["sample_index"])
            if index not in seen:
                seen.add(index)
                indices.append(index)
    return indices


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_lognp_compensation_modules(model, module_names,
                                      sensitivity_root, limit=4):
    """Select only ranked sensitive modules for calibration compensation."""
    module_names = set(module_names)
    path = Path(sensitivity_root) / model / "layer_quantization_metrics.csv"
    candidates = select_sensitive_modules(
        read_csv(path), limit=limit, config="MP_W4A4_base")
    selected = [row["module"] for row in candidates
                if row["module"] in module_names]
    if selected:
        return selected
    return sorted(module_names)[:int(limit)]


def replace_config_rows(existing, replacement, configs):
    configs = set(configs)
    preserved = [row for row in existing if row.get("config") not in configs]
    return preserved + list(replacement)


def merge_manifest_rows(existing, replacement, configs):
    return replace_config_rows(existing, replacement, configs)


LEGACY_PREDICTION_CONFIGS = (
        "W8A8_full", "W4A8_full", "W4A4_full",
        "HW_W4A8_full", "HW_W4A4_full",
)


def should_export_predictions(config_name, explicit_names=None):
    if explicit_names is not None:
        return config_name in explicit_names
    return config_name in LEGACY_PREDICTION_CONFIGS


def prediction_payload(gt, fp32, pred, sample_index, model, config):
    gt = np.asarray(gt)
    fp32 = np.asarray(fp32)
    pred = np.asarray(pred)
    if gt.shape != fp32.shape or gt.shape != pred.shape:
        raise ValueError("gt, fp32, and pred must have identical shapes")
    valid_gt = np.isfinite(gt) & (gt > 1e-4)
    nonfinite = valid_gt & ~np.isfinite(pred)
    abs_err = np.abs(pred - gt).astype(np.float32)
    abs_err[~valid_gt] = np.nan
    return {
        "gt": gt.astype(np.float32),
        "fp32": fp32.astype(np.float32),
        "pred": pred.astype(np.float32),
        "abs_err": abs_err,
        "valid_gt": valid_gt,
        "nonfinite": nonfinite,
        "sample_index": np.array(int(sample_index)),
        "model": np.array(model),
        "config": np.array(config),
    }


def prepare_prediction_dir(out_dir, config):
    prediction_dir = Path(out_dir) / "predictions" / config
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for path in prediction_dir.glob("sample_*.npz"):
        path.unlink()
    return prediction_dir


def write_prediction_payload(prediction_dir, payload):
    path = Path(prediction_dir) / ("sample_%05d.npz" %
                                   int(payload["sample_index"]))
    np.savez_compressed(str(path), **payload)
    return path


def aggregate_region_rows(rows):
    totals = {}
    for row in rows:
        region = row["region"]
        current = totals.setdefault(region, {
            "region": region,
            "num_pixels": 0,
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "sum_abs_rel": 0.0,
        })
        for key in ("num_pixels", "sum_sq", "sum_abs", "sum_abs_rel"):
            current[key] += row[key]
    summary = []
    for region in sorted(totals):
        row = totals[region]
        count = int(row["num_pixels"])
        row["num_pixels"] = count
        row["RMSE"] = math.sqrt(row["sum_sq"] / count) if count else float("nan")
        row["MAE"] = row["sum_abs"] / count if count else float("nan")
        row["ABS_REL"] = row["sum_abs_rel"] / count if count else float("nan")
        summary.append(row)
    return summary


def write_csv(path, rows, preferred=()):
    if not rows:
        return
    keys = set()
    for row in rows:
        keys.update(row)
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def write_json(path, payload):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def detach_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [detach_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_cpu(item) for item in value)
    return value


class PropagationInputCapture(object):
    def __init__(self, model_name, module):
        self.model_name = model_name
        self.current = {}
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        if self.model_name == "cspn":
            names = ("guidance", "pred_init", "sparse_depth")
        else:
            names = ("pred_init", "guidance", "confidence", "sparse_depth")
        self.current = {}
        for name, value in zip(names, inputs):
            if value is not None:
                self.current[name] = detach_cpu(value)

    def close(self):
        self.handle.remove()


def seeded_sample(dataset, index, seed):
    np.random.seed(seed + int(index))
    torch.manual_seed(seed + int(index))
    return dataset[int(index)]


def batch_from_sample(sample):
    return dict((key, value.unsqueeze(0) if torch.is_tensor(value) else value)
                for key, value in sample.items())


def model_signals(output, adapter, input_capture):
    signals = dict((key, detach_cpu(value))
                   for key, value in extract_output_signals(output).items())
    signals.update(input_capture.current)
    states = adapter.last_states()
    if states:
        signals["propagation_states"] = states
    return signals


def calibration_dataset(saved_args):
    dataset_class = sweep.CspnOfficialDataset \
        if saved_args.model == "cspn" else sweep.NyuHdf5Dataset
    return dataset_class(
        csv_file=saved_args.train_list,
        root_dir=str(REPO_ROOT),
        split="train",
        n_sample=saved_args.n_sample,
        seed=saved_args.seed,
    )


def evaluation_dataset(saved_args):
    return sweep.NyuHdf5Dataset(
        csv_file=saved_args.eval_list,
        root_dir=str(REPO_ROOT),
        split="val",
        n_sample=saved_args.n_sample,
        seed=saved_args.seed,
    )


def capture_fp32_records(model, saved_args, dataset, indices, device,
                         adapter, input_capture, seed):
    records = []
    adapter.capture()
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            batch = batch_from_sample(sample)
            model_args, _ = sweep.batch_to_model_input(saved_args.model, batch, device)
            output = model(*model_args)
            pred = sweep.extract_pred(output).detach().cpu()[0, 0]
            gt = sample["depth"][0].clone()
            sparse = sample["rgbd"][3].clone()
            records.append({
                "sample_index": int(index),
                "sample": sample,
                "gt": gt,
                "sparse": sparse,
                "pred": pred,
                "signals": model_signals(output, adapter, input_capture),
            })
            print("FP32 %d/%d sample=%05d" % (rank, len(indices), index), flush=True)
    adapter.disable()
    return records


def evaluate_configuration(model, saved_args, records, device, config,
                           instrumentor, adapter, input_capture, out_dir,
                           export_predictions, merge_adapter=None):
    mitigation = instrumentor_options(config)
    instrumentor.configure(
        config["w_bits"], config["a_bits"], config["groups"], **mitigation)
    compensation_rows = []
    if config.get("compensation_method"):
        compensation_rows = instrumentor.apply_compensation(
            method=config["compensation_method"])
        for row in compensation_rows:
            row.update({"model": saved_args.model, "config": config["name"]})
    if merge_adapter is not None:
        merge_adapter.freeze(config["a_bits"])
        merge_adapter.quantize()
    if config["state_bits"] is None:
        adapter.disable()
    else:
        adapter.configure(config["state_bits"])

    sample_rows = []
    all_region_rows = []
    signal_rows = []
    prediction_dir = Path(out_dir) / "predictions" / config["name"]
    if export_predictions:
        prediction_dir = prepare_prediction_dir(out_dir, config["name"])

    with torch.no_grad():
        for rank, record in enumerate(records, 1):
            sample = record["sample"]
            batch = batch_from_sample(sample)
            model_args, _ = sweep.batch_to_model_input(saved_args.model, batch, device)
            output = model(*model_args)
            pred = sweep.extract_pred(output).detach().cpu()[0, 0]
            gt_np = record["gt"].numpy()
            pred_np = pred.numpy()
            sparse_np = record["sparse"].numpy()
            regions = regional_depth_metrics(gt_np, pred_np, sparse_np)
            all_region_rows.extend(regions)
            metrics = dict((row["region"], row) for row in regions)["all"]
            nonfinite = int(np.count_nonzero(~np.isfinite(pred_np)))
            sample_rows.append({
                "model": saved_args.model,
                "config": config["name"],
                "sample_index": record["sample_index"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "ABS_REL": metrics["ABS_REL"],
                "num_pixels": metrics["num_pixels"],
                "nonfinite_pixels": nonfinite,
            })
            candidate_signals = model_signals(output, adapter, input_capture)
            for row in compare_output_signals(record["signals"], candidate_signals):
                row.update({
                    "model": saved_args.model,
                    "config": config["name"],
                    "sample_index": record["sample_index"],
                })
                signal_rows.append(row)
            if export_predictions:
                write_prediction_payload(prediction_dir, prediction_payload(
                    gt_np, record["pred"].numpy(), pred_np,
                    record["sample_index"], saved_args.model, config["name"]))
            print("%s %d/%d sample=%05d RMSE=%.5f" % (
                config["name"], rank, len(records), record["sample_index"],
                metrics["RMSE"]), flush=True)

    region_summary = aggregate_region_rows(all_region_rows)
    for row in region_summary:
        row.update({"model": saved_args.model, "config": config["name"]})
    layer_rows = instrumentor.statistics()
    for row in layer_rows:
        row.update({"model": saved_args.model, "config": config["name"]})
    state_rows = adapter.statistics() if config["state_bits"] is not None else []
    for row in state_rows:
        row.update({"model": saved_args.model, "config": config["name"]})
    manifest_rows = []
    if config.get("activation_mode") == "lognp":
        for row in instrumentor.manifest():
            manifest_rows.append(dict(
                row, model=saved_args.model, config=config["name"]))
    return (sample_rows, region_summary, signal_rows, layer_rows, state_rows,
            manifest_rows, compensation_rows)


def baseline_rows(saved_args, records):
    sample_rows = []
    regions = []
    for record in records:
        current = regional_depth_metrics(
            record["gt"].numpy(), record["pred"].numpy(), record["sparse"].numpy())
        regions.extend(current)
        metrics = dict((row["region"], row) for row in current)["all"]
        sample_rows.append({
            "model": saved_args.model,
            "config": "FP32",
            "sample_index": record["sample_index"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "ABS_REL": metrics["ABS_REL"],
            "num_pixels": metrics["num_pixels"],
            "nonfinite_pixels": 0,
        })
    summary = aggregate_region_rows(regions)
    for row in summary:
        row.update({"model": saved_args.model, "config": "FP32"})
    return sample_rows, summary


def persist_tables(out_dir, sample_rows, region_rows, signal_rows,
                   layer_rows, state_rows):
    out_dir = Path(out_dir)
    write_csv(out_dir / "sample_metrics.csv", sample_rows,
              ("model", "config", "sample_index", "RMSE", "MAE", "ABS_REL"))
    write_csv(out_dir / "regional_metrics.csv", region_rows,
              ("model", "config", "region", "RMSE", "MAE", "ABS_REL"))
    write_csv(out_dir / "signal_metrics.csv", signal_rows,
              ("model", "config", "sample_index", "signal", "iteration"))
    write_csv(out_dir / "layer_quantization_metrics.csv", layer_rows,
              ("model", "config", "module", "group", "kind"))
    write_csv(out_dir / "state_quantization_metrics.csv", state_rows,
              ("model", "config", "iteration"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--sample-metrics", required=True)
    parser.add_argument("--out-dir", default="profile_logs/nyu_rtn_quantization")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--config-names", nargs="*", default=[])
    parser.add_argument("--export-prediction-configs", nargs="*", default=None,
                        help="configuration names whose predictions are saved")
    parser.add_argument("--append", action="store_true",
                        help="replace requested configs while preserving existing metrics")
    parser.add_argument("--quant-backend",
                        choices=("rtn", "hardware", "outlier", "mixed", "lognp"),
                        default="rtn")
    parser.add_argument("--outlier-profile-root",
                        default="profile_logs/nyu_activation_outliers")
    parser.add_argument("--sensitivity-root",
                        default="profile_logs/nyu_activation_outliers")
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--fold-max-error", type=float, default=0.05)
    parser.add_argument(
        "--skip-conv-bn-fold", action="store_true",
        help="keep official Conv-BN modules intact for numerically sensitive models")
    parser.add_argument("--lognp-alpha-factor", type=float, default=1.0)
    parser.add_argument("--lognp-max-z", type=float, default=24.0)
    parser.add_argument("--lognp-compensation-samples", type=int, default=8192)
    parser.add_argument("--lognp-compensation-limit", type=int, default=4)
    parser.add_argument("--lognp-sensitivity-root",
                        default="profile_logs/nyu_activation_bit_allocation")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    saved_args = prepare_args(load_run_args(run_dir), args)
    if saved_args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(saved_args.device)
    torch.backends.cudnn.benchmark = False
    model, meta = build_model(saved_args, checkpoint, device)
    trainset = calibration_dataset(saved_args)
    calibration_count = min(args.calibration_samples, len(trainset))
    calibration_indices = np.random.RandomState(args.seed).choice(
        len(trainset), calibration_count, replace=False).tolist()
    hardware_preparation = None
    merge_adapter = None
    group_fn = lambda name, module: classify_module(saved_args.model, name, module)
    if args.quant_backend in ("hardware", "outlier", "mixed", "lognp"):
        preparation_sample = seeded_sample(
            trainset, calibration_indices[0], args.seed)
        preparation_batch = batch_from_sample(preparation_sample)
        preparation_args, _ = sweep.batch_to_model_input(
            saved_args.model, preparation_batch, device)
        excluded_pairs = [("conv1_1", "bn1")] \
            if saved_args.model == "cspn" else []
        hardware_preparation = prepare_hardware_model(
            model, preparation_args, excluded_pairs=excluded_pairs,
            fold=not args.skip_conv_bn_fold)
        if hardware_preparation["max_abs_error"] > args.fold_max_error:
            raise RuntimeError("Conv-BN fold changed FP32 output by %.8f" %
                               hardware_preparation["max_abs_error"])
        instrumentor = HardwareAlignedInstrumentor(
            model, group_fn, hardware_preparation["fused_relu_producers"])
        merge_adapter = CallIndexedConcatAdapter(model)
    else:
        instrumentor = RTNInstrumentor(model, group_fn)
    adapter = install_state_adapter(saved_args.model, model)
    input_capture = PropagationInputCapture(saved_args.model, adapter.module)

    model_out_dir = Path(args.out_dir) / saved_args.model
    model_out_dir.mkdir(parents=True, exist_ok=True)
    table_filenames = {
        "samples": "sample_metrics.csv",
        "regions": "regional_metrics.csv",
        "signals": "signal_metrics.csv",
        "layers": "layer_quantization_metrics.csv",
        "states": "state_quantization_metrics.csv",
    }
    existing_tables = dict((key, read_csv(model_out_dir / filename))
                           for key, filename in table_filenames.items()) \
        if args.append else dict((key, []) for key in table_filenames)
    existing_lognp_manifest = read_csv(
        model_out_dir / "lognp_manifest.csv") if args.append else []
    existing_lognp_compensation = read_csv(
        model_out_dir / "lognp_compensation.csv") if args.append else []
    metadata_path = model_out_dir / "metadata.json"
    existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) \
        if args.append and metadata_path.exists() else None
    groups = sorted(set(instrumentor.module_groups().values()),
                    key=lambda group: MODULE_GROUP_ORDER.index(group))
    if args.quant_backend == "hardware":
        configs = build_hardware_configurations(groups)
    elif args.quant_backend == "outlier":
        configs = build_outlier_configurations(groups)
        profile_root = Path(args.outlier_profile_root) / saved_args.model
        channel_maxima = None
        for config in configs:
            if "percentile" in config:
                config["activation_overrides"] = load_percentile_overrides(
                    profile_root, config["percentile"])
            if "smooth_alpha" in config:
                if channel_maxima is None:
                    channel_maxima = load_input_channel_maxima(profile_root)
                config["smooth_channel_maxima"] = channel_maxima
    elif args.quant_backend == "mixed":
        sensitivity_path = Path(args.sensitivity_root) / saved_args.model / \
            "layer_quantization_metrics.csv"
        candidates = select_sensitive_modules(
            read_csv(sensitivity_path), limit=args.candidate_limit)
        configs = build_mixed_configurations(
            instrumentor.module_groups(), candidates)
    elif args.quant_backend == "lognp":
        configs = build_lognp_configurations(groups)
        for config in configs:
            if config.get("activation_mode") == "lognp":
                config["alpha_factor"] = args.lognp_alpha_factor
                config["max_z"] = args.lognp_max_z
    else:
        configs = build_configurations(groups)
    if args.config_names:
        requested = set(args.config_names)
        configs = [config for config in configs if config["name"] in requested]
        missing = requested - set(config["name"] for config in configs)
        if missing:
            raise ValueError("unknown configs: %s" % sorted(missing))
    export_prediction_configs = None
    if args.export_prediction_configs is not None:
        export_prediction_configs = set(args.export_prediction_configs)
        missing_exports = export_prediction_configs - set(
            config["name"] for config in configs)
        if missing_exports:
            raise ValueError("prediction export configs were not selected: %s" %
                             sorted(missing_exports))
    if args.quant_backend == "mixed":
        mixed_manifest_path = model_out_dir / "mixed_precision_configs.csv"
        existing_manifest = read_csv(mixed_manifest_path) if args.append else []
        selected_configs = set(config["name"] for config in configs)
        mixed_manifest = merge_manifest_rows(
            existing_manifest, config_manifest_rows(configs), selected_configs)
        write_csv(mixed_manifest_path, mixed_manifest,
                  ("config", "selection", "module", "activation_bits",
                   "default_activation_bits", "weight_bits", "state_bits"))
    replacing_configs = set(config["name"] for config in configs
                            if config["name"] != "FP32")
    lognp_manifest_rows = [
        row for row in existing_lognp_manifest
        if row.get("config") not in replacing_configs]
    lognp_compensation_rows = [
        row for row in existing_lognp_compensation
        if row.get("config") not in replacing_configs]

    manifest_rows = [
        {"module": name, "group": group}
        for name, group in sorted(instrumentor.module_groups().items())
    ]
    write_csv(model_out_dir / "module_manifest.csv", manifest_rows, ("module", "group"))

    if args.quant_backend == "lognp":
        compensation_modules = select_lognp_compensation_modules(
            saved_args.model, instrumentor.modules,
            args.lognp_sensitivity_root, limit=args.lognp_compensation_limit)
        instrumentor.enable_compensation_capture(
            modules=compensation_modules,
            sample_limit=args.lognp_compensation_samples)
        instrumentor.observe(activation_mode="lognp")
    else:
        instrumentor.observe()
    adapter.observe()
    if merge_adapter is not None:
        merge_adapter.observe()
    t0 = time.time()
    with torch.no_grad():
        for rank, index in enumerate(calibration_indices, 1):
            sample = seeded_sample(trainset, index, args.seed)
            batch = batch_from_sample(sample)
            model_args, _ = sweep.batch_to_model_input(saved_args.model, batch, device)
            model(*model_args)
            if rank % 16 == 0 or rank == calibration_count:
                print("calibration %d/%d" % (rank, calibration_count), flush=True)
    instrumentor.freeze()
    adapter.freeze()
    if merge_adapter is not None:
        merge_adapter.disable()

    calibration_rows = []
    for (name, kind), observer in sorted(instrumentor.observers.items()):
        calibration_rows.append({
            "module": name,
            "group": instrumentor.module_groups()[name],
            "kind": kind,
            "observed": int(observer.observed),
            "minimum": observer.minimum,
            "maximum": observer.maximum,
        })
    if args.quant_backend in ("hardware", "outlier", "mixed", "lognp"):
        for name, observer in sorted(instrumentor.relu_observers.items()):
            calibration_rows.append({
                "module": name,
                "group": "relu",
                "kind": "relu_output",
                "observed": int(observer.observed),
                "minimum": observer.minimum,
                "maximum": observer.maximum,
            })
    write_csv(model_out_dir / "calibration_ranges.csv", calibration_rows,
              ("module", "group", "kind", "observed", "minimum", "maximum"))

    indices = load_sample_indices(args.sample_metrics)
    if args.max_eval_samples:
        indices = indices[:args.max_eval_samples]
    valset = evaluation_dataset(saved_args)
    if existing_metadata is not None:
        if existing_metadata.get("model") != saved_args.model:
            raise ValueError("existing metadata belongs to a different model")
        if existing_metadata.get("evaluation_indices") != indices:
            raise ValueError("append requires identical evaluation indices")
    instrumentor.disable()
    if merge_adapter is not None:
        merge_adapter.disable()
    records = capture_fp32_records(
        model, saved_args, valset, indices, device, adapter, input_capture, args.seed)
    if should_export_predictions("FP32", export_prediction_configs):
        prediction_dir = prepare_prediction_dir(model_out_dir, "FP32")
        for record in records:
            gt_np = record["gt"].numpy()
            fp32_np = record["pred"].numpy()
            write_prediction_payload(prediction_dir, prediction_payload(
                gt_np, fp32_np, fp32_np, record["sample_index"],
                saved_args.model, "FP32"))
    baseline_sample_rows, baseline_region_rows = baseline_rows(saved_args, records)
    sample_rows = replace_config_rows(
        existing_tables["samples"], [], replacing_configs)
    region_rows = replace_config_rows(
        existing_tables["regions"], [], replacing_configs)
    signal_rows = replace_config_rows(
        existing_tables["signals"], [], replacing_configs)
    layer_rows = replace_config_rows(
        existing_tables["layers"], [], replacing_configs)
    state_rows = replace_config_rows(
        existing_tables["states"], [], replacing_configs)
    if not any(row.get("config") == "FP32" for row in sample_rows):
        sample_rows.extend(baseline_sample_rows)
    if not any(row.get("config") == "FP32" for row in region_rows):
        region_rows.extend(baseline_region_rows)
    persist_tables(model_out_dir, sample_rows, region_rows, signal_rows,
                   layer_rows, state_rows)

    for config in configs:
        if config["name"] == "FP32":
            continue
        export_predictions = should_export_predictions(
            config["name"], export_prediction_configs)
        current = evaluate_configuration(
            model, saved_args, records, device, config, instrumentor, adapter,
            input_capture, model_out_dir, export_predictions, merge_adapter)
        sample_rows.extend(current[0])
        region_rows.extend(current[1])
        signal_rows.extend(current[2])
        layer_rows.extend(current[3])
        state_rows.extend(current[4])
        lognp_manifest_rows.extend(current[5])
        lognp_compensation_rows.extend(current[6])
        persist_tables(model_out_dir, sample_rows, region_rows, signal_rows,
                       layer_rows, state_rows)
        if args.quant_backend == "lognp":
            write_csv(model_out_dir / "lognp_manifest.csv",
                      lognp_manifest_rows)
            write_csv(model_out_dir / "lognp_compensation.csv",
                      lognp_compensation_rows)
        print("completed config=%s" % config["name"], flush=True)

    if args.quant_backend in ("hardware", "outlier", "mixed", "lognp"):
        hardware_manifest = []
        for row in hardware_preparation["folded_pairs"]:
            hardware_manifest.append({
                "kind": "conv_bn_fold", "module": row["conv"],
                "target": row["bn"],
            })
        for row in hardware_preparation["unfolded_fanout_pairs"]:
            hardware_manifest.append({
                "kind": "unfolded_bn_fanout", "module": row["conv"],
                "target": row["bn"],
            })
        for row in hardware_preparation["unfolded_conv_bn_pairs"]:
            hardware_manifest.append({
                "kind": "conv_bn_unfolded", "module": row["conv"],
                "target": row["bn"],
            })
        for row in instrumentor.manifest():
            hardware_manifest.append(dict(row, target=""))
        for row in merge_adapter.manifest():
            hardware_manifest.append({
                "kind": "merge", "module": row["merge"], "target": "",
                "bits": row["bits"], "unsigned": row["unsigned"],
                "qmin": row["qmin"], "qmax": row["qmax"],
                "scale": row["scale"], "zero_point": row["zero_point"],
            })
        write_csv(model_out_dir / "hardware_manifest.csv", hardware_manifest,
                  ("kind", "module", "target", "bits", "unsigned",
                   "qmin", "qmax", "scale", "zero_point"))

    completed_configs = [config["name"] for config in configs]
    if existing_metadata is not None:
        completed_configs = list(existing_metadata.get("configs", [])) + [
            name for name in completed_configs
            if name not in existing_metadata.get("configs", [])
        ]
    elapsed = time.time() - t0
    if existing_metadata is not None:
        elapsed += float(existing_metadata.get("elapsed_seconds", 0.0))
    metadata = {
        "model": saved_args.model,
        "iteration": saved_args.iteration,
        "architecture": meta,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "calibration_samples": calibration_count,
        "calibration_indices": calibration_indices,
        "evaluation_samples": len(indices),
        "evaluation_indices": indices,
        "groups": groups,
        "configs": completed_configs,
        "quant_backend": args.quant_backend,
        "quantization_execution": (
            "float_qdq_reference" if args.quant_backend == "lognp"
            else "hardware_aligned_qdq"),
        "elapsed_seconds": elapsed,
        "state_range": {
            "minimum": adapter.controller.observer.minimum,
            "maximum": adapter.controller.observer.maximum,
        },
    }
    if hardware_preparation is not None:
        metadata["hardware_alignment"] = {
            "folded_pairs": hardware_preparation["folded_pairs"],
            "unfolded_fanout_pairs": hardware_preparation[
                "unfolded_fanout_pairs"],
            "unfolded_conv_bn_pairs": hardware_preparation[
                "unfolded_conv_bn_pairs"],
            "folded_fp32_max_abs_error": hardware_preparation["max_abs_error"],
            "merge_sites": len(merge_adapter.manifest()),
            "merge_contract": {
                "concat": "common output scale before consuming integer op",
                "add": "independent input scales and one requantized output scale",
                "direct_concat": "covered by consuming Conv/Linear input QDQ",
            },
            "relu_sites": len(instrumentor.relu_observers),
            "bias_contract": (
                "reference_float_reconstruction"
                if args.quant_backend == "lognp"
                else "int32 scale=sx*sw[o]"),
        }
    write_json(metadata_path, metadata)
    instrumentor.close()
    if merge_adapter is not None:
        merge_adapter.close()
    adapter.close()
    input_capture.close()
    print("model=%s samples=%d out=%s elapsed=%.1fs" % (
        saved_args.model, len(indices), model_out_dir, metadata["elapsed_seconds"]),
        flush=True)


if __name__ == "__main__":
    main()
