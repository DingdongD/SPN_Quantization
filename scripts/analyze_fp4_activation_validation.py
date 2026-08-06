#!/usr/bin/env python3
"""Validate and summarize the controlled FP4 activation experiment."""

from __future__ import division, print_function

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fp4_activation_validation import (
    FP4_CONFIG_NAMES,
    a4_to_a8_recovery,
    paired_bootstrap_rmse_difference,
)


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
PAIRED_CONFIGS = (
    ("W8", "FP4V_W8A4", "FP4V_W8E2M1", "FP4V_W8A8"),
    ("W4", "FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8"),
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((field, row[field]) for field in fieldnames))


def validate_sample_rows(rows, model, configs, expected_indices):
    expected = set((config, int(index))
                   for config in configs for index in expected_indices)
    observed = []
    for row in rows:
        if row["model"] != model:
            raise ValueError("sample row model mismatch for %s" % model)
        key = (row["config"], int(row["sample_index"]))
        observed.append(key)
        metrics = np.asarray([
            float(row["RMSE"]), float(row["MAE"]),
            float(row["ABS_REL"]),
        ])
        if not np.isfinite(metrics).all():
            raise ValueError("sample rows contain non-finite metrics")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("sample rows do not match required configurations")


def aggregate_sample_rows(rows, model, configs):
    output = []
    for config in configs:
        current = [row for row in rows if row["config"] == config]
        pixels = np.asarray(
            [int(float(row["num_pixels"])) for row in current],
            dtype=np.float64)
        rmse = np.asarray([float(row["RMSE"]) for row in current])
        mae = np.asarray([float(row["MAE"]) for row in current])
        abs_rel = np.asarray([float(row["ABS_REL"]) for row in current])
        nonfinite = np.asarray(
            [int(float(row["nonfinite_pixels"])) for row in current])
        total_pixels = float(pixels.sum())
        output.append({
            "model": model,
            "config": config,
            "samples": len(current),
            "mean_sample_RMSE": float(rmse.mean()),
            "pooled_pixel_RMSE": float(
                np.sqrt(np.sum(rmse ** 2 * pixels) / total_pixels)),
            "pooled_pixel_MAE": float(np.sum(mae * pixels) / total_pixels),
            "pooled_pixel_ABS_REL": float(
                np.sum(abs_rel * pixels) / total_pixels),
            "nonfinite_samples": int(np.count_nonzero(nonfinite)),
            "nonfinite_pixels": int(nonfinite.sum()),
            "finite_pixels": int(total_pixels),
        })
    return output


def _rmse_by_sample(rows, model, config):
    selected = [row for row in rows
                if row["model"] == model and row["config"] == config]
    return dict((int(row["sample_index"]), float(row["RMSE"]))
                for row in selected)


def _nonfinite_by_sample(rows, model, config):
    selected = [row for row in rows
                if row["model"] == model and row["config"] == config]
    return dict((int(row["sample_index"]),
                 int(float(row["nonfinite_pixels"])))
                for row in selected)


def build_paired_comparisons(rows, model, resamples, seed,
                             constraints_clear, diagnostics_clear):
    output = []
    for weight, int4_config, e2m1_config, a8_config in PAIRED_CONFIGS:
        int4 = _rmse_by_sample(rows, model, int4_config)
        e2m1 = _rmse_by_sample(rows, model, e2m1_config)
        a8 = _rmse_by_sample(rows, model, a8_config)
        if set(int4) != set(e2m1) or set(int4) != set(a8):
            raise ValueError("paired sample indices differ for %s %s" %
                             (model, weight))
        indices = sorted(int4)
        statistics = paired_bootstrap_rmse_difference(
            np.asarray([int4[index] for index in indices]),
            np.asarray([e2m1[index] for index in indices]),
            resamples=resamples, seed=seed)
        int4_mean = float(np.mean([int4[index] for index in indices]))
        e2m1_mean = float(np.mean([e2m1[index] for index in indices]))
        a8_mean = float(np.mean([a8[index] for index in indices]))
        recovery = a4_to_a8_recovery(int4_mean, e2m1_mean, a8_mean)
        nonfinite = _nonfinite_by_sample(rows, model, e2m1_config)
        finite_predictions = all(nonfinite[index] == 0 for index in indices)
        effective = (
            finite_predictions and constraints_clear[e2m1_config] and
            diagnostics_clear[e2m1_config] and
            statistics["mean_difference"] < 0.0 and
            statistics["ci_upper"] < 0.0 and
            recovery is not None and recovery > 0.0)
        output.append({
            "model": model,
            "weight_bits": weight,
            "int4_config": int4_config,
            "e2m1_config": e2m1_config,
            "a8_config": a8_config,
            "int4_mean_RMSE": int4_mean,
            "e2m1_mean_RMSE": e2m1_mean,
            "a8_mean_RMSE": a8_mean,
            "mean_difference": statistics["mean_difference"],
            "ci_lower": statistics["ci_lower"],
            "ci_upper": statistics["ci_upper"],
            "recovery": recovery,
            "finite_predictions": finite_predictions,
            "constraints_clear": constraints_clear[e2m1_config],
            "diagnostics_clear": diagnostics_clear[e2m1_config],
            "effective": effective,
        })
    return output


def propagation_constraints(rows, configs):
    output = {}
    for config in configs:
        current = [row for row in rows if row["config"] == config]
        if not current:
            raise ValueError("missing propagation metrics for %s" % config)
        nonfinite = [float(row["nonfinite_ratio"]) for row in current
                     if row["nonfinite_ratio"] != ""]
        coefficient = [float(row["coefficient_sum_max_error"])
                       for row in current
                       if row["coefficient_sum_max_error"] != ""]
        contraction = [float(row["contraction_violation_rate"])
                       for row in current
                       if row["contraction_violation_rate"] != ""]
        if not nonfinite or not coefficient or not contraction:
            raise ValueError("incomplete propagation constraints for %s" %
                             config)
        values = np.asarray(nonfinite + coefficient + contraction)
        if not np.isfinite(values).all():
            raise ValueError("non-finite propagation constraints for %s" %
                             config)
        clear = (max(nonfinite) == 0.0 and
                 max(coefficient) <= 1e-6 and
                 max(contraction) <= 1e-6)
        if "anchor_max_error" in current[0]:
            anchor = [float(row["anchor_max_error"]) for row in current
                      if row["anchor_max_error"] != ""]
            if not anchor:
                raise ValueError("missing anchor metrics for %s" % config)
            clear = clear and max(anchor) <= 1e-6
        output[config] = clear
    return output


def aggregate_layer_rows(rows, model):
    keys = sorted(set((row["config"], row["group"]) for row in rows))
    output = []
    for config, group in keys:
        current = [row for row in rows
                   if row["config"] == config and row["group"] == group]
        numel = sum(int(float(row["numel"])) for row in current)
        error_sq = sum(float(row["error_sq"]) for row in current)
        signal_sq = sum(float(row["signal_sq"]) for row in current)
        sqnr = (10.0 * math.log10(signal_sq / error_sq)
                if error_sq > 0.0 else float("inf"))
        output.append({
            "model": model,
            "config": config,
            "group": group,
            "sqnr_db": sqnr,
            "zero_code_rate": sum(
                float(row["zero_code_rate"]) * int(float(row["numel"]))
                for row in current) / float(numel),
            "saturation_rate": sum(
                float(row["saturation_rate"]) * int(float(row["numel"]))
                for row in current) / float(numel),
            "nonfinite_rate": sum(
                float(row["nonfinite_rate"]) * int(float(row["numel"]))
                for row in current) / float(numel),
            "numel": numel,
        })
    return output


def aggregate_propagation_steps(rows, model):
    selected = [row for row in rows if row["signal"] == "propagation_states"]
    keys = sorted(set((row["config"], int(row["iteration"]))
                      for row in selected))
    output = []
    for config, iteration in keys:
        current = [row for row in selected
                   if row["config"] == config and
                   int(row["iteration"]) == iteration]
        rmse = np.asarray([float(row["rmse"]) for row in current])
        if not np.isfinite(rmse).all():
            raise ValueError("non-finite propagation-step RMSE")
        output.append({
            "model": model,
            "config": config,
            "iteration": iteration,
            "mean_RMSE": float(rmse.mean()),
            "samples": len(current),
        })
    return output


def propagation_diagnostics_clear(rows, config):
    current = [row for row in rows if row["config"] == config]
    initial = [float(row["rmse"]) for row in current
               if row["signal"] == "pred_init"]
    final = [float(row["rmse"]) for row in current
             if row["signal"] == "pred"]
    states = [row for row in current
              if row["signal"] == "propagation_states"]
    if not initial or not final or not states:
        raise ValueError("incomplete propagation diagnostics for %s" % config)
    iterations = sorted(set(int(row["iteration"]) for row in states))
    first = [float(row["rmse"]) for row in states
             if int(row["iteration"]) == iterations[0]]
    last = [float(row["rmse"]) for row in states
            if int(row["iteration"]) == iterations[-1]]
    values = np.asarray(initial + final + first + last)
    if not np.isfinite(values).all():
        return False
    return (float(np.mean(final)) <= float(np.mean(initial)) and
            float(np.mean(last)) <= float(np.mean(first)))


def validate_metadata(metadata, model, expected_samples):
    if metadata["model"] != model or metadata["quant_backend"] != "fp4":
        raise ValueError("FP4 metadata identity mismatch for %s" % model)
    if tuple(metadata["configs"]) != FP4_CONFIG_NAMES:
        raise ValueError("FP4 configuration order mismatch for %s" % model)
    if int(metadata["evaluation_samples"]) != int(expected_samples):
        raise ValueError("evaluation sample count mismatch for %s" % model)
    if len(metadata["evaluation_indices"]) != int(expected_samples):
        raise ValueError("evaluation index count mismatch for %s" % model)
    contract = metadata["fp4_validation"]
    if contract["activation_format"] != "scaled_e2m1_rne":
        raise ValueError("E2M1 contract mismatch for %s" % model)
    if contract["bias_contract"] != "fp32_isolation":
        raise ValueError("bias contract mismatch for %s" % model)
    if contract["propagation_signals"] != "a8":
        raise ValueError("propagation contract mismatch for %s" % model)
    provenance = metadata["model_provenance"]
    for field in ("source_git_commit", "source_sha256",
                  "checkpoint_sha256", "source_path"):
        if not provenance[field]:
            raise ValueError("missing provenance %s for %s" % (field, model))


def validate_prediction_payloads(model_root, model, configs,
                                 expected_indices):
    expected = set(int(index) for index in expected_indices)
    for config in configs:
        paths = sorted((model_root / "predictions" / config).glob(
            "sample_*.npz"))
        observed = []
        for path in paths:
            with np.load(str(path), allow_pickle=False) as payload:
                index = int(payload["sample_index"])
                observed.append(index)
                if str(payload["model"]) != model:
                    raise ValueError("prediction model mismatch in %s" % path)
                if str(payload["config"]) != config:
                    raise ValueError("prediction config mismatch in %s" % path)
                arrays = [payload[field] for field in
                          ("gt", "fp32", "pred", "abs_err",
                           "valid_gt", "nonfinite")]
                if len(set(array.shape for array in arrays)) != 1:
                    raise ValueError("prediction shape mismatch in %s" % path)
                if not np.isfinite(payload["pred"]).all():
                    raise ValueError("non-finite prediction in %s" % path)
                if np.asarray(payload["nonfinite"], dtype=bool).any():
                    raise ValueError("non-finite mask is set in %s" % path)
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("prediction sample mismatch for %s %s" %
                             (model, config))


def analyze(root, out_dir, expected_samples, resamples, seed):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []
    comparison_rows = []
    layer_rows = []
    step_rows = []
    shared_calibration = None
    shared_evaluation = None

    for model in MODEL_ORDER:
        model_root = root / model
        metadata = json.loads(
            (model_root / "metadata.json").read_text(encoding="utf-8"))
        validate_metadata(metadata, model, expected_samples)
        calibration = tuple(int(index)
                            for index in metadata["calibration_indices"])
        evaluation = tuple(int(index)
                           for index in metadata["evaluation_indices"])
        if shared_calibration is None:
            shared_calibration = calibration
            shared_evaluation = evaluation
        if calibration != shared_calibration or evaluation != shared_evaluation:
            raise ValueError("formal sample indices differ for %s" % model)

        samples = read_csv(model_root / "sample_metrics.csv")
        validate_sample_rows(samples, model, FP4_CONFIG_NAMES, evaluation)
        validate_prediction_payloads(
            model_root, model, FP4_CONFIG_NAMES, evaluation)
        constraints = propagation_constraints(
            read_csv(model_root / "propagation_quantization_metrics.csv"),
            FP4_CONFIG_NAMES[1:])
        signals = read_csv(model_root / "signal_metrics.csv")
        diagnostics = dict((config, propagation_diagnostics_clear(
            signals, config)) for config in FP4_CONFIG_NAMES[1:])
        aggregate_rows.extend(aggregate_sample_rows(
            samples, model, FP4_CONFIG_NAMES))
        comparison_rows.extend(build_paired_comparisons(
            samples, model, resamples, seed, constraints, diagnostics))
        layer_rows.extend(aggregate_layer_rows(
            read_csv(model_root / "layer_quantization_metrics.csv"), model))
        step_rows.extend(aggregate_propagation_steps(
            signals, model))

    write_csv(out_dir / "configuration_summary.csv", aggregate_rows, (
        "model", "config", "samples", "mean_sample_RMSE",
        "pooled_pixel_RMSE", "pooled_pixel_MAE", "pooled_pixel_ABS_REL",
        "nonfinite_samples", "nonfinite_pixels", "finite_pixels"))
    write_csv(out_dir / "paired_comparisons.csv", comparison_rows, (
        "model", "weight_bits", "int4_config", "e2m1_config", "a8_config",
        "int4_mean_RMSE", "e2m1_mean_RMSE", "a8_mean_RMSE",
        "mean_difference", "ci_lower", "ci_upper", "recovery",
        "finite_predictions", "constraints_clear", "diagnostics_clear",
        "effective"))
    write_csv(out_dir / "group_activation_summary.csv", layer_rows, (
        "model", "config", "group", "sqnr_db", "zero_code_rate",
        "saturation_rate", "nonfinite_rate", "numel"))
    write_csv(out_dir / "propagation_step_summary.csv", step_rows, (
        "model", "config", "iteration", "mean_RMSE", "samples"))
    report = {
        "models": list(MODEL_ORDER),
        "configs": list(FP4_CONFIG_NAMES),
        "calibration_indices": list(shared_calibration),
        "evaluation_indices": list(shared_evaluation),
        "expected_samples": int(expected_samples),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "comparisons": comparison_rows,
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--bootstrap-resamples", required=True, type=int)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    args = parser.parse_args()
    report = analyze(
        args.root, args.out_dir, args.expected_samples,
        args.bootstrap_resamples, args.bootstrap_seed)
    effective = sum(int(row["effective"])
                    for row in report["comparisons"])
    print("validated %d models and %d paired comparisons; %d effective" %
          (len(report["models"]), len(report["comparisons"]), effective))


if __name__ == "__main__":
    main()
