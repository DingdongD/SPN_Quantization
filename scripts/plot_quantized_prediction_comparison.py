#!/usr/bin/env python3
"""Render GT, FP32, and hardware-aligned quantized NYU predictions."""

from __future__ import division, print_function

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantized_prediction_analysis import (
    cross_validate_metrics,
    prediction_metrics,
    select_representative_samples,
    validate_sample_sets,
)


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}
SPARSE_CONFIGS = {
    "cspn": "MP_heads_A8",
    "dyspn": "MP_encoder_A8",
    "nlspn": "MP_site01_A8",
    "completionformer": "MP_heads_A8",
}
CONFIG_LABELS = {
    "GT": "GT",
    "FP32": "FP32",
    "MP_W8A8_full": "W8A8",
    "MP_W4A4_base": "W4A4 baseline",
    "MP_heads_A8": "Best sparse A8",
    "MP_encoder_A8": "Best sparse A8",
    "MP_site01_A8": "Best sparse A8",
    "MP_W4A8_full": "W4A8 full",
}
INVALID_GT_RGBA = np.array([0.85, 0.85, 0.85, 1.0])
NONFINITE_RGBA = np.array([1.0, 0.0, 1.0, 1.0])
DEPTH_RANGE = (0.0, 10.0)
ERROR_RANGE = (0.0, 3.0)


def panel_specifications(model):
    configs = (
        "GT", "FP32", "MP_W8A8_full", "MP_W4A4_base",
        SPARSE_CONFIGS[model], "MP_W4A8_full",
    )
    return [{
        "config": config,
        "label": CONFIG_LABELS[config],
        "rotation": 0,
    } for config in configs]


def _semantic_rgba(values, valid_gt, nonfinite, cmap_name, value_range):
    values = np.asarray(values)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    nonfinite = np.asarray(nonfinite, dtype=bool)
    if values.shape != valid_gt.shape or values.shape != nonfinite.shape:
        raise ValueError("values and masks must have identical shapes")
    lo, hi = value_range
    normalized = np.clip((np.nan_to_num(values, nan=lo, posinf=hi,
                                        neginf=lo) - lo) / (hi - lo), 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(normalized)
    rgba[~valid_gt] = INVALID_GT_RGBA
    rgba[valid_gt & nonfinite] = NONFINITE_RGBA
    return rgba


def depth_rgba(depth, valid_gt, nonfinite=None):
    if nonfinite is None:
        nonfinite = np.zeros_like(valid_gt, dtype=bool)
    return _semantic_rgba(
        depth, valid_gt, nonfinite, "viridis", DEPTH_RANGE)


def error_rgba(error, valid_gt, nonfinite=None):
    if nonfinite is None:
        nonfinite = np.zeros_like(valid_gt, dtype=bool)
    return _semantic_rgba(
        error, valid_gt, nonfinite, "magma", ERROR_RANGE)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((field, row.get(field, ""))
                                 for field in fieldnames))


def load_payload(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return dict((key, payload[key]) for key in payload.files)


def model_configs(model):
    return [row["config"] for row in panel_specifications(model)
            if row["config"] != "GT"]


def load_model_predictions(root, model, expected_indices):
    prediction_root = Path(root) / model / "predictions"
    configs = model_configs(model)
    paths = validate_sample_sets(
        prediction_root, configs, expected_indices)
    output = dict((int(index), {}) for index in expected_indices)
    for config in configs:
        for path in paths[config]:
            payload = load_payload(path)
            index = int(payload["sample_index"])
            if str(payload["model"]) != model or str(payload["config"]) != config:
                raise ValueError("embedded metadata mismatch in %s" % path)
            shapes = [payload[key].shape for key in
                      ("gt", "fp32", "pred", "abs_err", "valid_gt", "nonfinite")]
            if len(set(shapes)) != 1:
                raise ValueError("array shape mismatch in %s" % path)
            output[index][config] = payload
    return output


def computed_metric_rows(model, predictions):
    rows = []
    for sample_index in sorted(predictions):
        for config in model_configs(model):
            payload = predictions[sample_index][config]
            row = prediction_metrics(payload["gt"], payload["pred"])
            row.update({
                "model": model,
                "config": config,
                "sample_index": sample_index,
            })
            rows.append(row)
    return rows


def selection_rows(model, metric_rows):
    roles = {
        "MP_W4A4_base": "w4a4",
        SPARSE_CONFIGS[model]: "sparse_a8",
    }
    return [dict(row, role=roles[row["config"]])
            for row in metric_rows if row["config"] in roles]


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def _draw_semantic_key(ax, sample_index, reason, model=None):
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    prefix = (MODEL_NAMES[model] + "\n") if model else ""
    ax.text(0.04, 0.78, "%s#%05d\n%s" %
            (prefix, sample_index, reason.replace("_", " ")),
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.add_patch(plt.Rectangle((0.04, 0.20), 0.09, 0.14,
                               color=INVALID_GT_RGBA, transform=ax.transAxes))
    ax.text(0.16, 0.27, "invalid GT", transform=ax.transAxes,
            va="center", fontsize=8)
    ax.add_patch(plt.Rectangle((0.04, 0.02), 0.09, 0.14,
                               color=NONFINITE_RGBA, transform=ax.transAxes))
    ax.text(0.16, 0.09, "NaN / Inf", transform=ax.transAxes,
            va="center", fontsize=8)


def _draw_comparison(entries, predictions_by_model, out_path):
    nrows = 2 * len(entries)
    fig, axes = plt.subplots(
        nrows, 6, figsize=(15.0, 2.15 * nrows), squeeze=False,
        constrained_layout=True)
    depth_axes = []
    error_axes = []
    for entry_index, entry in enumerate(entries):
        model = entry["model"]
        sample_index = int(entry["sample_index"])
        payloads = predictions_by_model[model][sample_index]
        reference = payloads["FP32"]
        valid_gt = reference["valid_gt"].astype(bool)
        specs = panel_specifications(model)
        depth_row = 2 * entry_index
        error_row = depth_row + 1
        for column, spec in enumerate(specs):
            depth_ax = axes[depth_row, column]
            error_ax = axes[error_row, column]
            depth_axes.append(depth_ax)
            if spec["config"] == "GT":
                rgba = depth_rgba(reference["gt"], valid_gt)
                depth_ax.imshow(rgba, aspect="auto", interpolation="nearest")
                _draw_semantic_key(
                    error_ax, sample_index, entry["reason"],
                    model if len(entries) > 1 and len(set(
                        row["model"] for row in entries)) > 1 else None)
            else:
                payload = payloads[spec["config"]]
                nonfinite = payload["nonfinite"].astype(bool)
                metric = prediction_metrics(payload["gt"], payload["pred"])
                depth_ax.imshow(depth_rgba(
                    payload["pred"], valid_gt, nonfinite),
                    aspect="auto", interpolation="nearest")
                error_ax.imshow(error_rgba(
                    payload["abs_err"], valid_gt, nonfinite),
                    aspect="auto", interpolation="nearest")
                error_axes.append(error_ax)
                depth_ax.set_xlabel(
                    "RMSE %.3f m | invalid %.2f%%" %
                    (metric["RMSE"], 100.0 * metric["nonfinite_rate"]),
                    rotation=0)
            if entry_index == 0:
                depth_ax.set_title(spec["label"], rotation=spec["rotation"])
            for ax in (depth_ax, error_ax):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("#777777")

    depth_map = ScalarMappable(
        norm=Normalize(*DEPTH_RANGE), cmap=plt.get_cmap("viridis"))
    error_map = ScalarMappable(
        norm=Normalize(*ERROR_RANGE), cmap=plt.get_cmap("magma"))
    depth_bar = fig.colorbar(depth_map, ax=depth_axes, fraction=0.010,
                             pad=0.008, aspect=35)
    depth_bar.set_label("Depth (m)")
    error_bar = fig.colorbar(error_map, ax=error_axes, fraction=0.010,
                             pad=0.008, aspect=35)
    error_bar.set_label("Absolute error (m)")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",
                        default="profile_logs/nyu_activation_bit_allocation")
    parser.add_argument(
        "--out-dir",
        default="profile_logs/nyu_activation_bit_allocation/prediction_comparison")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    predictions = {}
    metrics = []
    representatives = []
    common_evaluation = None
    common_calibration = None
    for model in MODEL_ORDER:
        metadata = json.loads(
            (root / model / "metadata.json").read_text(encoding="utf-8"))
        evaluation = [int(index) for index in metadata["evaluation_indices"]]
        calibration = [int(index) for index in metadata["calibration_indices"]]
        if common_evaluation is None:
            common_evaluation = evaluation
            common_calibration = calibration
        if evaluation != common_evaluation or calibration != common_calibration:
            raise ValueError("formal indices differ for %s" % model)
        predictions[model] = load_model_predictions(root, model, evaluation)
        model_metrics = computed_metric_rows(model, predictions[model])
        cross_validate_metrics(
            model_metrics, read_csv(root / model / "sample_metrics.csv"))
        metrics.extend(model_metrics)
        representatives.extend(select_representative_samples(
            selection_rows(model, model_metrics), model))

    write_csv(out_dir / "prediction_metrics.csv", metrics, (
        "model", "config", "sample_index", "RMSE", "MAE",
        "nonfinite_pixels", "num_pixels", "valid_gt_pixels",
        "nonfinite_rate"))
    write_csv(out_dir / "representative_samples.csv", representatives,
              ("model", "sample_index", "reason"))

    for model in MODEL_ORDER:
        selected = [row for row in representatives if row["model"] == model]
        _draw_comparison(
            selected, predictions,
            out_dir / ("%s_quantized_prediction_comparison.png" % model))
        sample_dir = out_dir / "samples" / model
        sample_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in common_evaluation:
            _draw_comparison([{
                "model": model,
                "sample_index": sample_index,
                "reason": "all samples",
            }], predictions, sample_dir / (
                "sample_%05d_quantized_prediction.png" % sample_index))

    overview = []
    for model in MODEL_ORDER:
        overview.append(next(
            row for row in representatives
            if row["model"] == model and row["reason"] == "p90_w4a4"))
    _draw_comparison(
        overview, predictions, out_dir / "quantized_prediction_overview.png")
    print("wrote %d metrics, %d representative samples, and 261 figures to %s" %
          (len(metrics), len(representatives), out_dir))


if __name__ == "__main__":
    main()
