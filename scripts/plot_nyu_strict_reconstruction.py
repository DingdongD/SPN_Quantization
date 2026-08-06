#!/usr/bin/env python3
"""Summarize strict AdaRound/BRECQ W4A8 deployment evaluations."""

from __future__ import division, print_function

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}
METHOD_ORDER = ("rtn", "adaround", "brecq")
METHOD_NAMES = {
    "rtn": "RTN W4A8",
    "adaround": "AdaRound W4A8",
    "brecq": "BRECQ W4A8",
}
METHOD_COLORS = {
    "rtn": "#5B8FF9",
    "adaround": "#61A534",
    "brecq": "#E45756",
}
DEPTH_RANGE = (0.0, 10.0)
ERROR_RANGE = (0.0, 3.0)
INVALID_GT_RGBA = np.array([0.85, 0.85, 0.85, 1.0])
NONFINITE_RGBA = np.array([1.0, 0.0, 1.0, 1.0])


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config_rows(root, method, model, config):
    rows = read_csv(
        Path(root) / method / model / "sample_metrics.csv")
    selected = [row for row in rows if row["config"] == config]
    if not selected:
        raise ValueError(
            "missing %s rows for %s/%s" % (config, method, model))
    return selected


def _aggregate(rows):
    values = [float(row["RMSE"]) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    invalid_samples = sum(
        int(float(row["nonfinite_pixels"])) > 0 or
        not math.isfinite(float(row["RMSE"]))
        for row in rows)
    if not finite:
        mean_rmse = float("inf")
    else:
        mean_rmse = round(sum(finite) / len(finite), 12)
    return mean_rmse, invalid_samples


def summarize_evaluation(root, models=MODEL_ORDER, methods=METHOD_ORDER):
    output = []
    for model in models:
        rtn_rows = _config_rows(root, "rtn", model, "HW_W4A8_full")
        rtn_rmse, rtn_invalid = _aggregate(rtn_rows)
        for method in methods:
            fp32_rows = _config_rows(root, method, model, "FP32")
            quant_rows = _config_rows(
                root, method, model, "HW_W4A8_full")
            fp32_rmse, fp32_invalid = _aggregate(fp32_rows)
            mean_rmse, invalid_samples = _aggregate(quant_rows)
            if fp32_invalid:
                raise ValueError("FP32 contains nonfinite samples: %s" % model)
            if method == "rtn":
                status = "baseline"
            elif invalid_samples:
                status = "rejected_nonfinite"
            elif rtn_invalid or mean_rmse <= rtn_rmse:
                status = "accepted"
            else:
                status = "rejected_regression"
            output.append({
                "model": model,
                "method": method,
                "label": METHOD_NAMES[method],
                "fp32_rmse": fp32_rmse,
                "mean_rmse": mean_rmse,
                "delta_vs_fp32": mean_rmse - fp32_rmse,
                "delta_vs_rtn": mean_rmse - rtn_rmse,
                "invalid_samples": invalid_samples,
                "evaluation_samples": len(quant_rows),
                "deployment_status": status,
            })
    return output


def representative_index(root, model, methods=METHOD_ORDER):
    by_method = {}
    common = None
    for method in methods:
        rows = _config_rows(root, method, model, "HW_W4A8_full")
        values = dict(
            (int(row["sample_index"]), float(row["RMSE"]))
            for row in rows)
        indices = set(values)
        if common is None:
            common = indices
        elif indices != common:
            raise ValueError("evaluation indices differ for %s" % model)
        by_method[method] = values

    def score(index):
        values = [by_method[method][index] for method in methods]
        if any(not math.isfinite(value) for value in values):
            return float("inf")
        return max(values) - min(values)

    return max(sorted(common), key=score)


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlepad": 8,
    })


def plot_rmse(path, rows):
    lookup = dict(
        ((row["model"], row["method"]), row) for row in rows)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    axis.set_axisbelow(True)
    for offset, method in enumerate(METHOD_ORDER):
        values = [lookup[model, method]["mean_rmse"]
                  for model in MODEL_ORDER]
        bars = axis.bar(
            x + (offset - 1) * width, values, width,
            color=METHOD_COLORS[method], label=METHOD_NAMES[method],
            zorder=3)
        for model, bar in zip(MODEL_ORDER, bars):
            row = lookup[model, method]
            if row["invalid_samples"]:
                bar.set_hatch("//")
                bar.set_edgecolor("#8B0000")
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.08,
                "%.3f" % row["mean_rmse"],
                ha="center", va="bottom", fontsize=9,
                rotation=0, zorder=4)
    axis.set_yscale("log")
    axis.set_ylabel("RMSE (m, log scale)")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [MODEL_NAMES[model] for model in MODEL_ORDER], rotation=0)
    axis.grid(axis="y", alpha=0.25, zorder=0)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    finite = [row["mean_rmse"] for row in rows
              if math.isfinite(row["mean_rmse"])]
    axis.set_ylim(0.09, max(finite) * 2.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_payload(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return dict((name, payload[name]) for name in payload.files)


def _prediction_path(root, method, model, config, sample_index):
    return (Path(root) / method / model / "predictions" / config /
            ("sample_%05d.npz" % sample_index))


def _semantic_rgba(values, valid_gt, nonfinite, cmap_name, value_range):
    lower, upper = value_range
    normalized = np.clip(
        (np.nan_to_num(values, nan=lower, posinf=upper, neginf=lower) -
         lower) / (upper - lower), 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(normalized)
    rgba[~valid_gt] = INVALID_GT_RGBA
    rgba[valid_gt & nonfinite] = NONFINITE_RGBA
    return rgba


def plot_predictions(path, root):
    columns = (
        "GT", "FP32", "RTN W4A8", "AdaRound W4A8", "BRECQ W4A8",
        "RTN |error|", "AdaRound |error|", "BRECQ |error|",
    )
    fig, axes = plt.subplots(
        len(MODEL_ORDER), len(columns),
        figsize=(20.0, 2.55 * len(MODEL_ORDER)), squeeze=False)
    for row_index, model in enumerate(MODEL_ORDER):
        sample_index = representative_index(root, model)
        fp32 = _load_payload(_prediction_path(
            root, "rtn", model, "FP32", sample_index))
        quantized = dict(
            (method, _load_payload(_prediction_path(
                root, method, model, "HW_W4A8_full", sample_index)))
            for method in METHOD_ORDER)
        valid_gt = fp32["valid_gt"].astype(bool)
        panels = [
            (fp32["gt"], np.zeros_like(valid_gt), False),
            (fp32["pred"], fp32["nonfinite"].astype(bool), False),
        ]
        for method in METHOD_ORDER:
            payload = quantized[method]
            panels.append((
                payload["pred"], payload["nonfinite"].astype(bool), False))
        for method in METHOD_ORDER:
            payload = quantized[method]
            panels.append((
                payload["abs_err"], payload["nonfinite"].astype(bool), True))

        for column_index, (values, nonfinite, is_error) in enumerate(panels):
            axis = axes[row_index, column_index]
            rgba = _semantic_rgba(
                values, valid_gt, nonfinite,
                "magma" if is_error else "viridis",
                ERROR_RANGE if is_error else DEPTH_RANGE)
            axis.imshow(rgba, aspect="auto", interpolation="nearest", zorder=2)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(columns[column_index])
            if column_index == 0:
                axis.set_ylabel(
                    "%s\n#%05d" % (MODEL_NAMES[model], sample_index),
                    rotation=0, ha="right", va="center")
    depth_map = ScalarMappable(
        norm=Normalize(*DEPTH_RANGE), cmap=plt.get_cmap("viridis"))
    error_map = ScalarMappable(
        norm=Normalize(*ERROR_RANGE), cmap=plt.get_cmap("magma"))
    fig.subplots_adjust(
        left=0.08, right=0.92, bottom=0.05, top=0.94,
        wspace=0.20, hspace=0.20)
    depth_bar = fig.colorbar(
        depth_map, cax=fig.add_axes([0.94, 0.55, 0.008, 0.34]))
    depth_bar.set_label("Depth (m)")
    error_bar = fig.colorbar(
        error_map, cax=fig.add_axes([0.94, 0.12, 0.008, 0.34]))
    error_bar.set_label("Absolute error (m)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(path, rows):
    lines = [
        "# Strict W4A8 Deployment Evaluation",
        "",
        "Acceptance requires zero nonfinite samples and mean RMSE no worse "
        "than the same-run RTN W4A8 baseline.",
        "",
        "| Model | Method | FP32 RMSE | W4A8 RMSE | Invalid | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %.6f | %.6f | %d/%d | %s |" % (
                MODEL_NAMES[row["model"]], row["label"],
                row["fp32_rmse"], row["mean_rmse"],
                row["invalid_samples"], row["evaluation_samples"],
                row["deployment_status"]))
    lines.extend([
        "",
        "Hatched bars indicate at least one sample with a nonfinite output.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True,
        help="evaluation root containing rtn, adaround, and brecq")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    rows = summarize_evaluation(root)
    write_csv(out_dir / "strict_w4a8_summary.csv", rows)
    write_report(out_dir / "strict_w4a8_deployment_report.md", rows)
    plot_rmse(out_dir / "strict_w4a8_rmse.png", rows)
    plot_predictions(out_dir / "strict_w4a8_predictions.png", root)
    print("summary=%s" % out_dir, flush=True)


if __name__ == "__main__":
    main()
