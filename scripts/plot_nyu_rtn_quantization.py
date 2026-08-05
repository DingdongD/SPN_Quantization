#!/usr/bin/env python3
"""Aggregate and plot four-model NYU RTN quantization results."""

from __future__ import print_function

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_quantization_analysis import MODULE_GROUP_ORDER  # noqa: E402
from scripts.visualize_nyu_prediction_comparison import collect  # noqa: E402


MODEL_ORDER = ["cspn", "dyspn", "nlspn", "completionformer"]
FULL_CONFIGS = ["FP32", "W8A8_full", "W4A8_full", "W4A4_full"]
QUANTIZED_FULL_CONFIGS = FULL_CONFIGS[1:]
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}
SIGNAL_ORDER = ["pred", "pred_init", "guidance", "affinity", "offset", "confidence"]
SIGNAL_NAMES = {
    "pred": "Final depth",
    "pred_init": "Initial depth",
    "guidance": "Guide",
    "affinity": "Affinity",
    "offset": "Offset",
    "confidence": "Conf.",
}


def read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    return list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))


def collect_tables(root):
    root = Path(root)
    mapping = {
        "regional": "regional_metrics.csv",
        "samples": "sample_metrics.csv",
        "signals": "signal_metrics.csv",
        "layers": "layer_quantization_metrics.csv",
        "states": "state_quantization_metrics.csv",
    }
    tables = dict((key, []) for key in mapping)
    for model in MODEL_ORDER:
        for key, filename in mapping.items():
            tables[key].extend(read_csv(root / model / filename))
    return tables


def full_quantization_summary(regional_rows, sample_rows):
    baselines = {}
    for row in regional_rows:
        if row.get("region") == "all" and row.get("config") == "FP32":
            baselines[row["model"]] = float(row["RMSE"])
    nonfinite = {}
    for row in sample_rows:
        key = (row["model"], row["config"])
        values = nonfinite.setdefault(key, [0, 0])
        bad = int(float(row.get("nonfinite_pixels", 0)))
        finite = int(float(row.get("num_pixels", 0)))
        values[0] += bad
        values[1] += bad + finite
    output = []
    for row in regional_rows:
        if row.get("region") != "all" or row.get("config") not in FULL_CONFIGS:
            continue
        model = row["model"]
        config = row["config"]
        rmse = float(row["RMSE"])
        bad, total = nonfinite.get((model, config), (0, 0))
        output.append({
            "model": model,
            "config": config,
            "RMSE": rmse,
            "delta_percent": 100.0 * (rmse - baselines[model]) / baselines[model],
            "nonfinite_pixels": bad,
            "nonfinite_rate": bad / float(total) if total else 0.0,
        })
    return sorted(output, key=lambda row: (
        MODEL_ORDER.index(row["model"]), FULL_CONFIGS.index(row["config"])))


def module_sensitivity_rows(regional_rows, sample_rows=None, bits=4):
    prefix = "W%dA%d_" % (bits, bits)
    suffix = "_only"
    baselines = dict((row["model"], float(row["RMSE"]))
                     for row in regional_rows
                     if row.get("region") == "all" and row.get("config") == "FP32")
    nonfinite = {}
    for row in sample_rows or []:
        key = (row["model"], row["config"])
        values = nonfinite.setdefault(key, [0, 0])
        bad = int(float(row.get("nonfinite_pixels", 0)))
        finite = int(float(row.get("num_pixels", 0)))
        values[0] += bad
        values[1] += bad + finite
    output = []
    for row in regional_rows:
        config = row.get("config", "")
        if row.get("region") != "all" or not config.startswith(prefix) \
                or not config.endswith(suffix):
            continue
        group = config[len(prefix):-len(suffix)]
        rmse = float(row["RMSE"])
        bad, total = nonfinite.get((row["model"], config), (0, 0))
        output.append({
            "model": row["model"],
            "group": group,
            "bits": bits,
            "RMSE": rmse,
            "rmse_ratio": rmse / baselines[row["model"]],
            "nonfinite_pixels": bad,
            "nonfinite_rate": bad / float(total) if total else 0.0,
        })
    return sorted(output, key=lambda row: (
        MODEL_ORDER.index(row["model"]), MODULE_GROUP_ORDER.index(row["group"])))


def median(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def median_optional(values):
    return median([value for value in values if value not in (None, "")])


def signal_summary_rows(rows):
    grouped = {}
    for row in rows:
        if row.get("config") not in QUANTIZED_FULL_CONFIGS \
                or row.get("iteration") != "0":
            continue
        key = (row["model"], row["config"], row["signal"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (model, config, signal), points in sorted(grouped.items()):
        output.append({
            "model": model,
            "config": config,
            "signal": signal,
            "median_sqnr_db": median([point["sqnr_db"] for point in points]),
            "median_cosine": median([point["cosine"] for point in points]),
            "median_sign_flip_rate": median([point["sign_flip_rate"] for point in points]),
            "median_dominant_neighbor_change_rate": median_optional([
                point.get("dominant_neighbor_change_rate") for point in points]),
            "median_endpoint_error": median_optional([
                point.get("endpoint_error") for point in points]),
        })
    return output


def propagation_drift_rows(rows):
    configs = {"W8A8_full", "W4A4_full", "W8A8_full_stateA8", "W4A4_full_stateA4"}
    grouped = {}
    for row in rows:
        if row.get("signal") != "propagation_states" or row.get("config") not in configs:
            continue
        key = (row["model"], row["config"], int(row["iteration"]))
        grouped.setdefault(key, []).append(float(row["rmse"]))
    return [
        {"model": key[0], "config": key[1], "iteration": key[2],
         "median_state_RMSE": median(values)}
        for key, values in sorted(grouped.items())
    ]


def regional_ratio_rows(rows):
    configs = tuple(QUANTIZED_FULL_CONFIGS)
    baselines = dict(((row["model"], row["region"]), float(row["RMSE"]))
                     for row in rows if row.get("config") == "FP32")
    output = []
    for row in rows:
        if row.get("config") not in configs:
            continue
        baseline = baselines.get((row["model"], row["region"]))
        if baseline is None or baseline == 0.0:
            continue
        rmse = float(row["RMSE"])
        output.append({
            "model": row["model"],
            "config": row["config"],
            "region": row["region"],
            "RMSE": rmse,
            "baseline_RMSE": baseline,
            "rmse_delta": rmse - baseline,
            "rmse_ratio": rmse / baseline,
        })
    return output


def layer_group_rows(rows):
    grouped = {}
    for row in rows:
        if row.get("config") not in QUANTIZED_FULL_CONFIGS:
            continue
        key = (row["model"], row["config"], row["group"], row["kind"])
        grouped.setdefault(key, []).append(row)
    output = []
    for key, points in sorted(grouped.items()):
        output.append({
            "model": key[0], "config": key[1], "group": key[2], "kind": key[3],
            "median_sqnr_db": median([point["sqnr_db"] for point in points]),
            "max_saturation_rate": max(float(point["saturation_rate"]) for point in points),
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def heatmap_annotation_color(value, minimum, maximum):
    rgba = plt.get_cmap("magma")(LogNorm(vmin=minimum, vmax=maximum)(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.55 else "white"


def format_heatmap_value(value, panel_max):
    if panel_max < 0.1:
        return "%.3f" % value
    if panel_max < 1.0:
        return "%.2f" % value
    return "%.1f" % value


def heatmap_figure_size(panel_count):
    return 6.4 * panel_count, 4.8


def plot_full_rmse(path, rows):
    configs = FULL_CONFIGS
    labels = ["FP32", "W8A8", "W4A8", "W4A4"]
    colors = ["#777777", "#2A9D8F", "#E9C46A", "#E76F51"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.set_axisbelow(True)
    lookup = dict(((row["model"], row["config"]), row) for row in rows)
    for index, (config, label, color) in enumerate(zip(configs, labels, colors)):
        values = [lookup[model, config]["RMSE"] for model in MODEL_ORDER]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label,
                      color=color, zorder=3)
        for model, bar in zip(MODEL_ORDER, bars):
            row = lookup[model, config]
            annotation = "%.2fx" % (row["RMSE"] / lookup[model, "FP32"]["RMSE"])
            if row["nonfinite_rate"] > 0:
                annotation += "\n%.1f%% NaN/Inf" % (100 * row["nonfinite_rate"])
                bar.set_hatch("//")
                bar.set_edgecolor("#8B0000")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
                    annotation, ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (m, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_ylim(0.09, max(row["RMSE"] for row in rows) * 1.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_module_heatmap(path, rows):
    groups = MODULE_GROUP_ORDER
    matrix = np.full((len(MODEL_ORDER), len(groups)), np.nan)
    for row in rows:
        matrix[MODEL_ORDER.index(row["model"]), groups.index(row["group"])] = row["rmse_ratio"]
    lookup = dict(((row["model"], row["group"]), row) for row in rows)
    finite = matrix[np.isfinite(matrix)]
    scale_min = 1.0
    scale_max = max(1.01, float(np.max(finite)))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="magma",
                      norm=LogNorm(vmin=scale_min, vmax=scale_max),
                      aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                color = heatmap_annotation_color(
                    matrix[row, col], scale_min, scale_max)
                details = lookup[MODEL_ORDER[row], groups[col]]
                annotation = "%.2fx" % matrix[row, col]
                if details["nonfinite_rate"] > 0:
                    annotation += "\n%.1f%% NaN/Inf" % (
                        100.0 * details["nonfinite_rate"])
                ax.text(col, row, annotation, ha="center", va="center",
                        color=color, fontsize=10)
            else:
                ax.text(col, row, "-", ha="center", va="center", color="#666666")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([name.replace("_", "\n") for name in groups])
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("W4A4 RMSE / FP32 RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heatmap(path, rows, value_key, configs, columns, column_names,
             colorbar_label, limits=None):
    fig, axes = plt.subplots(
        1, len(configs), figsize=heatmap_figure_size(len(configs)), squeeze=False)
    for axis, config in zip(axes[0], configs):
        matrix = np.full((len(MODEL_ORDER), len(columns)), np.nan)
        for row in rows:
            if row["config"] == config and row.get("signal", row.get("region")) in columns:
                column = row.get("signal", row.get("region"))
                matrix[MODEL_ORDER.index(row["model"]), columns.index(column)] = row[value_key]
        finite = matrix[np.isfinite(matrix)]
        if limits is None:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
        else:
            vmin, vmax = limits[config]
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm",
                            vmin=vmin, vmax=vmax, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                text = format_heatmap_value(value, vmax) if np.isfinite(value) else "-"
                axis.text(col_index, row_index, text, ha="center", va="center", fontsize=9)
        axis.set_xticks(np.arange(len(columns)))
        axis.set_xticklabels(column_names)
        axis.set_yticks(np.arange(len(MODEL_ORDER)))
        axis.set_yticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
        axis.set_title(config.replace("_full", ""))
        colorbar = fig.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_signal_sqnr(path, rows):
    _heatmap(
        path, rows, "median_sqnr_db", QUANTIZED_FULL_CONFIGS,
        SIGNAL_ORDER, [SIGNAL_NAMES[name].replace(" ", "\n") for name in SIGNAL_ORDER],
        "Median SQNR (dB)",
        limits={"W8A8_full": (0, 50), "W4A8_full": (-10, 40),
                "W4A4_full": (-10, 25)},
    )


def plot_regional_ratios(path, rows):
    regions = ["boundary", "smooth", "sparse_anchor", "holes"]
    _heatmap(
        path, rows, "rmse_delta", QUANTIZED_FULL_CONFIGS, regions,
        ["Boundary", "Smooth", "Sparse\nanchors", "Holes"],
        "RMSE increase (m)",
    )


def plot_propagation_drift(path, rows):
    colors = {
        "W8A8_full": "#2A9D8F",
        "W4A4_full": "#E76F51",
        "W8A8_full_stateA8": "#277DA1",
        "W4A4_full_stateA4": "#9D4EDD",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), squeeze=False)
    for axis, model in zip(axes.ravel(), MODEL_ORDER):
        for config, color in colors.items():
            points = sorted([row for row in rows
                             if row["model"] == model and row["config"] == config],
                            key=lambda row: row["iteration"])
            if points:
                axis.plot([row["iteration"] for row in points],
                          [row["median_state_RMSE"] for row in points],
                          marker="o", markersize=3, linewidth=1.8,
                          label=config.replace("_full", ""), color=color)
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Propagation iteration")
        axis.set_ylabel("State drift RMSE (m)")
        axis.grid(alpha=0.25, zorder=0)
        axis.set_axisbelow(True)
    handles = [Line2D([0], [0], color=color, marker="o", linewidth=1.8,
                      label=config.replace("_full", ""))
               for config, color in colors.items()]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_prediction_examples(path, quant_root, fp_pred_dir, indices):
    fp = collect(fp_pred_dir)
    rows = len(indices) * len(MODEL_ORDER)
    fig, axes = plt.subplots(rows, 7, figsize=(19, 2.0 * rows), squeeze=False)
    titles = ["Ground truth", "FP32", "W8A8", "W4A8", "W4A4",
              "W4A8 |error|", "W4A4 |error|"]
    row_index = 0
    for sample_index in indices:
        for model in MODEL_ORDER:
            item = fp[sample_index][model]
            w8 = np.load(str(Path(quant_root) / model / "predictions" / "W8A8_full"
                             / ("sample_%05d.npz" % sample_index)), allow_pickle=False)
            w4 = np.load(str(Path(quant_root) / model / "predictions" / "W4A4_full"
                             / ("sample_%05d.npz" % sample_index)), allow_pickle=False)
            w4a8 = np.load(str(Path(quant_root) / model / "predictions" / "W4A8_full"
                               / ("sample_%05d.npz" % sample_index)), allow_pickle=False)
            valid = item["valid"]
            panels = [
                (np.ma.masked_where(~valid, item["gt"]), "viridis", 0.0, 10.0),
                (np.ma.masked_where(~valid, item["pred"]), "viridis", 0.0, 10.0),
                (np.ma.masked_where(~valid, w8["pred"]), "viridis", 0.0, 10.0),
                (np.ma.masked_where(~valid, w4a8["pred"]), "viridis", 0.0, 10.0),
                (np.ma.masked_where(~valid, w4["pred"]), "viridis", 0.0, 10.0),
                (np.ma.masked_where(~valid, w4a8["abs_err"]), "magma", 0.0, 3.0),
                (np.ma.masked_where(~valid, w4["abs_err"]), "magma", 0.0, 3.0),
            ]
            for col, (image, cmap, vmin, vmax) in enumerate(panels):
                axes[row_index, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax,
                                            aspect="auto", interpolation="nearest")
                axes[row_index, col].set_xticks([])
                axes[row_index, col].set_yticks([])
                if row_index == 0:
                    axes[row_index, col].set_title(titles[col])
            axes[row_index, 0].set_ylabel("#%05d\n%s" % (
                sample_index, MODEL_NAMES[model]), rotation=0, ha="right", va="center")
            row_index += 1
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="profile_logs/nyu_rtn_quantization")
    parser.add_argument("--fp-pred-dir",
                        default="profile_logs/nyu_prediction_random64_fp32/predictions")
    parser.add_argument("--example-indices", nargs="+", type=int,
                        default=[386, 558, 384, 542])
    args = parser.parse_args()
    root = Path(args.root)
    set_style()
    tables = collect_tables(root)
    full = full_quantization_summary(tables["regional"], tables["samples"])
    modules = module_sensitivity_rows(tables["regional"], tables["samples"], bits=4)
    signals = signal_summary_rows(tables["signals"])
    drift = propagation_drift_rows(tables["signals"])
    regional = regional_ratio_rows(tables["regional"])
    layers = layer_group_rows(tables["layers"])
    write_csv(root / "full_quantization_summary.csv", full)
    write_csv(root / "module_sensitivity_w4a4.csv", modules)
    write_csv(root / "signal_damage_summary.csv", signals)
    write_csv(root / "propagation_drift_summary.csv", drift)
    write_csv(root / "regional_sensitivity_summary.csv", regional)
    write_csv(root / "layer_group_summary.csv", layers)
    plot_full_rmse(root / "full_quantization_rmse.png", full)
    plot_module_heatmap(root / "w4a4_module_sensitivity_heatmap.png", modules)
    plot_signal_sqnr(root / "signal_sqnr_full_quantization.png", signals)
    plot_regional_ratios(root / "regional_rmse_increase.png", regional)
    plot_propagation_drift(root / "propagation_state_drift.png", drift)
    plot_prediction_examples(root / "quantized_prediction_examples.png", root,
                             args.fp_pred_dir, args.example_indices)
    print("plots=%s" % root, flush=True)


if __name__ == "__main__":
    main()
