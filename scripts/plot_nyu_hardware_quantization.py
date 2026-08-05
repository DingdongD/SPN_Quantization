#!/usr/bin/env python3
"""Compare unfused RTN and hardware-aligned NYU quantization results."""

from __future__ import print_function

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_nyu_rtn_quantization import (
    MODEL_NAMES,
    MODEL_ORDER,
    SIGNAL_NAMES,
    SIGNAL_ORDER,
    collect_tables,
)
from scripts.nyu_quantization_analysis import MODULE_GROUP_ORDER
from scripts.visualize_nyu_prediction_comparison import collect


CONFIG_SPECS = (
    ("FP32", "FP32", "rtn"),
    ("W4A8_full", "RTN W4A8", "rtn"),
    ("HW_W4A8_full", "HW W4A8", "hardware"),
    ("W4A4_full", "RTN W4A4", "rtn"),
    ("HW_W4A4_full", "HW W4A4", "hardware"),
)
HARDWARE_CONFIGS = ("HW_W4A8_full", "HW_W4A4_full")


def _median(values):
    finite = [float(value) for value in values
              if value not in (None, "") and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def _nonfinite_rates(rows):
    totals = {}
    for row in rows:
        key = (row["model"], row["config"])
        bad, valid = totals.setdefault(key, [0, 0])
        totals[key][0] = bad + int(float(row.get("nonfinite_pixels", 0)))
        totals[key][1] = valid + int(float(row.get("num_pixels", 0)))
    return dict((key, bad / float(bad + valid) if bad + valid else 0.0)
                for key, (bad, valid) in totals.items())


def comparison_summary_rows(rtn_regional, rtn_samples,
                            hardware_regional, hardware_samples):
    sources = {"rtn": rtn_regional, "hardware": hardware_regional}
    rates = _nonfinite_rates(list(rtn_samples) + list(hardware_samples))
    lookup = {}
    for source, rows in sources.items():
        for row in rows:
            if row.get("region") == "all":
                lookup[(source, row["model"], row["config"])] = float(row["RMSE"])
    output = []
    for model in MODEL_ORDER:
        baseline = lookup.get(("rtn", model, "FP32"))
        if baseline is None:
            continue
        for config, label, source in CONFIG_SPECS:
            rmse = lookup.get((source, model, config))
            if rmse is None:
                continue
            output.append({
                "model": model,
                "config": config,
                "label": label,
                "backend": source,
                "RMSE": rmse,
                "rmse_ratio": rmse / baseline,
                "nonfinite_rate": rates.get((model, config), 0.0),
            })
    return output


def signal_damage_rows(rows):
    grouped = {}
    for row in rows:
        if row.get("config") not in HARDWARE_CONFIGS \
                or row.get("iteration") != "0" \
                or row.get("signal") not in SIGNAL_ORDER:
            continue
        key = (row["model"], row["config"], row["signal"])
        grouped.setdefault(key, []).append(row)
    return [{
        "model": key[0],
        "config": key[1],
        "signal": key[2],
        "median_sqnr_db": _median([row.get("sqnr_db") for row in points]),
        "median_zeroed_rate": _median([row.get("zeroed_rate") for row in points]),
        "median_sign_flip_rate": _median([
            row.get("sign_flip_rate") for row in points]),
    } for key, points in sorted(grouped.items())]


def layer_damage_rows(rows):
    grouped = {}
    for row in rows:
        if row.get("config") not in HARDWARE_CONFIGS:
            continue
        key = (row["model"], row["config"], row["group"], row["kind"])
        grouped.setdefault(key, []).append(row)
    return [{
        "model": key[0],
        "config": key[1],
        "group": key[2],
        "kind": key[3],
        "median_sqnr_db": _median([row.get("sqnr_db") for row in points]),
        "minimum_sqnr_db": min(float(row["sqnr_db"]) for row in points
                               if row.get("sqnr_db") not in (None, "")),
        "max_saturation_rate": max(float(row.get("saturation_rate", 0.0))
                                   for row in points),
        "module_count": len(points),
    } for key, points in sorted(grouped.items())]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def plot_rmse(path, rows):
    labels = [item[1] for item in CONFIG_SPECS]
    colors = ["#6C757D", "#5B8FF9", "#61DDAA", "#F6BD16", "#E45756"]
    lookup = dict(((row["model"], row["label"]), row) for row in rows)
    x = np.arange(len(MODEL_ORDER))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_axisbelow(True)
    for index, (label, color) in enumerate(zip(labels, colors)):
        values = [lookup[model, label]["RMSE"] for model in MODEL_ORDER]
        bars = ax.bar(x + (index - 2) * width, values, width,
                      color=color, label=label, zorder=3)
        for model, bar in zip(MODEL_ORDER, bars):
            row = lookup[model, label]
            annotation = "%.2fx" % row["rmse_ratio"]
            invalid = row["nonfinite_rate"]
            if invalid > 0:
                bar.set_hatch("//")
                bar.set_edgecolor("#8B0000")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.06,
                    annotation, ha="center", va="bottom", fontsize=8.5)
            if invalid > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(0.12, bar.get_height() / 1.8),
                        "%.1f%%\ninvalid" % (100.0 * invalid),
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (m, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False, ncol=5, loc="upper left")
    ax.set_ylim(0.09, max(row["RMSE"] for row in rows) * 2.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_damage_heatmap(path, rows, columns, column_labels, value_key, label):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.8), squeeze=False)
    for panel_index, (axis, config) in enumerate(zip(axes[0], HARDWARE_CONFIGS)):
        matrix = np.full((len(MODEL_ORDER), len(columns)), np.nan)
        for row in rows:
            column = row.get("signal", row.get("group"))
            if row["config"] == config and column in columns:
                matrix[MODEL_ORDER.index(row["model"]), columns.index(column)] = \
                    row[value_key]
        finite = matrix[np.isfinite(matrix)]
        vmin = min(-10.0, float(np.min(finite))) if finite.size else -10.0
        vmax = max(25.0, float(np.max(finite))) if finite.size else 25.0
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm",
                            vmin=vmin, vmax=vmax, aspect="auto", zorder=2)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                text = "%.1f" % value if np.isfinite(value) else "-"
                axis.text(col_index, row_index, text, ha="center", va="center",
                          fontsize=9, zorder=3)
        axis.set_xticks(np.arange(len(columns)))
        axis.set_xticklabels(column_labels)
        axis.set_yticks(np.arange(len(MODEL_ORDER)))
        axis.set_yticklabels(
            [MODEL_NAMES[model] for model in MODEL_ORDER]
            if panel_index == 0 else [])
        axis.set_title(config.replace("HW_", "").replace("_full", ""))
        colorbar = fig.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(label)
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.20,
                        top=0.88, wspace=0.38)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_signal_damage(path, rows):
    columns = [signal for signal in SIGNAL_ORDER
               if any(row["signal"] == signal for row in rows)]
    _plot_damage_heatmap(
        path, rows, columns,
        [SIGNAL_NAMES[signal].replace(" ", "\n") for signal in columns],
        "median_sqnr_db", "Median signal SQNR (dB)")


def plot_layer_damage(path, rows):
    activation_rows = [row for row in rows if row["kind"] == "input"]
    columns = [group for group in MODULE_GROUP_ORDER
               if any(row["group"] == group for row in activation_rows)]
    _plot_damage_heatmap(
        path, activation_rows, columns,
        [group.replace("_", "\n") for group in columns],
        "median_sqnr_db", "Median Conv/Linear input SQNR (dB)")


def _prediction(path, model, config, sample_index):
    return np.load(str(Path(path) / model / "predictions" / config /
                       ("sample_%05d.npz" % sample_index)), allow_pickle=False)


def plot_prediction_examples(path, rtn_root, hardware_root, fp_pred_dir, indices):
    fp = collect(fp_pred_dir)
    rows = len(indices) * len(MODEL_ORDER)
    titles = ["Ground truth", "FP32", "RTN W4A8", "HW W4A8",
              "RTN W4A4", "HW W4A4", "HW W4A4 |error|"]
    fig, axes = plt.subplots(rows, len(titles),
                             figsize=(18.5, 1.9 * rows), squeeze=False)
    depth_cmap = plt.get_cmap("viridis").copy()
    depth_cmap.set_bad("white")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("white")
    row_index = 0
    for sample_index in indices:
        for model in MODEL_ORDER:
            item = fp[sample_index][model]
            rtn_w4a8 = _prediction(rtn_root, model, "W4A8_full", sample_index)
            rtn_w4a4 = _prediction(rtn_root, model, "W4A4_full", sample_index)
            hw_w4a8 = _prediction(
                hardware_root, model, "HW_W4A8_full", sample_index)
            hw_w4a4 = _prediction(
                hardware_root, model, "HW_W4A4_full", sample_index)
            valid = item["valid"]
            panels = [item["gt"], item["pred"], rtn_w4a8["pred"], hw_w4a8["pred"],
                      rtn_w4a4["pred"], hw_w4a4["pred"], hw_w4a4["abs_err"]]
            for col, panel in enumerate(panels):
                finite = np.isfinite(panel)
                image = np.ma.masked_where(~valid | ~finite, panel)
                is_error = col == len(panels) - 1
                axes[row_index, col].imshow(
                    image, cmap=error_cmap if is_error else depth_cmap,
                    vmin=0.0, vmax=3.0 if is_error else 10.0,
                    aspect="auto", interpolation="nearest")
                axes[row_index, col].set_xticks([])
                axes[row_index, col].set_yticks([])
                if row_index == 0:
                    axes[row_index, col].set_title(titles[col])
            axes[row_index, 0].set_ylabel(
                "#%05d\n%s" % (sample_index, MODEL_NAMES[model]),
                rotation=0, ha="right", va="center")
            row_index += 1
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtn-root", default="profile_logs/nyu_rtn_quantization")
    parser.add_argument("--hardware-root",
                        default="profile_logs/nyu_hardware_aligned_quantization")
    parser.add_argument("--fp-pred-dir",
                        default="profile_logs/nyu_prediction_random64_fp32/predictions")
    parser.add_argument("--example-indices", nargs="+", type=int,
                        default=[386, 558])
    args = parser.parse_args()
    rtn_root = Path(args.rtn_root)
    hardware_root = Path(args.hardware_root)
    set_style()
    rtn = collect_tables(rtn_root)
    hardware = collect_tables(hardware_root)
    comparison = comparison_summary_rows(
        rtn["regional"], rtn["samples"],
        hardware["regional"], hardware["samples"])
    signals = signal_damage_rows(hardware["signals"])
    layers = layer_damage_rows(hardware["layers"])
    write_csv(hardware_root / "hardware_comparison_summary.csv", comparison)
    write_csv(hardware_root / "hardware_signal_damage_summary.csv", signals)
    write_csv(hardware_root / "hardware_layer_damage_summary.csv", layers)
    plot_rmse(hardware_root / "hardware_aligned_rmse_comparison.png", comparison)
    plot_signal_damage(hardware_root / "hardware_signal_sqnr.png", signals)
    plot_layer_damage(hardware_root / "hardware_layer_input_sqnr.png", layers)
    plot_prediction_examples(
        hardware_root / "hardware_prediction_examples.png", rtn_root,
        hardware_root, args.fp_pred_dir, args.example_indices)
    print("plots=%s" % hardware_root, flush=True)


if __name__ == "__main__":
    main()
