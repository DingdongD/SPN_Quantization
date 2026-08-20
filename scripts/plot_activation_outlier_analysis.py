#!/usr/bin/env python3
"""Aggregate and plot NYU activation-outlier diagnostics."""

from __future__ import division, print_function

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}
MODEL_TICK_NAMES = dict(MODEL_NAMES, completionformer="Completion\nFormer")
GROUP_ORDER = ("encoder", "attention", "decoder", "depth_head",
               "propagation_head")
GROUP_NAMES = {
    "encoder": "Encoder",
    "attention": "Attention",
    "decoder": "Decoder",
    "depth_head": "Depth head",
    "propagation_head": "Propagation head",
}
MEASURE_ORDER = ("parameter_elements", "activation_elements", "boundaries")
MEASURE_NAMES = {
    "parameter_elements": "Quantized parameters",
    "activation_elements": "Activation QDQ traffic",
    "boundaries": "Quantized boundaries",
}
def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nonfinite_rates(rows):
    totals = {}
    for row in rows:
        key = (row["model"], row["config"])
        bad, finite = totals.setdefault(key, [0, 0])
        totals[key][0] = bad + int(float(row.get("nonfinite_pixels", 0)))
        totals[key][1] = finite + int(float(row.get("num_pixels", 0)))
    return dict((key, bad / float(bad + finite) if bad + finite else 0.0)
                for key, (bad, finite) in totals.items())




def encoder_occupancy_rows(rows):
    selected = [row for row in rows if row["group"] == "encoder"]
    selected.sort(key=lambda row: (
        MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else 99,
        MEASURE_ORDER.index(row["measure"])
        if row["measure"] in MEASURE_ORDER else 99))
    return [{
        "model": row["model"],
        "config": row["config"],
        "measure": row["measure"],
        "encoder_value": float(row["value"]),
        "encoder_share": float(row["share"]),
    } for row in selected]


def _model_rows(root, filename):
    output = []
    for model in MODEL_ORDER:
        rows = read_csv(Path(root) / model / filename)
        for row in rows:
            row = dict(row)
            row.setdefault("model", model)
            output.append(row)
    return output


def _validate_metadata(root):
    metadata = []
    for model in MODEL_ORDER:
        with (Path(root) / model / "metadata.json").open(
                "r", encoding="utf-8") as stream:
            metadata.append(json.load(stream))
    calibration = [item["calibration_indices"] for item in metadata]
    evaluation = [item["evaluation_indices"] for item in metadata]
    if any(indices != calibration[0] for indices in calibration[1:]):
        raise ValueError("models do not share calibration indices")
    if any(indices != evaluation[0] for indices in evaluation[1:]):
        raise ValueError("models do not share evaluation indices")
    if len(calibration[0]) != 128 or len(evaluation[0]) != 64:
        raise ValueError("expected 128 calibration and 64 evaluation samples")
    return metadata


def combined_group_rows(root):
    rows = _model_rows(root, "group_outlier_summary.csv")
    numeric = (
        "sites", "median_max_over_p99_99", "maximum_max_over_p99_99",
        "median_p99_99_over_p99", "median_channel_max_over_median",
        "maximum_channel_max_over_median",
    )
    for row in rows:
        for key in numeric:
            row[key] = int(row[key]) if key == "sites" else float(row[key])
    return rows


def top_outlier_rows(root, limit=8):
    output = []
    metrics = (
        ("spatial_tail", "max_over_p99_99"),
        ("channel_imbalance", "channel_max_over_median"),
    )
    for model in MODEL_ORDER:
        rows = [row for row in read_csv(
            Path(root) / model / "activation_percentiles.csv")
                if row["kind"] == "input"]
        for metric_name, metric_key in metrics:
            ranked = sorted(
                [row for row in rows
                 if math.isfinite(float(row[metric_key]))],
                key=lambda row: float(row[metric_key]), reverse=True)[:limit]
            for rank, row in enumerate(ranked, 1):
                output.append({
                    "model": model,
                    "metric": metric_name,
                    "rank": rank,
                    "module": row["module"],
                    "site": row["site"],
                    "group": row["group"],
                    "ratio": float(row[metric_key]),
                    "p75": float(row["p75"]),
                    "p99": float(row["p99"]),
                    "p99_9": float(row["p99_9"]),
                    "p99_99": float(row["p99_99"]),
                    "maximum": float(row["maximum"]),
                })
    return output


def percentile_tail_rows(top_rows):
    percentiles = (
        ("p75", "p75"), ("p99", "p99"), ("p99.9", "p99_9"),
        ("p99.99", "p99_99"), ("max", "maximum"),
    )
    output = []
    for model in MODEL_ORDER:
        candidates = [row for row in top_rows
                      if row["model"] == model
                      and row["metric"] == "spatial_tail"]
        if not candidates:
            continue
        worst = min(candidates, key=lambda row: int(row["rank"]))
        denominator = float(worst["p99"])
        for label, key in percentiles:
            value = float(worst[key])
            output.append({
                "model": model,
                "site": worst["site"],
                "group": worst["group"],
                "percentile": label,
                "absolute_value": value,
                "over_p99": value / denominator if denominator else float("nan"),
            })
    return output


def low_bit_damage_rows(root, limit=8):
    output = []
    for model in MODEL_ORDER:
        rows = [row for row in read_csv(
            Path(root) / model / "layer_quantization_metrics.csv")
                if row["config"] == "HW_W4A4_MinMax"
                and row["kind"] == "input"
                and math.isfinite(float(row["sqnr_db"]))]
        for rank, row in enumerate(sorted(
                rows, key=lambda item: float(item["sqnr_db"]))[:limit], 1):
            output.append({
                "model": model,
                "rank": rank,
                "module": row["module"],
                "group": row["group"],
                "sqnr_db": float(row["sqnr_db"]),
                "mse": float(row["mse"]),
                "cosine": float(row["cosine"]),
            })
    return output


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
    })


def plot_encoder_occupancy(path, rows):
    lookup = dict(((row["model"], row["measure"]), row["encoder_share"])
                  for row in rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), sharey=True)
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B279A2"]
    x = np.arange(len(MODEL_ORDER))
    for index, (axis, measure) in enumerate(zip(axes, MEASURE_ORDER)):
        values = [100.0 * lookup[model, measure] for model in MODEL_ORDER]
        bars = axis.bar(x, values, width=0.68, color=colors, zorder=3)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25, zorder=0)
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_TICK_NAMES[model] for model in MODEL_ORDER],
                             fontsize=11)
        axis.set_xlabel(MEASURE_NAMES[measure])
        axis.set_ylim(0, 105)
        axis.bar_label(bars, labels=["%.1f%%" % value for value in values],
                       padding=3, fontsize=11)
        if index == 0:
            axis.set_ylabel("Encoder share (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_percentile_tails(path, rows):
    labels = ("p75", "p99", "p99.9", "p99.99", "max")
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.8), sharey=True)
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B279A2"]
    for index, (axis, model, color) in enumerate(
            zip(axes, MODEL_ORDER, colors)):
        points = [row for row in rows if row["model"] == model]
        values = [max(float(row["over_p99"]), 1e-3) for row in points]
        axis.plot(x, values, marker="o", linewidth=2.2, color=color, zorder=3)
        axis.fill_between(x, values, 1e-3, color=color, alpha=0.10, zorder=2)
        axis.axhline(1.0, color="#777777", linewidth=1.0,
                     linestyle="--", zorder=1)
        axis.set_yscale("log")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25, zorder=0)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=10)
        axis.set_xlabel(MODEL_NAMES[model])
        site = points[0]["site"] if points else ""
        if len(site) > 27:
            site = "..." + site[-24:]
        axis.text(0.5, 0.96, site, transform=axis.transAxes,
                  ha="center", va="top", fontsize=9)
        if index == 0:
            axis.set_ylabel("Activation magnitude / p99")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_group_heatmap(path, rows, key, colorbar_label, vmax=None):
    selected = [row for row in rows if row["kind"] == "input"]
    matrix = np.full((len(MODEL_ORDER), len(GROUP_ORDER)), np.nan)
    for row in selected:
        if row["group"] in GROUP_ORDER:
            matrix[MODEL_ORDER.index(row["model"]),
                   GROUP_ORDER.index(row["group"])] = float(row[key])
    finite = matrix[np.isfinite(matrix)]
    if vmax is None:
        vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    image_data = np.ma.masked_invalid(np.minimum(matrix, vmax))
    fig, axis = plt.subplots(figsize=(10.8, 5.3))
    image = axis.imshow(image_data, cmap="YlOrRd", vmin=1.0, vmax=vmax,
                        aspect="auto", zorder=2)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                label = "-"
            elif np.isinf(value):
                label = "inf"
            else:
                label = "%.2f" % value
            axis.text(col_index, row_index, label, ha="center", va="center",
                      fontsize=11, zorder=3)
    axis.set_xticks(np.arange(len(GROUP_ORDER)))
    axis.set_xticklabels([GROUP_NAMES[group] for group in GROUP_ORDER])
    axis.set_yticks(np.arange(len(MODEL_ORDER)))
    axis.set_yticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



def _input_group_lookup(group_rows):
    return dict(((row["model"], row["group"]), row) for row in group_rows
                if row["kind"] == "input")


def write_report(path, occupancy, groups, damage, percentile_tails,
                 top_outliers):
    occ = dict(((row["model"], row["measure"]), row["encoder_share"])
               for row in occupancy)
    outlier = _input_group_lookup(groups)
    lines = [
        "# Activation Outlier Findings",
        "",
        "Official CSPN, DySPN, NLSPN, and CompletionFormer checkpoints were "
        "profiled with the same NYU calibration protocol. Weights use signed "
        "symmetric per-output-channel quantization; activations use uniform "
        "MinMax contracts with unsigned ranges after ReLU.",
        "",
        "## Encoder Occupancy",
        "",
        "| Model | Parameters | Activation QDQ traffic | Boundaries |",
        "|---|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        lines.append("| %s | %.1f%% | %.1f%% | %.1f%% |" % (
            MODEL_NAMES[model],
            100.0 * occ[model, "parameter_elements"],
            100.0 * occ[model, "activation_elements"],
            100.0 * occ[model, "boundaries"]))
    lines.extend([
        "",
        "## Activation Tails",
        "",
        "| Model | Encoder median max/p99.99 | Encoder max max/p99.99 | "
        "Encoder median channel max/median |",
        "|---|---:|---:|---:|",
    ])
    for model in MODEL_ORDER:
        row = outlier[model, "encoder"]
        lines.append("| %s | %.2fx | %.2fx | %.2fx |" % (
            MODEL_NAMES[model], row["median_max_over_p99_99"],
            row["maximum_max_over_p99_99"],
            row["median_channel_max_over_median"]))
    lines.extend(["", "Worst spatial-tail inputs:", ""])
    tails = {}
    for row in percentile_tails:
        tails.setdefault(row["model"], []).append(row)
    for model in MODEL_ORDER:
        ordered = tails[model]
        lines.append("- **%s, %s:** %s" % (
            MODEL_NAMES[model], ordered[0]["site"], " / ".join(
                "%.4g" % row["absolute_value"] for row in ordered)))
    lines.extend(["", "Lowest-SQNR W4A4 inputs:", ""])
    for model in MODEL_ORDER:
        points = [row for row in damage if row["model"] == model][:3]
        lines.append("- **%s:** %s" % (MODEL_NAMES[model], "; ".join(
            "%s (%s, %.2f dB)" %
            (row["module"], row["group"], row["sqnr_db"])
            for row in points)))
    lines.extend(["", "Largest finite per-channel input imbalance:", ""])
    for model in MODEL_ORDER:
        row = min((item for item in top_outliers
                   if item["model"] == model and
                   item["metric"] == "channel_imbalance"),
                  key=lambda item: int(item["rank"]))
        lines.append("- **%s:** `%s`, %.2fx" % (
            MODEL_NAMES[model], row["site"], row["ratio"]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="profile_logs/nyu_activation_outliers")
    args = parser.parse_args()
    root = Path(args.root)
    _validate_metadata(root)
    set_style()

    occupancy_all = _model_rows(root, "occupancy.csv")
    occupancy = encoder_occupancy_rows(occupancy_all)
    groups = combined_group_rows(root)
    top = top_outlier_rows(root)
    percentile_tails = percentile_tail_rows(top)
    damage = low_bit_damage_rows(root)

    write_csv(root / "encoder_occupancy.csv", occupancy)
    write_csv(root / "activation_outlier_group_summary.csv", groups)
    write_csv(root / "activation_outlier_top_sites.csv", top)
    write_csv(root / "activation_percentile_tails.csv", percentile_tails)
    write_csv(root / "low_bit_damage_top_sites.csv", damage)
    plot_encoder_occupancy(root / "encoder_occupancy.png", occupancy)
    _plot_group_heatmap(
        root / "activation_tail_by_group.png", groups,
        "median_max_over_p99_99", "Median max / p99.99")
    _plot_group_heatmap(
        root / "channel_outlier_by_group.png", groups,
        "median_channel_max_over_median", "Median channel max / median")
    plot_percentile_tails(root / "activation_percentile_tails.png",
                          percentile_tails)
    write_report(root / "activation_outlier_findings.md", occupancy, groups,
                 damage, percentile_tails, top)
    print("Wrote activation-outlier analysis to %s" % root)


if __name__ == "__main__":
    main()
