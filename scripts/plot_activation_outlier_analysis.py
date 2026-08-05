#!/usr/bin/env python3
"""Aggregate and plot NYU activation-outlier mitigation experiments."""

from __future__ import division, print_function

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
POLICY_ORDER = ("FP32", "MinMax", "W8A4", "Percentile", "SmoothQuant",
                "AWQ-style")
POLICY_COLORS = {
    "FP32": "#6C757D",
    "MinMax": "#E45756",
    "W8A4": "#4C78A8",
    "Percentile": "#F2CF5B",
    "SmoothQuant": "#59A14F",
    "AWQ-style": "#B279A2",
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


def _config_policy(config):
    if config == "FP32":
        return "FP32", "FP32"
    if config == "HW_W4A4_MinMax":
        return "MinMax", "W4A4 MinMax"
    if config == "HW_W8A4_full":
        return "W8A4", "W8A4 MinMax"
    if "_P" in config:
        suffix = config.rsplit("_P", 1)[1]
        percentile = {"99": "99", "999": "99.9", "9999": "99.99"}[suffix]
        return "Percentile", "W4A4 P%s" % percentile
    if "_SQ_A" in config:
        alpha = int(config.rsplit("_SQ_A", 1)[1]) / 100.0
        return "SmoothQuant", "W4A4 SQ a=%.2g" % alpha
    if "_AWQ_C" in config:
        ratio = int(config.rsplit("_AWQ_C", 1)[1]) / 100.0
        return "AWQ-style", "W4A4 AWQ clip=%.2g" % ratio
    return config, config


def mitigation_summary_rows(regional_rows, sample_rows):
    rates = nonfinite_rates(sample_rows)
    baselines = dict((row["model"], float(row["RMSE"]))
                     for row in regional_rows
                     if row.get("region") == "all" and row["config"] == "FP32")
    output = []
    for row in regional_rows:
        if row.get("region") != "all":
            continue
        model = row["model"]
        policy, label = _config_policy(row["config"])
        rmse = float(row["RMSE"])
        output.append({
            "model": model,
            "config": row["config"],
            "policy": policy,
            "label": label,
            "RMSE": rmse,
            "MAE": float(row["MAE"]),
            "ABS_REL": float(row["ABS_REL"]),
            "rmse_over_fp32": rmse / baselines[model],
            "nonfinite_rate": rates.get((model, row["config"]), 0.0),
        })
    return sorted(output, key=lambda row: (
        MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else 99,
        POLICY_ORDER.index(row["policy"]) if row["policy"] in POLICY_ORDER else 99))


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


def plot_mitigation(path, rows):
    lookup = dict(((row["model"], row["policy"]), row) for row in rows)
    x = np.arange(len(MODEL_ORDER))
    width = 0.13
    fig, axis = plt.subplots(figsize=(14.8, 6.3))
    axis.set_axisbelow(True)
    for index, policy in enumerate(POLICY_ORDER):
        points = [lookup[(model, policy)] for model in MODEL_ORDER]
        positions = x + (index - 2.5) * width
        bars = axis.bar(positions, [row["RMSE"] for row in points], width,
                        label=policy, color=POLICY_COLORS[policy], zorder=3)
        for bar, row in zip(bars, points):
            invalid = row["nonfinite_rate"]
            if invalid > 0.0:
                bar.set_hatch("//")
                bar.set_edgecolor("#7F0000")
                axis.text(bar.get_x() + bar.get_width() / 2,
                          max(0.13, bar.get_height() * 0.62),
                          "%.1f%%\ninvalid" % (100.0 * invalid),
                          ha="center", va="center", rotation=90,
                          fontsize=8, color="#7F0000", zorder=4)
    axis.set_yscale("log")
    axis.set_ylabel("RMSE (m, log scale)")
    axis.set_xticks(x)
    axis.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    axis.grid(axis="y", alpha=0.25, zorder=0)
    handles = [Patch(facecolor=POLICY_COLORS[policy], label=policy)
               for policy in POLICY_ORDER]
    axis.legend(handles=handles, frameon=False, ncol=6, loc="upper center",
                bbox_to_anchor=(0.5, 1.12))
    axis.set_ylim(0.09, max(row["RMSE"] for row in rows) * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _input_group_lookup(group_rows):
    return dict(((row["model"], row["group"]), row) for row in group_rows
                if row["kind"] == "input")


def write_report(path, occupancy, groups, mitigation, damage, percentile_tails,
                 top_outliers):
    occ = dict(((row["model"], row["measure"]), row["encoder_share"])
               for row in occupancy)
    outlier = _input_group_lookup(groups)
    result = dict(((row["model"], row["policy"]), row) for row in mitigation)
    lines = [
        "# Activation Outlier and W4A4 Findings",
        "",
        "## Scope",
        "",
        "Official CSPN, DySPN, NLSPN, and CompletionFormer checkpoints were "
        "profiled on the same 128 NYU calibration samples and evaluated on the "
        "same fixed 64 validation samples. Weights use signed symmetric "
        "per-output-channel quantization; activations use per-tensor MinMax "
        "quantization (unsigned after ReLU, signed otherwise). The occupancy "
        "numbers below are footprint/traffic proxies, not measured CUDA latency.",
        "SmoothQuant adds static per-input-channel equalization before the same "
        "per-tensor activation quantizer; it does not change activation QDQ to "
        "per-channel quantization.",
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
        "DySPN and NLSPN are encoder-heavy in both parameters and activation "
        "traffic. CompletionFormer also has a large encoder, with additional "
        "attention cost. CSPN is not encoder-traffic dominated: only %.1f%% of "
        "its activation QDQ elements are in the encoder, while decoder and heads "
        "account for the remainder." % (100.0 * occ["cspn", "activation_elements"]),
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
    tail_lookup = {}
    for row in percentile_tails:
        tail_lookup.setdefault(row["model"], {})[row["percentile"]] = row
    lines.extend([
        "",
        "The worst spatial-tail input in each model has the following absolute "
        "distribution (p75 / p99 / p99.9 / p99.99 / max):",
        "",
    ])
    for model in MODEL_ORDER:
        points = tail_lookup[model]
        ordered = [points[key] for key in
                   ("p75", "p99", "p99.9", "p99.99", "max")]
        lines.append("- **%s, %s:** %s" % (
            MODEL_NAMES[model], ordered[0]["site"], " / ".join(
                "%.4g" % row["absolute_value"] for row in ordered)))
    lines.extend([
        "",
        "Largest finite per-channel input imbalance:",
        "",
        "| Model | Site | Group | Channel max/median |",
        "|---|---|---|---:|",
    ])
    for model in MODEL_ORDER:
        row = min((item for item in top_outliers
                   if item["model"] == model
                   and item["metric"] == "channel_imbalance"),
                  key=lambda item: int(item["rank"]))
        lines.append("| %s | `%s` | %s | %.2fx |" % (
            MODEL_NAMES[model], row["site"], GROUP_NAMES[row["group"]],
            row["ratio"]))
    lines.extend([
        "",
        "NLSPN has the strongest typical encoder spatial tail. CSPN and DySPN "
        "also contain isolated encoder sites around 5x max/p99.99. "
        "CompletionFormer's median spatial tail is milder, but its attention and "
        "decoder contain highly channel-localized outliers; zero-median channels "
        "produce infinite channel ratios at a few sites.",
        "",
        "## Fixed-64 Mitigation",
        "",
        "| Model | FP32 | W4A4 MinMax | Best tested W4A4 mitigation | "
        "W8A4 | Nonfinite warning |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for model in MODEL_ORDER:
        candidates = [result[model, policy] for policy in
                      ("Percentile", "SmoothQuant", "AWQ-style")]
        valid_candidates = [row for row in candidates
                            if row["nonfinite_rate"] == 0.0]
        invalid = result[model, "MinMax"]["nonfinite_rate"]
        warning = ("%.2f%% MinMax invalid" % (100.0 * invalid)
                   if invalid else "none")
        if valid_candidates:
            best = min(valid_candidates, key=lambda row: row["RMSE"])
            best_text = "%s: %.3f" % (best["label"], best["RMSE"])
        else:
            best = min(candidates, key=lambda row: row["nonfinite_rate"])
            best_text = "no valid result (min invalid %.2f%%)" % (
                100.0 * best["nonfinite_rate"])
        lines.append("| %s | %.3f | %.3f | %s | %.3f | %s |" % (
            MODEL_NAMES[model], result[model, "FP32"]["RMSE"],
            result[model, "MinMax"]["RMSE"], best_text,
            result[model, "W8A4"]["RMSE"], warning))
    lines.extend([
        "",
        "CSPN remains numerically invalid under every tested A4 policy, so its "
        "finite-only RMSE bars are not comparable to valid outputs. DySPN "
        "responds strongly to P99 activation clipping, NLSPN to P99.99 clipping, "
        "and CompletionFormer to SmoothQuant alpha=0.5. AWQ-style weight clipping "
        "is consistently weaker, indicating activation range resolution is the "
        "primary W4A4 failure mode.",
        "",
        "## Lowest-SQNR W4A4 Inputs",
        "",
    ])
    for model in MODEL_ORDER:
        points = [row for row in damage if row["model"] == model][:3]
        lines.append("- **%s:** %s" % (MODEL_NAMES[model], "; ".join(
            "%s (%s, %.2f dB)" % (row["module"], row["group"], row["sqnr_db"])
            for row in points)))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Encoder size and encoder outliers are separate effects. DySPN/NLSPN "
        "have both high encoder occupancy and poor encoder input SQNR; CSPN's "
        "failure cannot be attributed to encoder share alone, because most of its "
        "activation traffic is outside the encoder and its worst input SQNR is in "
        "the decoder. CompletionFormer is damaged in both transformer MLP inputs "
        "and encoder concat convolutions.",
        "",
        "SmoothQuant here is a best-case local QDQ simulation. Removing its "
        "runtime scaling requires folding scales into a unique producer; residual "
        "and multi-branch merges need explicit requantization. The AWQ-style run "
        "uses weight clipping only and is not a full reconstruction-loss AWQ "
        "search. These results measure accuracy, not packed-integer speed.",
    ])
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
    regional = _model_rows(root, "regional_metrics.csv")
    samples = _model_rows(root, "sample_metrics.csv")
    mitigation = mitigation_summary_rows(regional, samples)
    top = top_outlier_rows(root)
    percentile_tails = percentile_tail_rows(top)
    damage = low_bit_damage_rows(root)

    write_csv(root / "encoder_occupancy.csv", occupancy)
    write_csv(root / "activation_outlier_group_summary.csv", groups)
    write_csv(root / "activation_outlier_top_sites.csv", top)
    write_csv(root / "activation_percentile_tails.csv", percentile_tails)
    write_csv(root / "low_bit_damage_top_sites.csv", damage)
    write_csv(root / "mitigation_summary.csv", mitigation)
    plot_encoder_occupancy(root / "encoder_occupancy.png", occupancy)
    _plot_group_heatmap(
        root / "activation_tail_by_group.png", groups,
        "median_max_over_p99_99", "Median max / p99.99")
    _plot_group_heatmap(
        root / "channel_outlier_by_group.png", groups,
        "median_channel_max_over_median", "Median channel max / median")
    plot_percentile_tails(root / "activation_percentile_tails.png",
                          percentile_tails)
    plot_mitigation(root / "mitigation_rmse.png", mitigation)
    write_report(root / "activation_outlier_findings.md", occupancy, groups,
                 mitigation, damage, percentile_tails, top)
    print("Wrote activation-outlier analysis to %s" % root)


if __name__ == "__main__":
    main()
