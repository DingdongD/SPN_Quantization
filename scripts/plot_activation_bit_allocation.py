#!/usr/bin/env python3
"""Summarize and plot sensitivity-driven activation bit allocation."""

from __future__ import division, print_function

import argparse
import csv
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.activation_bit_allocation import (
    GROUP_ORDER,
    allocation_summary_rows,
    rank_allocation_rows,
)


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}
GROUP_COLORS = {
    "encoder": "#4C78A8",
    "attention": "#B279A2",
    "decoder": "#59A14F",
    "depth_head": "#F28E2B",
    "propagation_head": "#E15759",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def short_config_label(config):
    exact = {
        "FP32": "FP32",
        "MP_W4A4_base": "A4 base",
        "MP_W4A8_full": "A8 full",
        "MP_W4A4_stateA16": "A4 + state A16",
        "MP_W4A4_stateA8": "A4 + state A8",
        "MP_heads_A8": "Heads A8",
        "MP_propagation_head_A8": "Propagation head A8",
        "MP_depth_head_A8": "Depth head A8",
        "MP_encoder_A8": "Encoder A8",
        "MP_attention_A8": "Attention A8",
        "MP_decoder_A8": "Decoder A8",
    }
    if config in exact:
        return exact[config]
    site = re.match(r"MP_site0*([0-9]+)_A8$", config)
    if site:
        return "Site %d A8" % int(site.group(1))
    top = re.match(r"MP_top([0-9]+)_A8$", config)
    if top:
        return "Top %d A8" % int(top.group(1))
    return config.replace("MP_", "").replace("_", " ")


def select_best_sparse(rows):
    sparse = [row for row in rows if row.get("selection") in
              ("single_site", "group", "heads", "topk")]
    if not sparse:
        return None
    return min(sparse, key=lambda row: (
        float(row["nonfinite_rate"]), float(row["RMSE"]),
        float(row.get("extra_activation_ratio", float("inf")))))


def pareto_front(rows):
    finite = [row for row in rows
              if math.isfinite(float(row.get("extra_activation_ratio", "nan")))
              and math.isfinite(float(row.get("RMSE", "nan")))]
    ordered = sorted(finite, key=lambda row: (
        float(row["extra_activation_ratio"]),
        float(row["nonfinite_rate"]), float(row["RMSE"])))
    front = []
    best_quality = (float("inf"), float("inf"))
    for row in ordered:
        quality = (float(row["nonfinite_rate"]), float(row["RMSE"]))
        if quality < best_quality:
            front.append(row)
            best_quality = quality
    return front


def scatter_rows(rows):
    return [row for row in rows
            if row.get("selection") not in ("fp32", "state")]


def collect_tables(root):
    tables = {"regional": [], "samples": [], "manifest": [], "layers": []}
    filenames = {
        "regional": "regional_metrics.csv",
        "samples": "sample_metrics.csv",
        "manifest": "mixed_precision_configs.csv",
        "layers": "layer_quantization_metrics.csv",
    }
    for model in MODEL_ORDER:
        model_dir = Path(root) / model
        for key, filename in filenames.items():
            rows = read_csv(model_dir / filename)
            for row in rows:
                row.setdefault("model", model)
                if not row.get("model"):
                    row["model"] = model
            tables[key].extend(rows)
    return tables


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.axisbelow": True,
    })


def _by_model(rows):
    return dict((model, [row for row in rows if row["model"] == model])
                for model in MODEL_ORDER)


def plot_rmse_vs_traffic(path, rows):
    grouped = _by_model(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), squeeze=False)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        points = [row for row in scatter_rows(grouped[model])
                  if math.isfinite(float(row["extra_activation_ratio"]))]
        for row in points:
            x = 100.0 * float(row["extra_activation_ratio"])
            y = float(row["RMSE"])
            invalid = float(row["nonfinite_rate"])
            selection = row["selection"]
            marker = "o"
            color = "#E15759" if invalid > 0 else "#4C78A8"
            size = 72 if selection in ("baseline", "full_a8") else 52
            label = short_config_label(row["config"])
            if invalid > 0:
                label += " (%.1f%% invalid)" % (100.0 * invalid)
            ax.scatter(x, y, s=size, marker=marker, color=color,
                       edgecolor="white", linewidth=0.7, zorder=4,
                       label=label)
        front = pareto_front(points)
        if front:
            ax.plot([100.0 * float(row["extra_activation_ratio"])
                     for row in front],
                    [float(row["RMSE"]) for row in front],
                    color="#2F4B7C", linewidth=1.4, zorder=3)
        fp32 = next((row for row in grouped[model]
                     if row["config"] == "FP32"), None)
        if fp32:
            ax.axhline(float(fp32["RMSE"]), color="#6C757D",
                       linestyle="--", linewidth=1.1, zorder=1)
            ax.text(99, float(fp32["RMSE"]), "FP32", ha="right", va="bottom",
                    color="#555555", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlim(-3, 103)
        ax.set_xlabel("Extra Conv/Linear activation traffic (%)")
        ax.set_ylabel("RMSE (m, log scale)")
        ax.text(0.98, 0.96, MODEL_NAMES[model], transform=ax.transAxes,
                ha="right", va="top", fontweight="bold")
        ax.grid(axis="both", color="#D9D9D9", alpha=0.65,
                linewidth=0.7, zorder=0)
        legend = ax.legend(frameon=True, fontsize=8.2, ncol=2,
                           loc="lower left", handletextpad=0.4,
                           columnspacing=0.8)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_alpha(0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _representative_rows(rows):
    output = []
    grouped = _by_model(rows)
    for model in MODEL_ORDER:
        lookup = dict((row["config"], row) for row in grouped[model])
        sparse = select_best_sparse(grouped[model])
        for label, row in (
                ("FP32", lookup.get("FP32")),
                ("A4 base", lookup.get("MP_W4A4_base")),
                ("Best sparse A8", sparse),
                ("A8 full", lookup.get("MP_W4A8_full"))):
            if row is not None:
                item = dict(row)
                item["display"] = label
                output.append(item)
    return output


def plot_representative_summary(path, rows):
    representatives = _representative_rows(rows)
    lookup = dict(((row["model"], row["display"]), row)
                  for row in representatives)
    labels = ("FP32", "A4 base", "Best sparse A8", "A8 full")
    colors = ("#777777", "#E15759", "#F28E2B", "#4C78A8")
    x = np.arange(len(MODEL_ORDER))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    for index, (label, color) in enumerate(zip(labels, colors)):
        values = [float(lookup[model, label]["RMSE"])
                  for model in MODEL_ORDER]
        bars = ax.bar(x + (index - 1.5) * width, values, width,
                      color=color, label=label, zorder=3)
        for model, bar in zip(MODEL_ORDER, bars):
            row = lookup[model, label]
            invalid = float(row["nonfinite_rate"])
            if invalid > 0:
                bar.set_hatch("//")
                bar.set_edgecolor("#7A1F1F")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.08,
                        "%.1f%% invalid" % (100.0 * invalid),
                        ha="center", va="bottom", fontsize=7.5,
                        rotation=90)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (m, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    ax.grid(axis="y", color="#D9D9D9", alpha=0.7, zorder=0)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def selected_group_shares(model, manifest_rows, layer_rows):
    module_group = {}
    module_numel = {}
    for row in layer_rows:
        if row.get("model") != model or row.get("config") != "MP_W4A4_base" \
                or row.get("kind") not in ("input", "output"):
            continue
        module_group[row["module"]] = row["group"]
        module_numel[row["module"]] = module_numel.get(row["module"], 0.0) + \
            float(row["numel"])
    total = sum(module_numel.values())
    output = {}
    for row in manifest_rows:
        if row.get("model") not in (None, "", model) or not row.get("module"):
            continue
        config = row["config"]
        group = module_group.get(row["module"])
        if group is None:
            continue
        shares = output.setdefault(config, dict((item, 0.0)
                                                for item in GROUP_ORDER))
        shares[group] += module_numel.get(row["module"], 0.0) / total \
            if total else 0.0
    return output


def plot_selected_module_map(path, manifest_rows, layer_rows, summary_rows):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), squeeze=False)
    summary = _by_model(summary_rows)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        group_shares = selected_group_shares(model, manifest_rows, layer_rows)
        keep = [row["config"] for row in summary[model]
                if row["selection"] in ("single_site", "group", "heads", "topk")
                and row["config"] in group_shares]
        y = np.arange(len(keep))
        left = np.zeros(len(keep))
        for group in GROUP_ORDER:
            values = np.array([100.0 * group_shares[name].get(group, 0.0)
                               for name in keep])
            ax.barh(y, values, left=left, height=0.62,
                    color=GROUP_COLORS[group], label=group.replace("_", " "),
                    zorder=3)
            left += values
        ax.set_yticks(y)
        ax.set_yticklabels([short_config_label(name) for name in keep])
        ax.invert_yaxis()
        ax.set_xlabel("A8-upgraded Conv/Linear boundary traffic (%)")
        ax.text(0.98, 0.96, MODEL_NAMES[model], transform=ax.transAxes,
                ha="right", va="top", fontweight="bold")
        ax.grid(axis="x", color="#D9D9D9", alpha=0.7, zorder=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels),
               loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_findings(path, rows):
    grouped = _by_model(rows)
    lines = [
        "# Activation Bit Allocation Findings",
        "",
        "All metrics use the same 128 calibration and fixed 64 NYU validation "
        "samples. Weights remain signed symmetric per-output-channel W4. "
        "Activations are per-tensor MinMax; ReLU outputs are unsigned.",
        "",
        "The traffic metric counts Conv/Linear input and output tensor elements "
        "times activation bits. It excludes custom propagation, elementwise, "
        "merge and memory traffic, so it is a hardware-cost proxy rather than "
        "measured latency.",
        "",
        "| Model | FP32 RMSE | A4 base RMSE | A4 invalid | Best sparse | "
        "Sparse RMSE | Sparse invalid | Full A8 RMSE |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        lookup = dict((row["config"], row) for row in grouped[model])
        sparse = select_best_sparse(grouped[model])
        lines.append(
            "| %s | %.4f | %.4f | %.2f%% | %s | %.4f | %.2f%% | %.4f |" % (
                MODEL_NAMES[model], float(lookup["FP32"]["RMSE"]),
                float(lookup["MP_W4A4_base"]["RMSE"]),
                100.0 * float(lookup["MP_W4A4_base"]["nonfinite_rate"]),
                short_config_label(sparse["config"]), float(sparse["RMSE"]),
                100.0 * float(sparse["nonfinite_rate"]),
                float(lookup["MP_W4A8_full"]["RMSE"])))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Sparse A8 is useful only when it protects a complete sensitive path to "
        "the propagation input. A locally upgraded layer can be erased by the "
        "next A4 boundary, so lowest local SQNR alone is not a sufficient bit "
        "allocation rule.",
        "",
        "The baseline propagation loop is FP32. State A8/A16 configurations are "
        "downgrade feasibility tests, not upgrades; matching the A4 baseline "
        "therefore shows that pre-propagation Conv/Linear activations dominate "
        "the observed damage.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",
                        default="profile_logs/nyu_activation_bit_allocation")
    args = parser.parse_args()
    root = Path(args.root)
    tables = collect_tables(root)
    summary = allocation_summary_rows(
        tables["regional"], tables["samples"], tables["manifest"],
        tables["layers"])
    ranking = rank_allocation_rows(summary)
    write_csv(root / "allocation_summary.csv", summary)
    write_csv(root / "allocation_ranking.csv", ranking)
    set_style()
    plot_rmse_vs_traffic(root / "allocation_rmse_vs_traffic.png", summary)
    plot_representative_summary(
        root / "activation_allocation_summary.png", summary)
    plot_selected_module_map(
        root / "activation_selected_module_map.png", tables["manifest"],
        tables["layers"], summary)
    write_findings(root / "activation_bit_allocation_findings.md", summary)
    print("wrote activation allocation analysis to %s" % root)


if __name__ == "__main__":
    main()
