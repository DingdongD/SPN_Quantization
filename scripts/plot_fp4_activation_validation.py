#!/usr/bin/env python3
"""Render summaries and prediction comparisons for FP4 validation."""

from __future__ import division, print_function

import argparse
import csv
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
CONFIG_LABELS = {
    "FP32": "FP32",
    "FP4V_W8A4": "INT4",
    "FP4V_W8E2M1": "E2M1",
    "FP4V_W8A8": "A8",
    "FP4V_W4A4": "INT4",
    "FP4V_W4E2M1": "E2M1",
    "FP4V_W4A8": "A8",
}
COLORS = {
    "FP32": "#4d4d4d",
    "INT4": "#e45756",
    "E2M1": "#2a9d8f",
    "A8": "#4c78a8",
}
INVALID_GT_RGBA = np.array([0.85, 0.85, 0.85, 1.0])
NONFINITE_RGBA = np.array([1.0, 0.0, 1.0, 1.0])
DEPTH_RANGE = (0.0, 10.0)
ERROR_RANGE = (0.0, 3.0)


def panel_specifications(weight_bits):
    if weight_bits not in ("W8", "W4"):
        raise ValueError("weight_bits must be W8 or W4")
    return [
        {"config": "GT", "label": "GT"},
        {"config": "FP32", "label": "FP32"},
        {"config": "FP4V_%sA4" % weight_bits, "label": "INT4"},
        {"config": "FP4V_%sE2M1" % weight_bits, "label": "E2M1"},
        {"config": "FP4V_%sA8" % weight_bits, "label": "A8"},
    ]


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.axisbelow": True,
    })


def _semantic_rgba(values, valid_gt, nonfinite, cmap_name, value_range):
    values = np.asarray(values)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    nonfinite = np.asarray(nonfinite, dtype=bool)
    if values.shape != valid_gt.shape or values.shape != nonfinite.shape:
        raise ValueError("values and masks must have identical shapes")
    lo, hi = value_range
    normalized = np.clip((np.nan_to_num(
        values, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo), 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(normalized)
    rgba[~valid_gt] = INVALID_GT_RGBA
    rgba[valid_gt & nonfinite] = NONFINITE_RGBA
    return rgba


def depth_rgba(depth, valid_gt, nonfinite):
    return _semantic_rgba(
        depth, valid_gt, nonfinite, "viridis", DEPTH_RANGE)


def error_rgba(error, valid_gt, nonfinite):
    return _semantic_rgba(
        error, valid_gt, nonfinite, "magma", ERROR_RANGE)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_visual_indices(rows, weight_bits, count):
    int4_config = "FP4V_%sA4" % weight_bits
    e2m1_config = "FP4V_%sE2M1" % weight_bits
    int4 = dict((int(row["sample_index"]), float(row["RMSE"]))
                for row in rows if row["config"] == int4_config)
    e2m1 = dict((int(row["sample_index"]), float(row["RMSE"]))
                for row in rows if row["config"] == e2m1_config)
    if set(int4) != set(e2m1):
        raise ValueError("visual sample sets differ")
    if count <= 0 or count > len(int4):
        raise ValueError("invalid visual sample count")

    ordered = sorted(int4)
    values = np.asarray([int4[index] for index in ordered])
    candidates = []
    candidates.append(min(ordered, key=lambda index: (int4[index], index)))
    median = float(np.median(values))
    candidates.append(min(
        ordered, key=lambda index: (abs(int4[index] - median), index)))
    candidates.append(max(
        ordered, key=lambda index: (int4[index] - e2m1[index], -index)))
    candidates.append(max(ordered, key=lambda index: (int4[index], -index)))
    selected = []
    for index in candidates + ordered:
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def load_payload(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return dict((field, payload[field]) for field in payload.files)


def _lookup(rows, model, config):
    matches = [row for row in rows
               if row["model"] == model and row["config"] == config]
    if len(matches) != 1:
        raise ValueError("summary lookup mismatch for %s %s" %
                         (model, config))
    return matches[0]


def render_summary(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharey=True)
    positions = np.arange(len(MODEL_ORDER))
    width = 0.19
    for axis, weight_bits in zip(axes, ("W8", "W4")):
        configs = ("FP32", "FP4V_%sA4" % weight_bits,
                   "FP4V_%sE2M1" % weight_bits,
                   "FP4V_%sA8" % weight_bits)
        for offset, config in enumerate(configs):
            values = [float(_lookup(rows, model, config)["mean_sample_RMSE"])
                      for model in MODEL_ORDER]
            label = CONFIG_LABELS[config]
            axis.bar(positions + (offset - 1.5) * width, values, width,
                     color=COLORS[label], label=label, zorder=3)
        axis.set_xlabel("%s weights" % weight_bits)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [MODEL_NAMES[model] for model in MODEL_ORDER], rotation=0)
        axis.grid(True, axis="y", color="#d6d6d6", linewidth=0.8, zorder=0)
    axes[0].set_ylabel("Mean per-sample RMSE (m)")
    axes[1].legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_paired(rows, model, path):
    current = [row for row in rows if row["model"] == model]
    if len(current) != 2:
        raise ValueError("paired comparison count mismatch for %s" % model)
    current.sort(key=lambda row: row["weight_bits"], reverse=True)
    positions = np.arange(2)
    differences = np.asarray(
        [float(row["mean_difference"]) for row in current])
    lower = differences - np.asarray([float(row["ci_lower"])
                                      for row in current])
    upper = np.asarray([float(row["ci_upper"]) for row in current]) - differences
    recoveries = [float(row["recovery"]) * 100.0
                  if row["recovery"] != "" else np.nan for row in current]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].errorbar(
        positions, differences, yerr=np.vstack((lower, upper)), fmt="o",
        color=COLORS["E2M1"], capsize=5, linewidth=1.8, zorder=3)
    axes[0].axhline(0.0, color="#555555", linewidth=1.0, zorder=2)
    axes[0].set_ylabel("E2M1 - INT4 RMSE (m)")
    axes[1].bar(positions, recoveries, width=0.55,
                color=COLORS["E2M1"], zorder=3)
    axes[1].axhline(0.0, color="#555555", linewidth=1.0, zorder=2)
    axes[1].set_ylabel("A4-to-A8 recovery (%)")
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels([row["weight_bits"] for row in current], rotation=0)
        axis.grid(True, axis="y", color="#d6d6d6", linewidth=0.8, zorder=0)
        axis.set_xlabel(MODEL_NAMES[model])
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_groups(rows, model, path):
    current = [row for row in rows if row["model"] == model]
    groups = sorted(set(row["group"] for row in current))
    configs = ("FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5))
    fields = (("sqnr_db", "SQNR (dB)"),
              ("zero_code_rate", "Zero-code ratio"),
              ("saturation_rate", "Saturation ratio"))
    positions = np.arange(len(groups))
    width = 0.25
    for axis, (field, label) in zip(axes, fields):
        for offset, config in enumerate(configs):
            values = []
            for group in groups:
                matches = [row for row in current
                           if row["config"] == config and row["group"] == group]
                if len(matches) != 1:
                    raise ValueError("group summary mismatch")
                values.append(float(matches[0][field]))
            config_label = CONFIG_LABELS[config]
            axis.bar(positions + (offset - 1.0) * width, values, width,
                     color=COLORS[config_label], label=config_label, zorder=3)
        axis.set_xticks(positions)
        axis.set_xticklabels(groups, rotation=0)
        axis.set_ylabel(label)
        axis.grid(True, axis="y", color="#d6d6d6", linewidth=0.8, zorder=0)
    axes[1].set_xlabel(MODEL_NAMES[model])
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_propagation_steps(rows, model, path):
    current = [row for row in rows if row["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharey=True)
    for axis, weight_bits in zip(axes, ("W8", "W4")):
        for suffix in ("A4", "E2M1", "A8"):
            config = "FP4V_%s%s" % (weight_bits, suffix)
            selected = [row for row in current if row["config"] == config]
            selected.sort(key=lambda row: int(row["iteration"]))
            axis.plot(
                [int(row["iteration"]) for row in selected],
                [float(row["mean_RMSE"]) for row in selected],
                color=COLORS[CONFIG_LABELS[config]], marker="o", markersize=2.5,
                linewidth=1.5, label=CONFIG_LABELS[config], zorder=3)
        axis.set_xlabel("%s weights: propagation iteration" % weight_bits)
        axis.grid(True, color="#d6d6d6", linewidth=0.8, zorder=0)
    axes[0].set_ylabel("FP32-relative state RMSE (m)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def _hide_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#777777")


def render_predictions(root, rows, model, weight_bits, indices, path):
    panels = panel_specifications(weight_bits)
    fig, axes = plt.subplots(
        2 * len(indices), len(panels),
        figsize=(13.5, 3.35 * len(indices)), squeeze=False)
    for sample_row, sample_index in enumerate(indices):
        payloads = {}
        for panel in panels[1:]:
            config = panel["config"]
            payloads[config] = load_payload(
                Path(root) / model / "predictions" / config /
                ("sample_%05d.npz" % sample_index))
        reference = payloads["FP32"]
        valid = np.asarray(reference["valid_gt"], dtype=bool)
        for column, panel in enumerate(panels):
            depth_axis = axes[2 * sample_row, column]
            error_axis = axes[2 * sample_row + 1, column]
            if panel["config"] == "GT":
                empty = np.zeros_like(valid, dtype=bool)
                depth_axis.imshow(depth_rgba(reference["gt"], valid, empty),
                                  aspect="auto", interpolation="nearest")
                error_axis.text(
                    0.5, 0.5, "%s\n#%05d" %
                    (MODEL_NAMES[model], sample_index),
                    transform=error_axis.transAxes, ha="center", va="center")
            else:
                payload = payloads[panel["config"]]
                nonfinite = np.asarray(payload["nonfinite"], dtype=bool)
                depth_axis.imshow(depth_rgba(
                    payload["pred"], valid, nonfinite),
                    aspect="auto", interpolation="nearest")
                error_axis.imshow(error_rgba(
                    payload["abs_err"], valid, nonfinite),
                    aspect="auto", interpolation="nearest")
                metric = [row for row in rows
                          if row["config"] == panel["config"] and
                          int(row["sample_index"]) == sample_index]
                if len(metric) != 1:
                    raise ValueError("prediction metric lookup mismatch")
                depth_axis.set_xlabel(
                    "RMSE %.3f m" % float(metric[0]["RMSE"]), labelpad=2)
            if sample_row == 0:
                depth_axis.set_title(panel["label"], pad=3)
            _hide_axis(depth_axis)
            _hide_axis(error_axis)

    depth_map = ScalarMappable(
        norm=Normalize(*DEPTH_RANGE), cmap=plt.get_cmap("viridis"))
    error_map = ScalarMappable(
        norm=Normalize(*ERROR_RANGE), cmap=plt.get_cmap("magma"))
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.03, top=0.95,
                        wspace=0.18, hspace=0.38)
    depth_color_axis = fig.add_axes([0.92, 0.56, 0.012, 0.36])
    error_color_axis = fig.add_axes([0.92, 0.10, 0.012, 0.36])
    depth_bar = fig.colorbar(depth_map, cax=depth_color_axis)
    depth_bar.set_label("Depth (m)")
    error_bar = fig.colorbar(error_map, cax=error_color_axis)
    error_bar.set_label("Absolute error (m)")
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_all(root, analysis_dir, out_dir, visual_samples):
    root = Path(root)
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(analysis_dir / "configuration_summary.csv")
    paired = read_csv(analysis_dir / "paired_comparisons.csv")
    groups = read_csv(analysis_dir / "group_activation_summary.csv")
    steps = read_csv(analysis_dir / "propagation_step_summary.csv")
    render_summary(summary, out_dir / "fp4_four_model_summary.png")
    for model in MODEL_ORDER:
        render_paired(paired, model, out_dir / ("%s_paired.png" % model))
        render_groups(groups, model, out_dir / ("%s_groups.png" % model))
        render_propagation_steps(
            steps, model, out_dir / ("%s_propagation_steps.png" % model))
        sample_rows = read_csv(root / model / "sample_metrics.csv")
        for weight_bits in ("W8", "W4"):
            indices = select_visual_indices(
                sample_rows, weight_bits, visual_samples)
            render_predictions(
                root, sample_rows, model, weight_bits, indices,
                out_dir / ("%s_%s_predictions.png" %
                           (model, weight_bits.lower())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--visual-samples", required=True, type=int)
    args = parser.parse_args()
    set_style()
    render_all(
        args.root, args.analysis_dir, args.out_dir, args.visual_samples)
    print("wrote FP4 validation figures to %s" % args.out_dir)


if __name__ == "__main__":
    main()
