#!/usr/bin/env python3
"""Plot strict W4A4 and FP4 reconstruction evaluations."""

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
METHOD_ORDER = ("rtn", "adaround", "brecq")
METHOD_NAMES = {
    "rtn": "RTN",
    "adaround": "AdaRound",
    "brecq": "BRECQ",
}
CONFIG_ORDER = ("FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8")
CONFIG_NAMES = {
    "FP4V_W4A4": "A4",
    "FP4V_W4E2M1": "E2M1",
    "FP4V_W4A8": "A8",
}
COLORS = {
    "FP4V_W4A4": "#E45756",
    "FP4V_W4E2M1": "#2A9D8F",
    "FP4V_W4A8": "#4C78A8",
}
METHOD_HATCHES = {
    "rtn": "//",
    "adaround": "",
    "brecq": "xx",
}
INVALID_GT_RGBA = np.asarray([0.85, 0.85, 0.85, 1.0])
NONFINITE_RGBA = np.asarray([1.0, 0.0, 1.0, 1.0])
DEPTH_RANGE = (0.0, 10.0)
ERROR_RANGE = (0.0, 3.0)


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "axes.axisbelow": True,
    })


def prediction_panels():
    return (
        ("GT", "GT"),
        ("FP32", "FP32"),
        ("rtn:FP4V_W4A4", "RTN A4"),
        ("rtn:FP4V_W4E2M1", "RTN E2M1"),
        ("adaround:FP4V_W4A4", "AdaRound A4"),
        ("adaround:FP4V_W4E2M1", "AdaRound E2M1"),
        ("brecq:FP4V_W4A4", "BRECQ A4"),
        ("brecq:FP4V_W4E2M1", "BRECQ E2M1"),
    )


def _visual_series():
    return {
        panel for panel, label in prediction_panels()
        if panel not in ("GT", "FP32")}


def select_visual_index(rows):
    expected = _visual_series()
    by_sample = {}
    for row in rows:
        sample_index = int(row["sample_index"])
        series = "%s:%s" % (row["method"], row["config"])
        if sample_index not in by_sample:
            by_sample[sample_index] = {}
        if series in by_sample[sample_index]:
            raise ValueError("duplicate visual sample series")
        by_sample[sample_index][series] = float(row["RMSE"])
    for sample_index in by_sample:
        if set(by_sample[sample_index]) != expected:
            raise ValueError("visual sample series mismatch")
    finite = []
    for sample_index in sorted(by_sample):
        values = np.asarray(
            [by_sample[sample_index][series] for series in sorted(expected)],
            dtype=np.float64)
        if np.isfinite(values).all():
            finite.append((float(values.max() - values.min()), sample_index))
    if not finite:
        raise ValueError("no fully finite visual sample")
    return max(finite, key=lambda item: (item[0], -item[1]))[1]


def _semantic_rgba(values, valid_gt, nonfinite, cmap_name, limits):
    values = np.asarray(values)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    nonfinite = np.asarray(nonfinite, dtype=bool)
    if values.shape != valid_gt.shape or values.shape != nonfinite.shape:
        raise ValueError("values and masks must have identical shapes")
    lower, upper = limits
    normalized = np.clip(
        (np.nan_to_num(values, nan=lower, posinf=upper, neginf=lower) - lower) /
        (upper - lower), 0.0, 1.0)
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


def load_payload(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return {field: payload[field] for field in payload.files}


def _lookup(rows, model, method, config):
    matches = [
        row for row in rows
        if row["model"] == model and row["method"] == method and
        row["config"] == config]
    if len(matches) != 1:
        raise ValueError("plot lookup mismatch")
    return matches[0]


def _hide_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)


def render_rmse(rows, path):
    fig, axis = plt.subplots(figsize=(13.8, 6.4))
    positions = np.arange(len(MODEL_ORDER))
    width = 0.085
    series = [
        (method, config)
        for method in METHOD_ORDER for config in CONFIG_ORDER]
    for rank, (method, config) in enumerate(series):
        values = [
            float(_lookup(rows, model, method, config)["mean_rmse"])
            for model in MODEL_ORDER]
        offsets = positions + (rank - (len(series) - 1) / 2.0) * width
        axis.bar(
            offsets, values, width, color=COLORS[config],
            hatch=METHOD_HATCHES[method], edgecolor="#444444",
            linewidth=0.4, zorder=3,
            label="%s %s" % (METHOD_NAMES[method], CONFIG_NAMES[config]))
    axis.set_yscale("log")
    axis.set_ylabel("Mean per-sample RMSE (m, log scale)")
    axis.set_xticks(positions)
    axis.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    axis.grid(True, axis="y", color="#d6d6d6", linewidth=0.8, zorder=0)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_retention(rows, path):
    labels = [
        "%s %s" % (MODEL_NAMES[model], METHOD_NAMES[method])
        for model in MODEL_ORDER for method in METHOD_ORDER]
    values = np.asarray([
        [100.0 * float(_lookup(
            rows, model, method, config)["relative_rmse_degradation"])
         for config in CONFIG_ORDER]
        for model in MODEL_ORDER for method in METHOD_ORDER])
    bounded = np.clip(values, -10.0, 100.0)
    fig, axis = plt.subplots(figsize=(7.8, 8.0))
    image = axis.imshow(
        bounded, cmap="RdYlGn_r", vmin=-10.0, vmax=100.0, aspect="auto",
        zorder=1)
    axis.set_xticks(np.arange(len(CONFIG_ORDER)))
    axis.set_xticklabels([CONFIG_NAMES[config] for config in CONFIG_ORDER])
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    for row_rank in range(values.shape[0]):
        for column_rank in range(values.shape[1]):
            axis.text(
                column_rank, row_rank, "%+.1f%%" % values[row_rank, column_rank],
                ha="center", va="center", fontsize=10, color="#111111",
                zorder=3)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("RMSE degradation from FP32 (%)")
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_paired(rows, path):
    selected = [row for row in rows if row["comparison"] == "e2m1_minus_a4"]
    fig, axis = plt.subplots(figsize=(12.8, 5.8))
    positions = np.arange(len(MODEL_ORDER))
    offsets = (-0.22, 0.0, 0.22)
    for offset, method in zip(offsets, METHOD_ORDER):
        current = []
        for model in MODEL_ORDER:
            matches = [
                row for row in selected
                if row["model"] == model and row["method"] == method]
            if len(matches) != 1:
                raise ValueError("paired plot lookup mismatch")
            current.append(matches[0])
        means = np.asarray(
            [float(row["mean_difference"]) for row in current])
        lower = means - np.asarray([float(row["ci_lower"]) for row in current])
        upper = np.asarray([float(row["ci_upper"]) for row in current]) - means
        axis.errorbar(
            positions + offset, means, yerr=np.vstack((lower, upper)),
            fmt="o", capsize=4, linewidth=1.4,
            label=METHOD_NAMES[method], zorder=3)
    axis.axhline(0.0, color="#555555", linewidth=1.0, zorder=2)
    axis.set_ylabel("E2M1 - A4 mean RMSE (m), 95% CI")
    axis.set_xticks(positions)
    axis.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    axis.grid(True, axis="y", color="#d6d6d6", linewidth=0.8, zorder=0)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def _model_visual_rows(root, model):
    rows = []
    for method in METHOD_ORDER:
        current = read_csv(
            Path(root) / "primary" / method / model / "sample_metrics.csv")
        for row in current:
            if row["config"] in ("FP4V_W4A4", "FP4V_W4E2M1"):
                rows.append(dict(row, method=method))
    return rows


def _payload_path(root, model, panel, sample_index):
    if panel == "FP32":
        method, config = "rtn", "FP32"
    else:
        method, config = panel.split(":", 1)
    return Path(root) / "primary" / method / model / "predictions" / config / \
        ("sample_%05d.npz" % sample_index)


def render_predictions(root, path):
    panels = prediction_panels()
    fig, axes = plt.subplots(
        len(MODEL_ORDER), len(panels), figsize=(22.0, 11.0), squeeze=False)
    for model_rank, model in enumerate(MODEL_ORDER):
        sample_index = select_visual_index(_model_visual_rows(root, model))
        fp32 = load_payload(_payload_path(root, model, "FP32", sample_index))
        valid_gt = np.asarray(fp32["valid_gt"], dtype=bool)
        for panel_rank, (panel, label) in enumerate(panels):
            axis = axes[model_rank, panel_rank]
            if panel == "GT":
                values = fp32["gt"]
                nonfinite = ~np.isfinite(values)
            else:
                payload = load_payload(
                    _payload_path(root, model, panel, sample_index))
                values = payload["pred"]
                nonfinite = np.asarray(payload["nonfinite"], dtype=bool)
            axis.imshow(depth_rgba(values, valid_gt, nonfinite))
            if model_rank == 0:
                axis.set_title(label, fontsize=13)
            if panel_rank == 0:
                axis.set_ylabel(
                    "%s\n#%05d" % (MODEL_NAMES[model], sample_index),
                    rotation=0, ha="right", va="center")
            _hide_axis(axis)
    scalar = ScalarMappable(
        norm=Normalize(*DEPTH_RANGE), cmap=plt.get_cmap("viridis"))
    colorbar_axis = fig.add_axes([0.91, 0.20, 0.012, 0.60])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Depth (m)")
    fig.subplots_adjust(left=0.12, right=0.89, top=0.94, bottom=0.03,
                        wspace=0.08, hspace=0.10)
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_errors(root, path):
    panels = prediction_panels()[2:]
    fig, axes = plt.subplots(
        len(MODEL_ORDER), len(panels), figsize=(17.0, 11.0), squeeze=False)
    for model_rank, model in enumerate(MODEL_ORDER):
        sample_index = select_visual_index(_model_visual_rows(root, model))
        fp32 = load_payload(_payload_path(root, model, "FP32", sample_index))
        valid_gt = np.asarray(fp32["valid_gt"], dtype=bool)
        gt = fp32["gt"]
        for panel_rank, (panel, label) in enumerate(panels):
            payload = load_payload(
                _payload_path(root, model, panel, sample_index))
            nonfinite = np.asarray(payload["nonfinite"], dtype=bool)
            error = np.abs(payload["pred"] - gt)
            axis = axes[model_rank, panel_rank]
            axis.imshow(error_rgba(error, valid_gt, nonfinite))
            if model_rank == 0:
                axis.set_title("%s |error|" % label, fontsize=13)
            if panel_rank == 0:
                axis.set_ylabel(
                    "%s\n#%05d" % (MODEL_NAMES[model], sample_index),
                    rotation=0, ha="right", va="center")
            _hide_axis(axis)
    scalar = ScalarMappable(
        norm=Normalize(*ERROR_RANGE), cmap=plt.get_cmap("magma"))
    colorbar_axis = fig.add_axes([0.91, 0.20, 0.012, 0.60])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Absolute error (m)")
    fig.subplots_adjust(left=0.12, right=0.89, top=0.94, bottom=0.03,
                        wspace=0.08, hspace=0.10)
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_stress(rows, path):
    fig, axis = plt.subplots(figsize=(10.8, 5.6))
    positions = np.arange(len(MODEL_ORDER))
    width = 0.24
    colors = ("#5B8FF9", "#61A534", "#E45756")
    for rank, (method, color) in enumerate(zip(METHOD_ORDER, colors)):
        values = [
            float(_lookup(rows, model, method, "HW_W4A4_full")["mean_rmse"])
            for model in MODEL_ORDER]
        axis.bar(
            positions + (rank - 1) * width, values, width,
            color=color, label=METHOD_NAMES[method], zorder=3)
    axis.set_yscale("log")
    axis.set_ylabel("Integer W4A4 mean RMSE (m, log scale)")
    axis.set_xticks(positions)
    axis.set_xticklabels([MODEL_NAMES[model] for model in MODEL_ORDER])
    axis.grid(True, axis="y", color="#d6d6d6", linewidth=0.8, zorder=0)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_activation(rows, model, path):
    current = [row for row in rows if row["model"] == model]
    groups = sorted({row["group"] for row in current})
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    fields = (
        ("sqnr_db", "SQNR (dB)"),
        ("zero_code_rate", "Zero-code ratio"),
        ("saturation_rate", "Saturation ratio"),
    )
    series = [
        (method, config) for method in METHOD_ORDER
        for config in ("FP4V_W4A4", "FP4V_W4E2M1")]
    positions = np.arange(len(groups))
    width = 0.12
    for axis, (field, ylabel) in zip(axes, fields):
        for rank, (method, config) in enumerate(series):
            values = []
            for group in groups:
                matches = [
                    row for row in current
                    if row["method"] == method and row["config"] == config and
                    row["group"] == group]
                if len(matches) != 1:
                    raise ValueError("activation plot lookup mismatch")
                values.append(float(matches[0][field]))
            axis.bar(
                positions + (rank - 2.5) * width, values, width,
                color=COLORS[config], hatch=METHOD_HATCHES[method],
                edgecolor="#444444", linewidth=0.4, zorder=3,
                label="%s %s" % (METHOD_NAMES[method], CONFIG_NAMES[config]))
        axis.set_xticks(positions)
        axis.set_xticklabels(groups)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", color="#d6d6d6", linewidth=0.8, zorder=0)
    axes[2].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_propagation(rows, model, path):
    current = [row for row in rows if row["model"] == model]
    fig, axis = plt.subplots(figsize=(10.8, 5.6))
    for method in METHOD_ORDER:
        for config in ("FP4V_W4A4", "FP4V_W4E2M1"):
            selected = [
                row for row in current
                if row["method"] == method and row["config"] == config]
            selected.sort(key=lambda row: int(row["iteration"]))
            if not selected:
                raise ValueError("propagation plot series missing")
            axis.plot(
                [int(row["iteration"]) for row in selected],
                [float(row["mean_rmse"]) for row in selected],
                color=COLORS[config], linestyle="--" if method == "rtn" else
                "-" if method == "adaround" else ":",
                marker="o", markersize=3, linewidth=1.5,
                label="%s %s" % (METHOD_NAMES[method], CONFIG_NAMES[config]),
                zorder=3)
    axis.set_xlabel("Propagation iteration")
    axis.set_ylabel("FP32-relative state RMSE (m)")
    axis.grid(True, color="#d6d6d6", linewidth=0.8, zorder=0)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_all(root, analysis_dir, out_dir):
    set_style()
    root = Path(root)
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(analysis_dir / "strict_w4a4_fp4_summary.csv")
    paired = read_csv(analysis_dir / "strict_w4a4_fp4_paired.csv")
    activation = read_csv(
        analysis_dir / "strict_w4a4_fp4_activation_groups.csv")
    propagation = read_csv(
        analysis_dir / "strict_w4a4_fp4_propagation_steps.csv")
    stress = read_csv(analysis_dir / "strict_w4a4_integer_stress.csv")
    render_rmse(summary, out_dir / "strict_w4a4_fp4_rmse.png")
    render_retention(summary, out_dir / "strict_w4a4_fp4_retention.png")
    render_paired(paired, out_dir / "strict_w4a4_fp4_paired.png")
    render_predictions(root, out_dir / "strict_w4a4_fp4_predictions.png")
    render_errors(root, out_dir / "strict_w4a4_fp4_errors.png")
    render_stress(stress, out_dir / "strict_w4a4_integer_stress.png")
    for model in MODEL_ORDER:
        render_activation(
            activation, model,
            out_dir / ("strict_w4a4_fp4_activation_%s.png" % model))
        render_propagation(
            propagation, model,
            out_dir / ("strict_w4a4_fp4_propagation_%s.png" % model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    render_all(args.root, args.analysis_dir, args.out_dir)
    print("figures=%s" % Path(args.out_dir).resolve(), flush=True)


if __name__ == "__main__":
    main()
