#!/usr/bin/env python3
"""Plot CompletionFormer front-encoder W8A8 Pareto results."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


BASELINE_CONFIG = "JIQ_Joint_W4A4"
W4A8_CONFIG = "JIQ_W4A8"
FP32_CONFIG = "FP32"
SHARE_METRICS = {
    "mac": ("whole_model_mac_share", "W8A8 MAC share (%)"),
    "parameter": (
        "whole_model_parameter_share", "W8A8 parameter share (%)"),
    "operator": (
        "whole_model_operator_share", "W8A8 operator share (%)"),
}


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.axisbelow": True,
    })


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty result table: %s" % path)
    return rows


def _required_row(by_name, name):
    if name not in by_name:
        raise ValueError("missing aggregate config: %s" % name)
    return by_name[name]


def load_pareto_contract(root):
    root = Path(root)
    aggregate = read_csv(root / "front_encoder_final_aggregate.csv")
    names = [row["config"] for row in aggregate]
    if len(names) != len(set(names)):
        raise ValueError("duplicate aggregate configurations")
    by_name = dict((row["config"], row) for row in aggregate)
    fp32 = _required_row(by_name, FP32_CONFIG)
    baseline = _required_row(by_name, BASELINE_CONFIG)
    w4a8 = _required_row(by_name, W4A8_CONFIG)
    front = [row for row in aggregate
             if row["config"].startswith("FE_W8A8_")]
    if not front:
        raise ValueError("front encoder configurations are missing")

    for row in aggregate:
        rmse = float(row["mean_rmse"])
        samples = int(row["samples"])
        if not math.isfinite(rmse) or samples <= 0:
            raise ValueError("invalid aggregate row: %s" % row["config"])
    candidates = [baseline] + front
    for row in candidates:
        for key, label in SHARE_METRICS.values():
            value = float(row[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "invalid %s: %s" % (label, row["config"]))

    frontier = read_csv(root / "front_encoder_pareto.csv")
    frontier_names = [row["config"] for row in frontier]
    if len(frontier_names) != len(set(frontier_names)):
        raise ValueError("duplicate Pareto configurations")
    expected_frontier = [row["config"] for row in candidates
                         if int(row["is_pareto"]) == 1]
    if set(frontier_names) != set(expected_frontier):
        raise ValueError("Pareto table disagrees with aggregate table")
    for row in frontier:
        if int(row["is_pareto"]) != 1:
            raise ValueError("Pareto row is not marked as Pareto")
    frontier.sort(key=lambda row: float(row["whole_model_mac_share"]))

    metadata = json.loads(
        (root / "metadata.json").read_text(encoding="utf-8"))
    selection = metadata["front_encoder_w8a8_pareto"]
    knee_config = selection["knee_config"]
    best_config = selection["best_config"]
    if knee_config not in frontier_names:
        raise ValueError("knee config is not Pareto: %s" % knee_config)
    if best_config not in by_name or not best_config.startswith("FE_W8A8_"):
        raise ValueError("invalid best config: %s" % best_config)

    dominated = [row for row in front
                 if row["config"] not in set(frontier_names)]
    return {
        "aggregate": aggregate,
        "by_name": by_name,
        "frontier": frontier,
        "dominated": dominated,
        "fp32": fp32,
        "baseline": baseline,
        "w4a8": w4a8,
        "knee_config": knee_config,
        "best_config": best_config,
    }


def _prediction_configs(contract):
    ordered = (
        FP32_CONFIG,
        BASELINE_CONFIG,
        W4A8_CONFIG,
        contract["knee_config"],
        contract["best_config"],
    )
    output = []
    for name in ordered:
        if name not in output:
            output.append(name)
    return tuple(output)


def load_prediction_grid(root, contract, expected_samples):
    root = Path(root)
    expected_samples = int(expected_samples)
    configs = _prediction_configs(contract)
    by_sample = {}
    expected_indices = None
    for config in configs:
        paths = sorted((root / "predictions" / config).glob("sample_*.npz"))
        if len(paths) != expected_samples:
            raise ValueError(
                "%s expected %d predictions, found %d" % (
                    config, expected_samples, len(paths)))
        indices = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sample_index = int(payload["sample_index"])
                payload_config = str(payload["config"])
                gt = np.asarray(payload["gt"], dtype=np.float32)
                pred = np.asarray(payload["pred"], dtype=np.float32)
            if payload_config != config:
                raise ValueError("prediction config mismatch: %s" % path)
            if gt.shape != pred.shape:
                raise ValueError("prediction shape mismatch: %s" % path)
            if not bool(np.all(np.isfinite(gt))):
                raise ValueError("ground truth is nonfinite: %s" % path)
            if not bool(np.all(np.isfinite(pred))):
                raise ValueError("prediction is nonfinite: %s" % path)
            sample = by_sample.setdefault(sample_index, {})
            if config in sample:
                raise ValueError("duplicate prediction: %s" % path)
            sample[config] = {"gt": gt, "pred": pred}
            indices.append(sample_index)
        if expected_indices is None:
            expected_indices = indices
        elif indices != expected_indices:
            raise ValueError("prediction sample order differs for %s" % config)

    for sample_index in expected_indices:
        sample = by_sample[sample_index]
        if set(sample) != set(configs):
            raise ValueError("prediction configurations are incomplete")
        reference = sample[FP32_CONFIG]["gt"]
        for config in configs:
            if not np.array_equal(reference, sample[config]["gt"]):
                raise ValueError(
                    "ground truth differs at sample %d" % sample_index)
    return by_sample


def pareto_point_labels(frontier):
    labels = []
    for index, row in enumerate(frontier):
        config = row["config"]
        units = "W4A4" if config == BASELINE_CONFIG else \
            config[len("FE_W8A8_"):]
        labels.append(("P%d" % index, units))
    return labels


def plot_pareto(contract, metric, out_path, dpi):
    key, xlabel = SHARE_METRICS[metric]
    frontier = sorted(
        contract["frontier"], key=lambda row: float(row[key]))
    dominated = contract["dominated"]
    point_labels = pareto_point_labels(frontier)
    fig, (axis, key_axis) = plt.subplots(
        1, 2, figsize=(12.5, 6.2),
        gridspec_kw={"width_ratios": (2.35, 1.0)})
    if dominated:
        axis.scatter(
            [100.0 * float(row[key]) for row in dominated],
            [float(row["mean_rmse"]) for row in dominated],
            color="#A7A7A7", edgecolor="#555555", linewidth=0.5,
            s=42, label="Dominated", zorder=3)
    x_front = [100.0 * float(row[key]) for row in frontier]
    y_front = [float(row["mean_rmse"]) for row in frontier]
    axis.plot(
        x_front, y_front, color="#2F6B9A", marker="o", markersize=6,
        linewidth=1.8, label="Pareto frontier", zorder=4)
    knee = contract["by_name"][contract["knee_config"]]
    axis.scatter(
        [100.0 * float(knee[key])], [float(knee["mean_rmse"])],
        marker="*", s=180, color="#E07A1F", edgecolor="#333333",
        linewidth=0.7, label="Knee", zorder=6)
    best = contract["by_name"][contract["best_config"]]
    axis.scatter(
        [100.0 * float(best[key])], [float(best["mean_rmse"])],
        marker="D", s=64, facecolor="none", edgecolor="#C43C39",
        linewidth=1.4, label="Lowest RMSE front set", zorder=5)
    axis.axhline(
        float(contract["fp32"]["mean_rmse"]), color="#3B78A8",
        linestyle=":", linewidth=1.4, label="FP32", zorder=2)
    axis.axhline(
        float(contract["w4a8"]["mean_rmse"]), color="#4C9A5F",
        linestyle="--", linewidth=1.4, label="W4A8", zorder=2)
    for index, (row, (point, units)) in enumerate(
            zip(frontier, point_labels)):
        del units
        x_offset = -22 if index == len(frontier) - 1 else 5
        y_offset = 8 if index % 2 else -12
        axis.annotate(
            point,
            (100.0 * float(row[key]), float(row["mean_rmse"])),
            xytext=(x_offset, y_offset), textcoords="offset points",
            fontsize=10,
            zorder=7)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Mean RMSE (m)")
    axis.grid(color="#C8C8C8", linewidth=0.7, alpha=0.65, zorder=0)
    axis.set_axisbelow(True)
    for label in axis.get_xticklabels():
        label.set_rotation(0)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    key_axis.set_axis_off()
    key_axis.set_xlim(0.0, 1.0)
    key_axis.set_ylim(0.0, 1.0)
    line_height = 0.92 / float(len(point_labels))
    for index, (point, units) in enumerate(point_labels):
        key_axis.text(
            0.0, 0.97 - index * line_height,
            "%s  %s" % (point, units),
            ha="left", va="top", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    return out_path


def _display_name(config, contract):
    names = {
        FP32_CONFIG: "FP32",
        BASELINE_CONFIG: "W4A4",
        W4A8_CONFIG: "W4A8",
        contract["knee_config"]: "Knee",
        contract["best_config"]: "Best",
    }
    return names[config]


def plot_predictions(root, contract, out_path, expected_samples, dpi):
    grid = load_prediction_grid(root, contract, expected_samples)
    sample_indices = sorted(grid)
    configs = _prediction_configs(contract)
    columns = 1 + 2 * len(configs)
    rows = len(sample_indices)
    figure_height = max(4.0, rows * 1.05 + 0.5)
    fig = plt.figure(figsize=(2.0 * columns, figure_height))
    grid_spec = fig.add_gridspec(
        rows + 1, columns,
        height_ratios=[1.0] * rows + [0.16],
        wspace=0.025, hspace=0.08)
    axes = np.empty((rows, columns), dtype=object)
    for row_index in range(rows):
        for column_index in range(columns):
            axes[row_index, column_index] = fig.add_subplot(
                grid_spec[row_index, column_index])
    for row_index, sample_index in enumerate(sample_indices):
        sample = grid[sample_index]
        gt = sample[FP32_CONFIG]["gt"]
        valid = gt > 1e-4
        depth_axis = axes[row_index, 0]
        depth_axis.imshow(
            np.ma.masked_where(~valid, gt), cmap="viridis",
            vmin=0.0, vmax=10.0, interpolation="nearest",
            aspect="auto", rasterized=True, zorder=2)
        if row_index == 0:
            depth_axis.set_title("GT", pad=4.0)
        depth_axis.set_ylabel(
            "%05d" % sample_index, rotation=0, ha="right", va="center")
        for config_index, config in enumerate(configs):
            pred = sample[config]["pred"]
            error = np.abs(pred - gt)
            pred_axis = axes[row_index, 1 + 2 * config_index]
            error_axis = axes[row_index, 2 + 2 * config_index]
            pred_axis.imshow(
                np.ma.masked_where(~valid, pred), cmap="viridis",
                vmin=0.0, vmax=10.0, interpolation="nearest",
                aspect="auto", rasterized=True, zorder=2)
            error_axis.imshow(
                np.ma.masked_where(~valid, error), cmap="magma",
                vmin=0.0, vmax=2.0, interpolation="nearest",
                aspect="auto", rasterized=True, zorder=2)
            if row_index == 0:
                pred_axis.set_title(_display_name(config, contract), pad=4.0)
                error_axis.set_title("|Error|", pad=4.0)
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#666666")

    depth_map = ScalarMappable(norm=Normalize(0.0, 10.0), cmap="viridis")
    error_map = ScalarMappable(norm=Normalize(0.0, 2.0), cmap="magma")
    depth_map.set_array([])
    error_map.set_array([])
    depth_color_axis = fig.add_subplot(grid_spec[-1, 2:5])
    error_color_axis = fig.add_subplot(grid_spec[-1, 7:10])
    depth_colorbar = fig.colorbar(
        depth_map, cax=depth_color_axis, orientation="horizontal")
    depth_colorbar.set_label("Depth (m)")
    error_colorbar = fig.colorbar(
        error_map, cax=error_color_axis, orientation="horizontal")
    error_colorbar.set_label("Absolute error (m)")
    fig.subplots_adjust(
        left=0.045, right=0.995,
        bottom=0.55 / figure_height,
        top=1.0 - 0.35 / figure_height)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    return out_path


def generate_figures(root, out_dir, expected_samples, dpi):
    set_style()
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = load_pareto_contract(root)
    for row in contract["aggregate"]:
        if int(row["samples"]) != int(expected_samples):
            raise ValueError(
                "%s expected %d aggregate samples, found %s" % (
                    row["config"], int(expected_samples), row["samples"]))
    return {
        "mac": plot_pareto(
            contract, "mac", out_dir / "rmse_vs_w8a8_mac_share.png", dpi),
        "parameter": plot_pareto(
            contract, "parameter",
            out_dir / "rmse_vs_w8a8_parameter_share.png", dpi),
        "operator": plot_pareto(
            contract, "operator",
            out_dir / "rmse_vs_w8a8_operator_share.png", dpi),
        "predictions": plot_predictions(
            root, contract, out_dir / "prediction_comparison_64.png",
            expected_samples, dpi),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    paths = generate_figures(
        Path(args.root), Path(args.out_dir), args.expected_samples, args.dpi)
    for name in ("mac", "parameter", "operator", "predictions"):
        print("%s=%s" % (name, paths[name]), flush=True)


if __name__ == "__main__":
    main()
