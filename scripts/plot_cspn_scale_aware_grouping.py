#!/usr/bin/env python3
"""Plot CSPN scale-aware Group-8 metrics and predictions."""

from __future__ import annotations

import argparse
import csv
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

from scripts.plot_cspn_activation_resolution_predictions import (  # noqa: E402
    DEPTH_MAX,
    DEPTH_MIN,
    _draw,
    export_pdf,
    sample_rmse,
    set_style,
)
from scripts.plot_cspn_static_calibration import (  # noqa: E402
    load_predictions as load_prediction_configs,
)


BASELINE = "W4A4_G8_MINMAX"
SCALE_AWARE = "W4A4_G8_SCALE_AWARE"
CONFIGS = (BASELINE, SCALE_AWARE)
LABELS = {
    BASELINE: "Contiguous MinMax",
    SCALE_AWARE: "Scale-aware",
}


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_predictions(experiment_dir, expected_samples=64):
    return load_prediction_configs(
        experiment_dir, CONFIGS, expected_samples)


def sample_delta_rows(predictions):
    rows = []
    for index in sorted(predictions):
        baseline = sample_rmse(predictions[index][BASELINE])
        scale_aware = sample_rmse(predictions[index][SCALE_AWARE])
        rows.append({
            "sample_index": index,
            "baseline_rmse": baseline,
            "scale_aware_rmse": scale_aware,
            "rmse_delta": scale_aware - baseline,
        })
    return rows


def _representatives(predictions):
    rows = sample_delta_rows(predictions)
    by_index = dict((row["sample_index"], row) for row in rows)
    baseline_values = np.asarray(
        [row["baseline_rmse"] for row in rows], dtype=np.float64)
    median = float(np.median(baseline_values))
    rankings = (
        sorted(by_index, key=lambda index: (
            -by_index[index]["baseline_rmse"], index)),
        sorted(by_index, key=lambda index: (
            by_index[index]["rmse_delta"], index)),
        sorted(by_index, key=lambda index: (
            -by_index[index]["rmse_delta"], index)),
        sorted(by_index, key=lambda index: (
            abs(by_index[index]["baseline_rmse"] - median), index)),
    )
    selected = []
    for ranked in rankings:
        for index in ranked:
            if index not in selected:
                selected.append(index)
                break
        if len(selected) == min(4, len(predictions)):
            break
    return tuple(selected)


def render_metrics(model_dir, out_path, dpi=160):
    rows = dict(
        (row["config"], row) for row in
        _read_rows(Path(model_dir) / "aggregate_metrics.csv"))
    if set(rows) != set(CONFIGS):
        raise ValueError("aggregate metric configurations do not match")
    metrics = ("RMSE", "MAE", "ABS_REL", "boundary_RMSE")
    labels = ("RMSE", "MAE", "AbsRel", "Boundary RMSE")
    x = np.arange(len(metrics))
    width = 0.36
    set_style(12.0)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for offset, config, color in (
            (-width / 2.0, BASELINE, "#4c78a8"),
            (width / 2.0, SCALE_AWARE, "#e45756")):
        values = np.asarray([
            float(rows[config][metric]) / float(rows[BASELINE][metric])
            for metric in metrics])
        bars = ax.bar(
            x + offset, values, width=width, color=color,
            label=LABELS[config], zorder=3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, value,
                "%.3f" % value, ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#333333", linewidth=1.0, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Metric / contiguous baseline")
    ax.set_ylim(0.96, 1.04)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def render_sample_deltas(predictions, out_path, dpi=160):
    rows = sorted(sample_delta_rows(predictions),
                  key=lambda row: row["rmse_delta"])
    deltas = np.asarray([row["rmse_delta"] for row in rows])
    colors = np.where(deltas <= 0.0, "#54a24b", "#e45756")
    set_style(11.0)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(np.arange(len(rows)), deltas, color=colors, width=0.82, zorder=3)
    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=2)
    ax.set_xlabel("Evaluation samples sorted by RMSE delta")
    ax.set_ylabel("Scale-aware - contiguous RMSE (m)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def render_grouping_impact(input_dir, out_path, dpi=160):
    summaries = _read_rows(Path(input_dir) / "analysis" /
                           "grouping_summary.csv")
    activation = _read_rows(Path(input_dir) / "cspn" /
                            "activation_resolution_metrics.csv")
    aware = dict(
        ((row["module"], row["kind"]), row) for row in activation
        if row["config"] == SCALE_AWARE)
    x = np.asarray([
        float(row["dispersion_sum_reduction"]) * 100.0
        for row in summaries])
    y = np.asarray([
        float(aware[(row["module"], row["kind"])]["total_error_energy"])
        for row in summaries])
    set_style(11.0)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.scatter(x, y, s=40, color="#4c78a8", alpha=0.82, zorder=3)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("RMS dispersion reduction (%)")
    ax.set_ylabel("Added consumer-input error energy")
    ax.grid(color="#d9d9d9", linewidth=0.8, zorder=0)
    ranked = np.argsort(y)[-3:][::-1]
    offsets = ((5, 7, "left"), (-5, -14, "right"), (5, 8, "left"))
    for index, (x_offset, y_offset, alignment) in zip(ranked, offsets):
        ax.annotate(
            summaries[int(index)]["module"], (x[index], y[index]),
            xytext=(x_offset, y_offset), textcoords="offset points",
            ha=alignment, fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def _error_limit(predictions):
    values = []
    for payloads in predictions.values():
        valid = payloads[BASELINE]["valid_gt"]
        for config in CONFIGS:
            values.append(payloads[config]["abs_err"][valid])
    return max(float(np.quantile(np.concatenate(values), 0.99)), 0.25)


def render_prediction_detail(predictions, out_path, dpi=160):
    selected = _representatives(predictions)
    error_max = _error_limit(predictions)
    set_style(9.0)
    fig, axes = plt.subplots(
        len(selected), 7, figsize=(18.0, 3.2 * len(selected)),
        squeeze=False)
    for row, index in enumerate(selected):
        payloads = predictions[index]
        baseline = payloads[BASELINE]
        aware = payloads[SCALE_AWARE]
        valid = baseline["valid_gt"]
        panels = (
            (baseline["rgb"], None, None, None, "#%05d RGB" % index),
            (baseline["gt"], "viridis", DEPTH_MIN, DEPTH_MAX, "GT"),
            (baseline["fp32"], "viridis", DEPTH_MIN, DEPTH_MAX, "FP32"),
            (baseline["pred"], "viridis", DEPTH_MIN, DEPTH_MAX,
             "MinMax %.3f" % sample_rmse(baseline)),
            (aware["pred"], "viridis", DEPTH_MIN, DEPTH_MAX,
             "Scale-aware %.3f" % sample_rmse(aware)),
            (baseline["abs_err"], "magma", 0.0, error_max,
             "MinMax error"),
            (aware["abs_err"], "magma", 0.0, error_max,
             "Scale-aware error"),
        )
        for column, (image, cmap, lower, upper, label) in enumerate(panels):
            displayed = image if column == 0 else \
                np.ma.masked_where(~valid, image)
            _draw(axes[row, column], displayed, cmap, lower, upper, label)
    fig.subplots_adjust(
        left=0.02, right=0.955, bottom=0.04, top=0.97,
        wspace=0.04, hspace=0.18)
    depth_axis = fig.add_axes((0.963, 0.55, 0.008, 0.37))
    depth = ScalarMappable(
        norm=Normalize(DEPTH_MIN, DEPTH_MAX), cmap="viridis")
    depth.set_array([])
    fig.colorbar(depth, cax=depth_axis).set_label("Depth (m)")
    error_axis = fig.add_axes((0.963, 0.08, 0.008, 0.37))
    error = ScalarMappable(norm=Normalize(0.0, error_max), cmap="magma")
    error.set_array([])
    fig.colorbar(error, cax=error_axis).set_label("Absolute error (m)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def render_contact_sheet(predictions, out_path, dpi=120):
    indices = sorted(predictions)
    columns = 4
    panels = 4
    rows = int(np.ceil(len(indices) / float(columns)))
    set_style(6.5)
    fig, axes = plt.subplots(
        rows, columns * panels, figsize=(38.0, 30.0), squeeze=False)
    for rank, index in enumerate(indices):
        row = rank // columns
        start = (rank % columns) * panels
        payloads = predictions[index]
        valid = payloads[BASELINE]["valid_gt"]
        images = (
            (payloads[BASELINE]["gt"], "#%05d GT" % index),
            (payloads[BASELINE]["fp32"], "FP32"),
            (payloads[BASELINE]["pred"], "MinMax\n%.3f" %
             sample_rmse(payloads[BASELINE])),
            (payloads[SCALE_AWARE]["pred"], "Scale-aware\n%.3f" %
             sample_rmse(payloads[SCALE_AWARE])),
        )
        for offset, (image, label) in enumerate(images):
            _draw(axes[row, start + offset],
                  np.ma.masked_where(~valid, image), "viridis",
                  DEPTH_MIN, DEPTH_MAX, label)
    fig.subplots_adjust(
        left=0.008, right=0.972, bottom=0.008, top=0.992,
        wspace=0.035, hspace=0.24)
    color_axis = fig.add_axes((0.978, 0.04, 0.006, 0.92))
    scalar = ScalarMappable(
        norm=Normalize(DEPTH_MIN, DEPTH_MAX), cmap="viridis")
    scalar.set_array([])
    fig.colorbar(scalar, cax=color_axis).set_label("Depth (m)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--dpi", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output = Path(args.out_dir)
    predictions = load_predictions(
        input_dir / "cspn", args.expected_samples)
    paths = (
        render_metrics(
            input_dir / "cspn",
            output / "cspn_scale_aware_metrics.png", args.dpi),
        render_sample_deltas(
            predictions,
            output / "cspn_scale_aware_sample_deltas.png", args.dpi),
        render_grouping_impact(
            input_dir,
            output / "cspn_scale_aware_grouping_impact.png", args.dpi),
        render_prediction_detail(
            predictions,
            output / "cspn_scale_aware_prediction_detail.png", args.dpi),
        render_contact_sheet(
            predictions,
            output / "cspn_scale_aware_contact_sheet.png", 120),
    )
    for path in paths:
        export_pdf(path, path.with_suffix(".pdf"), args.dpi)


if __name__ == "__main__":
    main()
