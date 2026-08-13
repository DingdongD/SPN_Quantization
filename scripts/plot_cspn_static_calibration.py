#!/usr/bin/env python3
"""Plot CSPN static calibration metrics and predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    _load_payload,
    export_pdf,
    sample_rmse,
    set_style,
)


CONFIG_ORDER = (
    "W4A4_G8_MINMAX",
    "W4A4_G8_PERCENTILE_P999",
    "W4A4_G8_PERCENTILE_P9999",
    "W4A4_G8_HIST_MSE",
)
CONFIG_LABELS = {
    "W4A4_G8_MINMAX": "MinMax",
    "W4A4_G8_PERCENTILE_P999": "P99.9",
    "W4A4_G8_PERCENTILE_P9999": "P99.99",
    "W4A4_G8_HIST_MSE": "Histogram-MSE",
}


def load_predictions(experiment_dir, configs, expected_samples=64):
    configs = tuple(configs)
    root = Path(experiment_dir) / "predictions"
    directories = {path.name for path in root.iterdir() if path.is_dir()}
    if directories != set(configs):
        raise ValueError("prediction configuration directories do not match")
    by_config = {}
    for config in configs:
        payloads = {}
        for path in sorted((root / config).glob("sample_*.npz")):
            payload = _load_payload(path, config)
            index = int(payload["sample_index"])
            if index in payloads:
                raise ValueError("duplicate prediction sample index")
            payloads[index] = payload
        if len(payloads) != int(expected_samples):
            raise ValueError("prediction sample count does not match")
        by_config[config] = payloads
    indices = set(by_config[configs[0]])
    for config in configs[1:]:
        if set(by_config[config]) != indices:
            raise ValueError("prediction sample indices do not match")
    output = {}
    for index in sorted(indices):
        payloads = dict(
            (config, by_config[config][index]) for config in configs)
        reference = payloads[configs[0]]
        for config in configs[1:]:
            current = payloads[config]
            for key in ("gt", "fp32", "valid_gt", "sparse", "rgb"):
                if not np.array_equal(reference[key], current[key]):
                    raise ValueError("prediction payloads are inconsistent")
        output[index] = payloads
    return output


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def render_rmse(experiment_dir, out_path, dpi=180):
    rows = dict(
        (row["config"], row) for row in
        _read_rows(Path(experiment_dir) / "aggregate_metrics.csv"))
    if set(rows) != set(CONFIG_ORDER):
        raise ValueError("aggregate metric configurations do not match")
    values = [float(rows[config]["RMSE"]) for config in CONFIG_ORDER]
    set_style(12.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(
        np.arange(len(CONFIG_ORDER)), values,
        color=("#4c78a8", "#f58518", "#54a24b", "#e45756"),
        width=0.66, zorder=3)
    ax.set_xticks(
        np.arange(len(CONFIG_ORDER)),
        [CONFIG_LABELS[config] for config in CONFIG_ORDER])
    ax.set_ylabel("RMSE (m)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value,
                "%.3f" % value, ha="center", va="bottom")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def render_error_composition(experiment_dir, out_path, dpi=180):
    rows = _read_rows(
        Path(experiment_dir) / "activation_resolution_metrics.csv")
    totals = dict((config, {
        "signal": 0.0, "total": 0.0, "clipping": 0.0,
        "rounding": 0.0, "zero": 0.0,
    }) for config in CONFIG_ORDER)
    for row in rows:
        config = row["config"]
        if config not in totals:
            raise ValueError("unknown activation metric configuration")
        totals[config]["signal"] += float(row["signal_energy"])
        totals[config]["total"] += float(row["total_error_energy"])
        totals[config]["clipping"] += float(row["clipping_error_energy"])
        totals[config]["rounding"] += float(row["rounding_error_energy"])
        totals[config]["zero"] += float(row["zero_collapse_error_energy"])
    x = np.arange(len(CONFIG_ORDER))
    zero = np.asarray([
        totals[config]["zero"] / totals[config]["total"] * 100.0
        for config in CONFIG_ORDER])
    rounding = np.asarray([
        totals[config]["rounding"] / totals[config]["total"] * 100.0
        for config in CONFIG_ORDER])
    clipping = np.asarray([
        totals[config]["clipping"] / totals[config]["total"] * 100.0
        for config in CONFIG_ORDER])
    set_style(12.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x, zero, width=0.66, label="Zero collapse",
           color="#72b7b2", zorder=3)
    ax.bar(x, rounding, width=0.66, bottom=zero, label="Rounding",
           color="#f2cf5b", zorder=3)
    ax.bar(x, clipping, width=0.66, bottom=zero + rounding,
           label="Clipping", color="#e45756", zorder=3)
    ax.set_xticks(x, [CONFIG_LABELS[config] for config in CONFIG_ORDER])
    ax.set_ylabel("Activation error energy (%)")
    ax.set_ylim(0.0, 108.0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for index, config in enumerate(CONFIG_ORDER):
        sqnr = 10.0 * math.log10(
            totals[config]["signal"] / totals[config]["total"])
        ax.text(index, 101.0, "%.1f dB" % sqnr,
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def _first_unused(ranked, used):
    for index in ranked:
        if index not in used:
            return index
    raise ValueError("representative samples must be distinct")


def _representatives(predictions, baseline, candidate):
    baseline_rmse = dict(
        (index, sample_rmse(predictions[index][baseline]))
        for index in predictions)
    candidate_rmse = dict(
        (index, sample_rmse(predictions[index][candidate]))
        for index in predictions)
    median = float(np.median(tuple(baseline_rmse.values())))
    rankings = (
        ("MinMax worst", sorted(
            baseline_rmse,
            key=lambda index: (-baseline_rmse[index], index))),
        ("Largest candidate gain", sorted(
            baseline_rmse,
            key=lambda index: (-(baseline_rmse[index] -
                                 candidate_rmse[index]), index))),
        ("Largest candidate loss", sorted(
            baseline_rmse,
            key=lambda index: (-(candidate_rmse[index] -
                                 baseline_rmse[index]), index))),
        ("MinMax median", sorted(
            baseline_rmse,
            key=lambda index: (abs(baseline_rmse[index] - median), index))),
    )
    used = set()
    output = []
    for label, ranked in rankings:
        index = _first_unused(ranked, used)
        output.append((label, index))
        used.add(index)
    return tuple(output)


def render_prediction_detail(predictions, baseline, candidate,
                             out_path, dpi=150):
    selected = _representatives(predictions, baseline, candidate)
    errors = []
    for payloads in predictions.values():
        valid = payloads[baseline]["valid_gt"]
        errors.extend((payloads[baseline]["abs_err"][valid],
                       payloads[candidate]["abs_err"][valid]))
    error_max = max(float(np.percentile(np.concatenate(errors), 99.0)), 1e-3)
    set_style(9.5)
    columns = (
        "RGB", "Sparse", "GT", "FP32", "MinMax",
        "MinMax |error|", CONFIG_LABELS[candidate],
        "%s |error|" % CONFIG_LABELS[candidate],
    )
    fig, axes = plt.subplots(
        len(selected), len(columns), figsize=(20.0, 8.5), squeeze=False)
    for row, (label, index) in enumerate(selected):
        payloads = predictions[index]
        reference = payloads[baseline]
        valid = reference["valid_gt"]
        panels = (
            (reference["rgb"], None, None, None),
            (np.ma.masked_where(reference["sparse"] <= 1e-4,
                                reference["sparse"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, reference["gt"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, reference["fp32"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, payloads[baseline]["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, payloads[baseline]["abs_err"]),
             "magma", 0.0, error_max),
            (np.ma.masked_where(~valid, payloads[candidate]["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, payloads[candidate]["abs_err"]),
             "magma", 0.0, error_max),
        )
        for column, panel in enumerate(panels):
            _draw(axes[row, column], panel[0], panel[1], panel[2], panel[3],
                  columns[column] if row == 0 else "")
        axes[row, 0].set_ylabel(
            "%s\n#%05d" % (label, index), labelpad=5.0)
        axes[row, 4].set_xlabel(
            "RMSE %.3f m" % sample_rmse(payloads[baseline]))
        axes[row, 6].set_xlabel(
            "RMSE %.3f m" % sample_rmse(payloads[candidate]))
    fig.subplots_adjust(
        left=0.065, right=0.96, bottom=0.055, top=0.95,
        wspace=0.055, hspace=0.20)
    depth_axis = fig.add_axes((0.968, 0.54, 0.008, 0.36))
    depth = ScalarMappable(
        norm=Normalize(DEPTH_MIN, DEPTH_MAX), cmap="viridis")
    depth.set_array([])
    fig.colorbar(depth, cax=depth_axis).set_label("Depth (m)")
    error_axis = fig.add_axes((0.968, 0.09, 0.008, 0.36))
    error = ScalarMappable(norm=Normalize(0.0, error_max), cmap="magma")
    error.set_array([])
    fig.colorbar(error, cax=error_axis).set_label("Absolute error (m)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def render_contact_sheet(predictions, baseline, candidate,
                         out_path, dpi=120):
    indices = sorted(predictions)
    sample_columns = 4
    panels_per_sample = 4
    rows = int(np.ceil(len(indices) / float(sample_columns)))
    set_style(6.5)
    fig, axes = plt.subplots(
        rows, sample_columns * panels_per_sample,
        figsize=(38.0, 30.0), squeeze=False)
    for rank, index in enumerate(indices):
        row = rank // sample_columns
        first = (rank % sample_columns) * panels_per_sample
        payloads = predictions[index]
        valid = payloads[baseline]["valid_gt"]
        panels = (
            (payloads[baseline]["gt"], "#%05d GT" % index),
            (payloads[baseline]["fp32"], "FP32"),
            (payloads[baseline]["pred"], "MinMax\n%.3f" %
             sample_rmse(payloads[baseline])),
            (payloads[candidate]["pred"], "%s\n%.3f" %
             (CONFIG_LABELS[candidate], sample_rmse(payloads[candidate]))),
        )
        for offset, (image, label) in enumerate(panels):
            _draw(axes[row, first + offset],
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
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, choices=(64,),
                        required=True)
    parser.add_argument("--dpi", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    metadata = json.loads(
        (experiment_dir / "metadata.json").read_text(encoding="utf-8"))
    configs = tuple(metadata["prediction_configs"])
    baseline = "W4A4_G8_MINMAX"
    candidate = str(metadata["selected_non_minmax_prediction"])
    predictions = load_predictions(
        experiment_dir, configs, args.expected_samples)
    output = Path(args.output_dir)
    paths = (
        render_rmse(
            experiment_dir, output / "cspn_static_calibration_rmse.png",
            args.dpi),
        render_error_composition(
            experiment_dir,
            output / "cspn_static_calibration_error_composition.png",
            args.dpi),
        render_prediction_detail(
            predictions, baseline, candidate,
            output / "cspn_static_calibration_detail.png", args.dpi),
        render_contact_sheet(
            predictions, baseline, candidate,
            output / "cspn_static_calibration_contact_sheet.png", args.dpi),
    )
    for path in paths:
        export_pdf(path, path.with_suffix(".pdf"), args.dpi)


if __name__ == "__main__":
    main()
