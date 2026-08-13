#!/usr/bin/env python3
"""Plot CSPN SmoothQuant Group-A4 metrics and predictions."""

from __future__ import annotations

import argparse
import csv
import json
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

from scripts.plot_cspn_activation_resolution_predictions import (
    DEPTH_MAX,
    DEPTH_MIN,
    _draw,
    _load_payload,
    export_pdf,
    sample_rmse,
    set_style,
)


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
    predictions = {}
    for index in sorted(indices):
        payloads = dict(
            (config, by_config[config][index]) for config in configs)
        reference = payloads["FP32"]
        for config in configs[1:]:
            current = payloads[config]
            for key in ("gt", "fp32", "valid_gt", "sparse", "rgb"):
                if not np.array_equal(reference[key], current[key]):
                    raise ValueError("prediction payloads are inconsistent")
        predictions[index] = payloads
    return predictions


def _first_unused(ranked, used):
    for index in ranked:
        if index not in used:
            return index
    raise ValueError("representative samples must be distinct")


def select_representative_samples(predictions, smooth_config):
    if len(predictions) < 4:
        raise ValueError("prediction detail requires four samples")
    rtn = dict(
        (index, sample_rmse(predictions[index]["W4A4_GROUP8"]))
        for index in predictions)
    smooth = dict(
        (index, sample_rmse(predictions[index][smooth_config]))
        for index in predictions)
    median = float(np.median(tuple(rtn.values())))
    rankings = (
        ("RTN worst", sorted(rtn, key=lambda index: (-rtn[index], index))),
        ("Largest SQ gain", sorted(
            rtn, key=lambda index: (-(rtn[index] - smooth[index]), index))),
        ("Largest SQ loss", sorted(
            rtn, key=lambda index: (-(smooth[index] - rtn[index]), index))),
        ("RTN median", sorted(
            rtn, key=lambda index: (abs(rtn[index] - median), index))),
    )
    used = set()
    selected = []
    for label, ranked in rankings:
        index = _first_unused(ranked, used)
        selected.append((label, index))
        used.add(index)
    return tuple(selected)


def _global_error_limit(predictions, smooth_config):
    errors = []
    for payloads in predictions.values():
        valid = payloads["FP32"]["valid_gt"]
        errors.append(payloads["W4A4_GROUP8"]["abs_err"][valid])
        errors.append(payloads[smooth_config]["abs_err"][valid])
    return max(float(np.percentile(np.concatenate(errors), 99.0)), 1e-3)


def render_prediction_detail(predictions, smooth_config, out_path, dpi=150):
    set_style(9.5)
    selected = select_representative_samples(predictions, smooth_config)
    error_max = _global_error_limit(predictions, smooth_config)
    columns = (
        "RGB", "Sparse", "GT", "FP32", "G8 RTN",
        "G8 RTN |error|", "G8 SmoothQuant", "G8 SQ |error|",
    )
    fig, axes = plt.subplots(
        len(selected), len(columns), figsize=(20.0, 8.5), squeeze=False)
    for row, (label, index) in enumerate(selected):
        payloads = predictions[index]
        reference = payloads["FP32"]
        valid = reference["valid_gt"]
        panels = (
            (reference["rgb"], None, None, None),
            (np.ma.masked_where(reference["sparse"] <= 1e-4,
                                reference["sparse"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, reference["gt"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, reference["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(
                ~valid, payloads["W4A4_GROUP8"]["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(
                ~valid, payloads["W4A4_GROUP8"]["abs_err"]),
             "magma", 0.0, error_max),
            (np.ma.masked_where(~valid, payloads[smooth_config]["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, payloads[smooth_config]["abs_err"]),
             "magma", 0.0, error_max),
        )
        for column, panel in enumerate(panels):
            title = columns[column] if row == 0 else ""
            _draw(axes[row, column], panel[0], panel[1], panel[2], panel[3],
                  title)
        axes[row, 0].set_ylabel(
            "%s\n#%05d" % (label, index), labelpad=5.0)
        axes[row, 4].set_xlabel(
            "RMSE %.3f m" % sample_rmse(payloads["W4A4_GROUP8"]))
        axes[row, 6].set_xlabel(
            "RMSE %.3f m" % sample_rmse(payloads[smooth_config]))
    fig.subplots_adjust(
        left=0.06, right=0.96, bottom=0.055, top=0.95,
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


def render_contact_sheet(predictions, smooth_config, out_path, dpi=120):
    set_style(6.5)
    indices = sorted(predictions)
    sample_columns = 4
    panels_per_sample = 4
    rows = int(np.ceil(len(indices) / float(sample_columns)))
    fig, axes = plt.subplots(
        rows, sample_columns * panels_per_sample,
        figsize=(38.0, 30.0), squeeze=False)
    for rank, index in enumerate(indices):
        row = rank // sample_columns
        first = (rank % sample_columns) * panels_per_sample
        payloads = predictions[index]
        valid = payloads["FP32"]["valid_gt"]
        panels = (
            (payloads["FP32"]["gt"], "#%05d GT" % index),
            (payloads["FP32"]["pred"], "FP32\n%.3f" %
             sample_rmse(payloads["FP32"])),
            (payloads["W4A4_GROUP8"]["pred"], "G8 RTN\n%.3f" %
             sample_rmse(payloads["W4A4_GROUP8"])),
            (payloads[smooth_config]["pred"], "G8 SQ\n%.3f" %
             sample_rmse(payloads[smooth_config])),
        )
        for offset, (image, title) in enumerate(panels):
            _draw(axes[row, first + offset],
                  np.ma.masked_where(~valid, image), "viridis",
                  DEPTH_MIN, DEPTH_MAX, title)
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


def render_rmse(experiment_dir, out_path, dpi=180):
    with (Path(experiment_dir) / "aggregate_metrics.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = dict(
            (row["config"], row) for row in csv.DictReader(stream))
    labels = ("RTN", "alpha=0.25", "alpha=0.50", "alpha=0.75")
    set_style(12.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(labels))
    for group_size, color, marker in (
            (16, "#4c78a8", "o"), (8, "#e45756", "s")):
        names = (
            "W4A4_GROUP%d" % group_size,
            "SQ_W4A4_GROUP%d_A025" % group_size,
            "SQ_W4A4_GROUP%d_A050" % group_size,
            "SQ_W4A4_GROUP%d_A075" % group_size,
        )
        values = [float(rows[name]["RMSE"]) for name in names]
        ax.plot(x, values, color=color, marker=marker, linewidth=2.0,
                markersize=7.0, label="Group %d" % group_size, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (m)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def render_scale_pareto(experiment_dir, out_path, dpi=180):
    experiment_dir = Path(experiment_dir)
    with (experiment_dir / "aggregate_metrics.csv").open(
            newline="", encoding="utf-8") as stream:
        metrics = dict(
            (row["config"], row) for row in csv.DictReader(stream))
    with (experiment_dir / "config_manifest.csv").open(
            newline="", encoding="utf-8") as stream:
        manifest = dict(
            (row["config"], row) for row in csv.DictReader(stream))
    methods = (
        ("RTN", "W4A4_GROUP%d", "#4c78a8", "o"),
        ("alpha=0.25", "SQ_W4A4_GROUP%d_A025", "#f58518", "s"),
        ("alpha=0.50", "SQ_W4A4_GROUP%d_A050", "#54a24b", "^"),
        ("alpha=0.75", "SQ_W4A4_GROUP%d_A075", "#e45756", "D"),
    )
    set_style(12.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    scale_ticks = set()
    for label, template, color, marker in methods:
        names = tuple(template % group_size for group_size in (16, 8))
        scales = [int(manifest[name]["activation_scales"])
                  for name in names]
        scale_ticks.update(scales)
        values = [float(metrics[name]["RMSE"]) for name in names]
        ax.plot(scales, values, color=color, marker=marker, linewidth=2.0,
                markersize=7.0, label=label, zorder=3)
    ax.set_xlabel("Activation scale count")
    ax.set_ylabel("RMSE (m)")
    ax.set_xticks(sorted(scale_ticks))
    ax.grid(color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
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
    smooth_config = str(metadata["selected_group8"])
    predictions = load_predictions(
        experiment_dir, configs, args.expected_samples)
    output = Path(args.output_dir)
    detail = render_prediction_detail(
        predictions, smooth_config, output / "cspn_smoothquant_detail.png",
        args.dpi)
    contact = render_contact_sheet(
        predictions, smooth_config,
        output / "cspn_smoothquant_contact_sheet.png", args.dpi)
    rmse = render_rmse(
        experiment_dir, output / "cspn_smoothquant_rmse.png", args.dpi)
    pareto = render_scale_pareto(
        experiment_dir, output / "cspn_smoothquant_scale_pareto.png",
        args.dpi)
    export_pdf(detail, detail.with_suffix(".pdf"), args.dpi)
    export_pdf(contact, contact.with_suffix(".pdf"), args.dpi)
    export_pdf(rmse, rmse.with_suffix(".pdf"), args.dpi)
    export_pdf(pareto, pareto.with_suffix(".pdf"), args.dpi)


if __name__ == "__main__":
    main()
