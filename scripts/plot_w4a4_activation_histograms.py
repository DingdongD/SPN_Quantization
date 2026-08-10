#!/usr/bin/env python3
"""Plot strict W4A4 activation histogram profiles."""

from __future__ import division

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_LABELS = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}

COLORS = {
    "reference": "#3b6ea8",
    "magnitude": "#2a9d8f",
    "error": "#e76f51",
    "codes": "#f4a261",
}


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.axisbelow": True,
        "axes.titlepad": 0,
    })


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty comparison rows")
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_profile(model_dir):
    model_dir = Path(model_dir)
    index_rows = read_csv(model_dir / "histogram_index.csv")
    summary_rows = read_csv(model_dir / "outlier_summary.csv")
    with np.load(model_dir / "histogram_data.npz") as source:
        arrays = dict((key, source[key].copy()) for key in source.files)
    index = dict((row["site"], row) for row in index_rows)
    summary = dict((row["site"], row) for row in summary_rows)
    if set(index) != set(summary):
        raise ValueError("histogram index and summary sites differ")
    indexed_arrays = {
        row[field]
        for row in index_rows
        for field in row
        if field.endswith("_key")
    }
    if indexed_arrays != set(arrays):
        raise ValueError("histogram index and NPZ arrays differ")
    return {
        "model_dir": model_dir,
        "sites": sorted(index),
        "index": index,
        "summary": summary,
        "arrays": arrays,
    }


def _style_axis(axis):
    axis.grid(True, color="#d7dce2", linewidth=0.7, alpha=0.8, zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#8d96a0")


def _positive_log_limits(counts):
    maximum = max(1, int(np.max(counts)))
    return 0.8, maximum * 1.5


def finite_plot_values(values):
    original = np.asarray(values, dtype=np.float64)
    if bool(np.isnan(original).any()):
        raise ValueError("plot metric contains NaN")
    finite = original[np.isfinite(original)]
    if finite.size == 0:
        lower = -1.0
        upper = 1.0
    else:
        lower = float(finite.min())
        upper = float(finite.max())
    padding = max(1.0, abs(lower), abs(upper)) * 0.1
    plotted = original.copy()
    plotted[np.isposinf(plotted)] = upper + padding
    plotted[np.isneginf(plotted)] = lower - padding
    labels = [
        "Inf" if math.isinf(value) and value > 0.0 else
        "-Inf" if math.isinf(value) else ""
        for value in original
    ]
    return plotted, labels


def _horizontal_bars(axis, values, labels, color):
    plotted, annotations = finite_plot_values(values)
    positions = np.arange(len(values))
    bars = axis.barh(positions, plotted, color=color, zorder=3)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    for bar, annotation in zip(bars, annotations):
        if annotation:
            axis.text(
                bar.get_width(), bar.get_y() + bar.get_height() / 2.0,
                " " + annotation, va="center", fontsize=8, zorder=4)


def _vertical_bars(axis, values, labels, color):
    plotted, annotations = finite_plot_values(values)
    positions = np.arange(len(values))
    bars = axis.bar(positions, plotted, color=color, zorder=3)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=0)
    for bar, annotation in zip(bars, annotations):
        if annotation:
            axis.text(
                bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                annotation, ha="center", va="bottom", fontsize=8, zorder=4)


def _plot_site(axis_row, profile, site):
    row = profile["index"][site]
    summary = profile["summary"][site]
    arrays = profile["arrays"]

    signed_edges = arrays[row["signed_edges_key"]]
    reference_counts = arrays[row["reference_counts_key"]]
    axis_row[0].stairs(
        reference_counts, signed_edges, color=COLORS["reference"],
        linewidth=1.4, zorder=3)
    axis_row[0].set_xlabel("Reference activation")
    axis_row[0].set_ylabel("Count")
    axis_row[0].text(
        0.02, 0.96, "zero=%s" % row["reference_zeros"],
        transform=axis_row[0].transAxes, va="top", fontsize=8)

    magnitude_edges = arrays[row["magnitude_edges_key"]] / \
        float(summary["magnitude_normalizer"])
    magnitude_counts = arrays[row["magnitude_counts_key"]]
    axis_row[1].stairs(
        magnitude_counts, magnitude_edges, color=COLORS["magnitude"],
        linewidth=1.4, zorder=3)
    axis_row[1].set_xscale("log", base=2)
    axis_row[1].set_yscale("log")
    axis_row[1].set_ylim(*_positive_log_limits(magnitude_counts))
    axis_row[1].set_xlabel("|x| / p99-normalizer")

    error_edges = arrays[row["error_edges_key"]]
    error_counts = arrays[row["error_counts_key"]]
    axis_row[2].stairs(
        error_counts, error_edges, color=COLORS["error"],
        linewidth=1.4, zorder=3)
    axis_row[2].axvline(0.0, color="#4b5563", linewidth=0.8, zorder=2)
    axis_row[2].set_xlabel("x - Q(x)")

    code_values = arrays[row["code_values_key"]]
    code_counts = arrays[row["code_counts_key"]]
    axis_row[3].bar(
        code_values, code_counts, width=0.8,
        color=COLORS["codes"], edgecolor="white", linewidth=0.3,
        zorder=3)
    axis_row[3].set_xlabel("Integer code")
    axis_row[3].tick_params(axis="x", labelrotation=0)

    axis_row[0].text(
        0.0, 1.12,
        "%s | %s | %s | %s-bit %s" % (
            site, row["group"], row["kind"], row["bits"],
            row["granularity"]),
        transform=axis_row[0].transAxes, fontsize=9, va="bottom")
    for axis in axis_row:
        _style_axis(axis)


def plot_all_sites(profile, sites_per_page):
    if sites_per_page <= 0:
        raise ValueError("sites per page must be positive")
    destination = profile["model_dir"] / "all_sites_histograms.pdf"
    sites = profile["sites"]
    with PdfPages(destination) as document:
        for offset in range(0, len(sites), sites_per_page):
            page_sites = sites[offset:offset + sites_per_page]
            figure, axes = plt.subplots(
                sites_per_page, 4,
                figsize=(16, 4.0 * sites_per_page), squeeze=False)
            figure.subplots_adjust(
                left=0.07, right=0.98, top=0.95, bottom=0.08,
                hspace=0.62, wspace=0.34)
            for row_index, site in enumerate(page_sites):
                _plot_site(axes[row_index], profile, site)
            for row_index in range(len(page_sites), sites_per_page):
                for axis in axes[row_index]:
                    axis.axis("off")
            document.savefig(figure, dpi=160)
            plt.close(figure)
    return destination


def _ranked_rows(profile, rank_field, limit):
    rows = [
        row for row in profile["summary"].values()
        if int(row["critical_selection_eligible"]) == 1
    ]
    rows.sort(key=lambda row: int(row[rank_field]))
    return rows[:limit]


def plot_critical_layers(profile, limit):
    if limit <= 0:
        raise ValueError("critical layer limit must be positive")
    specifications = (
        ("error_energy_rank", "local_error_energy_share", "Error energy share"),
        ("sqnr_rank", "sqnr_db", "SQNR (dB)"),
        ("tail_rank", "p99_99_over_p99", "p99.99 / p99"),
        ("channel_imbalance_rank", "channel_max_over_median",
         "Channel max / median"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    for axis, specification in zip(axes.reshape(-1), specifications):
        rank_field, metric, label = specification
        rows = list(reversed(_ranked_rows(profile, rank_field, limit)))
        values = [float(row[metric]) for row in rows]
        labels = [row["site"] for row in rows]
        _horizontal_bars(axis, values, labels, COLORS["reference"])
        axis.set_xlabel(label)
        _style_axis(axis)
    figure.subplots_adjust(
        left=0.28, right=0.98, top=0.97, bottom=0.08,
        hspace=0.32, wspace=0.42)
    destination = profile["model_dir"] / "critical_layers.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_rgb_depth_inputs(profile):
    rows = [
        row for row in profile["index"].values()
        if row["module"] in ("input_rgb", "input_depth")
    ]
    rows.sort(key=lambda row: row["module"], reverse=True)
    if [row["module"] for row in rows] != ["input_rgb", "input_depth"]:
        raise ValueError("profile must contain one RGB and one depth input slice")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    arrays = profile["arrays"]
    for row_index, row in enumerate(rows):
        summary = profile["summary"][row["site"]]
        color = COLORS["reference"] if row["module"] == "input_rgb" \
            else COLORS["error"]
        edges = arrays[row["magnitude_edges_key"]] / \
            float(summary["magnitude_normalizer"])
        counts = arrays[row["magnitude_counts_key"]]
        axes[row_index, 0].stairs(
            counts, edges, color=color, linewidth=1.6, zorder=3)
        axes[row_index, 0].set_xscale("log", base=2)
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylim(*_positive_log_limits(counts))
        axes[row_index, 0].set_xlabel("|x| / p99-normalizer")
        axes[row_index, 0].set_ylabel(
            "RGB count" if row["module"] == "input_rgb" else "Depth count")
        axes[row_index, 0].text(
            0.02, 0.96,
            "zero=%.4f  saturation=%.4f" % (
                float(summary["reference_zero_ratio"]),
                float(summary["saturation_ratio"])),
            transform=axes[row_index, 0].transAxes, va="top", fontsize=9)

        codes = arrays[row["code_values_key"]]
        code_counts = arrays[row["code_counts_key"]]
        axes[row_index, 1].bar(
            codes, code_counts, width=0.8,
            color=color, edgecolor="white", linewidth=0.3, zorder=3)
        axes[row_index, 1].set_xlabel("Integer code")
        axes[row_index, 1].tick_params(axis="x", labelrotation=0)
        for axis in axes[row_index]:
            _style_axis(axis)
    figure.subplots_adjust(
        left=0.10, right=0.98, top=0.97, bottom=0.09,
        hspace=0.38, wspace=0.28)
    destination = profile["model_dir"] / "rgb_depth_input_histograms.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _group_rows(profile):
    rows = [
        row for row in profile["summary"].values()
        if int(row["synthetic_slice"]) == 0
    ]
    groups = sorted(set(row["group"] for row in rows))
    result = []
    for group in groups:
        selected = [row for row in rows if row["group"] == group]
        shares = set(float(row["group_error_energy_share"])
                     for row in selected)
        if len(shares) != 1:
            raise ValueError("group error shares are inconsistent")
        result.append({
            "group": group,
            "sites": len(selected),
            "error_share": shares.pop(),
            "median_tail": float(np.median([
                float(row["p99_99_over_p99"]) for row in selected])),
            "median_channel_imbalance": float(np.median([
                float(row["channel_max_over_median"]) for row in selected])),
        })
    return result


def plot_group_distribution(profile):
    rows = _group_rows(profile)
    labels = [row["group"] for row in rows]
    metrics = (
        ("sites", "Site count", COLORS["reference"]),
        ("error_share", "Error energy share", COLORS["error"]),
        ("median_tail", "Median p99.99 / p99", COLORS["magnitude"]),
        ("median_channel_imbalance", "Median channel max / median",
         COLORS["codes"]),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for axis, metric in zip(axes.reshape(-1), metrics):
        field, label, color = metric
        _vertical_bars(
            axis, [row[field] for row in rows], labels, color)
        axis.set_ylabel(label)
        _style_axis(axis)
    figure.subplots_adjust(
        left=0.09, right=0.98, top=0.97, bottom=0.10,
        hspace=0.34, wspace=0.28)
    destination = profile["model_dir"] / "group_outlier_distribution.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_model_profile(model_dir, sites_per_page=2, critical_limit=10):
    configure_style()
    profile = load_profile(model_dir)
    plot_all_sites(profile, sites_per_page)
    plot_critical_layers(profile, critical_limit)
    plot_rgb_depth_inputs(profile)
    plot_group_distribution(profile)
    return profile


def _model_comparison_row(root, model_name):
    if model_name not in MODEL_LABELS:
        raise ValueError("unknown model: %s" % model_name)
    profile = load_profile(Path(root) / model_name)
    rows = [
        row for row in profile["summary"].values()
        if int(row["synthetic_slice"]) == 0
    ]
    group_shares = {}
    for row in rows:
        group_shares[row["group"]] = float(row["group_error_energy_share"])
    front_end_share = sum(
        group_shares[group] for group in ("encoder", "attention")
        if group in group_shares)
    finite_sqnr = [float(row["sqnr_db"]) for row in rows
                   if math.isfinite(float(row["sqnr_db"]))]
    if not finite_sqnr:
        raise ValueError("model profile has no finite SQNR values")
    return {
        "model": MODEL_LABELS[model_name],
        "front_end_error_share": front_end_share,
        "worst_p99_99_over_p99": max(
            float(row["p99_99_over_p99"]) for row in rows),
        "worst_channel_max_over_median": max(
            float(row["channel_max_over_median"]) for row in rows),
        "median_sqnr_db": float(np.median(finite_sqnr)),
        "activation_sites": len(rows),
    }


def plot_root_comparison(root, model_names):
    configure_style()
    root = Path(root)
    rows = [_model_comparison_row(root, model_name)
            for model_name in model_names]
    labels = [row["model"] for row in rows]
    metrics = (
        ("front_end_error_share", "Encoder + attention error share",
         COLORS["reference"]),
        ("worst_p99_99_over_p99", "Worst p99.99 / p99",
         COLORS["magnitude"]),
        ("worst_channel_max_over_median", "Worst channel max / median",
         COLORS["error"]),
        ("median_sqnr_db", "Median SQNR (dB)", COLORS["codes"]),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for axis, metric in zip(axes.reshape(-1), metrics):
        field, label, color = metric
        _vertical_bars(
            axis, [row[field] for row in rows], labels, color)
        axis.set_ylabel(label)
        _style_axis(axis)
    figure.subplots_adjust(
        left=0.10, right=0.98, top=0.97, bottom=0.10,
        hspace=0.34, wspace=0.28)
    figure.savefig(
        root / "w4a4_activation_outlier_comparison.png", dpi=180)
    write_csv(root / "w4a4_activation_outlier_comparison.csv", rows)
    return figure, rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sites-per-page", type=int, default=2)
    parser.add_argument("--critical-limit", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root)
    for model_name in args.models:
        plot_model_profile(
            root / model_name,
            sites_per_page=args.sites_per_page,
            critical_limit=args.critical_limit)
    figure, _ = plot_root_comparison(root, args.models)
    plt.close(figure)


if __name__ == "__main__":
    main()
