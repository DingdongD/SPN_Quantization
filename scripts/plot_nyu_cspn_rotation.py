#!/usr/bin/env python3
"""Analyze CSPN selective channel-rotation evaluation results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METRICS = (
    "RMSE", "MAE", "ABS_REL", "IRMSE", "flat_RMSE", "boundary_RMSE",
)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("analysis summary must not be empty")
    fields = tuple(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_rotation_config(name: str) -> bool:
    return name.startswith("RANDOM_") or name.startswith("HADAMARD_")


def aggregate_results(rows):
    grouped = {}
    order = []
    for row in rows:
        name = row["config"]
        if name not in grouped:
            grouped[name] = dict((metric, []) for metric in METRICS)
            order.append(name)
        for metric in METRICS:
            grouped[name][metric].append(float(row[metric]))
    summary = []
    for name in order:
        current = {
            "config": name,
            "is_rotation": int(is_rotation_config(name)),
        }
        for metric in METRICS:
            current["mean_%s" % metric] = float(
                np.mean(np.asarray(grouped[name][metric], dtype=np.float64)))
        summary.append(current)
    candidates = [row for row in summary if row["is_rotation"] == 1]
    if not candidates:
        raise ValueError("no rotation configuration was evaluated")
    best = min(
        candidates,
        key=lambda row: (row["mean_RMSE"], row["config"]))["config"]
    for row in summary:
        row["selected_rotation"] = int(row["config"] == best)
    return summary, best


def _set_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })


def plot_rmse(summary, path):
    names = [row["config"] for row in summary]
    values = [row["mean_RMSE"] for row in summary]
    colors = [
        "#4E79A7" if not row["selected_rotation"] else "#E15759"
        for row in summary]
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    axis.bar(np.arange(len(names)), values, color=colors, zorder=3)
    axis.set_ylabel("RMSE (m)")
    axis.set_xticks(np.arange(len(names)), names, rotation=0)
    axis.tick_params(axis="x", labelsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_activation(boundary_rows, best, path):
    selected_names = ["RTN_W4A4", "RANDOM_both", "HADAMARD_both", best]
    selected_names = list(dict.fromkeys(selected_names))
    boundaries = ("decoder_entry", "layer4_signed_skip")
    metrics = (
        ("p99_99", "P99.99 |Activation|"),
        ("sqnr", "A4 SQNR (dB)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    for column, boundary in enumerate(boundaries):
        current = dict(
            (row["config"], row) for row in boundary_rows
            if row["boundary"] == boundary)
        names = [name for name in selected_names if name in current]
        for row_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column]
            values = [float(current[name][metric]) for name in names]
            colors = ["#E15759" if name == best else "#4E79A7"
                      for name in names]
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
            axis.bar(np.arange(len(names)), values, color=colors, zorder=3)
            axis.set_ylabel(label)
            axis.set_xlabel(
                "Decoder Entry" if boundary == "decoder_entry"
                else "Layer4 Signed Skip")
            axis.set_xticks(np.arange(len(names)), names, rotation=0)
            axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_predictions(input_dir, best, path, samples=4):
    input_dir = Path(input_dir)
    fp_paths = sorted((input_dir / "predictions" / "FP32").glob(
        "sample_*.npz"))[:int(samples)]
    if not fp_paths:
        raise FileNotFoundError("FP32 prediction payloads are missing")
    configurations = ("FP32", "RTN_W4A4", best)
    figure, axes = plt.subplots(
        len(fp_paths), 4, figsize=(16, 3.7 * len(fp_paths)),
        squeeze=False)
    labels = ("GT", "FP32", "RTN W4A4", best)
    for row_index, fp_path in enumerate(fp_paths):
        sample_name = fp_path.name
        fp_payload = np.load(fp_path)
        values = [fp_payload["gt"]]
        for config in configurations:
            payload = np.load(
                input_dir / "predictions" / config / sample_name)
            values.append(payload["pred"])
        valid = fp_payload["valid_gt"].astype(bool)
        depth = fp_payload["gt"][valid]
        minimum = float(depth.min())
        maximum = float(depth.max())
        for column, value in enumerate(values):
            axis = axes[row_index, column]
            image = axis.imshow(
                value, cmap="viridis", vmin=minimum, vmax=maximum)
            axis.set_xlabel(labels[column])
            axis.set_xticks([])
            axis.set_yticks([])
            image.set_zorder(2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_analysis(input_dir, out_dir):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, best = aggregate_results(
        read_csv(input_dir / "sample_metrics.csv"))
    boundary_rows = read_csv(input_dir / "boundary_metrics.csv")
    write_csv(out_dir / "summary.csv", summary)
    _set_style()
    plot_rmse(summary, out_dir / "rmse_comparison.png")
    plot_activation(
        boundary_rows, best,
        out_dir / "activation_rotation_comparison.png")
    plot_predictions(
        input_dir, best, out_dir / "prediction_comparison.png")
    return best


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    best = write_analysis(args.input_dir, args.out_dir)
    print("Selected rotation: %s" % best, flush=True)


if __name__ == "__main__":
    main()
