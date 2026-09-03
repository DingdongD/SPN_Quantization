#!/usr/bin/env python3
"""Plot the strict CSPN LSQ+, HAWQ, and baseline comparison."""

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
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.evaluate_nyu_cspn_lsqplus_hawq import (  # noqa: E402
    CONFIGURATIONS,
)


DISPLAY_LABELS = (
    "FP32",
    "PA-RTN W4A4",
    "PA-RTN W6A6",
    "LSQ+ W4A4",
    "LSQ+ W6A6",
    "HAWQ Mixed<=6",
    "Mixed Task-aware QAT",
)
XTICK_ROTATION = 0


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def _read_metrics(root: Path):
    with (Path(root) / "aggregate_metrics.csv").open(
            "r", newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if tuple(row["configuration"] for row in rows) != CONFIGURATIONS:
        raise ValueError("aggregate metric configuration order changed")
    return rows


def _payload(root: Path, configuration: str, index: int):
    path = Path(root) / "shards" / configuration / \
        ("sample_%05d.npz" % int(index))
    with np.load(path, allow_pickle=False) as source:
        return dict((key, source[key]) for key in source.files)


def plot_rmse(root: Path, output: Path) -> None:
    rows = _read_metrics(root)
    values = np.asarray([float(row["RMSE"]) for row in rows])
    colors = (
        "#4C78A8", "#A0A0A0", "#7F7F7F", "#F58518",
        "#E45756", "#54A24B", "#B279A2")
    figure, axis = plt.subplots(figsize=(15, 6))
    positions = np.arange(len(values))
    axis.bar(positions, values, color=colors, width=0.72, zorder=3)
    axis.set_ylabel("RMSE (m)")
    axis.set_xticks(positions, DISPLAY_LABELS, rotation=XTICK_ROTATION)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    maximum = max(float(values.max()) * 1.16, 0.01)
    axis.set_ylim(0.0, maximum)
    for position, value in zip(positions, values):
        axis.text(position, value + maximum * 0.015, "%.4f" % value,
                  ha="center", va="bottom", fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "quantization_rmse_comparison.png", dpi=180)
    figure.savefig(output / "quantization_rmse_comparison.pdf")
    plt.close(figure)


def plot_details(
        root: Path, indices, output: Path,
        depth_min: float = 0.0, depth_max: float = 10.0) -> None:
    indices = tuple(int(index) for index in indices)
    columns = ("RGB", "Sparse", "GT") + DISPLAY_LABELS
    figure, axes = plt.subplots(
        len(indices), len(columns),
        figsize=(2.7 * len(columns), 2.25 * len(indices)),
        squeeze=False,
    )
    depth_image = None
    for row_index, index in enumerate(indices):
        reference = _payload(root, "FP32", index)
        axes[row_index, 0].imshow(np.clip(reference["rgb"], 0.0, 1.0))
        sparse = np.ma.masked_where(reference["sparse"] <= 0.0,
                                    reference["sparse"])
        axes[row_index, 1].imshow(
            sparse, cmap="viridis", vmin=depth_min, vmax=depth_max)
        depth_image = axes[row_index, 2].imshow(
            reference["gt"], cmap="viridis",
            vmin=depth_min, vmax=depth_max)
        for column_index, configuration in enumerate(CONFIGURATIONS, 3):
            payload = _payload(root, configuration, index)
            axes[row_index, column_index].imshow(
                payload["pred"], cmap="viridis",
                vmin=depth_min, vmax=depth_max)
        for column_index, label in enumerate(columns):
            axis = axes[row_index, column_index]
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(label, fontsize=12)
            if column_index == 0:
                axis.set_ylabel("#%d" % index, rotation=0,
                                labelpad=22, va="center")
    figure.subplots_adjust(
        left=0.035, right=0.975, top=0.965, bottom=0.025,
        wspace=0.035, hspace=0.06)
    colorbar_axis = figure.add_axes((0.982, 0.08, 0.008, 0.84))
    figure.colorbar(depth_image, cax=colorbar_axis, label="Depth (m)")
    figure.savefig(output / "prediction_details.png", dpi=160)
    figure.savefig(output / "prediction_details.pdf")
    plt.close(figure)


def plot_contact_sheet(
        root: Path, indices, output: Path,
        depth_min: float = 0.0, depth_max: float = 10.0) -> None:
    indices = tuple(int(index) for index in indices)
    columns = min(8, len(indices))
    rows = int(math.ceil(len(indices) / float(columns)))
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.2 * columns, 1.75 * rows),
        squeeze=False)
    depth_image = None
    for position, index in enumerate(indices):
        row, column = divmod(position, columns)
        payload = _payload(root, "HAWQ_MIXED_LE6", index)
        depth_image = axes[row, column].imshow(
            payload["pred"], cmap="viridis",
            vmin=depth_min, vmax=depth_max)
        axes[row, column].set_title("#%d" % index, fontsize=10)
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
    for position in range(len(indices), rows * columns):
        row, column = divmod(position, columns)
        axes[row, column].axis("off")
    figure.subplots_adjust(
        left=0.015, right=0.965, top=0.95, bottom=0.025,
        wspace=0.03, hspace=0.12)
    colorbar_axis = figure.add_axes((0.975, 0.08, 0.01, 0.84))
    figure.colorbar(depth_image, cax=colorbar_axis, label="Depth (m)")
    figure.savefig(output / "prediction_contact_sheet.png", dpi=180)
    figure.savefig(output / "prediction_contact_sheet.pdf")
    plt.close(figure)


def generate_figures(root: Path, indices, detail_samples: int) -> None:
    _style()
    root = Path(root)
    indices = tuple(int(index) for index in indices)
    detail_samples = int(detail_samples)
    if detail_samples <= 0 or detail_samples > len(indices):
        raise ValueError("detail sample count is invalid")
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_rmse(root, output)
    plot_details(root, indices[:detail_samples], output)
    plot_contact_sheet(root, indices, output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--detail-samples", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    payload = json.loads(
        Path(args.calibration_metadata).read_text(encoding="utf-8"))
    generate_figures(
        Path(args.output_root), payload["evaluation_indices"],
        args.detail_samples)


if __name__ == "__main__":
    main()
