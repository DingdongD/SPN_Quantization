#!/usr/bin/env python3
"""Plot CompletionFormer joint integer quantization results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


CONFIG_ORDER = (
    "FP32",
    "JIQ_RTN_W4A4",
    "JIQ_Attention_W4A4",
    "JIQ_Concat_W4A4",
    "JIQ_Joint_W4A4",
    "JIQ_W4A8",
)
PANEL_ORDER = ("GT",) + CONFIG_ORDER
DISPLAY_NAMES = {
    "GT": "GT",
    "FP32": "FP32",
    "JIQ_RTN_W4A4": "RTN W4A4",
    "JIQ_Attention_W4A4": "Attention W4A4",
    "JIQ_Concat_W4A4": "Concat W4A4",
    "JIQ_Joint_W4A4": "Joint W4A4",
    "JIQ_W4A8": "Joint W4A8",
}
CONFIG_COLORS = {
    "FP32": "#4C78A8",
    "JIQ_RTN_W4A4": "#E45756",
    "JIQ_Attention_W4A4": "#72B7B2",
    "JIQ_Concat_W4A4": "#F2CF5B",
    "JIQ_Joint_W4A4": "#54A24B",
    "JIQ_W4A8": "#B279A2",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.axisbelow": True,
    })


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty result table: %s" % path)
    return rows


def load_prediction_grid(root: Path, expected_samples: int
                         ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    root = Path(root)
    expected_samples = int(expected_samples)
    by_sample = {}  # type: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    expected_indices = None
    for config in CONFIG_ORDER:
        directory = root / "predictions" / config
        paths = sorted(directory.glob("sample_*.npz"))
        if len(paths) != expected_samples:
            raise ValueError(
                "%s expected %d predictions, found %d" %
                (config, expected_samples, len(paths)))
        indices = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                sample_index = int(payload["sample_index"])
                payload_config = str(payload["config"])
                if payload_config != config:
                    raise ValueError("prediction config mismatch: %s" % path)
                gt = np.asarray(payload["gt"], dtype=np.float32)
                pred = np.asarray(payload["pred"], dtype=np.float32)
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
        if set(sample) != set(CONFIG_ORDER):
            raise ValueError("prediction configurations are incomplete")
        reference = sample["FP32"]["gt"]
        for config in CONFIG_ORDER:
            if not np.array_equal(reference, sample[config]["gt"]):
                raise ValueError(
                    "ground truth differs at sample %d" % sample_index)
    return by_sample


def aggregate_rmse_rows(root: Path, expected_samples: int
                        ) -> Dict[str, float]:
    rows = read_csv(Path(root) / "sample_metrics.csv")
    result = {}
    for config in CONFIG_ORDER:
        selected = [row for row in rows if row["config"] == config]
        if len(selected) != int(expected_samples):
            raise ValueError(
                "%s expected %d metric rows, found %d" %
                (config, int(expected_samples), len(selected)))
        indices = [int(row["sample_index"]) for row in selected]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate sample metrics for %s" % config)
        values = [float(row["RMSE"]) for row in selected]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite RMSE for %s" % config)
        result[config] = float(np.mean(values))
    return result


def plot_aggregate(root: Path, out_path: Path,
                   expected_samples: int, dpi: int) -> Path:
    values = aggregate_rmse_rows(root, expected_samples)
    fig, axis = plt.subplots(figsize=(10.8, 4.8))
    positions = np.arange(len(CONFIG_ORDER))
    bars = axis.bar(
        positions,
        [values[config] for config in CONFIG_ORDER],
        color=[CONFIG_COLORS[config] for config in CONFIG_ORDER],
        edgecolor="#333333", linewidth=0.6, zorder=3)
    axis.set_ylabel("Mean RMSE (m)")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [DISPLAY_NAMES[config] for config in CONFIG_ORDER], rotation=0)
    axis.grid(axis="y", color="#C8C8C8", linewidth=0.7, alpha=0.65,
              zorder=0)
    axis.set_axisbelow(True)
    for bar, config in zip(bars, CONFIG_ORDER):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(), "%.4f" % values[config],
            ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    return out_path


def _metric_series(rows: List[Dict[str, str]], metric: str
                   ) -> Dict[str, List[float]]:
    configs = []
    for row in rows:
        if row["config"] not in configs:
            configs.append(row["config"])
    output = {}
    for config in configs:
        selected = sorted(
            (row for row in rows if row["config"] == config),
            key=lambda row: row["module"])
        values = [float(row[metric]) for row in selected]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite %s for %s" % (metric, config))
        output[config] = values
    return output


def positive_log_values(values: List[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.where(array > 0.0, array, np.nan)


def plot_local_metrics(root: Path, out_path: Path, dpi: int) -> Path:
    attention_rows = read_csv(Path(root) / "attention_metrics.csv")
    concat_rows = read_csv(Path(root) / "concat_metrics.csv")
    panels = (
        (attention_rows, "context_mse", "Attention context MSE"),
        (attention_rows, "probability_kl", "Attention probability KL"),
        (concat_rows, "block_mse", "Concat block MSE"),
        (concat_rows, "partial_requantization_mse",
         "Concat requantization MSE"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), squeeze=False)
    for axis, (rows, metric, ylabel) in zip(axes.flat, panels):
        series = _metric_series(rows, metric)
        for config, values in series.items():
            indices = np.arange(1, len(values) + 1)
            axis.plot(
                indices, positive_log_values(values),
                marker="o", markersize=3.0,
                linewidth=1.4, label=DISPLAY_NAMES[config], zorder=3)
        axis.set_xlabel("Block index")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(color="#C8C8C8", linewidth=0.7, alpha=0.65, zorder=0)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    return out_path


def plot_predictions(root: Path, out_path: Path, expected_samples: int,
                     dpi: int) -> Path:
    grid = load_prediction_grid(root, expected_samples)
    sample_indices = sorted(grid)
    rows = len(sample_indices)
    columns = len(PANEL_ORDER)
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(15.0, max(2.5, rows * 1.15)),
        squeeze=False)
    for row_index, sample_index in enumerate(sample_indices):
        sample = grid[sample_index]
        gt = sample["FP32"]["gt"]
        valid = gt > 1e-4
        for column_index, panel in enumerate(PANEL_ORDER):
            axis = axes[row_index, column_index]
            image = gt if panel == "GT" else sample[panel]["pred"]
            masked = np.ma.masked_where(~valid, image)
            axis.imshow(
                masked, cmap="viridis", vmin=0.0, vmax=10.0,
                interpolation="nearest", aspect="auto", rasterized=True,
                zorder=2)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(DISPLAY_NAMES[panel], pad=4.0)
            if column_index == 0:
                axis.set_ylabel("%05d" % sample_index, rotation=0,
                                ha="right", va="center")
            for spine in axis.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#666666")
    scalar = ScalarMappable(norm=Normalize(0.0, 10.0), cmap="viridis")
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar, ax=list(axes.flat), fraction=0.012, pad=0.008)
    colorbar.set_label("Depth (m)")
    fig.subplots_adjust(
        left=0.055, right=0.965, bottom=0.008, top=0.985,
        wspace=0.025, hspace=0.08)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    return out_path


def generate_figures(root: Path, out_dir: Path, expected_samples: int,
                     dpi: int) -> Dict[str, Path]:
    set_style()
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "aggregate": plot_aggregate(
            root, out_dir / "aggregate_rmse.png", expected_samples, dpi),
        "local": plot_local_metrics(
            root, out_dir / "local_attention_concat_metrics.png", dpi),
        "predictions": plot_predictions(
            root, out_dir / "prediction_comparison_64.png",
            expected_samples, dpi),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    paths = generate_figures(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        expected_samples=args.expected_samples,
        dpi=args.dpi)
    print("aggregate=%s" % paths["aggregate"], flush=True)
    print("local=%s" % paths["local"], flush=True)
    print("predictions=%s" % paths["predictions"], flush=True)


if __name__ == "__main__":
    main()
