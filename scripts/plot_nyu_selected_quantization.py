#!/usr/bin/env python3
"""Render aligned formal predictions for selected NYU quantization methods."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.evaluate_nyu_selected_quantization import (  # noqa: E402
    EXPECTED_EVALUATION_SAMPLES,
    METHOD_LABELS,
    SELECTED_METHODS,
    load_formal_artifact_index,
    load_prediction_export,
    ordered_evaluation_identity,
    prediction_path,
)


@dataclass(frozen=True)
class AlignedSample:
    model: str
    sample_index: int
    evaluation_identity: str
    rgb: np.ndarray
    sparse: np.ndarray
    gt: np.ndarray
    predictions: tuple


@dataclass(frozen=True)
class SampleRanges:
    depth_min: float
    depth_max: float
    error_min: float
    error_max: float


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
    })


def load_aligned_sample(
        root: Path, model: str, sample_index: int,
        evaluation_identity: str, artifact_index_sha256: str,
        methods=SELECTED_METHODS) -> AlignedSample:
    methods = tuple(str(method) for method in methods)
    if methods != SELECTED_METHODS:
        raise ValueError("selected prediction panel method order changed")
    payloads = tuple(
        (method, load_prediction_export(
            prediction_path(root, method, sample_index),
            model,
            method,
            sample_index,
            evaluation_identity,
            artifact_index_sha256,
        ))
        for method in methods)
    reference = payloads[0][1]
    aligned_fields = (
        "rgb", "sparse", "gt", "valid_gt", "sparse_depth_max_m")
    for method, payload in payloads[1:]:
        if any(not np.array_equal(
                payload[field], reference[field])
                for field in aligned_fields):
            raise ValueError(
                "aligned input changed for %s sample %d" %
                (method, int(sample_index)))
    return AlignedSample(
        model=str(model),
        sample_index=int(sample_index),
        evaluation_identity=str(evaluation_identity),
        rgb=reference["rgb"],
        sparse=reference["sparse"],
        gt=reference["gt"],
        predictions=tuple(
            (method, payload["pred"], payload["abs_error"])
            for method, payload in payloads),
    )


def shared_sample_ranges(sample: AlignedSample) -> SampleRanges:
    valid_depth = [sample.gt[sample.gt > 1e-4]]
    sparse = sample.sparse[sample.sparse > 1e-4]
    if sparse.size:
        valid_depth.append(sparse)
    valid_depth.extend(
        prediction[np.isfinite(prediction)]
        for method, prediction, error in sample.predictions)
    depth = np.concatenate(
        tuple(np.asarray(values, dtype=np.float64).reshape(-1)
              for values in valid_depth))
    errors = np.concatenate(tuple(
        np.asarray(error[sample.gt > 1e-4], dtype=np.float64).reshape(-1)
        for method, prediction, error in sample.predictions))
    if depth.size == 0 or errors.size == 0 or not \
            bool(np.isfinite(depth).all()) or not \
            bool(np.isfinite(errors).all()):
        raise ValueError("aligned sample ranges require finite depth and error")
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    error_max = float(errors.max())
    if depth_max <= depth_min:
        depth_max = depth_min + 1e-6
    if error_max <= 0.0:
        error_max = 1e-6
    return SampleRanges(
        depth_min=depth_min,
        depth_max=depth_max,
        error_min=0.0,
        error_max=error_max,
    )


def render_prediction_panel(sample: AlignedSample, output: Path) -> None:
    _style()
    ranges = shared_sample_ranges(sample)
    columns = ("RGB", "Sparse", "GT") + tuple(
        METHOD_LABELS[method] for method, prediction, error
        in sample.predictions)
    figure, axes = plt.subplots(
        2,
        len(columns),
        figsize=(2.0 * len(columns), 4.1),
        squeeze=False,
    )
    axes[0, 0].imshow(np.clip(sample.rgb, 0.0, 1.0))
    sparse = np.ma.masked_where(sample.sparse <= 1e-4, sample.sparse)
    axes[0, 1].imshow(
        sparse,
        cmap="viridis",
        vmin=ranges.depth_min,
        vmax=ranges.depth_max,
    )
    depth_image = axes[0, 2].imshow(
        sample.gt,
        cmap="viridis",
        vmin=ranges.depth_min,
        vmax=ranges.depth_max,
    )
    for column, (method, prediction, error) in enumerate(
            sample.predictions, 3):
        del method, error
        axes[0, column].imshow(
            prediction,
            cmap="viridis",
            vmin=ranges.depth_min,
            vmax=ranges.depth_max,
        )
    for column in range(3):
        axes[1, column].axis("off")
    error_image = None
    for column, (method, prediction, error) in enumerate(
            sample.predictions, 3):
        del method, prediction
        error_image = axes[1, column].imshow(
            error,
            cmap="magma",
            vmin=ranges.error_min,
            vmax=ranges.error_max,
        )
    for column, label in enumerate(columns):
        axes[0, column].set_title(label)
        for row in (0, 1):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Depth")
    axes[1, 3].set_ylabel("Absolute error")
    figure.suptitle("%s sample %d" % (sample.model, sample.sample_index))
    figure.subplots_adjust(
        left=0.025,
        right=0.965,
        top=0.88,
        bottom=0.035,
        wspace=0.025,
        hspace=0.06,
    )
    depth_axis = figure.add_axes((0.971, 0.53, 0.006, 0.31))
    error_axis = figure.add_axes((0.971, 0.09, 0.006, 0.31))
    figure.colorbar(depth_image, cax=depth_axis, label="Depth (m)")
    figure.colorbar(error_image, cax=error_axis, label="Absolute error (m)")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def render_metric_comparison(metrics_path: Path, output: Path) -> None:
    with Path(metrics_path).open(
            "r", newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if tuple(str(row["method"]) for row in rows) != SELECTED_METHODS:
        raise ValueError("aggregate metric method order changed")
    if tuple(str(row["configuration"]) for row in rows) != tuple(
            METHOD_LABELS[method] for method in SELECTED_METHODS):
        raise ValueError("aggregate metric configuration labels changed")
    pooled = np.asarray(
        [float(row["pooled_rmse"]) for row in rows], dtype=np.float64)
    sample_mean = np.asarray(
        [float(row["mean_sample_rmse"]) for row in rows], dtype=np.float64)
    if not bool(np.isfinite(pooled).all()) or not bool(
            np.isfinite(sample_mean).all()) or bool(np.any(pooled < 0.0)) or \
            bool(np.any(sample_mean < 0.0)):
        raise ValueError("aggregate RMSE values must be finite and nonnegative")
    _style()
    colors = (
        "#4C78A8", "#72B7B2", "#54A24B", "#F2CF5B", "#F58518",
        "#E45756", "#B279A2", "#FF9DA6", "#9D755D", "#79706E",
    )
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(16, 5.8))
    axis.bar(
        positions,
        pooled,
        width=0.72,
        color=colors,
        label="Pooled RMSE (primary)",
        zorder=2,
    )
    axis.plot(
        positions,
        sample_mean,
        color="#202020",
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Mean per-sample RMSE (diagnostic)",
        zorder=3,
    )
    axis.set_ylabel("RMSE (m)")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        tuple(METHOD_LABELS[method] for method in SELECTED_METHODS),
        rotation=24,
        ha="right",
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    maximum = max(float(pooled.max()), float(sample_mean.max()), 1e-6)
    axis.set_ylim(0.0, maximum * 1.18)
    for position, value in zip(positions, pooled):
        axis.text(
            position,
            value + maximum * 0.018,
            "%.4f" % value,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def generate_prediction_panels(
        root: Path, model: str, indices,
        artifact_index_sha256: str,
        methods=SELECTED_METHODS) -> Tuple[Path, ...]:
    indices = tuple(int(index) for index in indices)
    if len(indices) != EXPECTED_EVALUATION_SAMPLES or \
            len(indices) != len(set(indices)):
        raise ValueError("formal prediction panels require 64 identities")
    identity = ordered_evaluation_identity(indices)
    output = Path(root) / "figures" / "prediction_panels"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "prediction panel directory must be empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in indices:
        sample = load_aligned_sample(
            root, model, index, identity, artifact_index_sha256, methods)
        path = output / ("sample_%05d.png" % index)
        render_prediction_panel(sample, path)
        paths.append(path)
    return tuple(paths)


def generate_figures(
        root: Path, model: str, indices,
        artifact_index_sha256: str,
        methods=SELECTED_METHODS) -> Tuple[Path, ...]:
    output = Path(root) / "figures" / "pooled_rmse_comparison.png"
    render_metric_comparison(
        Path(root) / "aggregate_metrics.csv", output)
    panels = generate_prediction_panels(
        root, model, indices, artifact_index_sha256, methods)
    return (output,) + panels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot aligned selected NYU quantization predictions")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    from spn_quant.experiment_config import load_selected_quantization_config
    selected = load_selected_quantization_config(args.config)
    models = tuple(model for model in selected.models
                   if model.model == args.model)
    if len(models) != 1:
        raise ValueError("selected plot model entry is not unique")
    artifact_index = load_formal_artifact_index(
        args.artifact_index, args.model, models[0].evaluation_indices)
    generate_figures(
        args.output_root, args.model, models[0].evaluation_indices,
        artifact_index.fingerprint)


if __name__ == "__main__":
    main()
