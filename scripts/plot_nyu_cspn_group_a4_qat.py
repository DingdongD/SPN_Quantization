#!/usr/bin/env python3
"""Plot paired CSPN PTQ/QAT predictions and errors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_nyu_cspn_group_a4_qat import CONFIGURATIONS


QUANTIZED_CONFIGURATIONS = CONFIGURATIONS[1:]


def set_style(font_size: int) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": int(font_size),
        "axes.titlesize": int(font_size),
        "axes.labelsize": int(font_size),
        "xtick.labelsize": int(font_size),
        "ytick.labelsize": int(font_size),
    })


def load_payload(path: Path, configuration: str):
    with np.load(path, allow_pickle=False) as source:
        payload = dict((key, source[key]) for key in source.files)
    required = {
        "gt", "fp32", "pred", "abs_err", "valid_gt", "sparse", "rgb",
        "sample_index", "model", "config",
    }
    if set(payload) != required:
        raise ValueError("prediction payload fields changed")
    if str(payload["config"].item()) != configuration:
        raise ValueError("prediction payload configuration changed")
    if str(payload["model"].item()) != "cspn":
        raise ValueError("prediction payload model changed")
    for key in ("gt", "fp32", "pred", "sparse", "rgb"):
        if not bool(np.isfinite(payload[key]).all()):
            raise ValueError("prediction payload contains non-finite values")
    valid = payload["valid_gt"].astype(bool)
    if not bool(np.isfinite(payload["abs_err"][valid]).all()):
        raise ValueError("prediction payload contains non-finite errors")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--font-size", type=int, required=True)
    parser.add_argument("--detail-count", type=int, required=True)
    return parser.parse_args(argv)


def _load_all(experiment_dir: Path, expected_samples: int):
    fp32_paths = sorted(
        (experiment_dir / "predictions" / "FP32").glob("sample_*.npz"))
    if len(fp32_paths) != expected_samples:
        raise RuntimeError("FP32 prediction count changed")
    indices = tuple(int(path.stem.split("_")[1]) for path in fp32_paths)
    payloads = {}
    for config in CONFIGURATIONS:
        current = {}
        for index in indices:
            path = experiment_dir / "predictions" / config / \
                ("sample_%05d.npz" % index)
            current[index] = load_payload(path, config)
        payloads[config] = current
    return indices, payloads


def _rgb_image(rgb: np.ndarray) -> np.ndarray:
    minimum = float(rgb.min())
    maximum = float(rgb.max())
    if minimum >= 0.0 and maximum <= 1.0:
        return rgb
    return np.clip((rgb - minimum) / (maximum - minimum), 0.0, 1.0)


def _detail_indices(indices, payloads, count: int):
    if count <= 0 or count > len(indices):
        raise ValueError("detail count must fit prediction count")
    scored = []
    for index in indices:
        errors = []
        for config in QUANTIZED_CONFIGURATIONS:
            payload = payloads[config][index]
            valid = payload["valid_gt"].astype(bool)
            errors.append(float(np.mean(payload["abs_err"][valid])))
        scored.append((max(errors), index))
    return tuple(index for _, index in sorted(scored, reverse=True)[:count])


def _hide_axis(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])


def plot_details(payloads, indices, output_dir: Path) -> None:
    columns = (
        "RGB", "Sparse", "GT", "FP32",
        "PTQ Static", "PTQ Static Error",
        "QAT Static", "QAT Static Error",
        "PTQ Dynamic", "PTQ Dynamic Error",
        "QAT Dynamic", "QAT Dynamic Error",
    )
    config_pairs = (
        "PTQ_STATIC_G8_W4A4", "QAT_STATIC_G8_W4A4",
        "PTQ_DYNAMIC_G8_W4A4", "QAT_DYNAMIC_G8_W4A4",
    )
    figure, axes = plt.subplots(
        len(indices), len(columns), figsize=(30, 2.8 * len(indices)),
        constrained_layout=True, squeeze=False)
    for row_index, index in enumerate(indices):
        source = payloads["FP32"][index]
        depth_max = float(np.quantile(source["gt"], 0.995))
        axes[row_index, 0].imshow(_rgb_image(source["rgb"]))
        axes[row_index, 1].imshow(
            source["sparse"], cmap="viridis", vmin=0.0, vmax=depth_max)
        axes[row_index, 2].imshow(
            source["gt"], cmap="viridis", vmin=0.0, vmax=depth_max)
        axes[row_index, 3].imshow(
            source["pred"], cmap="viridis", vmin=0.0, vmax=depth_max)
        column = 4
        for config in config_pairs:
            payload = payloads[config][index]
            axes[row_index, column].imshow(
                payload["pred"], cmap="viridis",
                vmin=0.0, vmax=depth_max)
            axes[row_index, column + 1].imshow(
                payload["abs_err"], cmap="magma", vmin=0.0, vmax=1.0)
            column += 2
        axes[row_index, 0].set_ylabel("Sample %05d" % index)
        for axis in axes[row_index]:
            _hide_axis(axis)
    for column, label in enumerate(columns):
        axes[0, column].set_title(label)
    figure.savefig(output_dir / "paired_prediction_details.png", dpi=180)
    figure.savefig(output_dir / "paired_prediction_details.pdf")
    plt.close(figure)


def plot_contact_sheet(payloads, indices, output_dir: Path) -> None:
    grid = int(np.ceil(np.sqrt(len(indices))))
    figure, axes = plt.subplots(
        grid, grid, figsize=(28, 22), constrained_layout=True,
        squeeze=False)
    order = (
        "FP32", "PTQ_STATIC_G8_W4A4", "QAT_STATIC_G8_W4A4",
        "PTQ_DYNAMIC_G8_W4A4", "QAT_DYNAMIC_G8_W4A4",
    )
    for position, index in enumerate(indices):
        row, column = divmod(position, grid)
        source = payloads["FP32"][index]
        panels = [source["gt"]]
        panels.extend(payloads[config][index]["pred"] for config in order)
        combined = np.concatenate(panels, axis=1)
        depth_max = float(np.quantile(source["gt"], 0.995))
        axes[row, column].imshow(
            combined, cmap="viridis", vmin=0.0, vmax=depth_max)
        axes[row, column].set_title("%05d" % index)
        _hide_axis(axes[row, column])
    for position in range(len(indices), grid * grid):
        row, column = divmod(position, grid)
        axes[row, column].axis("off")
    figure.savefig(output_dir / "paired_predictions_64.png", dpi=180)
    figure.savefig(output_dir / "paired_predictions_64.pdf")
    plt.close(figure)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.expected_samples != 64:
        raise ValueError("paired contact sheet requires 64 samples")
    if args.font_size <= 0:
        raise ValueError("font size must be positive")
    set_style(args.font_size)
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices, payloads = _load_all(
        experiment_dir, args.expected_samples)
    details = _detail_indices(indices, payloads, args.detail_count)
    plot_details(payloads, details, output_dir)
    plot_contact_sheet(payloads, indices, output_dir)
    print("paired CSPN prediction figures written", flush=True)


if __name__ == "__main__":
    main()
