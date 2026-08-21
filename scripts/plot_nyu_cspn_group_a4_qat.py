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

from scripts.evaluate_nyu_cspn_group_a4_qat import (
    CONFIGURATIONS,
    EXPECTED_CONFIGS,
    VISUAL_PREDICTION_FIELDS,
)


QUANTIZED_CONFIGURATIONS = CONFIGURATIONS[1:]
MIXED_COLUMNS = (
    "RGB",
    "Sparse depth",
    "GT",
    "FP32",
    "Uniform W6A6",
    "P3/T3",
    "Mixed QAT",
    "Mixed absolute error",
)


def prediction_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    valid = target > 0.0
    if not bool(valid.any()):
        raise ValueError("RMSE requires valid target depth")
    if not bool(np.isfinite(target).all()) or \
            not bool(np.isfinite(prediction).all()):
        raise ValueError("RMSE arrays must be finite")
    difference = prediction[valid].astype(np.float64) - \
        target[valid].astype(np.float64)
    return float(np.sqrt(np.mean(difference * difference)))


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
    if set(payload) != VISUAL_PREDICTION_FIELDS:
        raise ValueError("prediction payload fields changed")
    if str(payload["config"].item()) != configuration:
        raise ValueError("prediction payload configuration changed")
    if str(payload["model"].item()) != "cspn":
        raise ValueError("prediction payload model changed")
    for key in ("gt", "fp32", "pred", "sparse", "rgb", "model_rgb"):
        if not bool(np.isfinite(payload[key]).all()):
            raise ValueError("prediction payload contains non-finite values")
    valid = payload["valid_gt"].astype(bool)
    if not bool(np.isfinite(payload["abs_err"][valid]).all()):
        raise ValueError("prediction payload contains non-finite errors")
    if bool(payload["nonfinite"].any()):
        raise ValueError("prediction payload marks non-finite predictions")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--font-size", type=int, required=True)
    parser.add_argument("--detail-count", type=int, required=True)
    parser.add_argument("--mixed-protocol", action="store_true")
    return parser.parse_args(argv)


def _load_all(experiment_dir: Path, expected_samples: int):
    return _load_configurations(
        experiment_dir, expected_samples, CONFIGURATIONS)


def _load_mixed_all(experiment_dir: Path, expected_samples: int):
    return _load_configurations(
        experiment_dir, expected_samples, EXPECTED_CONFIGS)


def _load_configurations(
        experiment_dir: Path, expected_samples: int, configurations):
    fp32_paths = sorted(
        (experiment_dir / "predictions" / "FP32").glob("sample_*.npz"))
    if len(fp32_paths) != expected_samples:
        raise RuntimeError("FP32 prediction count changed")
    indices = tuple(int(path.stem.split("_")[1]) for path in fp32_paths)
    payloads = {}
    for config in configurations:
        current = {}
        for index in indices:
            path = experiment_dir / "predictions" / config / \
                ("sample_%05d.npz" % index)
            current[index] = load_payload(path, config)
        payloads[config] = current
    return indices, payloads


def _rgb_image(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or int(rgb.shape[-1]) != 3:
        raise ValueError("display RGB must be HWC with three channels")
    if not bool(np.isfinite(rgb).all()):
        raise ValueError("display RGB must be finite")
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise ValueError("display RGB must lie in [0, 1]")
    return rgb


def _sparse_image(sparse: np.ndarray) -> np.ma.MaskedArray:
    if sparse.ndim != 2:
        raise ValueError("sparse depth must be a 2D image")
    if not bool(np.isfinite(sparse).all()) or float(sparse.min()) < 0.0:
        raise ValueError("sparse depth must be finite and nonnegative")
    return np.ma.masked_equal(sparse, 0.0)


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


def _mixed_detail_indices(indices, payloads, count: int):
    if count <= 0 or count > len(indices):
        raise ValueError("detail count must fit prediction count")
    scored = []
    for index in indices:
        errors = tuple(
            prediction_rmse(
                payloads[config][index]["gt"],
                payloads[config][index]["pred"])
            for config in EXPECTED_CONFIGS[1:])
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
    sparse_cmap = matplotlib.colormaps["viridis"].copy()
    sparse_cmap.set_bad("#eeeeee")
    for row_index, index in enumerate(indices):
        source = payloads["FP32"][index]
        depth_max = float(np.quantile(source["gt"], 0.995))
        axes[row_index, 0].imshow(_rgb_image(source["rgb"]))
        axes[row_index, 1].imshow(
            _sparse_image(source["sparse"]), cmap=sparse_cmap,
            vmin=0.0, vmax=depth_max)
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


def plot_mixed_details(payloads, indices, output_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(indices), len(MIXED_COLUMNS),
        figsize=(25, max(4, 3.2 * len(indices))),
        constrained_layout=True,
        squeeze=False,
    )
    sparse_cmap = plt.get_cmap("viridis").copy()
    sparse_cmap.set_bad(color="white")
    depth_configs = (
        "FP32", "UNIFORM_W6A6", "P3_T3", "MIXED_TASK_AWARE_QAT")
    for row_index, index in enumerate(indices):
        source = payloads["FP32"][index]
        depth_max = max(
            float(np.quantile(source["gt"][source["gt"] > 0.0], 0.995)),
            *(float(np.quantile(
                payloads[config][index]["pred"], 0.995))
              for config in depth_configs),
        )
        axes[row_index, 0].imshow(_rgb_image(source["rgb"]))
        axes[row_index, 1].imshow(
            _sparse_image(source["sparse"]), cmap=sparse_cmap,
            vmin=0.0, vmax=depth_max)
        axes[row_index, 2].imshow(
            source["gt"], cmap="viridis", vmin=0.0, vmax=depth_max)
        for column, config in enumerate(depth_configs, 3):
            payload = payloads[config][index]
            axes[row_index, column].imshow(
                payload["pred"], cmap="viridis",
                vmin=0.0, vmax=depth_max)
            axes[row_index, column].set_xlabel(
                "RMSE %.3f m" % prediction_rmse(
                    payload["gt"], payload["pred"]))
        mixed = payloads["MIXED_TASK_AWARE_QAT"][index]
        axes[row_index, 7].imshow(
            mixed["abs_err"], cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_index, 0].set_ylabel("Sample %05d" % index)
        for axis in axes[row_index]:
            _hide_axis(axis)
    for column, label in enumerate(MIXED_COLUMNS):
        axes[0, column].set_title(label)
    figure.savefig(output_dir / "mixed_prediction_details.png", dpi=180)
    figure.savefig(output_dir / "mixed_prediction_details.pdf")
    plt.close(figure)


def plot_mixed_contact_sheet(payloads, indices, output_dir: Path) -> None:
    grid = int(np.ceil(np.sqrt(len(indices))))
    figure, axes = plt.subplots(
        grid, grid, figsize=(28, 22), constrained_layout=True,
        squeeze=False)
    for position, index in enumerate(indices):
        row, column = divmod(position, grid)
        source = payloads["FP32"][index]
        panels = [source["gt"]]
        panels.extend(
            payloads[config][index]["pred"] for config in EXPECTED_CONFIGS)
        combined = np.concatenate(panels, axis=1)
        depth_max = float(np.quantile(
            source["gt"][source["gt"] > 0.0], 0.995))
        axes[row, column].imshow(
            combined, cmap="viridis", vmin=0.0, vmax=depth_max)
        axes[row, column].set_title("%05d" % index)
        _hide_axis(axes[row, column])
    for position in range(len(indices), grid * grid):
        row, column = divmod(position, grid)
        axes[row, column].axis("off")
    figure.savefig(output_dir / "mixed_predictions_64.png", dpi=180)
    figure.savefig(output_dir / "mixed_predictions_64.pdf")
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
    if args.mixed_protocol:
        indices, payloads = _load_mixed_all(
            experiment_dir, args.expected_samples)
        details = _mixed_detail_indices(
            indices, payloads, args.detail_count)
        plot_mixed_details(payloads, details, output_dir)
        plot_mixed_contact_sheet(payloads, indices, output_dir)
    else:
        indices, payloads = _load_all(
            experiment_dir, args.expected_samples)
        details = _detail_indices(indices, payloads, args.detail_count)
        plot_details(payloads, details, output_dir)
        plot_contact_sheet(payloads, indices, output_dir)
    print("paired CSPN prediction figures written", flush=True)


if __name__ == "__main__":
    main()
