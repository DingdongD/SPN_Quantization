#!/usr/bin/env python3
"""Render aligned QDrop W4A4 prediction and aggregate comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
MODEL_LABELS = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_visual_samples(rows, seeds):
    seeds = tuple(int(seed) for seed in seeds)
    qdrop = [row for row in rows if row["method"] == "qdrop"]
    indices = sorted(set(int(row["sample_index"]) for row in qdrop))
    means = []
    for index in indices:
        current = [
            float(row["RMSE"]) for row in qdrop
            if int(row["sample_index"]) == index and
            int(row["seed"]) in seeds]
        if len(current) != len(seeds):
            raise ValueError("QDrop visual sample has an incomplete seed set")
        means.append((index, float(np.mean(current))))
    if not means:
        raise ValueError("QDrop visual sample metrics are empty")
    ordered = sorted(means, key=lambda row: (row[1], row[0]))
    median = ordered[(len(ordered) - 1) // 2][0]
    worst = ordered[-1][0]
    return median, worst


def validate_prediction_arrays(arrays):
    required = ("rgb", "sparse", "gt", "fp32", "rtn", "brecq")
    if any(name not in arrays for name in required):
        raise KeyError("prediction comparison arrays are incomplete")
    shape = tuple(np.asarray(arrays["gt"]).shape)
    if len(shape) != 2:
        raise ValueError("prediction depth arrays must be two-dimensional")
    rgb = np.asarray(arrays["rgb"])
    if rgb.ndim != 3 or tuple(rgb.shape[:2]) != shape or rgb.shape[2] != 3:
        raise ValueError("prediction RGB shape does not match depth")
    for name, value in arrays.items():
        if name == "rgb":
            continue
        array = np.asarray(value)
        if tuple(array.shape) != shape:
            raise ValueError("prediction array shapes do not match")
        if not np.isfinite(array).all():
            raise ValueError("prediction arrays contain non-finite values")


def _load_npz(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return dict((name, payload[name].copy()) for name in payload.files)


def _prediction_path(root, config, sample_index):
    return Path(root) / "predictions" / config / \
        ("sample_%05d.npz" % int(sample_index))


def load_prediction_arrays(root, baseline_root, model, sample_index, seeds,
                           qdrop_config):
    baseline_root = Path(baseline_root)
    fp32 = _load_npz(_prediction_path(
        baseline_root / "stress" / "rtn" / model,
        "FP32", sample_index))
    rtn = _load_npz(_prediction_path(
        baseline_root / "stress" / "rtn" / model,
        "HW_W4A4_full", sample_index))
    brecq = _load_npz(_prediction_path(
        baseline_root / "stress" / "brecq" / model,
        "HW_W4A4_full", sample_index))
    qdrop = {}
    for seed in seeds:
        qdrop[int(seed)] = _load_npz(_prediction_path(
            Path(root) / "evaluation" / "qdrop" / model /
            ("seed_%d" % int(seed)),
            qdrop_config, sample_index))
    first = qdrop[int(seeds[0])]
    if "rgb" not in first:
        raise KeyError("QDrop prediction payload is missing RGB")
    arrays = {
        "rgb": first["rgb"],
        "sparse": fp32["sparse"],
        "gt": fp32["gt"],
        "fp32": fp32["pred"],
        "rtn": rtn["pred"],
        "brecq": brecq["pred"],
    }
    for seed in seeds:
        arrays["qdrop_%d" % int(seed)] = qdrop[int(seed)]["pred"]
    arrays["qdrop_mean"] = np.mean(np.stack([
        arrays["qdrop_%d" % int(seed)] for seed in seeds
    ], axis=0), axis=0)
    validate_prediction_arrays(arrays)
    return arrays


def _masked_depth(value, valid):
    return np.ma.masked_where(~valid, np.asarray(value))


def plot_model_predictions(path, samples, seeds):
    labels = [
        "RGB", "Sparse", "GT", "FP32", "RTN W4A4",
        "BRECQ W4A4",
    ] + ["QDrop %d" % int(seed) for seed in seeds] + ["QDrop mean"]
    keys = [
        "rgb", "sparse", "gt", "fp32", "rtn", "brecq",
    ] + ["qdrop_%d" % int(seed) for seed in seeds] + ["qdrop_mean"]
    depth_values = []
    error_values = []
    for arrays in samples.values():
        valid = arrays["gt"] > 1.0e-4
        depth_values.append(arrays["gt"][valid])
        for key in keys[3:]:
            error_values.append(np.abs(
                arrays[key][valid] - arrays["gt"][valid]))
    depth = np.concatenate(depth_values)
    errors = np.concatenate(error_values)
    depth_min, depth_max = np.percentile(depth, (1.0, 99.0))
    error_max = max(float(np.percentile(errors, 99.0)), 1.0e-6)
    rows = 2 * len(samples)
    columns = len(keys)
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.45 * columns, 2.4 * rows),
        squeeze=False)
    depth_artist = None
    error_artist = None
    for sample_rank, (sample_name, arrays) in enumerate(samples.items()):
        valid = arrays["gt"] > 1.0e-4
        depth_row = 2 * sample_rank
        error_row = depth_row + 1
        for column, (label, key) in enumerate(zip(labels, keys)):
            depth_axis = axes[depth_row, column]
            error_axis = axes[error_row, column]
            for axis in (depth_axis, error_axis):
                axis.set_axisbelow(True)
                axis.grid(False)
                axis.set_xticks([])
                axis.set_yticks([])
            if key == "rgb":
                depth_axis.imshow(
                    np.clip(arrays[key], 0.0, 1.0), zorder=2)
                error_axis.axis("off")
            else:
                depth_artist = depth_axis.imshow(
                    _masked_depth(arrays[key], valid),
                    cmap="viridis", vmin=depth_min, vmax=depth_max,
                    zorder=2)
                if key in ("sparse", "gt"):
                    error_axis.axis("off")
                else:
                    error_artist = error_axis.imshow(
                        _masked_depth(
                            np.abs(arrays[key] - arrays["gt"]), valid),
                        cmap="magma", vmin=0.0, vmax=error_max,
                        zorder=2)
            if depth_row == 0:
                depth_axis.set_xlabel(label, fontsize=13)
                depth_axis.xaxis.set_label_position("top")
            if column == 0:
                depth_axis.set_ylabel(
                    "%s depth" % sample_name, fontsize=13)
                error_axis.set_ylabel(
                    "%s error" % sample_name, fontsize=13)
    if depth_artist is None or error_artist is None:
        raise RuntimeError("prediction figure has no depth or error artist")
    figure.subplots_adjust(
        left=0.05, right=0.96, bottom=0.04, top=0.94,
        wspace=0.04, hspace=0.08)
    depth_bar = figure.colorbar(
        depth_artist, ax=axes[::2, :].ravel().tolist(),
        fraction=0.012, pad=0.01)
    depth_bar.set_label("Depth (m)", fontsize=12)
    error_bar = figure.colorbar(
        error_artist, ax=axes[1::2, :].ravel().tolist(),
        fraction=0.012, pad=0.01)
    error_bar.set_label("Absolute error (m)", fontsize=12)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rmse_summary(path, seed_rows):
    methods = ("fp32", "rtn", "brecq", "qdrop")
    labels = ("FP32", "RTN W4A4", "BRECQ W4A4", "QDrop W4A4")
    x = np.arange(len(MODEL_ORDER), dtype=np.float64)
    width = 0.19
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    for method_index, (method, label) in enumerate(zip(methods, labels)):
        means = []
        errors = []
        for model in MODEL_ORDER:
            current = [
                float(row["mean_rmse"]) for row in seed_rows
                if row["model"] == model and row["method"] == method]
            if not current:
                raise ValueError("RMSE summary is incomplete")
            means.append(float(np.mean(current)))
            errors.append(float(np.std(current)))
        axis.bar(
            x + (method_index - 1.5) * width,
            means, width, yerr=errors, label=label,
            zorder=3, capsize=3)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.8, zorder=0)
    axis.set_ylabel("RMSE (m)", fontsize=14)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=0, fontsize=13)
    axis.legend(frameon=False, fontsize=11, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).resolve()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation = read_json(root / "qdrop_w4a4_evaluation.json")
    baseline_root = Path(evaluation["baseline_root"])
    seeds = tuple(int(seed) for seed in evaluation["seeds"])
    qdrop_config = str(evaluation["qdrop_evaluation_config"])
    sample_rows = read_csv(root / "qdrop_w4a4_sample_metrics.csv")
    seed_rows = read_csv(root / "qdrop_w4a4_seed_summary.csv")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
    })
    plot_rmse_summary(output / "qdrop_w4a4_rmse.png", seed_rows)
    selections = []
    for model in MODEL_ORDER:
        current = [row for row in sample_rows if row["model"] == model]
        median, worst = select_visual_samples(current, seeds)
        samples = {
            "Median": load_prediction_arrays(
                root, baseline_root, model, median, seeds, qdrop_config),
            "Worst": load_prediction_arrays(
                root, baseline_root, model, worst, seeds, qdrop_config),
        }
        plot_model_predictions(
            output / ("%s_qdrop_w4a4_predictions.png" % model),
            samples, seeds)
        selections.append({
            "model": model,
            "median_sample": median,
            "worst_sample": worst,
        })
    with (output / "qdrop_visual_samples.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("model", "median_sample", "worst_sample"))
        writer.writeheader()
        writer.writerows(selections)


if __name__ == "__main__":
    main()
