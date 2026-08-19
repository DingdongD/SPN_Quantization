#!/usr/bin/env python3
"""Render aligned CSPN BRECQ and QDrop W4A4/W6A6 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_nyu_qdrop_w4a4 import artifact_hashes


PREDICTION_CONFIGS = {
    "W4A4": "PA_W4A4_PROP_A8",
    "W6A6": "PA_W6A6_PROP_A8",
}
PREDICTION_COLUMNS = {
    "W4A4": ("gt", "fp32", "rtn", "brecq", "qdrop"),
    "W6A6": ("gt", "fp32", "rtn", "brecq", "qdrop", "p3_t3"),
}
COLUMN_LABELS = {
    "gt": "GT",
    "fp32": "FP32",
    "rtn": "RTN",
    "brecq": "BRECQ",
    "qdrop": "QDrop",
    "p3_t3": "P3/T3",
}
SUMMARY_SERIES = (
    ("fp32", "FP32", "FP32"),
    ("rtn", "W4A4", "RTN W4A4"),
    ("brecq", "W4A4", "BRECQ W4A4"),
    ("qdrop", "W4A4", "QDrop W4A4"),
    ("rtn", "W6A6", "RTN W6A6"),
    ("brecq", "W6A6", "BRECQ W6A6"),
    ("qdrop", "W6A6", "QDrop W6A6"),
    ("p3_t3", "P3T3", "P3/T3"),
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def select_visual_samples(rows, precision, selected_seed):
    current = [
        row for row in rows
        if row["method"] == "qdrop" and
        row["precision"] == precision and
        int(row["seed"]) == int(selected_seed)]
    values = sorted(
        ((int(row["sample_index"]), float(row["RMSE"]))
         for row in current),
        key=lambda row: (row[1], row[0]))
    if not values:
        raise ValueError("QDrop visual sample metrics are empty")
    if len(set(index for index, _ in values)) != len(values):
        raise ValueError("QDrop visual sample metrics contain duplicates")
    return values[(len(values) - 1) // 2][0], values[-1][0]


def validate_prediction_sources(sources):
    required = ("fp32", "rtn", "brecq", "qdrop", "p3_t3")
    if tuple(sources) != required:
        raise KeyError("prediction sources are incomplete or unordered")
    reference = sources["fp32"]
    for field in ("sample_index", "rgb", "sparse", "gt", "fp32"):
        if field not in reference:
            raise KeyError("FP32 prediction source is missing %s" % field)
        value = np.asarray(reference[field])
        if not np.isfinite(value).all():
            raise ValueError("reference %s contains non-finite values" % field)
        for name in required[1:]:
            if field not in sources[name]:
                raise KeyError("%s prediction source is missing %s" %
                               (name, field))
            if field == "fp32":
                aligned = np.allclose(
                    value, sources[name][field],
                    rtol=1.0e-6, atol=1.0e-7)
            else:
                aligned = np.array_equal(value, sources[name][field])
            if not aligned:
                raise ValueError("prediction source %s differs" % field)
    shape = tuple(np.asarray(reference["gt"]).shape)
    if len(shape) != 2:
        raise ValueError("prediction depth arrays must be two-dimensional")
    if tuple(np.asarray(reference["rgb"]).shape) != shape + (3,):
        raise ValueError("prediction RGB shape does not match depth")
    for name in required:
        if "pred" not in sources[name]:
            raise KeyError("%s prediction source is missing pred" % name)
        if tuple(np.asarray(sources[name]["pred"]).shape) != shape:
            raise ValueError("%s prediction shape does not match GT" % name)


def _load_npz(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return dict((name, payload[name].copy()) for name in payload.files)


def _prediction_path(root, config, sample_index):
    return Path(root) / "cspn" / "predictions" / config / \
        ("sample_%05d.npz" % int(sample_index))


def load_prediction_arrays(root, p3_root, sample_index, precision,
                           selected_seed):
    root = Path(root)
    config = PREDICTION_CONFIGS[precision]
    sources = {
        "fp32": _load_npz(_prediction_path(
            root / "evaluation" / "rtn", "FP32", sample_index)),
        "rtn": _load_npz(_prediction_path(
            root / "evaluation" / "rtn", config, sample_index)),
        "brecq": _load_npz(_prediction_path(
            root / "evaluation" / "brecq" / precision,
            config, sample_index)),
        "qdrop": _load_npz(_prediction_path(
            root / "evaluation" / "qdrop" / precision /
            ("seed_%d" % int(selected_seed)), config, sample_index)),
        "p3_t3": _load_npz(
            Path(p3_root) / "predictions" / "CONTEXT_P3_T3_W8A8" /
            ("sample_%05d.npz" % int(sample_index))),
    }
    validate_prediction_sources(sources)
    return {
        "rgb": sources["fp32"]["rgb"],
        "sparse": sources["fp32"]["sparse"],
        "gt": sources["fp32"]["gt"],
        "fp32": sources["fp32"]["pred"],
        "rtn": sources["rtn"]["pred"],
        "brecq": sources["brecq"]["pred"],
        "qdrop": sources["qdrop"]["pred"],
        "p3_t3": sources["p3_t3"]["pred"],
    }


def _masked(value, valid):
    array = np.asarray(value)
    return np.ma.masked_where(~valid | ~np.isfinite(array), array)


def plot_predictions(path, precision, samples):
    columns = PREDICTION_COLUMNS[precision]
    depth_values = []
    error_values = []
    for arrays in samples.values():
        valid = arrays["gt"] > 1.0e-4
        depth_values.append(arrays["gt"][valid])
        for name in columns[1:]:
            current = np.asarray(arrays[name])
            finite = valid & np.isfinite(current)
            depth_values.append(current[finite])
            error_values.append(np.abs(current[finite] - arrays["gt"][finite]))
    depth = np.concatenate(depth_values)
    errors = np.concatenate(error_values)
    depth_min, depth_max = np.percentile(depth, (1.0, 99.0))
    error_max = max(float(np.percentile(errors, 99.0)), 1.0e-6)
    figure, axes = plt.subplots(
        2 * len(samples), len(columns),
        figsize=(2.7 * len(columns), 4.7 * len(samples)), squeeze=False)
    depth_artist = None
    error_artist = None
    for rank, (sample_name, arrays) in enumerate(samples.items()):
        valid = arrays["gt"] > 1.0e-4
        depth_row = 2 * rank
        error_row = depth_row + 1
        for column, name in enumerate(columns):
            depth_axis = axes[depth_row, column]
            error_axis = axes[error_row, column]
            for axis in (depth_axis, error_axis):
                axis.set_axisbelow(True)
                axis.grid(False)
                axis.set_xticks([])
                axis.set_yticks([])
            depth_artist = depth_axis.imshow(
                _masked(arrays[name], valid), cmap="viridis",
                vmin=depth_min, vmax=depth_max, zorder=2)
            if name == "gt":
                error_axis.axis("off")
            else:
                error_artist = error_axis.imshow(
                    _masked(np.abs(arrays[name] - arrays["gt"]), valid),
                    cmap="magma", vmin=0.0, vmax=error_max, zorder=2)
            if depth_row == 0:
                depth_axis.set_xlabel(
                    COLUMN_LABELS[name], fontsize=15, labelpad=8)
                depth_axis.xaxis.set_label_position("top")
            if column == 0:
                depth_axis.set_ylabel("%s depth" % sample_name, fontsize=14)
                error_axis.set_ylabel("%s error" % sample_name, fontsize=14)
    if depth_artist is None or error_artist is None:
        raise RuntimeError("prediction figure has no depth or error artist")
    figure.subplots_adjust(
        left=0.07, right=0.94, bottom=0.04, top=0.92,
        wspace=0.03, hspace=0.08)
    depth_bar = figure.colorbar(
        depth_artist, ax=axes[::2, :].ravel().tolist(),
        fraction=0.014, pad=0.012)
    depth_bar.set_label("Depth (m)", fontsize=13)
    error_bar = figure.colorbar(
        error_artist, ax=axes[1::2, :].ravel().tolist(),
        fraction=0.014, pad=0.012)
    error_bar.set_label("Absolute error (m)", fontsize=13)
    for suffix in (".png", ".pdf"):
        figure.savefig(
            str(Path(path).with_suffix(suffix)), dpi=200,
            bbox_inches="tight")
    plt.close(figure)


def plot_rmse_summary(path, seed_rows):
    means = []
    errors = []
    invalid_ratios = []
    labels = []
    for method, precision, label in SUMMARY_SERIES:
        current_rows = [
            row for row in seed_rows
            if row["method"] == method and row["precision"] == precision]
        if not current_rows:
            raise ValueError("RMSE summary is missing %s" % label)
        current = [
            float(row["mean_rmse"]) for row in seed_rows
            if row["method"] == method and row["precision"] == precision]
        ratios = [float(row["nonfinite_ratio"]) for row in current_rows]
        if np.isfinite(current).all():
            if any(ratio != 0.0 for ratio in ratios):
                raise ValueError("finite RMSE has invalid outputs for %s" % label)
            means.append(float(np.mean(current)))
            errors.append(float(np.std(current, ddof=0)))
            invalid_ratios.append(0.0)
        else:
            if any(not np.isinf(value) for value in current) or \
                    any(ratio <= 0.0 for ratio in ratios):
                raise ValueError("invalid RMSE summary differs for %s" % label)
            means.append(float("nan"))
            errors.append(0.0)
            invalid_ratios.append(float(np.mean(ratios)))
        labels.append(label)
    x = np.arange(len(labels), dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    finite = np.isfinite(means)
    if not finite.any():
        raise ValueError("RMSE summary has no finite methods")
    figure, axis = plt.subplots(figsize=(14.5, 5.8))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.8, zorder=0)
    axis.bar(
        x[finite], means[finite], yerr=errors[finite],
        width=0.72, color="#4c78a8",
        edgecolor="#2f2f2f", linewidth=0.5, capsize=3, zorder=3)
    upper = float(np.max(means[finite] + errors[finite])) * 1.18
    axis.set_ylim(0.0, upper)
    for index, ratio in enumerate(invalid_ratios):
        if ratio == 0.0:
            continue
        axis.scatter(
            x[index], upper * 0.82, marker="x", s=55,
            linewidths=1.5, color="#c83e3e", zorder=4)
        axis.text(
            x[index], upper * 0.76,
            "Invalid\n%.1f%%" % (100.0 * ratio),
            ha="center", va="top", fontsize=10, color="#9b2f2f")
    axis.set_ylabel("RMSE (m)", fontsize=15)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=0, fontsize=12)
    axis.tick_params(axis="y", labelsize=12)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(
            str(Path(path).with_suffix(suffix)), dpi=200,
            bbox_inches="tight")
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
    manifest = read_json(root / "manifest.json")
    p3_root = Path(manifest["p3_t3_root"])
    selected = read_json(root / "selected_qdrop_seeds.json")
    sample_rows = read_csv(root / "sample_metrics.csv")
    seed_rows = read_csv(root / "seed_summary.csv")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
    })
    plot_rmse_summary(output / "cspn_rmse_comparison", seed_rows)
    selections = []
    for precision in ("W4A4", "W6A6"):
        seed = int(selected[precision])
        median, worst = select_visual_samples(
            sample_rows, precision, seed)
        samples = {
            "Median": load_prediction_arrays(
                root, p3_root, median, precision, seed),
            "Worst": load_prediction_arrays(
                root, p3_root, worst, precision, seed),
        }
        plot_predictions(
            output / ("cspn_%s_predictions" % precision.lower()),
            precision, samples)
        selections.append({
            "precision": precision,
            "selected_seed": seed,
            "median_sample": median,
            "worst_sample": worst,
        })
    with (output / "visual_samples.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "precision", "selected_seed",
                "median_sample", "worst_sample"))
        writer.writeheader()
        writer.writerows(selections)
    manifest["artifacts"] = artifact_hashes(root)
    write_json(root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
