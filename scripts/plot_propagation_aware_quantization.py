#!/usr/bin/env python3
"""Render static NYU propagation-aware quantization comparisons."""

from __future__ import division, print_function

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
CONFIGS = (
    "FP32",
    "PA_Generic_W4A4",
    "PA_Constraint",
    "PA_OffsetA8",
    "PA_StateA8",
    "PA_W4A8",
    "PA_W8A8",
)
PA_W4A4_CONFIGS = ("PA_Constraint", "PA_OffsetA8", "PA_StateA8")
LABELS = {
    "FP32": "FP32",
    "PA_Generic_W4A4": "Generic W4A4",
    "PA_Constraint": "PA constraint",
    "PA_OffsetA8": "PA offset A8",
    "PA_StateA8": "PA state A8",
    "PA_W4A8": "W4A8",
    "PA_W8A8": "W8A8",
}
INVALID_GT_RGBA = np.array([0.85, 0.85, 0.85, 1.0])
NONFINITE_RGBA = np.array([1.0, 0.0, 1.0, 1.0])


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.axisbelow": True,
    })


def select_detail_samples(rows, seed=20260804, random_count=4,
                          worst_count=4):
    generic = [row for row in rows
               if row["config"] == "PA_Generic_W4A4"]
    by_index = dict((int(row["sample_index"]), row) for row in generic)
    if len(by_index) < int(random_count) + int(worst_count):
        raise ValueError("not enough Generic W4A4 samples for detail selection")
    def failure_rank(index):
        row = by_index[index]
        nonfinite = int(float(row.get("nonfinite_pixels", 0) or 0))
        rmse = float(row["RMSE"])
        if not math.isfinite(rmse):
            rmse = float("inf")
        return -nonfinite, -rmse, index

    ranked = sorted(by_index, key=failure_rank)
    worst = ranked[:int(worst_count)]
    remaining = sorted(set(by_index) - set(worst))
    rng = np.random.RandomState(int(seed))
    random_indices = sorted(int(index) for index in rng.choice(
        remaining, size=int(random_count), replace=False))
    output = [{
        "sample_index": index,
        "reason": "random",
    } for index in random_indices]
    output.extend({
        "sample_index": index,
        "reason": "worst_generic_w4a4",
    } for index in worst)
    return output


def _load_payload(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return dict((key, payload[key]) for key in payload.files)


def load_predictions(model_root, expected_samples):
    output = {}
    expected_indices = None
    for config in CONFIGS:
        paths = sorted((Path(model_root) / "predictions" / config).glob(
            "sample_*.npz"))
        indices = [int(_load_payload(path)["sample_index"]) for path in paths]
        if len(indices) != int(expected_samples) or len(set(indices)) != len(indices):
            raise ValueError("%s prediction count is %d, expected %d" % (
                config, len(indices), expected_samples))
        if expected_indices is None:
            expected_indices = indices
            output = dict((index, {}) for index in indices)
        elif indices != expected_indices:
            raise ValueError("prediction sample sets differ for %s" % config)
        for path, index in zip(paths, indices):
            payload = _load_payload(path)
            if "sparse" not in payload:
                raise ValueError("prediction payload lacks sparse depth: %s" % path)
            output[index][config] = payload
    return output


def select_best_pa(sample_rows):
    means = {}
    for config in PA_W4A4_CONFIGS:
        values = [float(row["RMSE"]) for row in sample_rows
                  if row["config"] == config and
                  math.isfinite(float(row["RMSE"]))]
        if values:
            means[config] = float(np.mean(values))
    if not means:
        raise ValueError("no finite propagation-aware W4A4 metrics")
    return min(means, key=lambda config: (means[config], config))


def _hide_axis(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#666666")


def _semantic_rgba(values, valid, maximum, cmap):
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("values and validity mask must have identical shapes")
    finite = np.isfinite(values)
    normalized = np.clip(
        np.nan_to_num(values, nan=0.0, posinf=maximum, neginf=0.0) /
        float(maximum),
        0.0, 1.0)
    rgba = plt.get_cmap(cmap)(normalized)
    rgba[~valid] = INVALID_GT_RGBA
    rgba[valid & ~finite] = NONFINITE_RGBA
    return rgba


def depth_rgba(values, valid):
    return _semantic_rgba(values, valid, 10.0, "viridis")


def error_rgba(values, valid, maximum):
    return _semantic_rgba(values, valid, maximum, "magma")


def _depth_image(axis, values, valid):
    axis.imshow(depth_rgba(values, valid), interpolation="nearest",
                aspect="auto")
    _hide_axis(axis)


def contact_annotations(sample_index, panel, sample_rank, sample_columns):
    label = "GT" if panel == "GT" else LABELS[panel]
    title = label if int(sample_rank) < int(sample_columns) else ""
    return title, "#%05d" % int(sample_index)


def _render_contact(predictions, best_pa, path, sample_columns=4):
    indices = sorted(predictions)
    blocks_per_row = int(sample_columns)
    rows = int(math.ceil(len(indices) / float(blocks_per_row)))
    panels = (
        "GT", "FP32", "PA_Generic_W4A4", best_pa,
        "PA_W4A8", "PA_W8A8",
    )
    fig, axes = plt.subplots(
        rows, blocks_per_row * len(panels),
        figsize=(4.2 * blocks_per_row, 2.45 * rows), squeeze=False)
    for rank, index in enumerate(indices):
        row = rank // blocks_per_row
        first = (rank % blocks_per_row) * len(panels)
        payload = predictions[index]["FP32"]
        valid = payload["valid_gt"].astype(bool)
        values = {
            "GT": payload["gt"],
            "FP32": payload["pred"],
            "PA_Generic_W4A4": predictions[index]["PA_Generic_W4A4"]["pred"],
            best_pa: predictions[index][best_pa]["pred"],
            "PA_W4A8": predictions[index]["PA_W4A8"]["pred"],
            "PA_W8A8": predictions[index]["PA_W8A8"]["pred"],
        }
        for offset, panel in enumerate(panels):
            axis = axes[row, first + offset]
            _depth_image(axis, values[panel], valid)
            title, sample_label = contact_annotations(
                index, panel, rank, sample_columns)
            if title:
                axis.set_title(title, pad=2.0)
            if panel == "GT":
                axis.text(
                    0.03, 0.97, sample_label, transform=axis.transAxes,
                    ha="left", va="top", fontsize=7, color="#111111",
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.78, "pad": 1.0})
    used = len(indices) * len(panels)
    for flat_index, axis in enumerate(axes.flat):
        if flat_index >= used:
            axis.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                        wspace=0.03, hspace=0.20)
    fig.savefig(str(path), dpi=150, facecolor="white")
    plt.close(fig)


def _render_details(predictions, selected, best_pa, path):
    columns = (
        "Sparse", "GT", "FP32", "Generic W4A4", LABELS[best_pa],
        "W4A8", "W8A8", "Generic abs error", "PA W4A4 abs error",
        "W4A8 abs error", "W8A8 abs error",
    )
    fig, axes = plt.subplots(
        len(selected), len(columns),
        figsize=(23.5, 2.35 * len(selected)), squeeze=False)
    for row, selection in enumerate(selected):
        index = int(selection["sample_index"])
        payload = predictions[index]["FP32"]
        generic = predictions[index]["PA_Generic_W4A4"]
        pa = predictions[index][best_pa]
        w4a8 = predictions[index]["PA_W4A8"]
        w8a8 = predictions[index]["PA_W8A8"]
        valid = payload["valid_gt"].astype(bool)
        depth_values = (
            payload["sparse"], payload["gt"], payload["pred"],
            generic["pred"], pa["pred"], w4a8["pred"], w8a8["pred"],
        )
        errors = (
            generic["abs_err"], pa["abs_err"],
            w4a8["abs_err"], w8a8["abs_err"],
        )
        finite_errors = np.concatenate([
            value[np.isfinite(value) & valid] for value in errors])
        error_max = max(0.1, float(np.quantile(finite_errors, 0.99))) \
            if finite_errors.size else 1.0
        for column, values in enumerate(depth_values):
            _depth_image(axes[row, column], values, valid)
        for column, values in enumerate(errors, start=len(depth_values)):
            axes[row, column].imshow(
                error_rgba(values, valid, error_max),
                interpolation="nearest", aspect="auto")
            _hide_axis(axes[row, column])
        axes[row, 0].set_ylabel(
            "#%05d\n%s" % (index, selection["reason"].replace("_", " ")),
            rotation=0, ha="right", va="center")
        if row == 0:
            for column, label in enumerate(columns):
                axes[row, column].set_title(label, pad=3.0)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.02, top=0.97,
                        wspace=0.04, hspace=0.10)
    fig.savefig(str(path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def step_series(rows, config):
    current = [row for row in rows
               if row.get("config") == config and
               row.get("signal") == "propagation_states"]
    iterations = sorted(set(int(row["iteration"]) for row in current))
    means = []
    failure_rates = []
    for iteration in iterations:
        values = [float(row["rmse"]) for row in current
                  if int(row["iteration"]) == iteration]
        finite = [value for value in values if math.isfinite(value)]
        means.append(float(np.mean(finite)) if finite else float("nan"))
        failure_rates.append(
            float(sum(not math.isfinite(value) for value in values)) /
            float(len(values)))
    return iterations, means, failure_rates


def _render_step_error(rows, path):
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    failure_series = []
    for config in CONFIGS[1:]:
        iterations, values, failure_rates = step_series(rows, config)
        if not iterations:
            continue
        axis.plot(iterations, values, marker="o", markersize=3,
                  linewidth=1.5, label=LABELS[config], zorder=3)
        if any(rate > 0.0 for rate in failure_rates):
            failure_series.append((config, iterations, failure_rates))
    axis.set_xlabel("Propagation iteration")
    axis.set_ylabel("FP32-relative state RMSE (m)")
    axis.grid(True, color="#d8d8d8", linewidth=0.7, zorder=0)
    handles, labels = axis.get_legend_handles_labels()
    if failure_series:
        failure_axis = axis.twinx()
        for config, iterations, failure_rates in failure_series:
            line = failure_axis.plot(
                iterations, np.asarray(failure_rates) * 100.0,
                color="#111111", linestyle="--", marker="x",
                markersize=4, linewidth=1.2,
                label="%s non-finite" % LABELS[config], zorder=4)[0]
            handles.append(line)
            labels.append(line.get_label())
        failure_axis.set_ylabel("Non-finite sample rate (%)")
        failure_axis.set_ylim(0.0, 105.0)
    axis.legend(handles, labels, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def _render_constraints(rows, path):
    configs = [config for config in CONFIGS[2:]
               if any(row.get("config") == config for row in rows)]
    sum_error = []
    contraction = []
    for config in configs:
        current = [row for row in rows
                   if row.get("config") == config and
                   row.get("signal") == "affinity_constraints"]
        sum_error.append(max(float(row.get(
            "coefficient_sum_max_error", 0.0) or 0.0) for row in current))
        contraction.append(np.mean([float(row.get(
            "contraction_violation_rate", 0.0) or 0.0) for row in current]))
    positions = np.arange(len(configs))
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    width = 0.36
    axis.bar(positions - width / 2, sum_error, width,
             label="Coefficient sum max error", zorder=3)
    axis.bar(positions + width / 2, contraction, width,
             label="Contraction violation rate", zorder=3)
    axis.set_xticks(positions)
    axis.set_xticklabels([LABELS[config] for config in configs], rotation=0)
    axis.set_ylabel("Ratio")
    axis.grid(True, axis="y", color="#d8d8d8", linewidth=0.7, zorder=0)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180, facecolor="white")
    plt.close(fig)


def render_model(root, out_dir, model, expected_samples=64,
                 random_count=4, worst_count=4):
    model_root = Path(root) / model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = read_csv(model_root / "sample_metrics.csv")
    predictions = load_predictions(model_root, expected_samples)
    best_pa = select_best_pa(sample_rows)
    selected = select_detail_samples(
        sample_rows, random_count=random_count, worst_count=worst_count)
    write_csv(
        out_dir / ("%s_selected_visual_samples.csv" % model), selected,
        ("sample_index", "reason"))

    outputs = [
        out_dir / ("%s_prediction_contact_sheet.png" % model),
        out_dir / ("%s_prediction_details.png" % model),
        out_dir / ("%s_propagation_step_error.png" % model),
        out_dir / ("%s_constraint_violations.png" % model),
    ]
    _render_contact(predictions, best_pa, outputs[0])
    _render_details(predictions, selected, best_pa, outputs[1])
    _render_step_error(read_csv(model_root / "signal_metrics.csv"), outputs[2])
    _render_constraints(read_csv(
        model_root / "propagation_quantization_metrics.csv"), outputs[3])
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="profile_logs/nyu_propagation_aware_quantization")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--expected-samples", type=int, default=64)
    parser.add_argument("--models", nargs="*", default=list(MODEL_ORDER))
    args = parser.parse_args()
    set_style()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.root) / "figures"
    outputs = []
    for model in args.models:
        outputs.extend(render_model(
            args.root, out_dir, model,
            expected_samples=args.expected_samples))
    print("wrote %d figures to %s" % (len(outputs), out_dir), flush=True)


if __name__ == "__main__":
    main()
