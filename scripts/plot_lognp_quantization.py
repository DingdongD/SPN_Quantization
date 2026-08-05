#!/usr/bin/env python3
"""Aggregate and plot LogNP activation quantization reports."""

from __future__ import print_function

import argparse
import csv
import math
import os
from pathlib import Path


NUMERIC_FIELDS = ("mse", "p50", "p75", "p99", "p99_9",
                  "nonfinite_rate", "zero_code_rate", "saturation_rate",
                  "sqnr_db", "transformed_sqnr_db")


def _float(row, key, default=0.0):
    value = row.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _int(row, key, default=1):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def aggregate_lognp_rows(rows):
    """Aggregate layer rows with activation-error fields by model/config."""
    groups = {}
    for row in rows:
        if "model" not in row or "config" not in row:
            continue
        if not any(key in row for key in ("p50", "p99", "p99_9")):
            continue
        key = (row["model"], row["config"])
        group = groups.setdefault(key, {
            "model": row["model"], "config": row["config"],
            "numel": 0, "_weighted": dict((field, 0.0)
                                             for field in NUMERIC_FIELDS),
        })
        weight = max(_int(row, "numel"), 1)
        group["numel"] += weight
        for field in NUMERIC_FIELDS:
            group["_weighted"][field] += _float(row, field) * weight
    result = []
    for key in sorted(groups):
        group = groups[key]
        row = {"model": group["model"], "config": group["config"],
               "numel": group["numel"]}
        for field in NUMERIC_FIELDS:
            row[field] = group["_weighted"][field] / float(group["numel"])
        result.append(row)
    return result


def compare_lognp_configs(rows, baseline="LOGNP_W8A4_tensor"):
    """Add metric deltas against the selected per-model baseline."""
    baselines = dict((row["model"], row) for row in rows
                     if row.get("config") == baseline)
    compared = []
    for row in rows:
        output = dict(row)
        reference = baselines.get(row["model"])
        for field in ("mse", "p50", "p75", "p99", "p99_9",
                      "nonfinite_rate", "zero_code_rate",
                      "saturation_rate"):
            output[field + "_delta"] = (
                _float(row, field) - _float(reference, field)
                if reference is not None else float("nan"))
        compared.append(output)
    return compared


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    if not rows:
        return
    keys = set()
    for row in rows:
        keys.update(row)
    preferred = ["model", "config", "numel"]
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _sample_summary(root):
    rows = []
    for model_dir in sorted(Path(root).glob("*/sample_metrics.csv")):
        for row in _read_csv(model_dir):
            if not row.get("config", "").startswith("LOGNP_"):
                continue
            rows.append(row)
    groups = {}
    for row in rows:
        key = (row["model"], row["config"])
        group = groups.setdefault(key, {"model": row["model"],
                                        "config": row["config"],
                                        "count": 0, "RMSE": 0.0, "MAE": 0.0,
                                        "ABS_REL": 0.0,
                                        "nonfinite_pixels": 0.0,
                                        "num_pixels": 0.0})
        group["count"] += 1
        for field in ("RMSE", "MAE", "ABS_REL"):
            group[field] += _float(row, field)
        group["nonfinite_pixels"] += _float(row, "nonfinite_pixels")
        group["num_pixels"] += _float(row, "num_pixels")
    for group in groups.values():
        for field in ("RMSE", "MAE", "ABS_REL"):
            group[field] /= float(max(group["count"], 1))
        group["nonfinite_rate"] = group["nonfinite_pixels"] / float(
            max(group["num_pixels"], 1.0))
        del group["nonfinite_pixels"]
        del group["num_pixels"]
    return [groups[key] for key in sorted(groups)]


def plot_summary(summary, sample_summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    arial_path = "/root/.fonts/Arial.ttf"
    if os.path.exists(arial_path):
        font_manager.fontManager.addfont(arial_path)
        arial_name = font_manager.FontProperties(fname=arial_path).get_name()
    else:
        arial_name = "Arial"

    models = sorted(set(row["model"] for row in summary) |
                    set(row["model"] for row in sample_summary))
    if not models:
        raise RuntimeError("no LogNP rows to plot")
    plt.rcParams.update({
        "font.family": arial_name,
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
    figure, axes = plt.subplots(
        len(models), 1, figsize=(13, max(4.0, 3.4 * len(models))),
        squeeze=False)
    for axis, model in zip(axes[:, 0], models):
        current = [row for row in sample_summary if row["model"] == model]
        current = sorted(current, key=lambda row: row["config"])
        labels = [row["config"].replace("_", " ").title()
                  for row in current]
        x = list(range(len(labels)))
        width = 0.36
        rmse = [row["RMSE"] for row in current]
        mae = [row["MAE"] for row in current]
        bars_rmse = axis.bar(
            [value - width / 2 for value in x], rmse, width=width,
            label="RMSE", color="#4C78A8", zorder=2)
        bars_mae = axis.bar(
            [value + width / 2 for value in x], mae, width=width,
            label="MAE", color="#F58518", zorder=2)
        axis.set_ylabel(model.upper(), fontsize=14)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.8, zorder=0)
        axis.legend(frameon=False, ncol=2)
        for bar in list(bars_rmse) + list(bars_mae):
            bar.set_zorder(2)
    figure.tight_layout()
    figure.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="profile_logs/nyu_lognp_quantization")
    parser.add_argument("--out-dir", default="profile_logs/nyu_lognp_quantization/summary")
    args = parser.parse_args()
    root = Path(args.root)
    layer_rows = []
    for path in sorted(root.glob("*/layer_quantization_metrics.csv")):
        layer_rows.extend(_read_csv(path))
    summary = compare_lognp_configs(aggregate_lognp_rows(layer_rows))
    sample_summary = _sample_summary(root)
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "lognp_layer_summary.csv", summary)
    _write_csv(out_dir / "lognp_prediction_summary.csv", sample_summary)
    plot_summary(summary, sample_summary, out_dir / "lognp_prediction_metrics.png")


if __name__ == "__main__":
    main()
