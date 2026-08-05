#!/usr/bin/env python3
"""Build side-by-side NYU prediction and error-map comparisons."""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["cspn", "dyspn", "nlspn", "completionformer"]
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}


def load_prediction(path):
    data = np.load(str(path), allow_pickle=False)
    metrics = json.loads(str(data["metrics"]))
    return {
        "path": str(path),
        "rgb": data["rgb"],
        "sparse": data["sparse"],
        "gt": data["gt"],
        "pred": data["pred"],
        "abs_err": data["abs_err"],
        "valid": data["valid"].astype(bool),
        "label": str(data["label"]),
        "model": str(data["model"]),
        "iteration": int(data["iteration"]),
        "sample_index": int(data["sample_index"]),
        "metrics": metrics,
    }


def collect(pred_dir):
    by_sample = {}
    for path in sorted(Path(pred_dir).glob("sample_*.npz")):
        pred = load_prediction(path)
        by_sample.setdefault(pred["sample_index"], {})[pred["model"]] = pred
    return by_sample


def model_sort_key(model):
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER), model


def write_metrics_csv(path, rows):
    fieldnames = [
        "sample_index", "model", "iteration", "RMSE", "MAE", "ABS_REL",
        "MSE", "npz_path",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((k, row.get(k, "")) for k in fieldnames))


def summarize(rows):
    out = []
    for model in sorted(set(row["model"] for row in rows), key=model_sort_key):
        points = [row for row in rows if row["model"] == model]
        out.append({
            "model": model,
            "iteration": points[0]["iteration"],
            "mean_RMSE": float(np.mean([row["RMSE"] for row in points])),
            "mean_MAE": float(np.mean([row["MAE"] for row in points])),
            "mean_ABS_REL": float(np.mean([row["ABS_REL"] for row in points])),
            "num_samples": len(points),
        })
    return out


def write_summary_csv(path, rows):
    fieldnames = ["model", "iteration", "mean_RMSE", "mean_MAE", "mean_ABS_REL", "num_samples"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def set_matplotlib_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def draw_panel(ax, img, cmap=None, vmin=None, vmax=None, title="", xlabel=""):
    if cmap is None:
        im = ax.imshow(np.clip(img, 0.0, 1.0), aspect="auto")
    else:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def make_sample_figure(sample_index, preds, out_path):
    models = sorted(preds.keys(), key=model_sort_key)
    ref = preds[models[0]]
    valid = ref["valid"]
    gt = ref["gt"]
    if np.any(valid):
        vmin, vmax = float(np.min(gt[valid])), float(np.max(gt[valid]))
    else:
        vmin, vmax = 0.0, 1.0
    emax = max([float(np.max(preds[m]["abs_err"][valid])) for m in models] + [1e-3])

    ncols = 3 + 2 * len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(2.45 * ncols, 2.75))
    panels = [
        (ref["rgb"], None, None, None, "RGB", ""),
        (np.ma.masked_where(ref["sparse"] <= 1e-4, ref["sparse"]),
         "viridis", vmin, vmax, "Sparse", ""),
        (np.ma.masked_where(~valid, gt), "viridis", vmin, vmax, "GT", ""),
    ]
    for model in models:
        item = preds[model]
        name = MODEL_NAMES.get(model, model)
        metric = item["metrics"]
        panels.extend([
            (item["pred"], "viridis", vmin, vmax,
             "%s pred" % name, "iter=%d RMSE=%.3f" % (item["iteration"], metric["RMSE"])),
            (np.ma.masked_where(~valid, item["abs_err"]), "magma", 0.0, emax,
             "%s |err|" % name, "MAE=%.3f" % metric["MAE"]),
        ])

    for ax, (img, cmap, lo, hi, title, xlabel) in zip(axes, panels):
        draw_panel(ax, img, cmap, lo, hi, title, xlabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_grid(by_sample, out_path):
    sample_ids = sorted(by_sample.keys())
    models = sorted(set(m for preds in by_sample.values() for m in preds.keys()),
                    key=model_sort_key)
    ncols = 3 + 2 * len(models)
    nrows = len(sample_ids)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.45 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    for r, sample_index in enumerate(sample_ids):
        preds = by_sample[sample_index]
        ref = preds[models[0]]
        valid = ref["valid"]
        gt = ref["gt"]
        if np.any(valid):
            vmin, vmax = float(np.min(gt[valid])), float(np.max(gt[valid]))
        else:
            vmin, vmax = 0.0, 1.0
        emax = max([float(np.max(preds[m]["abs_err"][valid])) for m in models if m in preds] + [1e-3])
        panels = [
            (ref["rgb"], None, None, None, "RGB", "%05d" % sample_index),
            (np.ma.masked_where(ref["sparse"] <= 1e-4, ref["sparse"]),
             "viridis", vmin, vmax, "Sparse", ""),
            (np.ma.masked_where(~valid, gt), "viridis", vmin, vmax, "GT", ""),
        ]
        for model in models:
            item = preds[model]
            name = MODEL_NAMES.get(model, model)
            metric = item["metrics"]
            panels.extend([
                (item["pred"], "viridis", vmin, vmax,
                 "%s pred" % name, "RMSE=%.3f" % metric["RMSE"]),
                (np.ma.masked_where(~valid, item["abs_err"]), "magma", 0.0, emax,
                 "%s |err|" % name, "MAE=%.3f" % metric["MAE"]),
            ])
        for c, (img, cmap, lo, hi, title, xlabel) in enumerate(panels):
            draw_panel(axes[r, c], img, cmap, lo, hi,
                       title if r == 0 else "", xlabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_bar_chart(path, summary):
    models = [MODEL_NAMES.get(row["model"], row["model"]) for row in summary]
    rmse = [row["mean_RMSE"] for row in summary]
    mae = [row["mean_MAE"] for row in summary]
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_axisbelow(True)
    ax.bar(x - width / 2, rmse, width, label="RMSE", zorder=3)
    ax.bar(x + width / 2, mae, width, label="MAE", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Error (m)")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="profile_logs/nyu_prediction_comparison/predictions")
    parser.add_argument("--out-dir", default="profile_logs/nyu_prediction_comparison")
    args = parser.parse_args()

    set_matplotlib_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_sample = collect(args.pred_dir)
    if not by_sample:
        raise RuntimeError("no prediction npz files found in %s" % args.pred_dir)

    rows = []
    for sample_index, preds in sorted(by_sample.items()):
        for model in sorted(preds.keys(), key=model_sort_key):
            item = preds[model]
            row = {
                "sample_index": sample_index,
                "model": model,
                "iteration": item["iteration"],
                "npz_path": item["path"],
            }
            row.update(item["metrics"])
            rows.append(row)
        make_sample_figure(sample_index, preds,
                           out_dir / ("sample_%05d_prediction_error.png" % sample_index))
    make_grid(by_sample, out_dir / "prediction_error_comparison_grid.png")

    summary = summarize(rows)
    write_metrics_csv(out_dir / "sample_error_metrics.csv", rows)
    write_summary_csv(out_dir / "model_error_summary.csv", summary)
    (out_dir / "model_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    make_bar_chart(out_dir / "model_error_summary_bar.png", summary)

    print("samples=%d models=%d out_dir=%s" % (
        len(by_sample), len(summary), out_dir), flush=True)
    for row in summary:
        print("%s iter=%d mean_rmse=%.5f mean_mae=%.5f" % (
            row["model"], row["iteration"], row["mean_RMSE"], row["mean_MAE"]), flush=True)


if __name__ == "__main__":
    main()
