#!/usr/bin/env python3
"""Plot persisted CSPN PA-W8A8 Im2Col diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


K_AXIS_METRICS = (
    ("activation_rms", "Activation RMS"),
    ("activation_p99", "Activation p99"),
    ("activation_sqnr_db", "Activation SQNR (dB)"),
    ("activation_new_zero_rate", "Activation new-zero rate"),
    ("weight_rms", "Weight RMS"),
    ("weight_sqnr_db", "Weight SQNR (dB)"),
)
SPATIAL_METRICS = (
    ("patch_rms", "Patch RMS"),
    ("activation_error", "A8 input error energy"),
    ("local_output_error", "Local Conv output error"),
)
SPATIAL_FIELDS = {
    "module", "sample_index", "patch_rms", "patch_p99",
    "patch_maximum_abs", "activation_error",
    "activation_new_zero_count", "activation_saturation_count",
    "local_output_error",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font-size", type=int, required=True)
    parser.add_argument("--dpi", type=int, required=True)
    parser.add_argument("--row-stride", type=int, required=True)
    return parser.parse_args(argv)


def set_style(font_size: int) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": int(font_size),
        "axes.titlesize": int(font_size),
        "axes.labelsize": int(font_size),
        "xtick.labelsize": int(font_size),
        "ytick.labelsize": int(font_size),
        "axes.axisbelow": True,
    })


def _read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("plot input table is empty: %s" % path)
    return rows


def _write_manifest(path: Path, rows) -> None:
    fields = (
        "kind", "module", "sample_index", "source", "png", "pdf",
        "row_stride")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _module_directory(root: Path, module: str) -> Path:
    return root.joinpath(*str(module).split("."))


def _finite(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("plot metric must be finite: %s" % label)
    return values


def plot_k_axis(module: str, rows, output_root: Path, dpi: int):
    selected = [row for row in rows if row["module"] == module]
    if not selected:
        raise ValueError("missing K-axis rows: %s" % module)
    offsets = sorted(set(int(row["kernel_offset"]) for row in selected))
    channels = sorted(set(int(row["channel"]) for row in selected))
    expected = set((channel, offset) for channel in channels for offset in offsets)
    observed = set(
        (int(row["channel"]), int(row["kernel_offset"]))
        for row in selected)
    if observed != expected:
        raise ValueError("K-axis channel-offset grid is incomplete: %s" % module)
    by_coordinate = dict(
        ((int(row["channel"]), int(row["kernel_offset"])), row)
        for row in selected)
    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    colors = matplotlib.colormaps["tab10"](
        np.linspace(0.0, 1.0, len(offsets)))
    for panel, (metric, label) in enumerate(K_AXIS_METRICS, 1):
        axis = figure.add_subplot(2, 3, panel, projection="3d")
        for color, offset in zip(colors, offsets):
            z = _finite(np.asarray([
                float(by_coordinate[(channel, offset)][metric])
                for channel in channels]), metric)
            axis.plot(
                channels, np.full(len(channels), offset), z,
                color=color, linewidth=1.4, zorder=3,
                label="Offset %d" % offset)
        axis.set_xlabel("Input channel")
        axis.set_ylabel("Kernel offset")
        axis.set_zlabel(label)
        axis.set_title(label)
        axis.view_init(elev=25, azim=-58)
        axis.grid(True)
    directory = _module_directory(output_root / "k_axis_3d", module)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / "distribution.png"
    pdf = directory / "distribution.pdf"
    figure.savefig(png, dpi=int(dpi))
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def load_spatial(path: Path, module: str, sample_index: int):
    with np.load(path, allow_pickle=False) as source:
        payload = dict((key, source[key]) for key in source.files)
    if set(payload) != SPATIAL_FIELDS:
        raise ValueError("spatial diagnostic schema changed")
    if str(payload["module"].item()) != module or \
            int(payload["sample_index"].item()) != int(sample_index):
        raise ValueError("spatial diagnostic identity changed")
    shape = payload["patch_rms"].shape
    if len(shape) != 2:
        raise ValueError("spatial diagnostic arrays must be Hout x Wout")
    for key in SPATIAL_FIELDS - {"module", "sample_index"}:
        if payload[key].shape != shape or \
                not bool(np.isfinite(payload[key]).all()):
            raise ValueError("spatial diagnostic array changed: %s" % key)
    return payload


def plot_spatial(module: str, sample_index: int, payload,
                 output_root: Path, dpi: int, row_stride: int):
    height, width = payload["patch_rms"].shape
    columns = np.arange(width)
    rows = tuple(range(0, height, int(row_stride)))
    if rows[-1] != height - 1:
        rows = rows + (height - 1,)
    colors = matplotlib.colormaps["viridis"](
        np.linspace(0.0, 1.0, len(rows)))
    figure = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    for panel, (metric, label) in enumerate(SPATIAL_METRICS, 1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        values = _finite(payload[metric], metric)
        for color, row in zip(colors, rows):
            axis.plot(
                columns, np.full(width, row), values[row],
                color=color, linewidth=1.0, zorder=3)
        axis.set_xlabel("Output column")
        axis.set_ylabel("Output row")
        axis.set_zlabel(label)
        axis.set_title(label)
        axis.view_init(elev=28, azim=-62)
        axis.grid(True)
    directory = _module_directory(output_root / "spatial_3d", module)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / ("sample_%05d.png" % int(sample_index))
    pdf = directory / ("sample_%05d.pdf" % int(sample_index))
    figure.savefig(png, dpi=int(dpi))
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main(argv=None):
    args = parse_args(argv)
    if args.font_size <= 0 or args.dpi <= 0 or args.row_stride <= 0:
        raise ValueError("plot dimensions and stride must be positive")
    experiment = Path(args.experiment_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(str(output))
    output.mkdir(parents=True)
    manifest = json.loads(
        (experiment / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != "cspn" or \
            manifest["configuration"] != "PA_W8A8":
        raise ValueError("plot input must be CSPN PA_W8A8")
    modules = tuple(str(module) for module in manifest["selected_plot_modules"])
    samples = tuple(int(index) for index in manifest["selected_plot_samples"])
    if not modules or not samples:
        raise ValueError("plot selections must be nonempty")
    k_rows = _read_csv(experiment / "channel_offset_metrics.csv")
    spatial_rows = _read_csv(experiment / "spatial_manifest.csv")
    spatial_by_identity = dict(
        ((row["module"], int(row["sample_index"])), row)
        for row in spatial_rows)
    set_style(args.font_size)
    figure_rows = []
    for module in modules:
        png, pdf = plot_k_axis(module, k_rows, output, args.dpi)
        figure_rows.append({
            "kind": "k_axis_3d",
            "module": module,
            "sample_index": "",
            "source": "channel_offset_metrics.csv",
            "png": str(png.relative_to(output)),
            "pdf": str(pdf.relative_to(output)),
            "row_stride": "",
        })
        for sample_index in samples:
            identity = (module, sample_index)
            if identity not in spatial_by_identity:
                raise ValueError(
                    "missing spatial manifest identity: %s %d" % identity)
            source_path = experiment / spatial_by_identity[identity]["path"]
            payload = load_spatial(source_path, module, sample_index)
            png, pdf = plot_spatial(
                module, sample_index, payload, output,
                args.dpi, args.row_stride)
            figure_rows.append({
                "kind": "spatial_3d",
                "module": module,
                "sample_index": sample_index,
                "source": str(source_path.relative_to(experiment)),
                "png": str(png.relative_to(output)),
                "pdf": str(pdf.relative_to(output)),
                "row_stride": int(args.row_stride),
            })
    _write_manifest(output / "figure_manifest.csv", figure_rows)
    print("CSPN PA_W8A8 Im2Col figures written", flush=True)


if __name__ == "__main__":
    main()
