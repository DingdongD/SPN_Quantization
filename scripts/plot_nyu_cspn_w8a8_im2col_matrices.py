#!/usr/bin/env python3
"""Render complete CSPN Conv weight and activation Im2Col matrices."""

from __future__ import annotations

import argparse
from io import BytesIO
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spn_quant.im2col_matrix_visualization import (  # noqa: E402
    ConvMatrixCapture,
    full_line_curtain,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--font-size", type=int, required=True)
    parser.add_argument("--dpi", type=int, required=True)
    parser.add_argument("--line-width", type=float, required=True)
    parser.add_argument("--elevation", type=float, required=True)
    parser.add_argument("--azimuth", type=float, required=True)
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
        raise ValueError("matrix plot source table is empty: %s" % path)
    return rows


def _write_csv(path: Path, rows) -> None:
    if not rows:
        raise ValueError("matrix figure manifest must be nonempty")
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _module_directory(root: Path, module: str) -> Path:
    return root.joinpath(*str(module).split("."))


def _activation_matrix(capture: ConvMatrixCapture,
                       tensor) -> np.ndarray:
    layout = capture.geometry.layout(capture.module)
    matrix = layout.unfold(tensor)[0].transpose(0, 1).contiguous()
    return matrix.numpy()


def _weight_matrix(capture: ConvMatrixCapture,
                   tensor) -> np.ndarray:
    layout = capture.geometry.layout(capture.module)
    return layout.flatten_weight(tensor).contiguous().numpy()


def _panel_title(title: str, maximum: float) -> str:
    return str(title) if float(maximum) > 0.0 else \
        "%s (all zero)" % str(title)


def _panel_image(matrix: np.ndarray, transpose: bool, title: str,
                 x_label: str, y_label: str, z_label: str,
                 maximum: float, dpi: int, line_width: float,
                 elevation: float, azimuth: float):
    plotted = np.ascontiguousarray(
        matrix.transpose() if transpose else matrix,
        dtype=np.float32)
    curtain = full_line_curtain(plotted)
    norm_maximum = float(maximum) if float(maximum) > 0.0 else 1.0
    normalization = colors.Normalize(vmin=0.0, vmax=norm_maximum)
    colormap = matplotlib.colormaps["viridis"]
    peaks = np.asarray([
        float(line[:, 2].max()) for line in curtain.lines],
        dtype=np.float32)
    collection = Line3DCollection(
        curtain.lines,
        colors=colormap(normalization(peaks)),
        linewidths=float(line_width), alpha=0.82,
        rasterized=True, zorder=3)
    figure = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(collection)
    rows, columns = plotted.shape
    axis.set_xlim(0, max(columns - 1, 1))
    axis.set_ylim(0, max(rows - 1, 1))
    axis.set_zlim(0, norm_maximum)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_zlabel(z_label)
    axis.set_title(_panel_title(title, maximum))
    axis.view_init(elev=float(elevation), azim=float(azimuth))
    axis.set_box_aspect((1.45, 1.0, 0.8))
    axis.grid(True)
    scalar = ScalarMappable(norm=normalization, cmap=colormap)
    scalar.set_array(peaks)
    colorbar = figure.colorbar(scalar, ax=axis, shrink=0.62, pad=0.1)
    if float(maximum) == 0.0:
        colorbar.set_label("All values = 0")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=int(dpi))
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as source:
        image = source.convert("RGB").copy()
    buffer.close()
    return image, curtain.orientation, curtain.rendered_elements


def _compose(images, png: Path, pdf: Path, dpi: int) -> None:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    combined = Image.new("RGB", (width, height), "white")
    left = 0
    for image in images:
        combined.paste(image, (left, 0))
        left += image.width
    combined.save(png, format="PNG", dpi=(int(dpi), int(dpi)))
    combined.save(pdf, format="PDF", resolution=float(dpi))
    combined.close()
    for image in images:
        image.close()


def _weight_triptych(capture: ConvMatrixCapture, output: Path,
                      args):
    shared = max(
        float(capture.original_weight.abs().max().item()),
        float(capture.quantized_weight.abs().max().item()))
    error = float((
        capture.original_weight - capture.quantized_weight
    ).abs().max().item())
    specifications = (
        (capture.original_weight, "FP32 |W_col|", shared),
        (capture.quantized_weight, "W8 QDQ |W_col|", shared),
        (capture.original_weight - capture.quantized_weight,
         "Absolute W8 error", error),
    )
    images = []
    orientation = None
    rendered = None
    for tensor, title, maximum in specifications:
        matrix = np.abs(_weight_matrix(capture, tensor)).astype(
            np.float32, copy=False)
        image, current_orientation, current_rendered = _panel_image(
            matrix, False, title,
            "K = input channel x kernel offset", "Output channel",
            "Absolute weight", maximum, args.dpi, args.line_width,
            args.elevation, args.azimuth)
        images.append(image)
        if orientation is None:
            orientation = current_orientation
            rendered = current_rendered
        elif orientation != current_orientation or rendered != current_rendered:
            raise RuntimeError("weight panel matrix identity changed")
    png = output / ("sample_%05d_weights.png" % capture.sample_index)
    pdf = output / ("sample_%05d_weights.pdf" % capture.sample_index)
    _compose(images, png, pdf, args.dpi)
    rows = capture.geometry.out_channels
    columns = capture.geometry.in_channels * \
        capture.geometry.kernel_size[0] * capture.geometry.kernel_size[1]
    return png, pdf, rows, columns, orientation, rendered, shared, error


def _activation_triptych(capture: ConvMatrixCapture, output: Path,
                          args):
    shared = max(
        float(capture.reference_input.abs().max().item()),
        float(capture.quantized_input.abs().max().item()))
    difference = capture.reference_input - capture.quantized_input
    error = float(difference.abs().max().item())
    specifications = (
        (capture.reference_input, "Pre-QDQ |X_col|", shared),
        (capture.quantized_input, "A8 QDQ |X_col|", shared),
        (difference, "Absolute A8 error", error),
    )
    images = []
    orientation = None
    rendered = None
    semantic_shape = None
    for tensor, title, maximum in specifications:
        matrix = np.abs(_activation_matrix(capture, tensor)).astype(
            np.float32, copy=False)
        if semantic_shape is None:
            semantic_shape = matrix.shape
        elif semantic_shape != matrix.shape:
            raise RuntimeError("activation panel matrix identity changed")
        image, current_orientation, current_rendered = _panel_image(
            matrix, True, title, "Spatial patch / token",
            "K = input channel x kernel offset", "Absolute activation",
            maximum, args.dpi, args.line_width,
            args.elevation, args.azimuth)
        images.append(image)
        if orientation is None:
            orientation = current_orientation
            rendered = current_rendered
        elif orientation != current_orientation or rendered != current_rendered:
            raise RuntimeError("activation panel matrix identity changed")
    png = output / ("sample_%05d_activations.png" % capture.sample_index)
    pdf = output / ("sample_%05d_activations.pdf" % capture.sample_index)
    _compose(images, png, pdf, args.dpi)
    return (png, pdf, semantic_shape[0], semantic_shape[1],
            orientation, rendered, shared, error)


def main(argv=None):
    args = parse_args(argv)
    values = (
        args.font_size, args.dpi, args.line_width,
        args.elevation, abs(args.azimuth))
    if any(float(value) <= 0.0 for value in values):
        raise ValueError("matrix plot dimensions and camera must be nonzero")
    capture_root = Path(args.capture_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    output.mkdir(parents=True)
    source_rows = _read_csv(capture_root / "capture_manifest.csv")
    set_style(args.font_size)
    figure_rows = []
    for row in source_rows:
        capture = ConvMatrixCapture.load(capture_root / row["path"])
        if capture.module != row["module"] or \
                capture.sample_index != int(row["sample_index"]):
            raise ValueError("capture manifest identity changed")
        directory = _module_directory(output, capture.module)
        directory.mkdir(parents=True, exist_ok=True)
        for kind, result in (
                ("weight", _weight_triptych(capture, directory, args)),
                ("activation", _activation_triptych(
                    capture, directory, args))):
            png, pdf, rows, columns, orientation, rendered, shared, error = result
            elements = int(rows) * int(columns)
            if int(rendered) != elements:
                raise RuntimeError("full matrix rendering sampled elements")
            figure_rows.append({
                "module": capture.module,
                "sample_index": capture.sample_index,
                "kind": kind,
                "matrix_rows": rows,
                "matrix_columns": columns,
                "matrix_elements": elements,
                "rendered_elements": rendered,
                "line_orientation": orientation,
                "sampling": "none",
                "shared_reference_qdq_maximum": shared,
                "error_maximum": error,
                "source": row["path"],
                "png": str(png.relative_to(output)),
                "pdf": str(pdf.relative_to(output)),
            })
        print("full matrix figures %s" % capture.module, flush=True)
    _write_csv(output / "figure_manifest.csv", figure_rows)
    print("CSPN full Im2Col matrix figures written", flush=True)


if __name__ == "__main__":
    main()
