#!/usr/bin/env python3
"""Render audited CSPN activation-resolution prediction comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image


CONFIG_ORDER = (
    "FP32", "W4A4_RTN", "W4A4_CHANNEL", "W4A4_CALIBRATED_SCALE",
)
CONFIG_NAMES = {
    "FP32": "FP32",
    "W4A4_RTN": "RTN W4A4",
    "W4A4_CHANNEL": "Per-channel W4A4",
    "W4A4_CALIBRATED_SCALE": "Calibrated-scale W4A4",
}
REQUIRED_KEYS = {
    "gt", "fp32", "pred", "abs_err", "valid_gt", "nonfinite",
    "sample_index", "model", "config", "sparse", "rgb",
}
DEPTH_MIN = 0.0
DEPTH_MAX = 10.0


def _load_payload(path: Path, config: str):
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != REQUIRED_KEYS:
            raise ValueError("prediction payload keys do not match contract")
        payload = dict((key, source[key]) for key in source.files)
    if str(payload["model"]) != "cspn":
        raise ValueError("prediction payload model must be cspn")
    if str(payload["config"]) != config:
        raise ValueError("prediction payload configuration mismatch")
    sample_index = int(payload["sample_index"])
    shape = payload["gt"].shape
    for key in ("fp32", "pred", "abs_err", "valid_gt", "nonfinite", "sparse"):
        if payload[key].shape != shape:
            raise ValueError("prediction payload spatial shape mismatch")
    if payload["rgb"].shape != shape + (3,):
        raise ValueError("prediction RGB shape mismatch")
    if bool(np.any(payload["nonfinite"])):
        raise ValueError("prediction payload contains non-finite output")
    for key in ("gt", "fp32", "pred", "abs_err", "sparse", "rgb"):
        if not bool(np.isfinite(payload[key]).all()):
            raise ValueError("prediction payload contains non-finite values")
    payload["sample_index"] = sample_index
    payload["valid_gt"] = payload["valid_gt"].astype(bool)
    return payload


def _validate_shared_payload(sample_index, payloads) -> None:
    reference = payloads["FP32"]
    for config in CONFIG_ORDER[1:]:
        current = payloads[config]
        for key in ("gt", "fp32", "valid_gt", "sparse", "rgb"):
            if not np.array_equal(reference[key], current[key]):
                raise ValueError(
                    "sample %d has inconsistent %s payload" %
                    (sample_index, key))


def load_predictions(experiment_dir: Path, expected_samples: int = 64):
    prediction_root = Path(experiment_dir) / "predictions"
    directories = {
        path.name for path in prediction_root.iterdir() if path.is_dir()
    }
    if directories != set(CONFIG_ORDER):
        raise ValueError("prediction configuration directories do not match contract")

    by_config = {}
    for config in CONFIG_ORDER:
        payloads = {}
        for path in sorted((prediction_root / config).glob("sample_*.npz")):
            payload = _load_payload(path, config)
            sample_index = payload["sample_index"]
            if sample_index in payloads:
                raise ValueError("duplicate prediction sample index")
            payloads[sample_index] = payload
        if len(payloads) != int(expected_samples):
            raise ValueError("prediction sample count does not match contract")
        by_config[config] = payloads

    expected_indices = set(by_config["FP32"])
    for config in CONFIG_ORDER[1:]:
        if set(by_config[config]) != expected_indices:
            raise ValueError("prediction sample indices do not match")

    by_sample = {}
    for sample_index in sorted(expected_indices):
        payloads = dict(
            (config, by_config[config][sample_index]) for config in CONFIG_ORDER)
        _validate_shared_payload(sample_index, payloads)
        by_sample[sample_index] = payloads
    return by_sample


def sample_rmse(payload) -> float:
    valid = payload["valid_gt"]
    if not bool(np.any(valid)):
        raise ValueError("prediction payload has no valid ground truth")
    error = payload["pred"][valid].astype(np.float64) - \
        payload["gt"][valid].astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def _first_unused(ranked, used):
    for sample_index in ranked:
        if sample_index not in used:
            return sample_index
    raise ValueError("representative selection requires four distinct samples")


def select_representative_samples(predictions):
    if len(predictions) < 4:
        raise ValueError("representative selection requires at least four samples")
    rtn = dict((index, sample_rmse(predictions[index]["W4A4_RTN"]))
               for index in predictions)
    channel = dict(
        (index, sample_rmse(predictions[index]["W4A4_CHANNEL"]))
        for index in predictions)
    median = float(np.median(tuple(rtn.values())))
    rankings = (
        ("RTN worst", sorted(rtn, key=lambda index: (-rtn[index], index))),
        ("Largest channel gain", sorted(
            rtn, key=lambda index: (-(rtn[index] - channel[index]), index))),
        ("RTN median", sorted(
            rtn, key=lambda index: (abs(rtn[index] - median), index))),
        ("Channel worst", sorted(
            channel, key=lambda index: (-channel[index], index))),
    )
    used = set()
    selected = []
    for label, ranked in rankings:
        sample_index = _first_unused(ranked, used)
        selected.append((label, sample_index))
        used.add(sample_index)
    return tuple(selected)


def global_error_limit(predictions) -> float:
    values = []
    for payloads in predictions.values():
        valid = payloads["FP32"]["valid_gt"]
        for config in CONFIG_ORDER[1:]:
            values.append(payloads[config]["abs_err"][valid])
    errors = np.concatenate(values)
    if errors.size == 0:
        raise ValueError("error range requires valid pixels")
    return max(float(np.percentile(errors, 99.0)), 1e-3)


def set_style(font_size: float) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    })


def _draw(ax, image, cmap=None, vmin=None, vmax=None, title=""):
    if cmap is None:
        ax.imshow(np.clip(image, 0.0, 1.0), interpolation="nearest",
                  aspect="auto", rasterized=True)
    else:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", aspect="auto", rasterized=True)
    ax.set_title(title, pad=2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#5f6368")


def _add_colorbars(fig, depth_rect, error_rect, error_max) -> None:
    depth_axis = fig.add_axes(depth_rect)
    depth = ScalarMappable(
        norm=Normalize(DEPTH_MIN, DEPTH_MAX), cmap="viridis")
    depth.set_array([])
    depth_bar = fig.colorbar(depth, cax=depth_axis)
    depth_bar.set_label("Depth (m)")

    error_axis = fig.add_axes(error_rect)
    error = ScalarMappable(norm=Normalize(0.0, error_max), cmap="magma")
    error.set_array([])
    error_bar = fig.colorbar(error, cax=error_axis)
    error_bar.set_label("Absolute error (m)")


def render_detail(predictions, out_path, dpi=150):
    set_style(9.5)
    selected = select_representative_samples(predictions)
    error_max = global_error_limit(predictions)
    columns = (
        "RGB", "Sparse", "GT", "FP32", "RTN W4A4",
        "RTN |error|", "Per-channel W4A4", "Per-channel |error|",
        "Calibrated W4A4", "Calibrated |error|",
    )
    fig, axes = plt.subplots(
        len(selected), len(columns), figsize=(25.0, 8.4), squeeze=False)
    for row, (selection, sample_index) in enumerate(selected):
        payloads = predictions[sample_index]
        reference = payloads["FP32"]
        valid = reference["valid_gt"]
        panels = (
            (reference["rgb"], None, None, None),
            (np.ma.masked_where(reference["sparse"] <= 1e-4,
                                reference["sparse"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, reference["gt"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, reference["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, payloads["W4A4_RTN"]["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, payloads["W4A4_RTN"]["abs_err"]),
             "magma", 0.0, error_max),
            (np.ma.masked_where(~valid, payloads["W4A4_CHANNEL"]["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(~valid, payloads["W4A4_CHANNEL"]["abs_err"]),
             "magma", 0.0, error_max),
            (np.ma.masked_where(
                ~valid, payloads["W4A4_CALIBRATED_SCALE"]["pred"]),
             "viridis", DEPTH_MIN, DEPTH_MAX),
            (np.ma.masked_where(
                ~valid, payloads["W4A4_CALIBRATED_SCALE"]["abs_err"]),
             "magma", 0.0, error_max),
        )
        for column, panel in enumerate(panels):
            title = columns[column] if row == 0 else ""
            _draw(axes[row, column], panel[0], panel[1], panel[2], panel[3],
                  title)
        axes[row, 0].set_ylabel(
            "%s\n#%05d" % (selection, sample_index), labelpad=5.0)
        axes[row, 4].set_xlabel(
            "RMSE %.3f m" % sample_rmse(payloads["W4A4_RTN"]))
        axes[row, 6].set_xlabel(
            "RMSE %.3f m" % sample_rmse(payloads["W4A4_CHANNEL"]))
        axes[row, 8].set_xlabel(
            "RMSE %.3f m" %
            sample_rmse(payloads["W4A4_CALIBRATED_SCALE"]))
    fig.subplots_adjust(
        left=0.055, right=0.965, bottom=0.055, top=0.955,
        wspace=0.055, hspace=0.20)
    _add_colorbars(
        fig, (0.972, 0.54, 0.007, 0.36), (0.972, 0.09, 0.007, 0.36),
        error_max)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def render_contact_sheet(predictions, out_path, expected_samples=64,
                         sample_columns=4, dpi=150):
    if len(predictions) != int(expected_samples):
        raise ValueError("contact sheet sample count does not match contract")
    set_style(6.5)
    sample_indices = sorted(predictions)
    sample_rows = int(np.ceil(len(sample_indices) / float(sample_columns)))
    panels_per_sample = 5
    fig, axes = plt.subplots(
        sample_rows, sample_columns * panels_per_sample,
        figsize=(48.0, 32.0), squeeze=False)
    for rank, sample_index in enumerate(sample_indices):
        row = rank // sample_columns
        block = rank % sample_columns
        first = block * panels_per_sample
        payloads = predictions[sample_index]
        reference = payloads["FP32"]
        valid = reference["valid_gt"]
        panels = (
            (reference["gt"], "#%05d GT" % sample_index),
            (reference["pred"], "FP32\n%.3f" % sample_rmse(reference)),
            (payloads["W4A4_RTN"]["pred"], "RTN\n%.3f" %
             sample_rmse(payloads["W4A4_RTN"])),
            (payloads["W4A4_CHANNEL"]["pred"], "Channel\n%.3f" %
             sample_rmse(payloads["W4A4_CHANNEL"])),
            (payloads["W4A4_CALIBRATED_SCALE"]["pred"], "Calibrated\n%.3f" %
             sample_rmse(payloads["W4A4_CALIBRATED_SCALE"])),
        )
        for offset, (image, title) in enumerate(panels):
            masked = np.ma.masked_where(~valid, image)
            _draw(axes[row, first + offset], masked, "viridis",
                  DEPTH_MIN, DEPTH_MAX, title)
    fig.subplots_adjust(
        left=0.008, right=0.972, bottom=0.008, top=0.992,
        wspace=0.035, hspace=0.24)
    depth_axis = fig.add_axes((0.978, 0.04, 0.006, 0.92))
    scalar = ScalarMappable(
        norm=Normalize(DEPTH_MIN, DEPTH_MAX), cmap="viridis")
    scalar.set_array([])
    bar = fig.colorbar(scalar, cax=depth_axis)
    bar.set_label("Depth (m)", fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def export_pdf(png_path, pdf_path, dpi=150):
    pdf_path = Path(pdf_path)
    with Image.open(png_path) as source:
        source.convert("RGB").save(
            pdf_path, "PDF", resolution=float(dpi))
    return pdf_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    predictions = load_predictions(
        Path(args.experiment_dir), args.expected_samples)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail = render_detail(
        predictions, output / "cspn_activation_prediction_details.png",
        dpi=args.dpi)
    contact = render_contact_sheet(
        predictions, output / "cspn_activation_prediction_contact_sheet.png",
        expected_samples=args.expected_samples, dpi=args.dpi)
    detail_pdf = export_pdf(
        detail, output / "cspn_activation_prediction_details.pdf", args.dpi)
    contact_pdf = export_pdf(
        contact, output / "cspn_activation_prediction_contact_sheet.pdf",
        args.dpi)
    selected = select_representative_samples(predictions)
    print("samples=%d error_p99=%.6f" % (
        len(predictions), global_error_limit(predictions)), flush=True)
    print("selected=%s" % ";".join(
        "%s:%05d" % row for row in selected), flush=True)
    print("detail=%s detail_pdf=%s" % (detail, detail_pdf), flush=True)
    print("contact=%s contact_pdf=%s" % (contact, contact_pdf), flush=True)


if __name__ == "__main__":
    main()
