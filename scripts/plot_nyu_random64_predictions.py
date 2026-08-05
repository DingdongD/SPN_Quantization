#!/usr/bin/env python3
"""Render all random-64 NYU ground truths and FP32 predictions in one figure."""

from __future__ import print_function

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_nyu_prediction_comparison import collect


MODEL_ORDER = ["cspn", "dyspn", "nlspn", "completionformer"]
MODEL_NAMES = {
    "cspn": "CSPN",
    "dyspn": "DySPN",
    "nlspn": "NLSPN",
    "completionformer": "CompletionFormer",
}
PANEL_ORDER = ["gt"] + MODEL_ORDER
PANEL_BOUNDS = (0.008, 0.966, 0.008, 0.993)
COLORBAR_RECT = (0.977, 0.04, 0.006, 0.92)


def sample_block_position(rank, columns=4):
    return rank // columns, rank % columns


def validate_complete_samples(by_sample, expected_samples=64):
    sample_ids = sorted(by_sample)
    if len(sample_ids) != expected_samples:
        raise ValueError("expected %d samples, found %d" % (
            expected_samples, len(sample_ids)))
    expected_models = set(MODEL_ORDER)
    for sample_index in sample_ids:
        actual = set(by_sample[sample_index])
        if actual != expected_models:
            missing = sorted(expected_models - actual)
            extra = sorted(actual - expected_models)
            raise ValueError("sample %d model mismatch: missing=%s extra=%s" % (
                sample_index, missing, extra))
    return sample_ids


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
    })


def _check_shared_ground_truth(sample_index, predictions):
    reference = predictions[MODEL_ORDER[0]]
    for model in MODEL_ORDER[1:]:
        current = predictions[model]
        if not np.array_equal(reference["valid"], current["valid"]):
            raise ValueError("sample %d has inconsistent valid masks" % sample_index)
        if not np.allclose(reference["gt"], current["gt"], rtol=0.0, atol=0.0):
            raise ValueError("sample %d has inconsistent ground truth" % sample_index)
    return reference


def render_contact_sheet(by_sample, out_path, expected_samples=64,
                         sample_columns=4, depth_min=0.0, depth_max=10.0,
                         dpi=150):
    sample_ids = validate_complete_samples(by_sample, expected_samples)
    sample_rows = int(np.ceil(len(sample_ids) / float(sample_columns)))
    panel_columns = sample_columns * len(PANEL_ORDER)
    fig, axes = plt.subplots(
        sample_rows,
        panel_columns,
        figsize=(48.0, 32.0),
        squeeze=False,
    )

    for rank, sample_index in enumerate(sample_ids):
        row, block = sample_block_position(rank, sample_columns)
        first_col = block * len(PANEL_ORDER)
        predictions = by_sample[sample_index]
        reference = _check_shared_ground_truth(sample_index, predictions)
        valid = reference["valid"]
        gt = np.ma.masked_where(~valid, reference["gt"])

        panels = [(gt, "#%05d  GT" % sample_index)]
        for model in MODEL_ORDER:
            item = predictions[model]
            panels.append((
                np.ma.masked_where(~valid, item["pred"]),
                "%s\nRMSE %.3f" % (MODEL_NAMES[model], item["metrics"]["RMSE"]),
            ))

        for offset, (image, title) in enumerate(panels):
            ax = axes[row, first_col + offset]
            ax.imshow(image, cmap="viridis", vmin=depth_min, vmax=depth_max,
                      interpolation="nearest", aspect="auto", rasterized=True)
            ax.set_title(title, pad=2.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#5f6368")
        axes[row, first_col].spines["left"].set_linewidth(1.2)
        axes[row, first_col].spines["left"].set_color("#202124")

    used = len(sample_ids) * len(PANEL_ORDER)
    for flat_index, ax in enumerate(axes.flat):
        if flat_index >= used:
            ax.axis("off")

    scalar = ScalarMappable(norm=Normalize(depth_min, depth_max), cmap="viridis")
    scalar.set_array([])
    colorbar_axis = fig.add_axes(COLORBAR_RECT)
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Depth (m)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=PANEL_BOUNDS[0], right=PANEL_BOUNDS[1],
                        bottom=PANEL_BOUNDS[2], top=PANEL_BOUNDS[3],
                        wspace=0.035, hspace=0.24)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def export_pdf(png_path, pdf_path, dpi=150):
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    with Image.open(str(png_path)) as source:
        rgb = source.convert("RGB")
        rgb.save(str(pdf_path), "PDF", resolution=float(dpi))
    return pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        default="profile_logs/nyu_prediction_random64_fp32/predictions",
    )
    parser.add_argument(
        "--out-dir",
        default="profile_logs/nyu_prediction_random64_fp32",
    )
    parser.add_argument("--expected-samples", type=int, default=64)
    parser.add_argument("--sample-columns", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    set_style()
    by_sample = collect(args.pred_dir)
    out_dir = Path(args.out_dir)
    png = render_contact_sheet(
        by_sample,
        out_dir / "random64_gt_prediction_contact_sheet.png",
        expected_samples=args.expected_samples,
        sample_columns=args.sample_columns,
        dpi=args.dpi,
    )
    pdf = export_pdf(
        png,
        out_dir / "random64_gt_prediction_contact_sheet.pdf",
        dpi=args.dpi,
    )
    print("samples=%d png=%s pdf=%s" % (len(by_sample), png, pdf), flush=True)


if __name__ == "__main__":
    main()
