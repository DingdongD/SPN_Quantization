#!/usr/bin/env python3
"""Plot measured CSPN encoder-prefix and sensitive-tail W8A8 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_stem_precision as stem_runner
from scripts.run_nyu_rtn_quantization import write_json
from spn_quant.cspn_encoder_prefix import pareto_rows


EXPECTED_CELLS = set(
    (prefix_index, tail_index)
    for prefix_index in range(6)
    for tail_index in range(4))
PREFIX_LABELS = (
    "P0 None",
    "P1 Stem",
    "P2 Stem+L1",
    "P3 Stem+L1+L2",
    "P4 Stem+L1+L2+L3",
    "P5 Stem+L1+L2+L3+L4",
)
TAIL_LABELS = (
    "T0\nNone",
    "T1\nDecoder4",
    "T2\nDepth",
    "T3\nBoth",
)
TAIL_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
)


def load_csv(path: Path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _validate_matrix_rows(
        rows: Sequence[Mapping[str, object]],
        value_fields: Sequence[str]):
    if not rows:
        raise ValueError("matrix plot requires rows")
    cells = set()
    names = set()
    output = []
    for source in rows:
        source["config"]
        source["prefix_index"]
        source["tail_index"]
        for field in value_fields:
            source[field]
        name = str(source["config"])
        prefix_index = int(source["prefix_index"])
        tail_index = int(source["tail_index"])
        cell = prefix_index, tail_index
        if name in names or cell in cells:
            raise ValueError("matrix rows contain duplicate values")
        row = dict(source)
        row["prefix_index"] = prefix_index
        row["tail_index"] = tail_index
        for field in value_fields:
            value = float(source[field])
            if not math.isfinite(value):
                raise ValueError("matrix values must be finite")
            row[field] = value
        names.add(name)
        cells.add(cell)
        output.append(row)
    if cells != EXPECTED_CELLS:
        raise ValueError("matrix coverage mismatch")
    return sorted(
        output,
        key=lambda row: (row["prefix_index"], row["tail_index"]))


def validate_aggregate_rows(rows: Sequence[Mapping[str, object]]):
    return _validate_matrix_rows(rows, (
        "RMSE",
        "normalized_added_bit_cost",
        "w8_weight_mac_fraction",
    ))


def validate_interaction_rows(rows: Sequence[Mapping[str, object]]):
    return _validate_matrix_rows(rows, ("RMSE", "interaction_rmse"))


def validate_pareto_rows(
        rows: Sequence[Mapping[str, object]], cost_field: str):
    if not rows:
        raise ValueError("Pareto plot requires rows")
    output = []
    names = set()
    for source in rows:
        source["config"]
        source["RMSE"]
        source[cost_field]
        name = str(source["config"])
        rmse = float(source["RMSE"])
        cost = float(source[cost_field])
        if name in names:
            raise ValueError("Pareto rows contain duplicate configurations")
        if not math.isfinite(rmse) or not math.isfinite(cost) or cost < 0.0:
            raise ValueError("Pareto values must be finite and nonnegative")
        row = dict(source)
        row["RMSE"] = rmse
        row[cost_field] = cost
        row["prefix_index"] = int(source["prefix_index"])
        row["tail_index"] = int(source["tail_index"])
        names.add(name)
        output.append(row)
    expected = pareto_rows(output, cost_field)
    if {str(row["config"]) for row in expected} != names:
        raise ValueError("Pareto input is not a non-dominated set")
    return sorted(
        output,
        key=lambda row: (
            row[cost_field], row["RMSE"], str(row["config"])))


def _configure_font() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })


def _matrix(rows, field: str):
    values = np.empty((6, 4), dtype=np.float64)
    for row in rows:
        values[row["prefix_index"], row["tail_index"]] = row[field]
    return values


def plot_heatmap(
        rows, field: str, color_label: str, output_png: Path,
        output_pdf: Path, cmap: str, digits: int) -> None:
    _configure_font()
    values = _matrix(rows, field)
    figure, axis = plt.subplots(figsize=(9.6, 6.5))
    image = axis.imshow(values, cmap=cmap, aspect="auto", zorder=1)
    axis.set_xticks(np.arange(4), labels=TAIL_LABELS)
    axis.set_yticks(np.arange(6), labels=PREFIX_LABELS)
    axis.tick_params(axis="x", labelrotation=0)
    axis.set_xlabel("Sensitive Tail W8A8 State")
    axis.set_ylabel("Encoder W8A8 Prefix")
    colorbar = figure.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label(color_label)
    midpoint = float(values.min() + values.max()) / 2.0
    for prefix_index in range(6):
        for tail_index in range(4):
            value = float(values[prefix_index, tail_index])
            color = "white" if value < midpoint else "black"
            axis.text(
                tail_index, prefix_index, ("%." + str(digits) + "f") % value,
                ha="center", va="center", color=color, fontsize=11,
                zorder=2)
    figure.tight_layout()
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def plot_pareto(
        rows, cost_field: str, x_label: str, output_png: Path,
        output_pdf: Path) -> None:
    _configure_font()
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    axis.plot(
        [row[cost_field] * 100.0 for row in rows],
        [row["RMSE"] for row in rows],
        color="#7f7f7f", linewidth=1.1, zorder=2)
    for tail_index in range(4):
        selected = [
            row for row in rows if row["tail_index"] == tail_index]
        if not selected:
            continue
        axis.scatter(
            [row[cost_field] * 100.0 for row in selected],
            [row["RMSE"] for row in selected],
            color=TAIL_COLORS[tail_index], edgecolor="white",
            linewidth=0.8, s=76, label="T%d" % tail_index, zorder=3)
    for index, row in enumerate(rows):
        offset = 7 if index % 2 == 0 else -11
        axis.annotate(
            "P%d/T%d" % (row["prefix_index"], row["tail_index"]),
            (row[cost_field] * 100.0, row["RMSE"]),
            xytext=(5, offset), textcoords="offset points",
            fontsize=9, ha="left",
            va="bottom" if offset > 0 else "top", zorder=4)
    axis.set_xlabel(x_label)
    axis.set_ylabel("RMSE (m)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.input_dir)
    aggregate = validate_aggregate_rows(load_csv(
        root / "aggregate_metrics.csv"))
    interactions = validate_interaction_rows(load_csv(
        root / "interaction_metrics.csv"))
    normalized_pareto = validate_pareto_rows(load_csv(
        root / "pareto_normalized_cost.csv"),
        "normalized_added_bit_cost")
    mac_pareto = validate_pareto_rows(load_csv(
        root / "pareto_w8_mac.csv"), "w8_weight_mac_fraction")
    plot_heatmap(
        aggregate, "RMSE", "RMSE (m)",
        root / "encoder_prefix_tail_rmse_heatmap.png",
        root / "encoder_prefix_tail_rmse_heatmap.pdf",
        "viridis", 4)
    plot_heatmap(
        interactions, "interaction_rmse", "Interaction RMSE (m)",
        root / "encoder_prefix_tail_interaction_heatmap.png",
        root / "encoder_prefix_tail_interaction_heatmap.pdf",
        "coolwarm", 4)
    plot_pareto(
        normalized_pareto, "normalized_added_bit_cost",
        "Normalized Added Bit-Element Cost (%)",
        root / "encoder_prefix_normalized_cost_pareto.png",
        root / "encoder_prefix_normalized_cost_pareto.pdf")
    plot_pareto(
        mac_pareto, "w8_weight_mac_fraction",
        "W8 Conv MAC Fraction (%)",
        root / "encoder_prefix_w8_mac_pareto.png",
        root / "encoder_prefix_w8_mac_pareto.pdf")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = stem_runner._artifact_hashes(root)
    write_json(manifest_path, manifest)
    print("CSPN encoder-prefix W8A8 plots complete: %s" % root,
          flush=True)


if __name__ == "__main__":
    main()
