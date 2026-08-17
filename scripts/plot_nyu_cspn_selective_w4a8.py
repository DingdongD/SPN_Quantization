#!/usr/bin/env python3
"""Plot persisted CSPN selective-W4A8 search results."""

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
from spn_quant.cspn_selective_w4a8 import pareto_rows


PAIR_LABELS = ("00", "10", "01", "11")


def load_csv(path: Path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_stage1_rows(rows: Sequence[Mapping[str, object]]):
    output = []
    masks = set()
    for source in rows:
        if str(source["role"]) != "primary":
            continue
        row = dict(source)
        mask = int(row["mask"])
        rmse = float(row["RMSE"])
        cost = float(row["normalized_added_bit_cost"])
        fraction = float(row["a8_activation_element_fraction"])
        if mask in masks:
            raise ValueError("Stage-1 rows contain duplicate masks")
        if not all(math.isfinite(value) for value in (rmse, cost, fraction)):
            raise ValueError("Stage-1 plot values must be finite")
        row["mask"] = mask
        row["RMSE"] = rmse
        row["normalized_added_bit_cost"] = cost
        row["a8_activation_element_fraction"] = fraction
        masks.add(mask)
        output.append(row)
    if masks != set(range(16)):
        raise ValueError("Stage-1 mask coverage mismatch")
    return sorted(output, key=lambda row: row["mask"])


def validate_boundary_rows(rows: Sequence[Mapping[str, object]]):
    output = []
    owners = set()
    ranks = []
    for source in rows:
        row = dict(source)
        rank = int(row["rank"])
        owner = str(row["module"]), str(row["kind"])
        score = float(row["score"])
        saved_cost = float(row["saved_cost"])
        if owner in owners:
            raise ValueError("boundary ranking contains duplicate owners")
        if not math.isfinite(score) or not math.isfinite(saved_cost) or \
                score < 0.0 or saved_cost <= 0.0:
            raise ValueError("boundary ranking values are invalid")
        row["rank"] = rank
        row["score"] = score
        row["saved_cost"] = saved_cost
        ranks.append(rank)
        owners.add(owner)
        output.append(row)
    output.sort(key=lambda row: row["rank"])
    if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("boundary ranking is not contiguous")
    if not output:
        raise ValueError("boundary ranking is empty")
    return output


def validate_path_rows(rows: Sequence[Mapping[str, object]]):
    output = []
    steps = set()
    for source in rows:
        row = dict(source)
        step = int(row["step"])
        rmse = float(row["RMSE"])
        cost = float(row["normalized_added_bit_cost"])
        fraction = float(row["a8_activation_element_fraction"])
        if step in steps:
            raise ValueError("path rows contain duplicate steps")
        if not all(math.isfinite(value) for value in (rmse, cost, fraction)):
            raise ValueError("path values must be finite")
        row["step"] = step
        row["RMSE"] = rmse
        row["normalized_added_bit_cost"] = cost
        row["a8_activation_element_fraction"] = fraction
        steps.add(step)
        output.append(row)
    output.sort(key=lambda row: row["step"])
    if not output or [row["step"] for row in output] != \
            list(range(len(output))):
        raise ValueError("path step coverage mismatch")
    for previous, current in zip(output, output[1:]):
        if current["normalized_added_bit_cost"] >= \
                previous["normalized_added_bit_cost"] or \
                current["a8_activation_element_fraction"] >= \
                previous["a8_activation_element_fraction"]:
            raise ValueError("path activation cost is not strictly decreasing")
    return output


def validate_pareto_rows(rows, cost_field: str):
    if not rows:
        raise ValueError("Pareto rows are empty")
    output = []
    names = set()
    for source in rows:
        row = dict(source)
        name = str(row["config"])
        row["RMSE"] = float(row["RMSE"])
        row[cost_field] = float(row[cost_field])
        if name in names:
            raise ValueError("Pareto rows contain duplicate configurations")
        names.add(name)
        output.append(row)
    expected = pareto_rows(output, cost_field)
    if {str(row["config"]) for row in expected} != names:
        raise ValueError("Pareto input contains dominated configurations")
    return sorted(output, key=lambda row: (
        row[cost_field], row["RMSE"], str(row["config"])))


def _configure_font() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })


def plot_stage1(rows, png: Path, pdf: Path) -> None:
    _configure_font()
    values = np.empty((4, 4), dtype=np.float64)
    for row in rows:
        mask = int(row["mask"])
        row_state = (mask & 1) + ((mask >> 1) & 1) * 2
        column_state = ((mask >> 2) & 1) + ((mask >> 3) & 1) * 2
        values[row_state, column_state] = float(row["RMSE"])
    figure, axis = plt.subplots(figsize=(7.4, 5.8))
    image = axis.imshow(values, cmap="viridis", aspect="auto", zorder=1)
    axis.set_xticks(range(4), labels=PAIR_LABELS)
    axis.set_yticks(range(4), labels=PAIR_LABELS)
    axis.tick_params(axis="x", labelrotation=0)
    axis.set_xlabel("Layer2 / Decoder4 A8 State")
    axis.set_ylabel("Stem / Layer1 A8 State")
    colorbar = figure.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label("RMSE (m)")
    midpoint = float(values.min() + values.max()) / 2.0
    for row_index in range(4):
        for column_index in range(4):
            value = float(values[row_index, column_index])
            axis.text(
                column_index, row_index, "%.4f" % value,
                ha="center", va="center",
                color="white" if value < midpoint else "black",
                zorder=2)
    figure.tight_layout()
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def plot_boundaries(rows, png: Path, pdf: Path) -> None:
    _configure_font()
    labels = [
        "%s::%s" % (row["module"], row["kind"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    height = max(5.5, 0.31 * len(rows))
    figure, axis = plt.subplots(figsize=(9.4, height))
    positions = np.arange(len(rows))
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.8, zorder=0)
    axis.barh(positions, scores, color="#4c78a8", zorder=2)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Propagation MSE Increase per Saved Cost")
    axis.set_ylabel("Demoted Activation Owner")
    figure.tight_layout()
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def plot_pareto(rows, cost_field: str, x_label: str,
                png: Path, pdf: Path) -> None:
    _configure_font()
    figure, axis = plt.subplots(figsize=(8.4, 5.8))
    x = [float(row[cost_field]) * 100.0 for row in rows]
    y = [float(row["RMSE"]) for row in rows]
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    axis.plot(x, y, color="#7f7f7f", linewidth=1.1, zorder=2)
    axis.scatter(x, y, color="#e45756", edgecolor="white",
                 linewidth=0.8, s=70, zorder=3)
    for index, row in enumerate(rows):
        offset = 7 if index % 2 == 0 else -10
        axis.annotate(
            str(row["config"]), (x[index], y[index]),
            xytext=(5, offset), textcoords="offset points",
            fontsize=8, ha="left",
            va="bottom" if offset > 0 else "top", zorder=4)
    axis.set_xlabel(x_label)
    axis.set_ylabel("RMSE (m)")
    figure.tight_layout()
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.input_dir)
    stage1 = validate_stage1_rows(load_csv(
        root / "stage1_aggregate_metrics.csv"))
    boundaries = validate_boundary_rows(load_csv(
        root / "stage2_boundary_ranking.csv"))
    validate_path_rows(load_csv(
        root / "stage2_path_aggregate_metrics.csv"))
    cost_pareto = validate_pareto_rows(load_csv(
        root / "pareto_normalized_cost.csv"),
        "normalized_added_bit_cost")
    a8_pareto = validate_pareto_rows(load_csv(
        root / "pareto_a8_fraction.csv"),
        "a8_activation_element_fraction")
    plot_stage1(
        stage1, root / "stage1_unit_mask_rmse.png",
        root / "stage1_unit_mask_rmse.pdf")
    plot_boundaries(
        boundaries, root / "stage2_boundary_demotion_sensitivity.png",
        root / "stage2_boundary_demotion_sensitivity.pdf")
    plot_pareto(
        cost_pareto, "normalized_added_bit_cost",
        "Normalized Added Bit-Element Cost (%)",
        root / "selective_w4a8_normalized_cost_pareto.png",
        root / "selective_w4a8_normalized_cost_pareto.pdf")
    plot_pareto(
        a8_pareto, "a8_activation_element_fraction",
        "A8 Activation Element Fraction (%)",
        root / "selective_w4a8_a8_fraction_pareto.png",
        root / "selective_w4a8_a8_fraction_pareto.pdf")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = stem_runner._artifact_hashes(root)
    write_json(manifest_path, manifest)
    print("CSPN selective W4A8 plots complete: %s" % root, flush=True)


if __name__ == "__main__":
    main()
