#!/usr/bin/env python3
"""Plot the measured CSPN decoder precision Pareto set."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_stem_precision as stem_runner
from scripts.run_nyu_rtn_quantization import write_json
from spn_quant.cspn_sensitivity import pareto_rows


REQUIRED_COLUMNS = (
    "config",
    "stage",
    "RMSE",
    "normalized_added_bit_cost",
)

STAGE_MARKERS = {
    "baseline": ("o", "#4c78a8"),
    "block": ("s", "#f58518"),
    "site": ("^", "#54a24b"),
    "cumulative": ("D", "#e45756"),
}


def load_pareto_rows(path: Path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_pareto_rows(rows: Sequence[Mapping[str, object]]):
    if not rows:
        raise ValueError("Pareto plot requires rows")
    validated = []
    names = set()
    for source in rows:
        for column in REQUIRED_COLUMNS:
            source[column]
        name = str(source["config"])
        stage = str(source["stage"])
        rmse = float(source["RMSE"])
        cost = float(source["normalized_added_bit_cost"])
        if name in names:
            raise ValueError("Pareto rows contain duplicate configurations")
        if stage not in STAGE_MARKERS:
            raise ValueError("Pareto row has unknown stage: %s" % stage)
        if not math.isfinite(rmse) or not math.isfinite(cost) or cost < 0.0:
            raise ValueError("Pareto values must be finite and nonnegative")
        names.add(name)
        row = dict(source)
        row["RMSE"] = rmse
        row["normalized_added_bit_cost"] = cost
        validated.append(row)
    expected = pareto_rows(validated)
    if {row["config"] for row in expected} != names:
        raise ValueError("Pareto input is not a non-dominated set")
    return sorted(
        validated,
        key=lambda row: (
            row["normalized_added_bit_cost"],
            row["RMSE"], row["config"]))


def plot_pareto(rows, output_png: Path, output_pdf: Path) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    stages = tuple(STAGE_MARKERS)
    for stage in stages:
        selected = [row for row in rows if row["stage"] == stage]
        if not selected:
            continue
        marker, color = STAGE_MARKERS[stage]
        axis.scatter(
            [row["normalized_added_bit_cost"] * 100.0 for row in selected],
            [row["RMSE"] for row in selected],
            marker=marker, color=color, edgecolor="white",
            linewidth=0.8, s=72, label=stage.replace("_", " ").title(),
            zorder=3)
    axis.plot(
        [row["normalized_added_bit_cost"] * 100.0 for row in rows],
        [row["RMSE"] for row in rows],
        color="#7f7f7f", linewidth=1.1, zorder=2)
    for index, row in enumerate(rows):
        offset = 8 if index % 2 == 0 else -13
        vertical = "bottom" if offset > 0 else "top"
        axis.annotate(
            row["config"],
            (row["normalized_added_bit_cost"] * 100.0, row["RMSE"]),
            xytext=(5, offset), textcoords="offset points",
            fontsize=9, ha="left", va=vertical, zorder=4)
    axis.set_xlabel("Normalized Added Bit-Element Cost (%)")
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
    rows = validate_pareto_rows(load_pareto_rows(
        root / "pareto_metrics.csv"))
    output_png = root / "decoder_sensitivity_pareto.png"
    output_pdf = root / "decoder_sensitivity_pareto.pdf"
    plot_pareto(rows, output_png, output_pdf)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = stem_runner._artifact_hashes(root)
    write_json(manifest_path, manifest)
    print("CSPN decoder sensitivity Pareto plot complete: %s" % root,
          flush=True)


if __name__ == "__main__":
    main()
