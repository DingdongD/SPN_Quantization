#!/usr/bin/env python3
"""Audit and plot CSPN task-sensitive mixed-bit results."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_task_sensitive_bits as runner
from scripts import run_nyu_cspn_stem_precision as stem_runner
from scripts.run_nyu_rtn_quantization import write_json
from spn_quant import cspn_task_sensitive_bits as allocation


VALIDATION_CONFIGS = (
    "FP32",
    "UNIFORM_W4A4",
    "UNIFORM_W6A6",
    "CONTEXT_P3_T3_W8A8",
    "FINAL",
)
METRIC_FIELDS = (
    "RMSE", "MAE", "ABS_REL", "IRMSE", "flat_RMSE", "boundary_RMSE",
)
COLORS = {
    2: "#59A14F",
    4: "#4E79A7",
    6: "#F28E2B",
    8: "#E15759",
}


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finite(value, field: str) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ValueError("%s must be finite" % field)
    return output


def validate_allocation_rows(rows):
    if not rows:
        raise ValueError("final allocation must not be empty")
    normalized = []
    identities = []
    for source in rows:
        tensor = str(source["tensor"])
        if tensor not in ("weight", "activation"):
            raise ValueError("unknown allocation tensor: %s" % tensor)
        bits = int(source["bits"])
        if bits not in allocation.BIT_OPTIONS:
            raise ValueError("allocation bit width is unsupported")
        cost = int(source["cost"])
        if cost <= 0:
            raise ValueError("allocation cost must be positive")
        fraction = _finite(source["cost_fraction"], "cost fraction")
        weighted = int(source["weighted_bits"])
        if weighted != bits * cost:
            raise ValueError("weighted bit cost differs")
        row = {
            "tensor": tensor,
            "block": str(source["block"]),
            "module": str(source["module"]),
            "kind": str(source["kind"]),
            "bits": bits,
            "cost": cost,
            "cost_fraction": fraction,
            "weighted_bits": weighted,
        }
        normalized.append(row)
        identities.append((tensor, row["module"], row["kind"]))
    if len(identities) != len(set(identities)):
        raise ValueError("final allocation contains duplicate tensors")
    for tensor in ("weight", "activation"):
        current = tuple(row for row in normalized if row["tensor"] == tensor)
        if not current:
            raise ValueError("final allocation tensor coverage is incomplete")
        denominator = sum(row["cost"] for row in current)
        for row in current:
            expected = row["cost"] / float(denominator)
            if not math.isclose(
                    row["cost_fraction"], expected,
                    rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("%s cost fraction differs" % tensor)
        if not math.isclose(
                sum(row["cost_fraction"] for row in current), 1.0,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("%s cost fraction sum differs" % tensor)
        average = sum(row["weighted_bits"] for row in current) / \
            float(denominator)
        if average > 4.0:
            raise ValueError("%s average bit budget exceeds four" % tensor)
    return tuple(normalized)


def summarize_bit_fractions(rows):
    output = {}
    for tensor in ("weight", "activation"):
        current = tuple(row for row in rows if row["tensor"] == tensor)
        denominator = sum(row["cost"] for row in current)
        output[tensor] = dict(
            (bits, sum(row["cost"] for row in current
                       if row["bits"] == bits) / float(denominator))
            for bits in allocation.BIT_OPTIONS)
    return output


def _assignment_from_payload(payload):
    return allocation.BitAssignment(
        weight_bits=tuple(
            (str(row["module"]), int(row["bits"]))
            for row in payload["weight_bits"]),
        activation_bits=tuple(
            ((str(row["module"]), str(row["kind"])), int(row["bits"]))
            for row in payload["activation_bits"]),
    )


def _cost_basis(root: Path):
    weight_rows = read_csv(root / "weight_cost_basis.csv")
    activation_rows = read_csv(root / "activation_cost_basis.csv")
    return allocation.CostBasis(
        weight_macs=tuple(
            (str(row["module"]), int(row["macs"]))
            for row in weight_rows),
        activation_elements=tuple(
            ((str(row["module"]), str(row["kind"])), int(row["elements"]))
            for row in activation_rows),
    )


def _audit_phase_rows(rows, manifest):
    counts = Counter(str(row["stage"]) for row in rows)
    declared = manifest["phase_counts"]
    for stage in ("single_block", "joint", "demotion", "refinement"):
        if counts[stage] != int(declared[stage]):
            raise ValueError("calibration phase count differs: %s" % stage)
    if counts["local"] != int(declared["local_candidates"]):
        raise ValueError("local candidate count differs")
    local_rounds = set(
        int(row["local_round"]) for row in rows
        if str(row["stage"]) == "local")
    if len(local_rounds) != int(declared["local_rounds"]):
        raise ValueError("local round count differs")
    for row in rows:
        for field in ("calibration_RMSE", "boundary_RMSE",
                      "propagation_MSE", "RMSE"):
            _finite(row[field], field)
        json.loads(row["assignment"])


def _audit_validation_rows(rows, final_assignment):
    names = tuple(str(row["config"]) for row in rows)
    if names != VALIDATION_CONFIGS:
        raise ValueError("validation configuration coverage differs")
    for row in rows:
        for field in METRIC_FIELDS:
            _finite(row[field], field)
    final_row = rows[-1]
    persisted = _assignment_from_payload(json.loads(final_row["assignment"]))
    if persisted != final_assignment:
        raise ValueError("final validation bits differ from assignment")
    for field in (
            "coefficient_sum_max_error", "contraction_violation_ratio",
            "anchor_max_error"):
        if float(final_row[field]) != 0.0:
            raise ValueError("final propagation invariant differs: %s" % field)


def _prediction_identities(root: Path):
    identities = {}
    for config in VALIDATION_CONFIGS:
        paths = sorted((root / "predictions" / config).glob("sample_*.npz"))
        if len(paths) != 64:
            raise ValueError("prediction count differs: %s" % config)
        current = []
        for path in paths:
            with np.load(path) as payload:
                index = int(payload["sample_index"])
                if payload["gt"].shape != payload["pred"].shape:
                    raise ValueError("prediction payload shape differs")
                if not np.isfinite(payload["gt"]).all():
                    raise ValueError("prediction GT contains non-finite values")
            current.append(index)
        if len(current) != len(set(current)):
            raise ValueError("prediction identities contain duplicates")
        identities[config] = tuple(current)
    reference = identities[VALIDATION_CONFIGS[0]]
    if any(identities[config] != reference
           for config in VALIDATION_CONFIGS[1:]):
        raise ValueError("prediction identities differ across configurations")
    return reference


def audit_result_root(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    actual_hashes = stem_runner._artifact_hashes(root)
    if actual_hashes != manifest["artifacts"]:
        raise ValueError("artifact hashes differ")
    protocol = runner.SearchProtocol(
        int(manifest["protocol"]["beam_width"]),
        int(manifest["protocol"]["joint_measured_limit"]),
        int(manifest["protocol"]["local_round_limit"]),
        int(manifest["protocol"]["refinement_block_limit"]),
        int(manifest["protocol"]["refinement_width"]),
        int(manifest["protocol"]["refinement_measured_limit"]),
    )
    runner.validate_production_protocol(protocol)
    rows = validate_allocation_rows(read_csv(root / "final_allocation.csv"))
    registry = runner.expected_registry()
    weight_rows = tuple(row for row in rows if row["tensor"] == "weight")
    activation_rows = tuple(
        row for row in rows if row["tensor"] == "activation")
    if set(row["module"] for row in weight_rows) != set(
            module for modules in registry.weights_by_block.values()
            for module in modules):
        raise ValueError("final weight coverage differs")
    if set((row["module"], row["kind"]) for row in activation_rows) != set(
            owner for owners in registry.activations_by_block.values()
            for owner in owners):
        raise ValueError("final activation coverage differs")
    final_assignment = _assignment_from_payload(json.loads(
        (root / "final_assignment.json").read_text()))
    runner.validate_assignment_contract(final_assignment)
    allocation_from_rows = allocation.BitAssignment(
        tuple((row["module"], row["bits"]) for row in weight_rows),
        tuple(((row["module"], row["kind"]), row["bits"])
              for row in activation_rows),
    )
    if allocation_from_rows != final_assignment:
        raise ValueError("final allocation bits differ from assignment")
    basis = _cost_basis(root)
    budget = allocation.audit_budget(final_assignment, basis)
    declared_budget = manifest["budget"]
    comparisons = (
        (budget.weight_numerator, int(declared_budget["weight_numerator"])),
        (budget.weight_denominator, int(declared_budget["weight_denominator"])),
        (budget.activation_numerator,
         int(declared_budget["activation_numerator"])),
        (budget.activation_denominator,
         int(declared_budget["activation_denominator"])),
    )
    if any(actual != expected for actual, expected in comparisons):
        raise ValueError("integer bit budget differs")
    if not budget.feasible:
        raise ValueError("final assignment exceeds bit budget")
    calibration_rows = read_csv(root / "calibration_metrics.csv")
    _audit_phase_rows(calibration_rows, manifest)
    validation_rows = read_csv(root / "validation_metrics.csv")
    if len(validation_rows) != int(manifest["phase_counts"]["validation"]):
        raise ValueError("validation phase count differs")
    _audit_validation_rows(validation_rows, final_assignment)
    identities = _prediction_identities(root)
    return {
        "average_weight_bits": budget.average_weight_bits,
        "average_activation_bits": budget.average_activation_bits,
        "prediction_samples": len(identities),
        "calibration_candidates": len(calibration_rows),
    }


def register_arial_font():
    paths = sorted(
        Path(path) for path in font_manager.findSystemFonts()
        if Path(path).name.lower() == "arial.ttf")
    if len(paths) != 1:
        raise RuntimeError(
            "expected exactly one Arial.ttf, found %d" % len(paths))
    font_manager.fontManager.addfont(str(paths[0]))
    name = font_manager.FontProperties(fname=str(paths[0])).get_name()
    if name != "Arial":
        raise RuntimeError("Arial.ttf reports unexpected family: %s" % name)
    return name


def _set_style():
    plt.rcParams.update({
        "font.family": register_arial_font(),
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })


def _save(figure, root: Path, name: str):
    figure.tight_layout()
    figure.savefig(root / (name + ".png"), dpi=220, bbox_inches="tight")
    figure.savefig(root / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_block_bits(rows, root: Path):
    blocks = allocation.BLOCK_ORDER
    values = {}
    for tensor in ("weight", "activation"):
        values[tensor] = []
        for block in blocks:
            bits = set(row["bits"] for row in rows
                       if row["tensor"] == tensor and row["block"] == block)
            if len(bits) != 1:
                raise ValueError("block bit assignment is not uniform")
            values[tensor].append(next(iter(bits)))
    positions = np.arange(len(blocks))
    figure, axis = plt.subplots(figsize=(16, 6))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    axis.bar(positions - 0.2, values["weight"], width=0.4,
             color="#4E79A7", label="Weight", zorder=3)
    axis.bar(positions + 0.2, values["activation"], width=0.4,
             color="#F28E2B", label="Activation", zorder=3)
    axis.set_ylabel("Bit Width")
    axis.set_xticks(positions, [block.replace("_", " ").title()
                               for block in blocks], rotation=0)
    axis.set_yticks(allocation.BIT_OPTIONS)
    axis.legend(frameon=False)
    _save(figure, root, "final_block_bits")


def _candidate_budget(row, basis):
    assignment = _assignment_from_payload(json.loads(row["assignment"]))
    return allocation.audit_budget(assignment, basis)


def plot_calibration_budget(rows, basis, root: Path):
    stages = tuple(dict.fromkeys(str(row["stage"]) for row in rows))
    palette = ("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, tensor, label in (
            (axes[0], "weight", "Average Weight Bits"),
            (axes[1], "activation", "Average Activation Bits")):
        axis.set_axisbelow(True)
        axis.grid(color="#D9D9D9", linewidth=0.8, zorder=0)
        for index, stage in enumerate(stages):
            current = tuple(row for row in rows if row["stage"] == stage)
            budgets = tuple(_candidate_budget(row, basis) for row in current)
            averages = [
                budget.average_weight_bits if tensor == "weight"
                else budget.average_activation_bits
                for budget in budgets]
            axis.scatter(
                averages,
                [float(row["calibration_RMSE"]) for row in current],
                s=28, alpha=0.75, color=palette[index], label=stage,
                zorder=3)
        axis.set_xlabel(label)
        axis.set_ylabel("Calibration RMSE (m)")
    axes[1].legend(frameon=False)
    _save(figure, root, "calibration_rmse_bit_budgets")


def plot_bit_fractions(rows, root: Path):
    summary = summarize_bit_fractions(rows)
    positions = np.arange(2)
    bottom = np.zeros(2, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    for bits in allocation.BIT_OPTIONS:
        values = np.asarray((summary["weight"][bits],
                             summary["activation"][bits]))
        axis.bar(positions, values * 100.0, bottom=bottom * 100.0,
                 color=COLORS[bits], label="%d-bit" % bits, zorder=3)
        bottom += values
    axis.set_ylabel("Logical Cost Fraction (%)")
    axis.set_xticks(positions, ("Weight MAC", "Activation Elements"),
                    rotation=0)
    axis.legend(frameon=False, ncol=2)
    _save(figure, root, "final_bit_cost_fractions")


def plot_predictions(root: Path):
    fp_paths = sorted((root / "predictions" / "FP32").glob("sample_*.npz"))
    figure, axes = plt.subplots(64, 6, figsize=(18, 128), squeeze=False)
    labels = ("GT", "FP32", "W4A4", "W6A6", "P3/T3", "Final")
    for row_index, fp_path in enumerate(fp_paths):
        payloads = []
        for config in VALIDATION_CONFIGS:
            payloads.append(np.load(
                root / "predictions" / config / fp_path.name))
        gt = payloads[0]["gt"]
        valid = payloads[0]["valid_gt"].astype(bool)
        minimum = float(gt[valid].min())
        maximum = float(gt[valid].max())
        values = (gt,) + tuple(payload["pred"] for payload in payloads)
        for column, value in enumerate(values):
            axis = axes[row_index, column]
            axis.imshow(value, cmap="viridis", vmin=minimum, vmax=maximum,
                        zorder=2)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_xlabel(labels[column])
                axis.xaxis.set_label_position("top")
        for payload in payloads:
            payload.close()
    _save(figure, root, "prediction_comparison_64")


def generate_plots(root):
    root = Path(root)
    rows = validate_allocation_rows(read_csv(root / "final_allocation.csv"))
    calibration_rows = read_csv(root / "calibration_metrics.csv")
    basis = _cost_basis(root)
    _set_style()
    plot_block_bits(rows, root)
    plot_calibration_budget(calibration_rows, basis, root)
    plot_bit_fractions(rows, root)
    plot_predictions(root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.input_dir)
    audit_result_root(root)
    generate_plots(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = stem_runner._artifact_hashes(root)
    write_json(manifest_path, manifest)
    report = audit_result_root(root)
    print("CSPN task-sensitive mixed-bit audit complete: %s" % report,
          flush=True)


if __name__ == "__main__":
    main()
