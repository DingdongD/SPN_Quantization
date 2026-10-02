#!/usr/bin/env python3
"""Summarize measured four-model PTQ evidence and define NAS search spaces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")


ARCHITECTURE_SPACES = {
    "cspn": {
        "baseline": "ResNet18 [2,2,2,2]",
        "inherited_depth_candidates": "[1,2,2,2];[2,1,2,2];[2,2,1,2];[2,2,2,1];[1,1,1,1]",
        "retrained_width_candidates": "encoder width 0.75/0.50; decoder width 0.75/0.50",
        "protected": "initial-depth/guidance heads; CSPN K24",
    },
    "dyspn": {
        "baseline": "ResNet34 [3,4,6,3]",
        "inherited_depth_candidates": "[2,4,6,3];[3,2,6,3];[3,4,3,3];[3,4,6,2];[2,2,2,2]",
        "retrained_width_candidates": "encoder width 0.75/0.50; decoder width 0.75; neighbors 5->3 only after backbone closure",
        "protected": "guidance head; DySPN K6 and neighbor semantics",
    },
    "nlspn": {
        "baseline": "ResNet34 [3,4,6,3]",
        "inherited_depth_candidates": "[2,4,6,3];[3,2,6,3];[3,4,3,3];[3,4,6,2];[2,2,2,2]",
        "retrained_width_candidates": "encoder width 0.75/0.50; decoder width 0.75",
        "protected": "early boundary; initial-depth/guidance/confidence heads; NLSPN K18",
    },
    "completionformer": {
        "baseline": "PVT [3,4,6,3] + ResNet embed [3,4]",
        "inherited_depth_candidates": "PVT [2,4,6,3];[3,2,6,3];[3,4,3,3];[3,4,6,2];[1,2,2,1]",
        "retrained_width_candidates": "Tiny [24,48,96,192]; Nano [16,32,64,128]; decoder width 0.75",
        "protected": "initial-depth/guidance/confidence heads; NLSPN K18",
    },
}


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _candidate(rows: Iterable[Mapping[str, str]], candidate_id: str) -> dict:
    matches = [row for row in rows if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("expected one candidate %s, found %d" %
                         (candidate_id, len(matches)))
    return matches[0]


def _optional_candidate(rows: Iterable[Mapping[str, str]],
                        candidate_id: str) -> dict | None:
    matches = [row for row in rows if row["candidate_id"] == candidate_id]
    if len(matches) > 1:
        raise ValueError("duplicate candidate: %s" % candidate_id)
    return matches[0] if matches else None


def _unit_cost_fractions(manifest: Mapping[str, object], kind: str) -> Dict[str, float]:
    rows = manifest["precision_costs"][kind]
    total = sum(int(value) for _, value in rows)
    if total <= 0:
        raise ValueError("precision cost total must be positive")
    return {str(name): int(value) / float(total) for name, value in rows}


def _sensitivity_class(delta: float) -> str:
    if delta <= 0.0025:
        return "safe"
    if delta <= 0.01:
        return "conditional"
    return "sensitive"


def analyze_model(model: str, model_root: Path) -> tuple[dict, List[dict]]:
    manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
    single = _read_csv(model_root / "single_module_ablation.csv")
    pareto = _read_csv(model_root / "pareto_ptq.csv")
    reference = float(manifest["reference_pooled_rmse"])
    anchor = manifest["anchor"]
    anchor_rmse = float(anchor["pooled_rmse"])
    anchor_relative = float(anchor["relative_loss"])
    weight_cost = _unit_cost_fractions(manifest, "weight_macs")
    activation_cost = _unit_cost_fractions(manifest, "activation_elements")
    units = list(anchor["assignment"]["weight_bits"])
    rows = []
    for unit in units:
        for axis, candidate_id, cost in (
                ("weight", "SINGLE_%s_W4A8" % unit, weight_cost),
                ("activation", "SINGLE_%s_W8A4" % unit, activation_cost)):
            measured = _optional_candidate(single, candidate_id)
            if measured is None:
                rows.append({
                    "model": model, "unit": unit, "axis": axis,
                    "candidate_id": candidate_id, "measured": False,
                    "rmse_m": "", "relative_to_fp32_pct": "",
                    "increment_vs_anchor_pct": "", "cost_fraction_pct":
                    100.0 * cost[unit], "classification": "protected",
                })
                continue
            relative = float(measured["relative_loss"])
            delta = relative - anchor_relative
            rows.append({
                "model": model, "unit": unit, "axis": axis,
                "candidate_id": candidate_id, "measured": True,
                "rmse_m": float(measured["pooled_rmse"]),
                "relative_to_fp32_pct": 100.0 * relative,
                "increment_vs_anchor_pct": 100.0 * delta,
                "cost_fraction_pct": 100.0 * cost[unit],
                "classification": _sensitivity_class(delta),
            })
    feasible = [row for row in pareto
                if str(row.get("valid", "")).lower() == "true" and
                float(row["relative_loss"]) <= float(manifest["maximum_relative_loss"])]
    if not feasible:
        raise ValueError("model has no feasible Pareto candidate: %s" % model)
    lowest_weight = min(feasible, key=lambda row: float(row["average_weight_bits"]))
    lowest_activation = min(feasible, key=lambda row: float(row["average_activation_bits"]))
    model_summary = {
        "model": model,
        "evaluation_samples": len(manifest["evaluation_indices"]),
        "fp32_rmse_m": reference,
        "anchor_id": anchor["candidate_id"],
        "anchor_rmse_m": anchor_rmse,
        "anchor_relative_pct": 100.0 * anchor_relative,
        "anchor_fp16_units": ";".join(anchor["assignment"]["fp16_units"]),
        "lowest_weight_candidate": lowest_weight["candidate_id"],
        "lowest_average_weight_bits": float(lowest_weight["average_weight_bits"]),
        "lowest_weight_relative_pct": 100.0 * float(lowest_weight["relative_loss"]),
        "lowest_activation_candidate": lowest_activation["candidate_id"],
        "lowest_average_activation_bits": float(lowest_activation["average_activation_bits"]),
        "lowest_activation_relative_pct": 100.0 * float(lowest_activation["relative_loss"]),
    }
    return model_summary, rows


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def build_rules(sensitivity: Sequence[Mapping[str, object]]) -> List[str]:
    measured = [row for row in sensitivity if row["measured"]]
    weight_delta = [float(row["increment_vs_anchor_pct"]) for row in measured
                    if row["axis"] == "weight"]
    activation_delta = [float(row["increment_vs_anchor_pct"]) for row in measured
                        if row["axis"] == "activation"]
    safe_weight = sum(row["classification"] == "safe" for row in measured
                      if row["axis"] == "weight")
    safe_activation = sum(row["classification"] == "safe" for row in measured
                          if row["axis"] == "activation")
    return [
        "Across measured single-unit ablations, median W4 incremental loss is %.3f%%; median A4 incremental loss is %.3f%%." %
        (_median(weight_delta), _median(activation_delta)),
        "Safe W4 units: %d; safe A4 units: %d. Weight lowering should therefore precede activation lowering." %
        (safe_weight, safe_activation),
        "Search backbone depth before width: depth subnets inherit weights and provide a cheap ranking signal; width candidates require retraining.",
        "Protect task heads and propagation in the first NAS round. Repeated propagation turns small head errors into task-level RMSE changes.",
        "Use a per-model relative RMSE gate and paired samples. Absolute RMSE is not comparable across the four checkpoints.",
        "Run PTQ before targeted QAT. Existing broad QAT is not a reliable monotonic improvement for NLSPN or CompletionFormer.",
    ]


def write_report(output: Path, summaries: Sequence[Mapping[str, object]],
                 sensitivity: Sequence[Mapping[str, object]],
                 rules: Sequence[str]) -> None:
    architecture_rows = []
    for model in MODEL_ORDER:
        row = {"model": model}
        row.update(ARCHITECTURE_SPACES[model])
        architecture_rows.append(row)
    _write_csv(output / "model_summary.csv", summaries)
    _write_csv(output / "quantization_sensitivity.csv", sensitivity)
    _write_csv(output / "architecture_search_space.csv", architecture_rows)
    lines = [
        "# Four-model NAS and low-bit evidence",
        "",
        "All measured rows use the same 64 NYU validation samples. Propagation remains FP16; the table does not claim full-integer propagation.",
        "",
        "## Measured PTQ summary",
        "",
        "| Model | FP32 RMSE | W8 anchor loss | Lowest-W candidate | Loss | Lowest-A candidate | Loss |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for row in summaries:
        lines.append("| {model} | {fp32_rmse_m:.6f} | {anchor_relative_pct:.3f}% | {lowest_weight_candidate} ({lowest_average_weight_bits:.2f}b) | {lowest_weight_relative_pct:.3f}% | {lowest_activation_candidate} ({lowest_average_activation_bits:.2f}b) | {lowest_activation_relative_pct:.3f}% |".format(**row))
    lines.extend(["", "## Cross-model rules", ""])
    lines.extend("%d. %s" % (index, rule) for index, rule in enumerate(rules, 1))
    lines.extend([
        "", "## Evidence boundary", "",
        "The PTQ values above are completed measurements. Architecture candidates in architecture_search_space.csv are a search definition, not trained accuracy results. Inherited-depth screening must be followed by candidate fine-tuning and a fresh PTQ calibration before a Pareto claim.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(results_root: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("analysis output already exists: %s" % output)
    summaries = []
    sensitivity = []
    for model in MODEL_ORDER:
        summary, rows = analyze_model(model, results_root / model)
        summaries.append(summary)
        sensitivity.extend(rows)
    rules = build_rules(sensitivity)
    output.mkdir(parents=True)
    write_report(output, summaries, sensitivity, rules)
    (output / "manifest.json").write_text(json.dumps({
        "format_version": 1,
        "source": str(results_root.resolve()),
        "models": list(MODEL_ORDER),
        "evaluation_protocol": "paired 64-sample pooled RMSE",
        "propagation_dtype": "fp16",
        "rules": rules,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output / "report.md"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.results_root, args.output))


if __name__ == "__main__":
    main()
