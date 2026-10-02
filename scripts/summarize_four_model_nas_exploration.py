#!/usr/bin/env python3
"""Build one evidence report from depth-NAS screens and calibrated PTQ runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selected_structure(rows: Sequence[Mapping[str, str]]) -> tuple[dict, bool]:
    candidates = [row for row in rows if row["candidate_id"] != "baseline"]
    feasible = [row for row in candidates
                if float(row["relative_rmse_pct"]) <= 2.0]
    if feasible:
        return max(feasible, key=lambda row: float(row["parameter_reduction_pct"])), True
    return min(candidates, key=lambda row: float(row["relative_rmse_pct"])), False


def _selected_low_bit(ptq_dir: Path, original_fp32: float) -> dict | None:
    targeted_path = ptq_dir / "targeted_lowbit.csv"
    if targeted_path.is_file():
        rows = [row for row in _csv(targeted_path)
                if str(row.get("valid", "")).lower() == "true" and
                str(row.get("passes_total_2pct", "")).lower() == "true"]
        if not rows:
            return None
        selected = min(rows, key=lambda row: (
            float(row["average_weight_bits"]) +
            float(row["average_activation_bits"]),
            float(row["pooled_rmse_m"])))
        return {
            "candidate_id": selected["candidate_id"],
            "pooled_rmse": selected["pooled_rmse_m"],
            "average_weight_bits": selected["average_weight_bits"],
            "average_activation_bits": selected["average_activation_bits"],
        }
    pareto_path = ptq_dir / "pareto_ptq.csv"
    if not pareto_path.is_file():
        return None
    rows = [row for row in _csv(pareto_path)
            if str(row.get("valid", "")).lower() == "true"]
    rows = [row for row in rows
            if 100.0 * (float(row["pooled_rmse"]) / original_fp32 - 1.0) <= 2.0]
    if not rows:
        return None
    return min(rows, key=lambda row: (
        float(row["average_weight_bits"]) +
        float(row["average_activation_bits"]),
        float(row["pooled_rmse"])))


def build_rows(baseline_ptq_root: Path, depth_root: Path,
               nas_quant_root: Path,
               nlspn_finetune_evaluation: Path | None = None) -> list[dict]:
    output = []
    for model in MODEL_ORDER:
        baseline_manifest = json.loads((
            baseline_ptq_root / model / "manifest.json").read_text(encoding="utf-8"))
        original_fp32 = float(baseline_manifest["reference_pooled_rmse"])
        screen = _csv(depth_root / model / "depth_screen.csv")
        baseline_screen = next(row for row in screen
                               if row["candidate_id"] == "baseline")
        selected, structure_feasible = _selected_structure(screen)
        finetuned = None
        if model == "nlspn" and nlspn_finetune_evaluation is not None:
            finetuned = json.loads(nlspn_finetune_evaluation.read_text(
                encoding="utf-8"))
        anchor_path = nas_quant_root / "anchors" / model / "manifest.json"
        anchor_manifest = json.loads(anchor_path.read_text(encoding="utf-8")) \
            if anchor_path.is_file() else None
        anchor = anchor_manifest.get("anchor") if anchor_manifest else None
        low_bit = _selected_low_bit(
            nas_quant_root / "targeted" / model, original_fp32)
        if low_bit is None:
            low_bit = _selected_low_bit(
                nas_quant_root / "ptq-search" / model, original_fp32)
        baseline_latency = float(baseline_screen["median_latency_ms"])
        selected_latency = float(selected["median_latency_ms"])
        selected_rmse = float(selected["pooled_rmse_m"])
        selected_name = selected["candidate_id"]
        if finetuned is not None:
            selected_rmse = float(finetuned["candidate"]["pooled_rmse_m"])
            selected_name += "_finetuned"
            structure_feasible = selected_rmse <= original_fp32 * 1.02
            # Replace the inherited-weight anchor with the recalibrated checkpoint row.
            anchor = None
            targeted_path = nas_quant_root / "targeted" / model / \
                "targeted_lowbit.csv"
            if targeted_path.is_file():
                targeted_rows = _csv(targeted_path)
                measured_anchor = next((row for row in targeted_rows
                                        if row["candidate_id"] ==
                                        "NAS_W8_ANCHOR"), None)
                if measured_anchor is not None:
                    anchor = {
                        "candidate_id": measured_anchor["candidate_id"],
                        "pooled_rmse": measured_anchor["pooled_rmse_m"],
                        "assignment": {"fp16_units": [
                            "initial_depth", "early_boundary"]},
                    }
        output.append({
            "model": model,
            "fixed64_pooled_fp32_rmse_m": original_fp32,
            "nas_candidate": selected_name,
            "nas_depths": selected["depths"],
            "nas_fp32_rmse_m": selected_rmse,
            "nas_relative_to_original_pct": 100.0 * (
                selected_rmse / original_fp32 - 1.0),
            "nas_parameter_reduction_pct": float(
                selected["parameter_reduction_pct"]),
            "a100_median_latency_reduction_pct": 100.0 * (
                1.0 - selected_latency / baseline_latency),
            "nas_passes_2pct": structure_feasible,
            "w8_anchor_id": anchor["candidate_id"] if anchor else "",
            "w8_rmse_m": float(anchor["pooled_rmse"]) if anchor else "",
            "w8_relative_to_original_pct": 100.0 * (
                float(anchor["pooled_rmse"]) / original_fp32 - 1.0)
                if anchor else "",
            "w8_fp16_units": ";".join(
                anchor["assignment"]["fp16_units"]) if anchor else "",
            "low_bit_candidate": low_bit["candidate_id"] if low_bit else "",
            "low_bit_rmse_m": float(low_bit["pooled_rmse"])
                if low_bit else "",
            "low_bit_relative_to_original_pct": 100.0 * (
                float(low_bit["pooled_rmse"]) / original_fp32 - 1.0)
                if low_bit else "",
            "low_bit_average_weight_bits": float(
                low_bit["average_weight_bits"]) if low_bit else "",
            "low_bit_average_activation_bits": float(
                low_bit["average_activation_bits"]) if low_bit else "",
        })
    return output


def write_report(output: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "combined_summary.csv", rows)
    lines = [
        "# Four-model NAS and low-bit exploration",
        "",
        "The structural screen uses the same fixed 64 NYU samples and pooled-pixel RMSE. These absolute RMSE values are not the official 654-image validation metric, which averages per-image RMSE. A100 latency is a model-only screening proxy, not U250 latency. W8/PTQ scales are calibration-derived; propagation stays FP16.",
        "",
        "| Model | Selected depth subnet | FP relative | Params | A100 latency | W8 total relative | Low-bit candidate | Avg W/A bits | Low-bit total relative |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        w8 = "pending" if row["w8_relative_to_original_pct"] == "" else \
            "%.3f%%" % row["w8_relative_to_original_pct"]
        low_bit = row["low_bit_candidate"] or "pending/not feasible"
        low_loss = "-" if row["low_bit_relative_to_original_pct"] == "" else \
            "%.3f%%" % row["low_bit_relative_to_original_pct"]
        low_bits = "-" if row["low_bit_average_weight_bits"] == "" else \
            "%.2f/%.2f" % (row["low_bit_average_weight_bits"],
                            row["low_bit_average_activation_bits"])
        lines.append(
            "| {model} | {nas_candidate} `{nas_depths}` | {nas_relative_to_original_pct:.3f}% | -{nas_parameter_reduction_pct:.2f}% | {a100_median_latency_reduction_pct:+.2f}% | {w8} | {low_bit} | {low_bits} | {low_loss} |".format(
                w8=w8, low_bit=low_bit, low_bits=low_bits,
                low_loss=low_loss, **row))
    lines.extend([
        "", "## Rules supported by measurements", "",
        "1. Search depth from the deepest encoder/transformer stage toward the input. The last stage is redundant in CSPN, DySPN, and CompletionFormer, while early-stage removal is consistently destructive.",
        "2. Architecture and quantization are not additive by assumption. Every selected subnet needs fresh calibration; the report only accepts total RMSE relative to the original FP32 model.",
        "3. Lower weights before activations. The separate four-model PTQ sweep found many safe W4 weight regions but substantially fewer safe A4 activation regions.",
        "4. Keep task heads and propagation at FP16 in the first pass. NLSPN additionally needs FP16 early-boundary and initial-depth islands under the measured policy.",
        "5. Output cosine is diagnostic, not a gate. NLSPN can retain cosine above 0.999 while exceeding the RMSE budget because propagation amplifies structured errors.",
        "6. Use QAT only to recover a selected structural or quantization boundary. The earlier broad QAT runs were non-monotonic and sometimes much worse than PTQ.",
        "7. Report parameter count, activation traffic, and MAC-weighted precision separately. Average W/A bits in this report are MAC/activation-element weighted and are not model-file byte counts.",
        "", "## Next gate", "",
        "Candidates marked outside 2% must first be fine-tuned in FP32. Candidates inside 2% proceed to calibrated PTQ, then 654-sample validation and target-hardware compilation. No U250 speed claim follows from the A100 proxy.",
    ])
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ptq-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--nas-quant-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nlspn-finetune-evaluation", type=Path)
    args = parser.parse_args(argv)
    rows = build_rows(args.baseline_ptq_root, args.depth_root,
                      args.nas_quant_root,
                      args.nlspn_finetune_evaluation)
    print(write_report(args.output, rows))


if __name__ == "__main__":
    main()
