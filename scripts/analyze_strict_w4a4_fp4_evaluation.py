#!/usr/bin/env python3
"""Validate and summarize strict W4A4 and FP4 evaluations."""

from __future__ import division, print_function

import argparse
import csv
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strict_w4a4_fp4_evaluation import analyze_result_root


OUTPUT_TABLES = (
    ("summary", "strict_w4a4_fp4_summary.csv"),
    ("paired", "strict_w4a4_fp4_paired.csv"),
    ("activation_groups", "strict_w4a4_fp4_activation_groups.csv"),
    ("propagation_steps", "strict_w4a4_fp4_propagation_steps.csv"),
    ("stress", "strict_w4a4_integer_stress.csv"),
)


def write_csv(path, rows):
    if not rows:
        raise ValueError("analysis table must not be empty: %s" % path)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    fields = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def render_report(summary, stress):
    lines = [
        "# Strict W4A4 and FP4 Evaluation",
        "",
        "Performance preservation requires zero nonfinite output, no more than "
        "10% mean-RMSE degradation from FP32, and no regression from RTN under "
        "the same activation format.",
        "",
        "## Matched FP4V Results",
        "",
        "| Model | Method | Configuration | FP32 RMSE | Quant RMSE | "
        "Delta vs RTN | Status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary:
        if row["config"] == "FP32":
            continue
        fp32_rows = [
            candidate for candidate in summary
            if candidate["model"] == row["model"] and
            candidate["method"] == row["method"] and
            candidate["config"] == "FP32"]
        if len(fp32_rows) != 1:
            raise ValueError("FP32 report lookup mismatch")
        lines.append(
            "| %s | %s | %s | %.6f | %.6f | %+.6f | %s |" % (
                row["model"], row["method"], row["config"],
                fp32_rows[0]["mean_rmse"], row["mean_rmse"],
                row["delta_vs_rtn"], row["status"]))
    lines.extend([
        "",
        "## Integer W4A4 Stress Baseline",
        "",
        "| Model | Method | Configuration | Mean RMSE | Nonfinite samples |",
        "|---|---|---|---:|---:|",
    ])
    for row in stress:
        if row["config"] == "FP32":
            continue
        lines.append(
            "| %s | %s | %s | %.6f | %d |" % (
                row["model"], row["method"], row["config"],
                row["mean_rmse"], row["nonfinite_samples"]))
    lines.extend([
        "",
        "E2M1 results use `float_e2m1_qdq_reference`; the integer stress "
        "baseline is reported separately.",
        "",
    ])
    return "\n".join(lines)


def write_analysis(root, out_dir, expected_samples,
                   bootstrap_resamples, bootstrap_seed):
    tables = analyze_result_root(
        root, expected_samples=expected_samples,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in OUTPUT_TABLES:
        write_csv(out_dir / filename, tables[key])
    report_path = out_dir / "strict_w4a4_fp4_report.md"
    temporary = report_path.with_suffix(report_path.suffix + ".part")
    temporary.write_text(
        render_report(tables["summary"], tables["stress"]),
        encoding="utf-8")
    os.replace(str(temporary), str(report_path))
    return tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    args = parser.parse_args()
    write_analysis(
        args.root, args.out_dir,
        expected_samples=args.expected_samples,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed)
    print("analysis=%s" % Path(args.out_dir).resolve(), flush=True)


if __name__ == "__main__":
    main()
