#!/usr/bin/env python3
"""Report paired CSPN encoder NAS non-inferiority on official validation data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spn_quant.nas.statistics import paired_hierarchical_bootstrap


METRICS = ("RMSE", "MAE", "ABS_REL", "DELTA1.25")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _read_metrics(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                key = (int(row["seed"]), int(row["sample_id"]))
                values = {metric: float(row[metric]) for metric in METRICS}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid metric row {line} in {path}: {error}") from error
            if key in rows:
                raise ValueError(f"duplicate seed/sample pair {key} in {path}")
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite metric at seed/sample pair {key}")
            rows[key] = values
    if not rows:
        raise ValueError(f"metric file is empty: {path}")
    return rows


def _matrix(
    rows: dict[tuple[int, int], dict[str, float]], metric: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    seeds = sorted({seed for seed, _ in rows})
    samples = sorted({sample for _, sample in rows})
    expected = {(seed, sample) for seed in seeds for sample in samples}
    if set(rows) != expected:
        raise ValueError("each training seed must contain the same sample IDs")
    return np.array([
        [rows[(seed, sample)][metric] for sample in samples]
        for seed in seeds
    ], dtype=np.float64), seeds, samples


def _metric_means(rows: dict[tuple[int, int], dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([row[metric] for row in rows.values()]))
        for metric in METRICS
    }


def _write_outputs(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output / "noninferiority.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    ni = result["noninferiority"]
    flat = {
        "status": "PASS" if ni["noninferior"] else "FAIL",
        **ni,
        **{f"control_{key}": value
           for key, value in result["control_metrics"].items()},
        **{f"candidate_{key}": value
           for key, value in result["candidate_metrics"].items()},
    }
    csv_path = output / "noninferiority.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    os.replace(temporary, csv_path)

    status = "PASS" if ni["noninferior"] else "FAIL"
    markdown = f"""# CSPN Encoder NAS Non-Inferiority

**Decision: {status}**

| Quantity | Value |
| --- | ---: |
| R18 RMSE | {ni['control_rmse']:.9f} m |
| Candidate RMSE | {ni['candidate_rmse']:.9f} m |
| Paired delta | {ni['delta_rmse']:.9f} m |
| 2% margin | {ni['margin']:.9f} m |
| One-sided {ni['confidence']:.0%} upper bound | {ni['upper_confidence_bound']:.9f} m |
| Training seeds | {ni['training_seed_count']} |
| Validation samples per seed | {ni['sample_count']} |
| Bootstrap replicates | {ni['replicates']} |

The candidate is non-inferior only when the paired upper confidence bound is
not greater than the pre-specified 2% R18 margin.
"""
    _atomic_text(output / "noninferiority.md", markdown)


def generate_report(
    control_path: Path,
    candidate_path: Path,
    output: Path,
    *,
    margin_ratio: float = 0.02,
    replicates: int = 10_000,
    seed: int = 20260920,
) -> dict[str, Any]:
    control_rows = _read_metrics(Path(control_path))
    candidate_rows = _read_metrics(Path(candidate_path))
    if set(control_rows) != set(candidate_rows):
        raise ValueError("control and candidate must have identical seed/sample pairs")
    control_rmse, seeds, samples = _matrix(control_rows, "RMSE")
    candidate_rmse, _, _ = _matrix(candidate_rows, "RMSE")
    result = {
        "control_file": str(Path(control_path).resolve()),
        "candidate_file": str(Path(candidate_path).resolve()),
        "seeds": seeds,
        "sample_ids": samples,
        "control_metrics": _metric_means(control_rows),
        "candidate_metrics": _metric_means(candidate_rows),
        "noninferiority": paired_hierarchical_bootstrap(
            control_rmse, candidate_rmse, margin_ratio=margin_ratio,
            replicates=replicates, seed=seed),
    }
    _write_outputs(Path(output), result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-ratio", type=float, default=0.02)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = generate_report(
        Path(args.control), Path(args.candidate), Path(args.output),
        margin_ratio=args.margin_ratio, replicates=args.replicates,
        seed=args.seed)
    print(json.dumps(result["noninferiority"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
