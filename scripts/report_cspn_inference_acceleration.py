#!/usr/bin/env python3
"""Select the fastest accuracy-qualified CSPN inference configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


def parse_candidate(value: str) -> tuple[str, str, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("candidate must have form NAME=REPORT=LATENCY")
    return parts[0], parts[1], parts[2]


def select_fastest(
    candidates: Iterable[dict[str, Any]],
    *,
    baseline_name: str,
) -> dict[str, Any]:
    rows = list(candidates)
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("candidate names must be unique")
    if baseline_name not in by_name:
        raise ValueError("baseline candidate is missing")
    passing = [
        row for row in rows
        if row.get("noninferior") is True and row.get("status") == "ok"
    ]
    if not passing:
        raise ValueError("no non-inferior successful candidate")
    selected = min(passing, key=lambda row: float(row["median_ms"]))
    baseline_ms = float(by_name[baseline_name]["median_ms"])
    selected_ms = float(selected["median_ms"])
    return {
        "baseline": baseline_name,
        "selected": selected["name"],
        "baseline_median_ms": baseline_ms,
        "selected_median_ms": selected_ms,
        "latency_reduction_ratio": 1.0 - selected_ms / baseline_ms,
        "speedup": baseline_ms / selected_ms,
        "candidates": rows,
    }


def _load_candidate(name: str, report_path: Path, latency_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    results = latency.get("results", [])
    if len(results) != 1:
        raise ValueError("strict latency file must contain exactly one result")
    timing = results[0]
    noninferiority = report["noninferiority"]
    return {
        "name": name,
        "status": timing["status"],
        "cspn_steps": timing["cspn_steps"],
        "precision": timing["precision"],
        "median_ms": timing.get("median_ms"),
        "p95_ms": timing.get("p95_ms"),
        "noninferior": noninferiority["noninferior"],
        "rmse": noninferiority["candidate_rmse"],
        "delta_rmse": noninferiority["delta_rmse"],
        "upper_confidence_bound": noninferiority["upper_confidence_bound"],
        "margin": noninferiority["margin"],
        "report": str(report_path.resolve()),
        "latency": str(latency_path.resolve()),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# CSPN Inference Acceleration",
        "",
        "Selected: **%s**" % decision["selected"],
        "",
        "| Configuration | Steps | Precision | RMSE (m) | Median (ms) | P95 (ms) | Non-inferior |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in decision["candidates"]:
        lines.append(
            "| {name} | {cspn_steps} | {precision} | {rmse:.9f} | "
            "{median_ms:.6f} | {p95_ms:.6f} | {noninferior} |".format(**row))
    lines.extend([
        "",
        "Latency reduction vs. %s: %.2f%% (%.3fx speedup)." % (
            decision["baseline"],
            100.0 * decision["latency_reduction_ratio"],
            decision["speedup"]),
        "",
        "TensorRT available: `%s`; ONNX available: `%s`; trtexec available: `%s`." % (
            decision["deployment"]["tensorrt_python"],
            decision["deployment"]["onnx_python"],
            decision["deployment"]["trtexec"]),
        "",
    ])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="NAME=NONINFERIORITY_JSON=STRICT_LATENCY_JSON")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    candidates = [
        _load_candidate(name, Path(report), Path(latency))
        for name, report, latency in (
            parse_candidate(value) for value in args.candidate)
    ]
    decision = select_fastest(candidates, baseline_name=args.baseline)
    decision["deployment"] = {
        "tensorrt_python": importlib.util.find_spec("tensorrt") is not None,
        "onnx_python": importlib.util.find_spec("onnx") is not None,
        "trtexec": shutil.which("trtexec") is not None,
    }
    output = Path(args.output)
    _atomic_text(
        output, json.dumps(decision, indent=2, sort_keys=True) + "\n")
    _atomic_text(output.with_name("README.md"), _markdown(decision))
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
