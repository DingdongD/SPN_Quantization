#!/usr/bin/env python3
"""Aggregate independent hardware-measured SPN energy runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, Sequence


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")
T95 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445,
       5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004,
       9: 2.262157, 10: 2.228139}


def _statistics(values: Sequence[float],
                allow_signed: bool = False) -> Dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) < 2:
        raise ValueError("at least two independent runs are required")
    if not all(math.isfinite(value) and (allow_signed or value >= 0.0)
               for value in values):
        raise ValueError("run values must be finite and non-negative")
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom not in T95:
        raise ValueError("unsupported run count for Student-t interval")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    half_width = T95[degrees_of_freedom] * sample_std / math.sqrt(len(values))
    return {
        "process_count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sample_std": sample_std,
        "minimum": min(values),
        "maximum": max(values),
        "cv_percent": 100.0 * sample_std / abs(mean) if mean else 0.0,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "run_values": values,
    }


def summarize_runs(runs: Sequence[Dict[str, Any]],
                   rail_name: str) -> Dict[str, Any]:
    """Summarize one like-for-like physical sensor rail across processes."""
    if len(runs) < 2:
        raise ValueError("at least two independent runs are required")
    scopes = {run["power"].get("sensor_scope") for run in runs}
    sources = {run["power"].get("source") for run in runs}
    models = {run.get("model") for run in runs}
    if len(scopes) != 1:
        raise ValueError("mixed sensor scope cannot be aggregated")
    if len(sources) != 1:
        raise ValueError("mixed power source cannot be aggregated")
    if len(models) != 1:
        raise ValueError("mixed models cannot be aggregated")
    rail_rows = []
    for run in runs:
        rails = run["power"].get("rails", {})
        if rail_name not in rails:
            raise ValueError("power rail %s is missing" % rail_name)
        rail_rows.append(rails[rail_name])

    return {
        "model": next(iter(models)),
        "source": next(iter(sources)),
        "sensor_scope": next(iter(scopes)),
        "rail": rail_name,
        "gross_j_per_forward": _statistics([
            row["gross_j_per_forward"] for row in rail_rows]),
        "incremental_j_per_forward": _statistics([
            row["incremental_j_per_forward"] for row in rail_rows],
            allow_signed=True),
        "idle_mean_power_w": _statistics([
            row["idle"]["mean_power_w"] for row in rail_rows]),
        "active_mean_power_w": _statistics([
            row["active"]["mean_power_w"] for row in rail_rows]),
        "completed_forwards": _statistics([
            run["workload"]["completed_forwards"] for run in runs]),
        "active_latency_mean_ms": _statistics([
            run["workload"]["latency_ms"]["mean"] for run in runs]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runs(paths: Iterable[Path]) -> list:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _validate_provenance(runs: Sequence[Dict[str, Any]]) -> None:
    for field in ("checkpoint_sha256", "precision"):
        values = {run.get(field) for run in runs}
        if None in values or len(values) != 1:
            raise ValueError("runs have inconsistent %s" % field)
    inputs = {
        run.get("fixed_input_artifact", {}).get("sha256") for run in runs}
    if None in inputs or len(inputs) != 1:
        raise ValueError("runs have inconsistent fixed input hashes")
    settings = {json.dumps(run.get("settings"), sort_keys=True) for run in runs}
    if len(settings) != 1:
        raise ValueError("runs have inconsistent energy settings")
    for run in runs:
        for rail in run["power"]["rails"].values():
            if rail["idle"].get("sample_count", 100) < 100 or \
                    rail["active"].get("sample_count", 100) < 100:
                raise ValueError("energy phase has fewer than 100 samples")


def build_summary(rtx_dir: Path, jetson_dir: Path) -> Dict[str, Any]:
    evidence_root = Path(os.path.commonpath((
        str(Path(rtx_dir)), str(Path(jetson_dir)))))
    layouts = {
        "rtx_3080_ti": (Path(rtx_dir), ("gpu_board",)),
        "jetson_xavier": (
            Path(jetson_dir),
            ("total_module", "gpu", "cpu", "soc", "cv", "vddrq", "sys5v")),
    }
    platforms = {}
    evidence = {}
    for platform_name, (repeat_dir, rails) in layouts.items():
        models = {}
        for model in MODELS:
            paths = sorted(repeat_dir.glob("rep*_%s.json" % model))
            if len(paths) != 5:
                raise ValueError("%s/%s requires five runs, found %d" % (
                    platform_name, model, len(paths)))
            runs = _load_runs(paths)
            _validate_provenance(runs)
            models[model] = {
                "precision": runs[0]["precision"],
                "checkpoint_sha256": runs[0]["checkpoint_sha256"],
                "fixed_input_sha256": runs[0]["fixed_input_artifact"]["sha256"],
                "rails": {
                    rail: summarize_runs(runs, rail) for rail in rails
                },
            }
            for path, run in zip(paths, runs):
                evidence[str(path.relative_to(evidence_root))] = _sha256(path)
                trace = path.with_suffix(".power.jsonl")
                if not trace.is_file():
                    trace = Path(run["raw_power_trace"]["path"])
                    if not trace.is_absolute():
                        trace = path.parent / trace
                if not trace.is_file():
                    raise ValueError("raw power trace is missing for %s" % path)
                if _sha256(trace) != run["raw_power_trace"]["sha256"]:
                    raise ValueError("raw trace hash mismatch for %s" % path)
                evidence[str(trace.relative_to(evidence_root))] = _sha256(trace)
        platforms[platform_name] = {"models": models}
    return {
        "schema_version": 1,
        "measurement_scope": (
            "batch-1 complete forward on fixed NYU sample 386; loaded model "
            "and inputs resident; propagation included"),
        "energy_definition": (
            "gross integrates active-window sensor power; incremental subtracts "
            "the loaded-idle mean power over the active duration"),
        "platforms": platforms,
        "evidence_sha256": dict(sorted(evidence.items())),
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtx-dir", type=Path, required=True)
    parser.add_argument("--jetson-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_summary(args.rtx_dir, args.jetson_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
