#!/usr/bin/env python3
"""Audit and aggregate Jetson Xavier pure-CPU SPN measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_spn_cross_platform_energy import (
    MODELS, _statistics, summarize_runs)


CATEGORIES = (
    "dense_compute", "attention", "sampling", "propagation",
    "elementwise_norm", "layout_memory", "other_unclassified",
)


def _consistent(runs, getter, label):
    values = {json.dumps(getter(run), sort_keys=True) for run in runs}
    if len(values) != 1:
        raise ValueError("inconsistent %s" % label)


def summarize_cpu_model(main_runs: Sequence[Dict[str, Any]],
                        energy_runs: Sequence[Dict[str, Any]],
                        profile_runs: Sequence[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    if len(main_runs) != 5 or len(energy_runs) != 5:
        raise ValueError("exactly five independent processes are required")
    _consistent(main_runs, lambda run: run["model"], "model")
    _consistent(main_runs, lambda run: run["precision"], "precision")
    _consistent(main_runs, lambda run: run["checkpoint_sha256"], "checkpoint")
    _consistent(main_runs, lambda run: run["config_sha256"], "config")
    _consistent(main_runs, lambda run: run["fixed_input_artifact"]["sha256"],
                "input")
    _consistent(main_runs, lambda run: run["settings"], "timing settings")
    _consistent(energy_runs, lambda run: run["settings"], "energy settings")
    model = main_runs[0]["model"]
    if {run["model"] for run in energy_runs} != {model}:
        raise ValueError("energy model differs from latency model")
    profile_runs = list(profile_runs) if profile_runs is not None else list(
        main_runs)
    if len(profile_runs) != 5 or {run["model"] for run in profile_runs} != {
            model}:
        raise ValueError("exactly five same-model profiler processes required")
    _consistent(profile_runs, lambda run: run["settings"],
                "profiler process settings")
    for run in main_runs:
        audit = run["cpu_execution_audit"]
        if audit["execution_device"] != "cpu" or audit["cuda_available"]:
            raise ValueError("run did not pass the CPU-only execution gate")
        if len(run["raw_latency_ms"]) != 9:
            raise ValueError("each process must contain nine latency samples")
    for run in profile_runs:
        audit = run["cpu_execution_audit"]
        if audit["execution_device"] != "cpu" or audit["cuda_available"]:
            raise ValueError("profiler did not pass the CPU-only execution gate")
        if run["breakdown"]["profiled_iterations"] != 2:
            raise ValueError("each process must contain two profiler forwards")
    for run in energy_runs:
        if run["workload"]["active_wall_seconds"] < 60.0:
            raise ValueError("energy active window is shorter than 60 seconds")
        for rail_name in ("cpu", "total_module"):
            rail = run["power"]["rails"].get(rail_name)
            if rail is None:
                raise ValueError("missing %s energy rail" % rail_name)
            if rail["idle"]["sample_count"] < 250 or \
                    rail["active"]["sample_count"] < 1000:
                raise ValueError("energy window has insufficient samples")

    category_means = {
        category: statistics.fmean(
            run["breakdown"]["category_mean_self_cpu_time_us"][category]
            for run in profile_runs)
        for category in CATEGORIES
    }
    total = sum(category_means.values())
    if total <= 0.0:
        raise ValueError("profiler category total must be positive")
    return {
        "model": model,
        "precision": main_runs[0]["precision"],
        "checkpoint_sha256": main_runs[0]["checkpoint_sha256"],
        "fixed_input_sha256": main_runs[0]["fixed_input_artifact"]["sha256"],
        "process_count": 5,
        "latency_sample_count": sum(
            len(run["raw_latency_ms"]) for run in main_runs),
        "latency_ms": _statistics([
            run["latency_ms"]["mean"] for run in main_runs]),
        "breakdown": {
            "profiled_forward_count": sum(
                run["breakdown"]["profiled_iterations"]
                for run in profile_runs),
            "category_mean_self_cpu_time_us": category_means,
            "category_share_percent": {
                category: 100.0 * value / total
                for category, value in category_means.items()
            },
            "self_cpu_time_sum_mean_us": total,
            "negative_self_time_event_count": sum(
                run["breakdown"]["negative_self_time_events"]["count"]
                for run in profile_runs),
            "negative_self_time_total_magnitude_us": sum(
                run["breakdown"]["negative_self_time_events"][
                    "total_magnitude_us"] for run in profile_runs),
        },
        "energy": {
            rail: summarize_runs(energy_runs, rail)
            for rail in ("cpu", "total_module")
        },
        "settings": {
            "timing": main_runs[0]["settings"],
            "energy": energy_runs[0]["settings"],
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_thread_pilot(pilot_dir: Path) -> Dict[str, Any]:
    rows = {}
    for threads in (1, 2, 4, 8):
        model_means = {}
        for model in MODELS:
            path = Path(pilot_dir) / ("%s_t%d.json" % (model, threads))
            if not path.is_file():
                raise ValueError("missing thread-pilot evidence: %s" % path)
            run = json.loads(path.read_text(encoding="utf-8"))
            audit = run["cpu_execution_audit"]
            if run["settings"]["cpu_threads"] != threads or \
                    audit["execution_device"] != "cpu" or \
                    audit["cuda_available"]:
                raise ValueError("invalid CPU thread-pilot run: %s" % path)
            model_means[model] = float(run["latency_ms"]["mean"])
        rows[str(threads)] = {
            "model_mean_latency_ms": model_means,
            "four_model_geometric_mean_ms": math.exp(statistics.fmean(
                math.log(value) for value in model_means.values())),
        }
    selected = min(rows, key=lambda key: rows[key][
        "four_model_geometric_mean_ms"])
    return {"selected_threads": int(selected), "candidates": rows}


def build_summary(input_dir: Path, profile_dir: Path = None,
                  pilot_dir: Path = None) -> Dict[str, Any]:
    input_dir = Path(input_dir)
    profile_dir = Path(profile_dir) if profile_dir is not None else input_dir
    models = {}
    evidence = {}
    for model in MODELS:
        mains = [input_dir / ("rep%d_%s.json" % (rep, model))
                 for rep in range(1, 6)]
        energies = [input_dir / ("rep%d_%s_energy.json" % (rep, model))
                    for rep in range(1, 6)]
        missing = [path for path in mains + energies if not path.is_file()]
        if missing:
            raise ValueError("missing formal evidence: %s" % missing[0])
        main_runs = [json.loads(path.read_text(encoding="utf-8"))
                     for path in mains]
        energy_runs = [json.loads(path.read_text(encoding="utf-8"))
                       for path in energies]
        profiles = [profile_dir / ("rep%d_%s_profile.json" % (rep, model))
                    for rep in range(1, 6)]
        if profile_dir == input_dir:
            profile_runs = main_runs
            profiles = []
        else:
            missing_profiles = [path for path in profiles if not path.is_file()]
            if missing_profiles:
                raise ValueError("missing profiler evidence: %s" %
                                 missing_profiles[0])
            profile_runs = [json.loads(path.read_text(encoding="utf-8"))
                            for path in profiles]
        models[model] = summarize_cpu_model(
            main_runs, energy_runs, profile_runs)
        for path, energy in zip(energies, energy_runs):
            trace = path.with_suffix(".power.jsonl")
            if not trace.is_file() or _sha256(trace) != \
                    energy["raw_power_trace"]["sha256"]:
                raise ValueError("raw power trace hash mismatch: %s" % trace)
        for path in mains + energies + [
                path.with_suffix(".power.jsonl") for path in energies]:
            evidence[path.name] = _sha256(path)
        for path in profiles:
            evidence["profiles_v2/" + path.name] = _sha256(path)
    first = json.loads((input_dir / "rep1_cspn.json").read_text(
        encoding="utf-8"))
    run_log = (input_dir / "run.log").read_text(encoding="utf-8")
    logged_hashes = {
        name: value for name, value in re.findall(
            r"source_sha256_([a-z_]+)=([0-9a-f]{64})", run_log)
    }
    if set(logged_hashes) != {"cpu_worker", "cpu_helpers"}:
        raise ValueError("formal run log lacks exact source hashes")
    result = {
        "schema_version": 1,
        "platform": "Jetson AGX Xavier 32 GB",
        "execution_device": "NVIDIA Carmel CPU",
        "measurement_scope": (
            "batch-1 complete FP32 forward on fixed NYU sample 386, including "
            "all propagation; preprocessed input resident in CPU memory"),
        "cpu_telemetry": first["cpu_before"],
        "software": first["software"],
        "models": models,
        "evidence_sha256": dict(sorted(evidence.items())),
        "implementation_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                REPO_ROOT / "scripts" / "benchmark_nyu_vanilla_cpu.py",
                REPO_ROOT / "scripts" / "nyu_model_runtime.py",
                REPO_ROOT / "spn_quant" / "cpu_benchmark.py",
                REPO_ROOT / "models" / "DCN.py",
            )
        },
        "formal_run_logged_sha256": logged_hashes,
    }
    if pilot_dir is not None:
        result["thread_pilot"] = summarize_thread_pilot(pilot_dir)
        for path in sorted(Path(pilot_dir).glob("*.json")):
            result["evidence_sha256"]["pilot/" + path.name] = _sha256(path)
        result["evidence_sha256"] = dict(sorted(
            result["evidence_sha256"].items()))
    return result


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_summary(
        args.input_dir, args.profile_dir, args.pilot_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
