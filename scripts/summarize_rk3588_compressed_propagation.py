#!/usr/bin/env python3
"""Validate and summarize controlled RK3588 compressed-SPN processes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


_FIXED_PROTOCOL = {
    "warmup_iterations": 30,
    "timed_iterations_per_round": 50,
    "rounds": 3,
}
_IDENTITY_FIELDS = (
    "model", "candidate", "precision", "model_sha256",
    "safe_onnx_sha256", "input_sha256",
    "api_version", "driver_version",
)


def enrich_board_process(payload: dict, *, candidate: str, precision: str,
                         model_sha256: str, safe_onnx_sha256: str,
                         input_sha256: Sequence[str],
                         process_index: int) -> dict:
    """Bind a board-emitted timing record to immutable host artifacts."""
    if payload.get("precision") != precision:
        raise ValueError("board precision does not match expected precision")
    output = dict(payload.get("output", {}))
    aliases = {
        "minimum": "finite_minimum",
        "maximum": "finite_maximum",
        "mean": "finite_mean",
    }
    for destination, source in aliases.items():
        if destination not in output and source in output:
            output[destination] = output[source]
    result = dict(payload)
    result.update({
        "candidate": str(candidate),
        "precision": str(precision),
        "model_sha256": str(model_sha256),
        "safe_onnx_sha256": str(safe_onnx_sha256),
        "input_sha256": [str(value) for value in input_sha256],
        "process_index": int(process_index),
        "output": output,
    })
    return result


def _finite_vector(payload: dict, key: str, expected: int) -> np.ndarray:
    values = np.asarray(payload.get(key, []), dtype=np.float64)
    if values.shape != (expected,):
        raise ValueError("%s must contain %d samples" % (key, expected))
    if not np.isfinite(values).all():
        raise ValueError("%s must contain only finite values" % key)
    return values


def _load_process(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in _FIXED_PROTOCOL.items():
        if payload.get(key) != value:
            raise ValueError("%s violates controlled protocol" % key)
    expected = (payload["timed_iterations_per_round"]
                * payload["rounds"])
    payload["_latency"] = _finite_vector(
        payload, "raw_latency_ms", expected)
    payload["_callback"] = _finite_vector(
        payload, "raw_grid_sample_cpu_ms", expected)
    payload["_calls"] = _finite_vector(
        payload, "raw_grid_sample_call_count", expected)
    output = payload.get("output", {})
    output_values = [output.get(key) for key in
                     ("elements", "minimum", "maximum", "mean")]
    if any(value is None or not math.isfinite(float(value))
           for value in output_values):
        raise ValueError("output statistics must be finite")
    for required in _IDENTITY_FIELDS + ("process_index",):
        if required not in payload:
            raise ValueError("missing required evidence field: %s" % required)
    return payload


def summarize_processes(paths: Sequence[Path]) -> dict:
    if len(paths) != 5:
        raise ValueError("exactly five independent processes are required")
    runs = [_load_process(path) for path in paths]
    expected_identity = {key: runs[0][key] for key in _IDENTITY_FIELDS}
    for run in runs[1:]:
        for key, value in expected_identity.items():
            if run[key] != value:
                raise ValueError("process identity mismatch: %s" % key)
    indices = [int(run["process_index"]) for run in runs]
    if sorted(indices) != list(range(5)):
        raise ValueError("process indices must be exactly 0 through 4")
    runs.sort(key=lambda item: int(item["process_index"]))

    process_means = np.asarray(
        [run["_latency"].mean() for run in runs], dtype=np.float64)
    all_latency = np.concatenate([run["_latency"] for run in runs])
    all_callback = np.concatenate([run["_callback"] for run in runs])
    all_calls = np.concatenate([run["_calls"] for run in runs])
    if not np.all(all_calls == all_calls[0]):
        raise ValueError("GridSample call count changed across forwards")
    return {
        "schema_version": 1,
        **expected_identity,
        "process_count": 5,
        "samples_per_process": int(runs[0]["_latency"].size),
        "process_mean_latency_ms": process_means.tolist(),
        "latency_ms": {
            "mean_of_process_means": float(process_means.mean()),
            "process_cv_percent": float(
                100.0 * process_means.std(ddof=1) / process_means.mean()),
            "median": float(np.median(all_latency)),
            "p05": float(np.quantile(all_latency, 0.05)),
            "p95": float(np.quantile(all_latency, 0.95)),
            "minimum": float(all_latency.min()),
            "maximum": float(all_latency.max()),
        },
        "grid_sample": {
            "calls_per_forward": int(all_calls[0]),
            "mean_cpu_ms": float(all_callback.mean()),
            "fraction_of_forward_percent": float(
                100.0 * all_callback.mean() / all_latency.mean()),
        },
        "runtime": {
            "api_version": runs[0]["api_version"],
            "driver_version": runs[0]["driver_version"],
        },
        "output": runs[0]["output"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=5, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = summarize_processes(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
