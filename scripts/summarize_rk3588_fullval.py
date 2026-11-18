#!/usr/bin/env python3
"""Validate and summarize RK3588 654-image FP16/W8A8 task metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Sequence


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")
PRECISIONS = ("fp16", "w8a8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_success(path: Path, model: str, precision: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != model:
        raise ValueError("model identity differs in %s" % path)
    if int(payload.get("sample_count", -1)) != 654:
        raise ValueError("%s must report 654 samples" % path)
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != 654:
        raise ValueError("%s must contain 654 sample rows" % path)
    if [int(row["index"]) for row in rows] != list(range(654)):
        raise ValueError("%s sample indices are not exactly 0..653" % path)
    if int(payload.get("nonfinite_prediction_pixels", -1)) != 0:
        raise ValueError("%s contains nonfinite prediction pixels" % path)
    valid_pixels = sum(int(row["valid_pixels"]) for row in rows)
    if valid_pixels != int(payload.get("valid_pixels", -1)):
        raise ValueError("%s valid-pixel aggregate differs" % path)
    mean_rmse = sum(float(row["rmse_m"]) for row in rows) / len(rows)
    pooled_rmse = math.sqrt(sum(
        float(row["rmse_m"]) ** 2 * int(row["valid_pixels"])
        for row in rows) / valid_pixels)
    if not math.isclose(mean_rmse, float(payload["mean_sample_rmse_m"]),
                        rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("%s mean sample RMSE differs" % path)
    if not math.isclose(pooled_rmse, float(payload["pooled_rmse_m"]),
                        rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("%s pooled RMSE differs" % path)
    return {
        "status": "complete",
        "sample_count": 654,
        "valid_pixels": valid_pixels,
        "mean_sample_rmse_m": float(payload["mean_sample_rmse_m"]),
        "pooled_rmse_m": float(payload["pooled_rmse_m"]),
        "mae_m": float(payload["mae_m"]),
        "abs_rel": float(payload["abs_rel"]),
        "nonfinite_prediction_pixels": 0,
        "nonpositive_prediction_pixels": int(
            payload["nonpositive_prediction_pixels"]),
        "mean_forward_latency_ms_health_only": float(
            payload["mean_forward_latency_ms"]),
        "api_version": str(payload["api_version"]),
        "driver_version": str(payload["driver_version"]),
        "evidence": path.name,
        "evidence_sha256": _sha256(path),
    }


def summarize(result_dir: Path, failure_path: Path) -> dict:
    result_dir = Path(result_dir)
    failure_path = Path(failure_path)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if failure.get("model") != "completionformer" or \
            failure.get("precision") != "fp16" or \
            failure.get("status") != "invalid_nonfinite_output":
        raise ValueError("failure evidence must describe CompletionFormer FP16")
    if int(failure.get("nonfinite_prediction_pixels", 0)) <= 0:
        raise ValueError("failure evidence must contain nonfinite pixels")

    runs = {model: {} for model in MODELS}
    successful = 0
    for model in MODELS:
        for precision in PRECISIONS:
            path = result_dir / (model + "_" + precision + ".json")
            if model == "completionformer" and precision == "fp16":
                if path.exists():
                    raise ValueError("failed CompletionFormer FP16 run has JSON")
                runs[model][precision] = dict(failure)
                runs[model][precision]["evidence"] = failure_path.name
                runs[model][precision]["evidence_sha256"] = _sha256(
                    failure_path)
                continue
            if not path.is_file():
                raise FileNotFoundError(path)
            runs[model][precision] = _validated_success(
                path, model, precision)
            successful += 1

    comparisons = {}
    for model in MODELS:
        fp16 = runs[model]["fp16"]
        w8a8 = runs[model]["w8a8"]
        if fp16["status"] != "complete" or w8a8["status"] != "complete":
            continue
        base = float(fp16["mean_sample_rmse_m"])
        quantized = float(w8a8["mean_sample_rmse_m"])
        comparisons[model] = {
            "fp16_mean_sample_rmse_m": base,
            "w8a8_mean_sample_rmse_m": quantized,
            "absolute_rmse_increase_m": quantized - base,
            "rmse_increase_pct": (quantized / base - 1.0) * 100.0,
        }
    return {
        "schema_version": 1,
        "dataset": "NYU Depth V2 official validation",
        "expected_sample_count_per_run": 654,
        "expected_run_count": 8,
        "successful_run_count": successful,
        "all_runs_accuracy_qualified": successful == 8,
        "runs": runs,
        "w8a8_vs_fp16": comparisons,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = summarize(args.result_dir, args.failure)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
