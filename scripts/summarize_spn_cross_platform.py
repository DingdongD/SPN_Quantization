#!/usr/bin/env python3
"""Build an auditable manifest for the three-platform SPN measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, Sequence


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")
CATEGORIES = (
    "dense_compute",
    "attention",
    "sampling",
    "propagation",
    "elementwise_norm",
    "layout_memory",
    "other_unclassified",
)
# Two-sided Student-t 95% critical values. The experiment uses five processes,
# but the nearby values make the helper fail clearly for accidental omissions.
T95 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445,
       5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004,
       9: 2.262157, 10: 2.228139}


def summarize_run_means(run_means: Sequence[float]) -> Dict[str, float]:
    """Summarize independent-process means without pooling raw iterations."""
    values = [float(value) for value in run_means]
    if len(values) < 2:
        raise ValueError("at least two independent process means are required")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("run means must be finite and positive")
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom not in T95:
        raise ValueError("unsupported process count for exact Student-t interval")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    half_width = T95[degrees_of_freedom] * sample_std / math.sqrt(len(values))
    return {
        "process_count": len(values),
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "sample_std_ms": sample_std,
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "cv_percent": 100.0 * sample_std / mean,
        "ci95_low_ms": mean - half_width,
        "ci95_high_ms": mean + half_width,
        "run_means_ms": values,
    }


def load_run_means(paths: Iterable[Path]) -> list[float]:
    means = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        output = payload.get("output")
        if isinstance(output, dict) and output.get("all_finite") is False:
            raise ValueError("%s records non-finite output" % path)
        latency = payload.get("latency_ms")
        if isinstance(latency, dict) and "mean" in latency:
            means.append(float(latency["mean"]))
            continue
        raw = payload.get("raw_latency_ms")
        if not isinstance(raw, list) or not raw:
            raise ValueError("%s has no measured latency samples" % path)
        means.append(statistics.fmean(float(value) for value in raw))
    return means


def validate_breakdown(payload: Dict[str, Any]) -> None:
    if "category_mean_gpu_activity_ms" in payload:
        categories = payload["category_mean_gpu_activity_ms"]
        total = float(payload["gpu_activity_sum_mean_ms"])
    elif "category_time_us" in payload:
        categories = payload["category_time_us"]
        total = float(payload["operator_time_sum_us"])
    else:
        raise ValueError("unknown breakdown schema")
    if set(categories) != set(CATEGORIES):
        raise ValueError("breakdown must contain every exclusive category")
    category_sum = sum(float(categories[name]) for name in CATEGORIES)
    if not math.isclose(category_sum, total, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            "category sum %.12g does not match direct total %.12g" %
            (category_sum, total))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _platform_layout(root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "jetson_xavier": {
            "precision": "fp32",
            "repeat_dir": root / "jetson_xavier" /
                "controlled_latency_fp32_sample386_20261022",
            "breakdown_dir": root / "jetson_xavier" /
                "strict_nsys_breakdown_fp32_sample386_20261022",
            "breakdown_suffix": ".json",
        },
        "rtx_3080_ti": {
            "precision": "fp32",
            "repeat_dir": root / "autodl_nvidia" / "controlled_repeats",
            "breakdown_dir": root / "autodl_nvidia" / "results",
            "breakdown_suffix": "_breakdown.json",
        },
        "rk3588": {
            "precision": "fp16",
            "repeat_dir": root / "rk3588" / "controlled_repeats",
            "breakdown_dir": root / "rk3588" / "runtime",
            "breakdown_suffix": "_perf_detail_breakdown.json",
        },
    }


def build_manifest(root: Path) -> Dict[str, Any]:
    root = Path(root)
    platforms = {}
    evidence = {}
    for platform, layout in _platform_layout(root).items():
        model_rows = {}
        for model in MODELS:
            repeat_files = sorted(layout["repeat_dir"].glob("rep*_%s.json" % model))
            if len(repeat_files) != 5:
                raise ValueError(
                    "%s/%s needs five independent process files, found %d" %
                    (platform, model, len(repeat_files)))
            if platform == "rtx_3080_ti":
                breakdown = layout["breakdown_dir"] / ("nsys_%s" % model) / \
                    (model + layout["breakdown_suffix"])
            else:
                breakdown = layout["breakdown_dir"] / \
                    (model + layout["breakdown_suffix"])
            breakdown_payload = json.loads(breakdown.read_text(encoding="utf-8"))
            validate_breakdown(breakdown_payload)
            model_rows[model] = {
                "latency": summarize_run_means(load_run_means(repeat_files)),
                "breakdown_file": str(breakdown.relative_to(root)),
            }
            for path in [*repeat_files, breakdown]:
                relative = str(path.relative_to(root))
                evidence[relative] = _sha256(path)
        platforms[platform] = {
            "precision": layout["precision"],
            "models": model_rows,
        }
    return {
        "schema_version": 1,
        "measurement_scope": (
            "batch-1 complete forward on fixed NYU sample 386; inputs resident; "
            "model loading, dataset decode, preprocessing, and disk I/O excluded"),
        "aggregation": (
            "mean, median, sample CV, and two-sided Student-t 95% interval over "
            "five independent process means; profiler categories remain direct "
            "unscaled device/operator activity"),
        "models": list(MODELS),
        "platforms": platforms,
        "evidence_sha256": dict(sorted(evidence.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
