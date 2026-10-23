#!/usr/bin/env python3
"""Create an exclusive GPU-activity breakdown from an Nsight Systems export.

CUDA activities are associated with their launching CPU operator through the
CUPTI correlation ID.  This is required because device execution is
asynchronous and can occur after the launching NVTX operator range has closed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CATEGORIES = (
    "dense_compute",
    "attention",
    "sampling",
    "propagation",
    "elementwise_norm",
    "layout_memory",
    "other_unclassified",
)


def category(name: str) -> str:
    value = name.lower()
    if any(token in value for token in (
            "adaptive_max_pool", "adaptive_avg_pool", "softmax",
            "scaled_dot_product", "flash_attention", "_efficient_attention",
            "attention", "bmm")):
        return "attention"
    if any(token in value for token in (
            "grid_sampler", "grid_sample", "gather", "deform", "upsample",
            "interpolate", "index_select", "embedding", "max_pool",
            "computeoffsets", "kern_precompute_indices")):
        return "sampling"
    if any(token in value for token in (
            "batch_norm", "batchnorm", "bn_fw", "layer_norm",
            "normalization", "norm_kernel")):
        return "elementwise_norm"
    if any(token in value for token in (
            "conv", "gemm", "matmul", "linear", "cublas", "cutlass",
            "xmma", "cudnn", "ampere_sgemm", "gemv", "winograd", "dgrad")):
        return "dense_compute"
    if any(token in value for token in (
            "cat", "copy", "contiguous", "permute", "transpose", "clone",
            "resize", "to_copy", "memcpy", "memset", "scalepackedtensor",
            "nchwtonhwc", "nhwctonchw")):
        return "layout_memory"
    if any(token in value for token in (
            "relu", "sigmoid", "tanh", "mul", "add", "sub", "div", "norm",
            "normalize", "exp", "sqrt", "clamp", "where", "sum", "mean",
            "elementwise", "vectorized", "binaryfunctor", "reduce_kernel",
            "pointwise")):
        return "elementwise_norm"
    return "other_unclassified"


def _contains(scope: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    return (scope["globalTid"] == event["globalTid"] and
            scope["start"] <= event["start"] and event["end"] <= scope["end"])


def attribute_activities(
        ranges: Sequence[Mapping[str, Any]],
        runtimes: Sequence[Mapping[str, Any]],
        activities: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Assign each correlated device activity exactly once to one forward."""
    forwards = sorted(
        (row for row in ranges if row["text"] == "SPN_PROFILE_INFERENCE"),
        key=lambda row: row["start"])
    operators = [row for row in ranges if str(row["text"]).startswith("aten::")]
    propagation = [row for row in ranges if row["text"] == "SPN_PROPAGATION"]
    runtime_by_correlation = {
        row["correlationId"]: row for row in runtimes
        if row.get("correlationId") is not None
    }
    output: List[List[Dict[str, Any]]] = [[] for _ in forwards]

    for activity in activities:
        runtime = runtime_by_correlation.get(activity.get("correlationId"))
        if runtime is None:
            continue
        forward_indices = [
            index for index, scope in enumerate(forwards)
            if _contains(scope, runtime)
        ]
        if len(forward_indices) != 1:
            continue
        if any(_contains(scope, runtime) for scope in propagation):
            assigned = "propagation"
            operator = "SPN_PROPAGATION"
        elif activity["activity"] in ("memcpy", "memset"):
            assigned = "layout_memory"
            operator = activity["name"]
        else:
            eligible = [scope for scope in operators if _contains(scope, runtime)]
            if eligible:
                innermost = min(eligible, key=lambda row: row["end"] - row["start"])
                operator = innermost["text"]
                assigned = category(operator.split(",", 1)[0])
            else:
                operator = activity["name"]
                assigned = category(operator)
        output[forward_indices[0]].append({
            **activity,
            "operator": operator,
            "category": assigned,
            "duration_ns": activity["end"] - activity["start"],
        })
    return output


def _rows(connection: sqlite3.Connection, query: str) -> List[Dict[str, Any]]:
    return [dict(row) for row in connection.execute(query)]


def _union_ns(rows: Iterable[Mapping[str, Any]]) -> int:
    intervals = sorted((row["start"], row["end"]) for row in rows)
    if not intervals:
        return 0
    total = 0
    low, high = intervals[0]
    for start, end in intervals[1:]:
        if start <= high:
            high = max(high, end)
        else:
            total += high - low
            low, high = start, end
    return total + high - low


def analyze(sqlite_path: Path, events_path: Path) -> Dict[str, Any]:
    events = json.loads(events_path.read_text(encoding="utf-8"))
    with sqlite3.connect(sqlite_path) as connection:
        connection.row_factory = sqlite3.Row
        ranges = _rows(connection,
                       "SELECT start,end,text,globalTid FROM NVTX_EVENTS "
                       "WHERE eventType=59 AND end IS NOT NULL")
        runtimes = _rows(connection,
                         "SELECT start,end,correlationId,globalTid "
                         "FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        activities = _rows(connection, """
            SELECT start,end,correlationId,s.value AS name,'kernel' AS activity
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.shortName
            UNION ALL
            SELECT start,end,correlationId,'[CUDA memcpy]','memcpy'
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            UNION ALL
            SELECT start,end,correlationId,'[CUDA memset]','memset'
            FROM CUPTI_ACTIVITY_KIND_MEMSET
            ORDER BY start
        """)

    attributed = attribute_activities(ranges, runtimes, activities)
    event_times = events["captured_cuda_event_ms"]
    if len(attributed) != len(event_times):
        raise RuntimeError(
            f"forward ranges ({len(attributed)}) != event timings ({len(event_times)})")
    per_forward = []
    for index, (rows, event_ms) in enumerate(zip(attributed, event_times), start=1):
        sums = defaultdict(int)
        counts = defaultdict(int)
        unclassified_names = defaultdict(int)
        for row in rows:
            sums[row["category"]] += row["duration_ns"]
            counts[row["category"]] += 1
            if row["category"] == "other_unclassified":
                unclassified_names[row["operator"]] += row["duration_ns"]
        union_ns = _union_ns(rows)
        sum_ns = sum(sums.values())
        per_forward.append({
            "iteration": index,
            "instrumented_cuda_event_ms": event_ms,
            "gpu_activity_sum_ms": sum_ns / 1e6,
            "gpu_busy_union_ms": union_ns / 1e6,
            "categories_ms": {
                key: sums.get(key, 0) / 1e6 for key in CATEGORIES
            },
            "category_activity_count": {
                key: counts.get(key, 0) for key in CATEGORIES
            },
            "other_unclassified_names_ms": {
                key: value / 1e6 for key, value in sorted(
                    unclassified_names.items(), key=lambda item: item[1], reverse=True)
            },
        })

    means = {
        key: sum(row["categories_ms"][key] for row in per_forward) /
        len(per_forward) for key in CATEGORIES
    }
    kernel_sum = sum(means.values())
    busy_union = sum(row["gpu_busy_union_ms"] for row in per_forward) / len(per_forward)
    return {
        "schema_version": 2,
        "model": events["model"],
        # The first Jetson capture predates the explicit board field. New
        # captures always provide it (including the RTX 3080 Ti run).
        "board": events.get("board", "Jetson AGX Xavier"),
        "precision": events["precision"],
        "sample_index": events["sample_index"],
        "input_resolution": events["input_resolution"],
        "profiled_forward_count": len(per_forward),
        "method": (
            "Nsight Systems CUPTI device activity durations; each kernel/memcpy/"
            "memset is mapped through correlationId to its CUDA runtime launch and "
            "then to the innermost aten NVTX range; propagation scope takes precedence"
        ),
        "semantics": (
            "Category milliseconds are direct device activity durations from the "
            "instrumented pass. They are exclusive and unscaled. They describe GPU "
            "service time, not an additive decomposition of uninstrumented end-to-end "
            "forward latency; profiler CPU overhead and launch gaps are not assigned."
        ),
        "instrumented_cuda_event_mean_ms": sum(event_times) / len(event_times),
        "gpu_activity_sum_mean_ms": kernel_sum,
        "gpu_busy_union_mean_ms": busy_union,
        "category_mean_gpu_activity_ms": means,
        "category_share_of_gpu_activity_sum_percent": {
            key: (100.0 * value / kernel_sum if kernel_sum else 0.0)
            for key, value in means.items()
        },
        "per_forward": per_forward,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
