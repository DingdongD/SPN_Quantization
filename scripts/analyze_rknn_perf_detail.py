#!/usr/bin/env python3
"""Summarize RKNN_QUERY_PERF_DETAIL into exclusive hardware categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple


CATEGORIES = (
    "dense_compute",
    "attention",
    "sampling",
    "propagation",
    "elementwise_norm",
    "layout_memory",
    "other_unclassified",
)
TARGETS = ("CPU", "NPU")
PROPAGATION_SCOPES = {
    "cspn": ("post_process_layer",),
    "dyspn": ("dyspn_6_5",),
    "nlspn": ("prop_layer",),
    "completionformer": ("prop_layer",),
}


def _metadata_value(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        return value


def parse_perf_detail(text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse the fixed-width table emitted by RKNN runtime 2.x."""
    lines = text.splitlines()
    metadata: Dict[str, Any] = {}
    for line in lines:
        if line == "perf_detail_begin":
            break
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = _metadata_value(value.strip())

    try:
        header_index = next(
            index for index, line in enumerate(lines)
            if line.startswith("ID") and "OpType" in line and "FullName" in line)
    except StopIteration as error:
        raise ValueError("RKNN performance detail table header not found") from error

    row_pattern = re.compile(
        r"^\s*(?P<id>\d+)\s+(?P<op_type>\S+)\s+"
        r"(?P<data_type>\S+)\s+(?P<target>CPU|NPU)\s+"
        r"(?P<input_shape>\S+)\s+(?P<output_shape>\S+)\s+"
        r"(?P<cycles>\S+)\s+(?P<time_us>\d+)\s*(?P<tail>.*)$")
    rows: List[Dict[str, Any]] = []
    for line in lines[header_index + 1:]:
        if line == "perf_detail_end":
            break
        if not line or set(line) == {"-"}:
            continue
        match = row_pattern.match(line)
        if match is None:
            continue
        row = match.groupdict()
        tail = row.pop("tail").split()
        if tail and "%" not in tail[0]:
            row["mac_usage"] = tail.pop(0)
        else:
            row["mac_usage"] = ""
        if len(tail) < 2 or "%" not in tail[0]:
            raise ValueError(f"malformed RKNN performance row: {line}")
        row["workload"] = tail.pop(0)
        row["rw_kb"] = tail.pop(0)
        row["full_name"] = " ".join(tail)
        row["id"] = int(row["id"])
        row["time_us"] = int(row["time_us"])
        rows.append(row)
    if not rows:
        raise ValueError("RKNN performance detail contains no operator rows")
    return metadata, rows


def category(
        row: Mapping[str, Any], model: str,
        propagation_id_range: Tuple[int, int] = None) -> str:
    op_type = str(row["op_type"]).lower()
    full_name = str(row.get("full_name", "")).lower()
    combined = f"{op_type} {full_name}"

    if (propagation_id_range is not None and
            propagation_id_range[0] <= int(row["id"]) <=
            propagation_id_range[1]):
        return "propagation"
    if any(scope in full_name for scope in PROPAGATION_SCOPES[model]):
        return "propagation"
    if any(token in combined for token in (
            "attention", "/attn", ".attn", "softmax")):
        return "attention"
    if any(token in combined for token in (
            "gridsample", "grid_sample", "gather", "deform", "resize",
            "interpolate", "upsample", "maxpool", "max_pool")):
        return "sampling"
    if any(token in op_type for token in (
            "conv", "matmul", "gemm", "fullyconnected", "linear")):
        return "dense_compute"
    if any(token in op_type for token in (
            "relu", "sigmoid", "tanh", "batchnormalization",
            "layernormalization", "normalize", "norm", "add", "sub",
            "mul", "div", "sqrt", "exp", "reduce", "where", "clip",
            "less", "greater")):
        return "elementwise_norm"
    if any(token in op_type for token in (
            "concat", "reshape", "transpose", "permute", "copy", "slice",
            "split", "cast", "flatten", "squeeze", "unsqueeze", "pad",
            "tile", "expand")):
        return "layout_memory"
    return "other_unclassified"


def analyze_text(
        text: str, model: str,
        propagation_id_range: Tuple[int, int] = None) -> Dict[str, Any]:
    if model not in PROPAGATION_SCOPES:
        raise ValueError(f"unsupported model: {model}")
    metadata, rows = parse_perf_detail(text)
    category_time = {name: 0 for name in CATEGORIES}
    category_count = {name: 0 for name in CATEGORIES}
    target_time = {target: 0 for target in TARGETS}
    target_count = {target: 0 for target in TARGETS}
    category_target_time = {
        name: {target: 0 for target in TARGETS} for name in CATEGORIES
    }
    cpu_operators = []
    for row in rows:
        assigned = category(row, model, propagation_id_range)
        target = row["target"]
        if target not in target_time:
            target_time[target] = 0
            target_count[target] = 0
            for values in category_target_time.values():
                values[target] = 0
        time_us = int(row["time_us"])
        category_time[assigned] += time_us
        category_count[assigned] += 1
        target_time[target] += time_us
        target_count[target] += 1
        category_target_time[assigned][target] += time_us
        if target == "CPU":
            cpu_operators.append({
                "id": row["id"],
                "op_type": row["op_type"],
                "time_us": time_us,
                "full_name": row["full_name"],
                "category": assigned,
            })

    total = sum(category_time.values())
    return {
        "schema_version": 1,
        "model": model,
        "precision": "fp16",
        "method": (
            "Direct per-operator times and compiler targets from "
            "RKNN_QUERY_PERF_DETAIL with RKNN_FLAG_COLLECT_PERF_MASK"
        ),
        "semantics": (
            "Times come from a single instrumented RKNN run. Categories are "
            "exclusive and unscaled; their sum is not substituted for the "
            "uninstrumented full-forward latency. Propagation scope takes "
            "precedence over primitive operator type."
        ),
        "runtime_metadata": metadata,
        "propagation_selector": {
            "scope_tokens": list(PROPAGATION_SCOPES[model]),
            "operator_id_range_inclusive": (
                list(propagation_id_range)
                if propagation_id_range is not None else None),
        },
        "operator_count": len(rows),
        "operator_time_sum_us": total,
        "category_time_us": category_time,
        "category_operator_count": category_count,
        "category_share_percent": {
            name: (100.0 * value / total if total else 0.0)
            for name, value in category_time.items()
        },
        "target_time_us": target_time,
        "target_operator_count": target_count,
        "target_share_percent": {
            target: (100.0 * value / total if total else 0.0)
            for target, value in target_time.items()
        },
        "category_target_time_us": category_target_time,
        "placement": {
            "cpu_operator_count": target_count.get("CPU", 0),
            "npu_operator_count": target_count.get("NPU", 0),
            "cpu_operators": cpu_operators,
        },
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(PROPAGATION_SCOPES), required=True)
    parser.add_argument("--propagation-start-id", type=int)
    parser.add_argument("--propagation-end-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.propagation_start_id is None) != \
            (args.propagation_end_id is None):
        parser.error("propagation start/end IDs must be provided together")
    propagation_range = None
    if args.propagation_start_id is not None:
        if args.propagation_start_id > args.propagation_end_id:
            parser.error("propagation start ID exceeds end ID")
        propagation_range = (
            args.propagation_start_id, args.propagation_end_id)
    result = analyze_text(
        args.input.read_text(encoding="utf-8", errors="replace"), args.model,
        propagation_id_range=propagation_range)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
