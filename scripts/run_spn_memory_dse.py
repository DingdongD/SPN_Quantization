#!/usr/bin/env python3
"""Run and validate the constrained four-model SPN memory/array DSE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.run_cacti_sram import CACTI_REVISION  # noqa: E402
from spn_quant.hardware.array_dse import ArrayDSEConfig  # noqa: E402
from spn_quant.hardware.cacti import CactiSramResult  # noqa: E402
from spn_quant.hardware.llmcompass_adapter import (  # noqa: E402
    LLMCOMPASS_REVISION,
    ArrayShape,
)
from spn_quant.hardware.memory_dse import (  # noqa: E402
    MemoryArchitecture,
    MemorySimulationResult,
    search_memory_mapping,
)
from spn_quant.hardware.workload import (  # noqa: E402
    WorkloadProfile,
    lower_layer_to_gemms,
)


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git",) + tuple(arguments), cwd=str(REPO_ROOT), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.stdout.strip()


def _write_csv(path: Path, fieldnames: Sequence[str],
               rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)


def _reference(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    repository_path = (REPO_ROOT / path).resolve()
    if repository_path.exists():
        return repository_path
    return (Path(config_path).resolve().parent / path).resolve()


def _load_study(config_path: Path,
                cacti_result: Any = None) -> Tuple[
        Mapping[str, Any], MemoryArchitecture, ArrayDSEConfig, Path, Path,
        Mapping[str, Any]]:
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cacti_path = (Path(cacti_result).resolve() if cacti_result is not None else
                  _reference(config_path, str(payload["cacti_result"])))
    compute_path = _reference(config_path, str(payload["compute_dse_config"]))
    cacti_payload = json.loads(cacti_path.read_text(encoding="utf-8"))
    if cacti_payload.get("cacti_revision") != CACTI_REVISION:
        raise ValueError("CACTI result revision does not match the pinned tool")
    cacti = CactiSramResult(**cacti_payload["result"])
    architecture = MemoryArchitecture.from_dict(payload, cacti)
    compute = ArrayDSEConfig.read_json(compute_path)
    if compute.dtype != architecture.dtype or compute.word_bytes != \
            architecture.operand_bytes or compute.dataflow != "os" or \
            compute.mac_per_cycle != 1:
        raise ValueError("compute and memory DSE configurations disagree")
    return (payload, architecture, compute, cacti_path, compute_path,
            cacti_payload)


def _simulate(
        profiles: Mapping[str, WorkloadProfile], compute: ArrayDSEConfig,
        architecture: MemoryArchitecture
        ) -> Tuple[Tuple[str, MemorySimulationResult], ...]:
    rows = []
    for model in MODEL_NAMES:
        for array in compute.array_shapes:
            for scenario in architecture.scenarios:
                for layer in profiles[model].layers:
                    for workload in lower_layer_to_gemms(layer):
                        rows.append((model, search_memory_mapping(
                            workload, array, architecture, scenario)))
    return tuple(rows)


def _layer_row(model: str, result: MemorySimulationResult,
               architecture: MemoryArchitecture) -> Dict[str, Any]:
    work, array, mapping = result.workload, result.array, result.mapping
    resource_floor = max(
        result.compute_cycles, result.activation_sram_cycles,
        result.weight_sram_cycles, result.dram_cycles)
    return {
        "model": model,
        "scenario": result.scenario.label,
        "dram_efficiency": result.scenario.efficiency,
        "primary_scenario": int(result.scenario.primary),
        "dram_bandwidth_bytes_per_second":
            result.scenario.bandwidth_bytes_per_second,
        "array_height": array.height,
        "array_width": array.width,
        "array_label": array.label,
        "pe_count": array.pe_count,
        "execution_index": work.execution_index,
        "layer": work.name,
        "kind": work.kind,
        "m": work.m,
        "n": work.n,
        "k": work.k,
        "repetitions": work.repetitions,
        "macs": work.macs,
        "tile_m": mapping.tile_m,
        "tile_n": mapping.tile_n,
        "tile_k": mapping.tile_k,
        "loop_order": mapping.loop_order,
        "double_buffered": int(mapping.double_buffered),
        "activation_residency": mapping.activation_residency,
        "weight_resident": int(mapping.weight_resident),
        "activation_capacity_bytes": mapping.activation_capacity_bytes,
        "weight_capacity_bytes": mapping.weight_capacity_bytes,
        "activation_capacity_fraction":
            mapping.activation_capacity_bytes /
            float(architecture.activation_buffer_bytes),
        "weight_capacity_fraction":
            mapping.weight_capacity_bytes /
            float(architecture.weight_buffer_bytes),
        "activation_dram_bytes": result.activation_dram_bytes,
        "weight_dram_bytes": result.weight_dram_bytes,
        "output_dram_bytes": result.output_dram_bytes,
        "partial_sum_dram_bytes": result.partial_sum_dram_bytes,
        "total_dram_bytes": result.total_dram_bytes,
        "activation_sram_bytes": result.activation_sram_bytes,
        "weight_sram_bytes": result.weight_sram_bytes,
        "compute_cycles": result.compute_cycles,
        "activation_sram_cycles": result.activation_sram_cycles,
        "weight_sram_cycles": result.weight_sram_cycles,
        "dram_cycles": result.dram_cycles,
        "resource_floor_cycles": resource_floor,
        "startup_cycles": result.startup_cycles,
        "drain_cycles": result.drain_cycles,
        "cycles": result.cycles,
        "latency_us": result.cycles / architecture.clock_hz * 1e6,
        "effective_array_utilization":
            work.macs / float(result.cycles * array.pe_count),
        "arithmetic_intensity_mac_per_dram_byte":
            work.macs / float(result.total_dram_bytes),
        "bottleneck": result.bottleneck,
    }


def _group(rows: Sequence[Mapping[str, Any]],
           fields: Sequence[str]) -> Dict[Tuple[Any, ...], list]:
    grouped: Dict[Tuple[Any, ...], list] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def _summary_rows(
        layer_rows: Sequence[Mapping[str, Any]],
        profiles: Mapping[str, WorkloadProfile],
        architecture: MemoryArchitecture) -> Tuple[Dict[str, Any], ...]:
    grouped = _group(
        layer_rows, ("model", "scenario", "array_height", "array_width"))
    summaries = []
    for (model, scenario, height, width), rows in sorted(grouped.items()):
        cycles = sum(int(row["cycles"]) for row in rows)
        compute_cycles = sum(int(row["compute_cycles"]) for row in rows)
        dense_macs = sum(int(row["macs"]) for row in rows)
        by_bottleneck: Dict[str, int] = {}
        for row in rows:
            name = str(row["bottleneck"])
            by_bottleneck[name] = by_bottleneck.get(name, 0) + int(row["cycles"])
        dominant = max(by_bottleneck, key=lambda name: (
            by_bottleneck[name], name))
        summaries.append({
            "model": model,
            "scenario": scenario,
            "dram_efficiency": float(rows[0]["dram_efficiency"]),
            "primary_scenario": int(rows[0]["primary_scenario"]),
            "array_height": int(height),
            "array_width": int(width),
            "array_label": "%sx%s" % (height, width),
            "pe_count": int(height) * int(width),
            "dense_macs": dense_macs,
            "irregular_macs": profiles[model].irregular_macs,
            "cycles": cycles,
            "latency_ms": cycles / architecture.clock_hz * 1e3,
            "compute_cycles": compute_cycles,
            "memory_slowdown": cycles / float(compute_cycles),
            "effective_array_utilization":
                dense_macs / float(cycles * int(height) * int(width)),
            "activation_dram_bytes": sum(
                int(row["activation_dram_bytes"]) for row in rows),
            "weight_dram_bytes": sum(
                int(row["weight_dram_bytes"]) for row in rows),
            "output_dram_bytes": sum(
                int(row["output_dram_bytes"]) for row in rows),
            "partial_sum_dram_bytes": sum(
                int(row["partial_sum_dram_bytes"]) for row in rows),
            "total_dram_bytes": sum(
                int(row["total_dram_bytes"]) for row in rows),
            "activation_sram_bytes": sum(
                int(row["activation_sram_bytes"]) for row in rows),
            "weight_sram_bytes": sum(
                int(row["weight_sram_bytes"]) for row in rows),
            "dominant_bottleneck": dominant,
            "gemm_count": len(rows),
        })
    return tuple(summaries)


def _ranking_rows(
        summaries: Sequence[Mapping[str, Any]],
        compute: ArrayDSEConfig) -> Tuple[Dict[str, Any], ...]:
    grouped = _group(summaries, ("scenario", "array_height", "array_width"))
    candidates = []
    for (scenario, height, width), rows in grouped.items():
        values = [float(row["effective_array_utilization"]) for row in rows]
        candidates.append({
            "scenario": scenario,
            "dram_efficiency": float(rows[0]["dram_efficiency"]),
            "primary_scenario": int(rows[0]["primary_scenario"]),
            "array_height": int(height),
            "array_width": int(width),
            "array_label": "%sx%s" % (height, width),
            "pe_count": int(height) * int(width),
            "model_count": len(rows),
            "total_cycles": sum(int(row["cycles"]) for row in rows),
            "total_latency_ms": sum(float(row["latency_ms"]) for row in rows),
            "geometric_mean_effective_utilization": math.exp(
                sum(math.log(value) for value in values) / len(values)),
            "best_within_iso_pe_group": 0,
        })
    by_scenario = _group(candidates, ("scenario",))
    output = []
    for _, rows in sorted(by_scenario.items()):
        iso_best = {}
        for pe_count, shapes in compute.iso_pe_groups:
            shape_set = {(shape.height, shape.width) for shape in shapes}
            eligible = [row for row in rows if (
                row["array_height"], row["array_width"]) in shape_set]
            if eligible:
                iso_best[int(pe_count)] = min(
                    eligible, key=lambda row: (
                        row["total_cycles"], -row["geometric_mean_effective_utilization"],
                        row["array_height"], row["array_width"]))["array_label"]
        ordered = sorted(rows, key=lambda row: (
            row["total_cycles"], row["pe_count"],
            row["array_height"], row["array_width"]))
        for rank, row in enumerate(ordered, 1):
            item = dict(row)
            item["rank"] = rank
            item["best_within_iso_pe_group"] = int(
                iso_best.get(item["pe_count"]) == item["array_label"])
            output.append(item)
    return tuple(output)


def _bottleneck_rows(
        layer_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    candidate_groups = _group(
        layer_rows, ("model", "scenario", "array_height", "array_width"))
    output = []
    for key, candidate_rows in sorted(candidate_groups.items()):
        total_cycles = sum(int(row["cycles"]) for row in candidate_rows)
        total_macs = sum(int(row["macs"]) for row in candidate_rows)
        bottlenecks = _group(candidate_rows, ("bottleneck",))
        for (bottleneck,), rows in sorted(bottlenecks.items()):
            output.append({
                "model": key[0], "scenario": key[1],
                "array_height": key[2], "array_width": key[3],
                "array_label": "%sx%s" % (key[2], key[3]),
                "bottleneck": bottleneck,
                "layer_count": len(rows),
                "layer_fraction": len(rows) / float(len(candidate_rows)),
                "cycle_fraction": sum(int(row["cycles"]) for row in rows) /
                    float(total_cycles),
                "mac_fraction": sum(int(row["macs"]) for row in rows) /
                    float(total_macs),
            })
    return tuple(output)


def _pareto_rows(
        summaries: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    output = []
    grouped = _group(summaries, ("model", "scenario"))
    for (model, scenario), rows in sorted(grouped.items()):
        for row in rows:
            dominated = any(
                int(other["pe_count"]) <= int(row["pe_count"]) and
                int(other["cycles"]) <= int(row["cycles"]) and
                (int(other["pe_count"]) < int(row["pe_count"]) or
                 int(other["cycles"]) < int(row["cycles"]))
                for other in rows if other is not row)
            if not dominated:
                output.append({
                    "model": model, "scenario": scenario,
                    "dram_efficiency": row["dram_efficiency"],
                    "array_height": row["array_height"],
                    "array_width": row["array_width"],
                    "array_label": row["array_label"],
                    "pe_count": row["pe_count"],
                    "cycles": row["cycles"],
                    "latency_ms": row["latency_ms"],
                    "effective_array_utilization":
                        row["effective_array_utilization"],
                })
    return tuple(output)


def _sensitivity_rows(
        summaries: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    primary = {
        (row["model"], row["array_height"], row["array_width"]): row
        for row in summaries if int(row["primary_scenario"])}
    return tuple({
        "model": row["model"], "scenario": row["scenario"],
        "dram_efficiency": row["dram_efficiency"],
        "array_height": row["array_height"],
        "array_width": row["array_width"],
        "array_label": row["array_label"],
        "pe_count": row["pe_count"], "cycles": row["cycles"],
        "latency_ms": row["latency_ms"],
        "cycles_relative_to_primary": int(row["cycles"]) / float(primary[
            (row["model"], row["array_height"], row["array_width"])]["cycles"]),
        "dominant_bottleneck": row["dominant_bottleneck"],
    } for row in summaries)


def _plot(path: Path, summaries: Sequence[Mapping[str, Any]],
          compute: ArrayDSEConfig) -> None:
    primary = [row for row in summaries if int(row["primary_scenario"])]
    by_model = {
        model: {(int(row["array_height"]), int(row["array_width"])): row
                for row in primary if row["model"] == model}
        for model in MODEL_NAMES}
    x = list(range(len(compute.array_shapes)))
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for model in MODEL_NAMES:
        axis.plot(x, [float(by_model[model][
            (shape.height, shape.width)]["latency_ms"])
            for shape in compute.array_shapes], marker="o", label=model)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [shape.label for shape in compute.array_shapes], rotation=45, ha="right")
    axis.set_xlabel("Array shape (height x width)")
    axis.set_ylabel("Dense mapped latency at 70% LPDDR4 efficiency (ms)")
    axis.set_yscale("log")
    axis.grid(True, which="both", axis="y", alpha=0.3)
    axis.legend(ncol=4, fontsize=9)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def run(config_path: Path, workload_paths: Mapping[str, Path],
        output: Path, cacti_result: Any = None) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError("DSE output already exists: %s" % output)
    if set(workload_paths) != set(MODEL_NAMES):
        raise ValueError("workload paths must cover exactly four SPN models")
    (config_payload, architecture, compute, cacti_path, compute_path,
     cacti_payload) = _load_study(config_path, cacti_result)
    resolved_workloads = {
        model: Path(workload_paths[model]).resolve() for model in MODEL_NAMES}
    profiles = {
        model: WorkloadProfile.read_json(resolved_workloads[model])
        for model in MODEL_NAMES}
    for model, profile in profiles.items():
        if profile.model_name != model:
            raise ValueError("workload model identity mismatch: %s" % model)
    revision = _git("rev-parse", "HEAD")
    dirty_before_output = bool(_git("status", "--porcelain"))
    simulations = _simulate(profiles, compute, architecture)
    layer_rows = tuple(
        _layer_row(model, result, architecture)
        for model, result in simulations)
    summaries = _summary_rows(layer_rows, profiles, architecture)
    rankings = _ranking_rows(summaries, compute)
    bottlenecks = _bottleneck_rows(layer_rows)
    frontier = _pareto_rows(summaries)
    sensitivity = _sensitivity_rows(summaries)

    output.mkdir(parents=True)
    _write_csv(
        output / "layer_memory_results.csv", tuple(layer_rows[0]), layer_rows)
    _write_csv(
        output / "model_memory_summary.csv", tuple(summaries[0]), summaries)
    _write_csv(
        output / "candidate_memory_ranking.csv", tuple(rankings[0]), rankings)
    _write_csv(
        output / "bottleneck_summary.csv", tuple(bottlenecks[0]), bottlenecks)
    _write_csv(
        output / "memory_pareto_frontier.csv", tuple(frontier[0]), frontier)
    _write_csv(
        output / "bandwidth_sensitivity.csv", tuple(sensitivity[0]), sensitivity)
    _plot(output / "memory_shape_latency.png", summaries, compute)
    metadata = {
        "format_version": 1,
        "repository_revision": revision,
        "repository_dirty_before_output": dirty_before_output,
        "llmcompass_revision": LLMCOMPASS_REVISION,
        "cacti_revision": cacti_payload["cacti_revision"],
        "cacti_compatibility_patch": cacti_payload["compatibility_patch"],
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": _sha256(Path(config_path).resolve()),
        "config": config_payload,
        "compute_config_path": str(compute_path),
        "compute_config_sha256": _sha256(compute_path),
        "cacti_result_path": str(cacti_path),
        "cacti_result_sha256": _sha256(cacti_path),
        "cacti_result": cacti_payload["result"],
        "input_paths": {
            model: str(resolved_workloads[model]) for model in MODEL_NAMES},
        "input_sha256": {
            model: _sha256(resolved_workloads[model]) for model in MODEL_NAMES},
        "array_shapes": [[shape.height, shape.width]
                         for shape in compute.array_shapes],
        "scenario_efficiencies": list(architecture.dram_efficiencies),
        "primary_dram_efficiency":
            architecture.primary_dram_efficiency,
        "derived": {
            "sram_bytes_per_cycle": architecture.sram_bytes_per_cycle,
            "sram_bandwidth_bytes_per_second":
                architecture.sram_bandwidth_bytes_per_second,
            "lpddr_peak_bandwidth_bytes_per_second":
                architecture.lpddr_peak_bandwidth_bytes_per_second,
            "two_sram_macro_area_mm2": 2 * architecture.cacti.area_mm2,
        },
        "schedule": {
            "single_buffered":
                "compute + dram + max(activation_sram, weight_sram)",
            "double_buffered":
                "max(compute, dram, activation_sram, weight_sram) + startup + drain",
        },
        "limitations": [
            "Dense GEMMs only; propagation, deformable sampling, and vector ops are excluded.",
            "One x32 LPDDR4-4266 channel and its sustained efficiency are assumptions.",
            "CACTI 32 nm SRAM timing and area are illustrative, not signoff macros.",
            "No inter-layer activation retention or operator fusion is modelled.",
        ],
        "confidence": {
            "capacity_and_aggregate_bandwidth_feasibility": "MEDIUM",
            "cycle_accurate_whole_model_latency": "LOW",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return output


def _read_csv(path: Path) -> Tuple[Dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def validate_artifacts(output: Path) -> str:
    output = Path(output)
    layers = _read_csv(output / "layer_memory_results.csv")
    summaries = _read_csv(output / "model_memory_summary.csv")
    rankings = _read_csv(output / "candidate_memory_ranking.csv")
    bottlenecks = _read_csv(output / "bottleneck_summary.csv")
    frontier = _read_csv(output / "memory_pareto_frontier.csv")
    sensitivity = _read_csv(output / "bandwidth_sensitivity.csv")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if not all((layers, summaries, rankings, bottlenecks, frontier, sensitivity)):
        raise ValueError("memory DSE artifacts contain an empty required table")
    config = metadata["config"]
    activation_capacity = int(config["activation_buffer_bytes"])
    weight_capacity = int(config["weight_buffer_bytes"])
    layer_groups: Dict[Tuple[str, str, int, int], list] = {}
    for row in layers:
        if int(row["activation_capacity_bytes"]) > activation_capacity:
            raise ValueError("layer mapping violates activation capacity")
        if int(row["weight_capacity_bytes"]) > weight_capacity:
            raise ValueError("layer mapping violates weight capacity")
        floors = (
            int(row["compute_cycles"]), int(row["activation_sram_cycles"]),
            int(row["weight_sram_cycles"]), int(row["dram_cycles"]))
        if int(row["cycles"]) < max(floors):
            raise ValueError("layer schedule violates a resource floor")
        expected_dram = sum(int(row[field]) for field in (
            "activation_dram_bytes", "weight_dram_bytes",
            "output_dram_bytes", "partial_sum_dram_bytes"))
        if expected_dram != int(row["total_dram_bytes"]):
            raise ValueError("layer DRAM traffic does not match its breakdown")
        key = (row["model"], row["scenario"], int(row["array_height"]),
               int(row["array_width"]))
        layer_groups.setdefault(key, []).append(row)
    expected_candidates = (
        len(MODEL_NAMES) * len(metadata["array_shapes"]) *
        len(metadata["scenario_efficiencies"]))
    if len(summaries) != expected_candidates:
        raise ValueError("model summary does not cover the complete DSE")
    summary_map = {}
    for row in summaries:
        key = (row["model"], row["scenario"], int(row["array_height"]),
               int(row["array_width"]))
        detail = layer_groups.get(key)
        if not detail:
            raise ValueError("model summary has no layer rows")
        if (sum(int(item["cycles"]) for item in detail) != int(row["cycles"]) or
                sum(int(item["macs"]) for item in detail) !=
                int(row["dense_macs"]) or
                sum(int(item["total_dram_bytes"]) for item in detail) !=
                int(row["total_dram_bytes"])):
            raise ValueError("model summary totals do not match layer rows")
        summary_map[key] = row
    ranking_groups = _group(rankings, ("scenario",))
    for (scenario,), rows in ranking_groups.items():
        ranks = sorted(int(row["rank"]) for row in rows)
        if ranks != list(range(1, len(rows) + 1)):
            raise ValueError("candidate ranking is not contiguous")
        for row in rows:
            height, width = int(row["array_height"]), int(row["array_width"])
            model_rows = [summary_map[(model, scenario, height, width)]
                          for model in MODEL_NAMES]
            if sum(int(item["cycles"]) for item in model_rows) != \
                    int(row["total_cycles"]):
                raise ValueError("candidate ranking totals do not match summaries")
            values = [float(item["effective_array_utilization"])
                      for item in model_rows]
            expected = math.exp(sum(math.log(value) for value in values) /
                                len(values))
            if not math.isclose(
                    expected,
                    float(row["geometric_mean_effective_utilization"]),
                    rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("candidate ranking utilization mismatch")
    if len(sensitivity) != len(summaries):
        raise ValueError("bandwidth sensitivity table is incomplete")
    if metadata.get("cacti_revision") != CACTI_REVISION or \
            metadata.get("llmcompass_revision") != LLMCOMPASS_REVISION:
        raise ValueError("tool provenance does not match pinned revisions")
    return "SPN_MEMORY_DSE_VALIDATION_OK"


def _parse_workloads(values: Sequence[str]) -> Dict[str, Path]:
    result = {}
    for value in values:
        model, separator, path = value.partition("=")
        if not separator or model not in MODEL_NAMES or not path:
            raise ValueError("workload must use model=/path/to/profile.json")
        if model in result:
            raise ValueError("duplicate workload model: %s" % model)
        result[model] = Path(path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cacti-result", type=Path)
    parser.add_argument("--validate-only", type=Path)
    return parser


def main(argv: Any = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only is not None:
        if (args.config is not None or args.workload or args.output is not None or
                args.cacti_result is not None):
            raise ValueError("--validate-only cannot be combined with run inputs")
        print(validate_artifacts(args.validate_only))
        return 0
    if args.config is None or args.output is None:
        raise ValueError("--config and --output are required for a DSE run")
    output = run(
        args.config, _parse_workloads(args.workload), args.output,
        cacti_result=args.cacti_result)
    print(validate_artifacts(output))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
