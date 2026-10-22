#!/usr/bin/env python3
"""Run a joint array/SRAM/LPDDR latency-area-energy DSE for four SPNs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from spn_quant.hardware.joint_dse import (  # noqa: E402
    ArchitecturePoint,
    CandidateEstimate,
    CostAssumptions,
    DesignConstraints,
    LayerEstimate,
    aggregate_candidate,
    aggregate_model,
    architecture_area,
    estimate_layer,
    normalize_candidate_fairness,
    pareto_frontier,
)
from spn_quant.hardware.llmcompass_adapter import LLMCOMPASS_REVISION  # noqa: E402
from spn_quant.hardware.memory_dse import (  # noqa: E402
    MemoryArchitecture,
    search_memory_mapping,
)
from spn_quant.hardware.workload import (  # noqa: E402
    WorkloadProfile,
    lower_layer_to_gemms,
)


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")
REQUIRED_ARCHITECTURE_CLASSES = frozenset(
    ("conservative", "balanced", "aggressive"))


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


def _reference(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    repository_path = (REPO_ROOT / path).resolve()
    if repository_path.exists():
        return repository_path
    return (Path(config_path).resolve().parent / path).resolve()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]],
               fieldnames: Any = None) -> None:
    fields = tuple(rows[0]) if rows else tuple(fieldnames or ())
    if not fields:
        raise ValueError("CSV schema is required for an empty table: %s" % path)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)


def _cost_assumptions(payload: Mapping[str, Any]) -> CostAssumptions:
    return CostAssumptions(**payload)


def _constraints(payload: Mapping[str, Any]) -> DesignConstraints:
    latency = payload.get("max_model_latency_ms", {})
    return DesignConstraints(
        max_total_area_mm2=payload.get("max_total_area_mm2"),
        max_model_average_dynamic_power_w=payload.get(
            "max_model_average_dynamic_power_w"),
        max_model_latency_ms=tuple(sorted(
            (str(name), float(value)) for name, value in latency.items())))


def _load_study(config_path: Path,
                sram_macro_paths: Any = None) -> Tuple[
        Mapping[str, Any], ArrayDSEConfig, CostAssumptions,
        DesignConstraints, Tuple[ArchitecturePoint, ...],
        Mapping[str, Path], Mapping[str, Mapping[str, Any]]]:
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("unsupported joint DSE config format")
    compute_path = _reference(config_path, payload["compute_dse_config"])
    compute = ArrayDSEConfig.read_json(compute_path)
    if (payload["dtype"] != compute.dtype or
            int(payload["operand_bytes"]) != compute.word_bytes or
            compute.dataflow != "os" or compute.mac_per_cycle != 1):
        raise ValueError("compute and joint DSE configurations disagree")
    assumptions = _cost_assumptions(payload["cost_model"])
    constraints = _constraints(payload.get("constraints", {}))

    configured_macro_paths = {
        name: _reference(config_path, value)
        for name, value in payload["sram_macros"].items()}
    if sram_macro_paths is None:
        macro_paths = configured_macro_paths
    else:
        macro_paths = {
            str(name): Path(path).resolve()
            for name, path in sram_macro_paths.items()}
        if set(macro_paths) != set(configured_macro_paths):
            raise ValueError(
                "SRAM macro overrides must cover every configured macro")
    if not macro_paths:
        raise ValueError("joint DSE requires SRAM macro results")
    macro_payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in macro_paths.items()}
    macros = {}
    for name, macro_payload in macro_payloads.items():
        if macro_payload.get("cacti_revision") != CACTI_REVISION:
            raise ValueError("CACTI result revision does not match pinned tool")
        technology = int(macro_payload["config"]["technology_nm"])
        if technology != assumptions.technology_nm:
            raise ValueError(
                "CACTI and logic technology node must match")
        macros[name] = CactiSramResult(**macro_payload["result"])

    profiles = tuple(payload["architecture_profiles"])
    if len(profiles) < 3:
        raise ValueError("joint DSE requires at least three architecture profiles")
    profile_ids = tuple(str(row["id"]) for row in profiles)
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("architecture profile identifiers must be unique")
    classes = frozenset(str(row["class"]) for row in profiles)
    if not REQUIRED_ARCHITECTURE_CLASSES.issubset(classes):
        raise ValueError(
            "profiles must include conservative, balanced, and aggressive classes")

    points = []
    lpddr = payload["lpddr4"]
    for profile in profiles:
        activation_name = str(profile["activation_sram"])
        weight_name = str(profile["weight_sram"])
        if activation_name not in macros or weight_name not in macros:
            raise ValueError("architecture profile references an unknown SRAM macro")
        activation, weight = macros[activation_name], macros[weight_name]
        efficiency = float(profile["sustained_efficiency"])
        memory_payload = {
            "format_version": 1,
            "dtype": payload["dtype"],
            "operand_bytes": payload["operand_bytes"],
            "output_bytes": payload["output_bytes"],
            "accumulator_bytes": payload["accumulator_bytes"],
            "clock_hz": payload["clock_hz"],
            "activation_buffer_bytes": activation.capacity_bytes,
            "weight_buffer_bytes": weight.capacity_bytes,
            "lpddr4": {
                "transfers_per_second": lpddr["transfers_per_second"],
                "bus_width_bits": lpddr["bus_width_bits"],
                "channels": profile["lpddr_channels"],
            },
            "dram_efficiencies": [efficiency],
            "primary_dram_efficiency": efficiency,
        }
        memory = MemoryArchitecture.from_dict(
            memory_payload, activation, weight)
        for array in compute.array_shapes:
            points.append(ArchitecturePoint(
                architecture_id="%s__%s" % (profile["id"], array.label),
                architecture_class=str(profile["class"]),
                array=array, memory=memory))
    return (payload, compute, assumptions, constraints, tuple(points),
            macro_paths, macro_payloads)


def _simulate_candidate(arguments: Tuple[
        str, WorkloadProfile, ArchitecturePoint, CostAssumptions]
        ) -> Tuple[LayerEstimate, ...]:
    model, profile, point, assumptions = arguments
    output = []
    for layer in profile.layers:
        for workload in lower_layer_to_gemms(layer):
            result = search_memory_mapping(
                workload, point.array, point.memory,
                point.memory.primary_scenario)
            output.append(estimate_layer(model, point, result, assumptions))
    if not output:
        raise ValueError("workload has no array-mappable GEMMs: %s" % model)
    return tuple(output)


def _simulate(
        profiles: Mapping[str, WorkloadProfile],
        points: Sequence[ArchitecturePoint], assumptions: CostAssumptions,
        workers: int) -> Tuple[LayerEstimate, ...]:
    if int(workers) < 1:
        raise ValueError("workers must be at least one")
    tasks = tuple(
        (model, profiles[model], point, assumptions)
        for model in MODEL_NAMES for point in points)
    if int(workers) == 1:
        chunks = map(_simulate_candidate, tasks)
        return tuple(row for chunk in chunks for row in chunk)
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        chunks = executor.map(_simulate_candidate, tasks, chunksize=1)
        return tuple(row for chunk in chunks for row in chunk)


def _layer_row(
        estimate: LayerEstimate, point: ArchitecturePoint) -> Dict[str, Any]:
    result, mapping = estimate.result, estimate.result.mapping
    memory = point.memory
    return {
        "model": estimate.model_name,
        "architecture_id": point.architecture_id,
        "architecture_class": point.architecture_class,
        "array_height": point.array.height,
        "array_width": point.array.width,
        "array_label": point.array.label,
        "pe_count": point.array.pe_count,
        "activation_buffer_bytes": memory.activation_buffer_bytes,
        "weight_buffer_bytes": memory.weight_buffer_bytes,
        "activation_port_bits": memory.activation_cacti.output_width_bits,
        "weight_port_bits": memory.weight_cacti.output_width_bits,
        "activation_bank_count": memory.activation_cacti.bank_count,
        "weight_bank_count": memory.weight_cacti.bank_count,
        "lpddr_channels": memory.lpddr_channels,
        "dram_efficiency": result.scenario.efficiency,
        "dram_bandwidth_bytes_per_second":
            result.scenario.bandwidth_bytes_per_second,
        "execution_index": result.workload.execution_index,
        "layer": result.workload.name,
        "kind": result.workload.kind,
        "m": result.workload.m,
        "n": result.workload.n,
        "k": result.workload.k,
        "repetitions": result.workload.repetitions,
        "macs": result.workload.macs,
        "tile_m": mapping.tile_m,
        "tile_n": mapping.tile_n,
        "tile_k": mapping.tile_k,
        "loop_order": mapping.loop_order,
        "double_buffered": int(mapping.double_buffered),
        "activation_residency": mapping.activation_residency,
        "weight_resident": int(mapping.weight_resident),
        "activation_capacity_bytes": mapping.activation_capacity_bytes,
        "weight_capacity_bytes": mapping.weight_capacity_bytes,
        "activation_dram_bytes": result.activation_dram_bytes,
        "weight_dram_bytes": result.weight_dram_bytes,
        "output_dram_bytes": result.output_dram_bytes,
        "partial_sum_dram_bytes": result.partial_sum_dram_bytes,
        "total_dram_bytes": result.total_dram_bytes,
        "activation_sram_read_bytes": result.activation_sram_read_bytes,
        "activation_sram_write_bytes": result.activation_sram_write_bytes,
        "weight_sram_read_bytes": result.weight_sram_read_bytes,
        "weight_sram_write_bytes": result.weight_sram_write_bytes,
        "activation_sram_bytes": result.activation_sram_bytes,
        "weight_sram_bytes": result.weight_sram_bytes,
        "compute_cycles": result.compute_cycles,
        "activation_sram_cycles": result.activation_sram_cycles,
        "weight_sram_cycles": result.weight_sram_cycles,
        "dram_cycles": result.dram_cycles,
        "startup_cycles": result.startup_cycles,
        "drain_cycles": result.drain_cycles,
        "cycles": result.cycles,
        "latency_ms": estimate.latency_ms,
        "bottleneck": result.bottleneck,
        "mac_energy_uj": estimate.mac_energy_uj,
        "activation_sram_energy_uj": estimate.activation_sram_energy_uj,
        "weight_sram_energy_uj": estimate.weight_sram_energy_uj,
        "dram_energy_uj": estimate.dram_energy_uj,
        "total_energy_uj": estimate.total_energy_uj,
        "average_dynamic_power_w": estimate.average_dynamic_power_w,
    }


def _catalog_row(
        point: ArchitecturePoint,
        assumptions: CostAssumptions) -> Dict[str, Any]:
    memory = point.memory
    area = architecture_area(point, assumptions)
    return {
        "architecture_id": point.architecture_id,
        "architecture_class": point.architecture_class,
        "array_height": point.array.height,
        "array_width": point.array.width,
        "array_label": point.array.label,
        "pe_count": point.array.pe_count,
        "activation_buffer_bytes": memory.activation_buffer_bytes,
        "weight_buffer_bytes": memory.weight_buffer_bytes,
        "activation_port_bits": memory.activation_cacti.output_width_bits,
        "weight_port_bits": memory.weight_cacti.output_width_bits,
        "activation_bank_count": memory.activation_cacti.bank_count,
        "weight_bank_count": memory.weight_cacti.bank_count,
        "activation_sram_bytes_per_cycle":
            memory.activation_sram_bytes_per_cycle,
        "weight_sram_bytes_per_cycle": memory.weight_sram_bytes_per_cycle,
        "lpddr_channels": memory.lpddr_channels,
        "dram_efficiency": memory.primary_scenario.efficiency,
        "dram_bandwidth_bytes_per_second":
            memory.primary_scenario.bandwidth_bytes_per_second,
        "array_area_mm2": area.array_area_mm2,
        "activation_sram_area_mm2": area.activation_sram_area_mm2,
        "weight_sram_area_mm2": area.weight_sram_area_mm2,
        "lpddr_io_area_mm2": area.lpddr_io_area_mm2,
        "fixed_logic_area_mm2": area.fixed_logic_area_mm2,
        "margin_area_mm2": area.margin_area_mm2,
        "total_area_mm2": area.total_area_mm2,
    }


def _model_row(
        model: Any, layer_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = tuple(layer_rows)
    return {
        "model": model.model_name,
        "architecture_id": model.architecture_id,
        "cycles": model.cycles,
        "latency_ms": model.latency_ms,
        "macs": sum(int(row["macs"]) for row in rows),
        "total_dram_bytes": sum(int(row["total_dram_bytes"]) for row in rows),
        "activation_sram_bytes": sum(
            int(row["activation_sram_bytes"]) for row in rows),
        "weight_sram_bytes": sum(
            int(row["weight_sram_bytes"]) for row in rows),
        "mac_energy_uj": sum(float(row["mac_energy_uj"]) for row in rows),
        "activation_sram_energy_uj": sum(
            float(row["activation_sram_energy_uj"]) for row in rows),
        "weight_sram_energy_uj": sum(
            float(row["weight_sram_energy_uj"]) for row in rows),
        "dram_energy_uj": sum(
            float(row["dram_energy_uj"]) for row in rows),
        "total_energy_uj": model.total_energy_uj,
        "average_dynamic_power_w": model.average_dynamic_power_w,
        "layer_count": model.layer_count,
    }


def _candidate_row(
        candidate: CandidateEstimate, pareto_ids: frozenset[str]
        ) -> Dict[str, Any]:
    return {
        "architecture_id": candidate.architecture_id,
        "architecture_class": candidate.architecture_class,
        "array_height": candidate.array.height,
        "array_width": candidate.array.width,
        "array_label": candidate.array.label,
        "pe_count": candidate.array.pe_count,
        "model_count": candidate.model_count,
        "geomean_latency_ms": candidate.geomean_latency_ms,
        "worst_normalized_latency": candidate.worst_normalized_latency,
        "mean_energy_uj": candidate.mean_energy_uj,
        "max_average_dynamic_power_w":
            candidate.max_average_dynamic_power_w,
        "total_area_mm2": candidate.total_area_mm2,
        "feasible": int(candidate.feasible),
        "constraint_violations": ";".join(candidate.constraint_violations),
        "pareto": int(candidate.architecture_id in pareto_ids),
    }


def _plot(
        path: Path, candidates: Sequence[CandidateEstimate],
        frontier: Sequence[CandidateEstimate]) -> None:
    feasible = tuple(row for row in candidates if row.feasible)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    if feasible:
        points = axis.scatter(
            [row.total_area_mm2 for row in feasible],
            [row.geomean_latency_ms for row in feasible],
            c=[row.mean_energy_uj for row in feasible], cmap="viridis",
            alpha=0.75, edgecolors="none")
        figure.colorbar(points, ax=axis, label="Mean energy per model (uJ)")
    if frontier:
        axis.scatter(
            [row.total_area_mm2 for row in frontier],
            [row.geomean_latency_ms for row in frontier],
            facecolors="none", edgecolors="red", linewidths=1.2,
            s=55, label="3-objective non-dominated points")
        axis.legend()
    axis.set_xlabel("Estimated total area (mm2)")
    axis.set_ylabel("Equal-model geometric-mean latency (ms)")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def run(config_path: Path, workload_paths: Mapping[str, Path],
        output: Path, workers: int = 1,
        sram_macro_paths: Any = None) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError("DSE output already exists: %s" % output)
    if set(workload_paths) != set(MODEL_NAMES):
        raise ValueError("workload paths must cover exactly four SPN models")
    (config, compute, assumptions, constraints, points, macro_paths,
     macro_payloads) = _load_study(config_path, sram_macro_paths)
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
    estimates = _simulate(profiles, points, assumptions, int(workers))
    point_by_id = {point.architecture_id: point for point in points}
    layer_rows = tuple(_layer_row(
        estimate, point_by_id[estimate.architecture_id])
        for estimate in estimates)
    estimate_groups: Dict[Tuple[str, str], list] = {}
    row_groups: Dict[Tuple[str, str], list] = {}
    for estimate, row in zip(estimates, layer_rows):
        key = (estimate.model_name, estimate.architecture_id)
        estimate_groups.setdefault(key, []).append(estimate)
        row_groups.setdefault(key, []).append(row)
    model_estimates = tuple(
        aggregate_model(
            model, point_by_id[architecture_id], rows)
        for (model, architecture_id), rows in sorted(estimate_groups.items()))
    model_rows = tuple(_model_row(
        estimate, row_groups[(estimate.model_name, estimate.architecture_id)])
        for estimate in model_estimates)
    by_point: Dict[str, list] = {}
    for estimate in model_estimates:
        by_point.setdefault(estimate.architecture_id, []).append(estimate)
    candidates = tuple(
        aggregate_candidate(
            point, by_point[point.architecture_id], assumptions, constraints)
        for point in points)
    candidates = normalize_candidate_fairness(candidates)
    frontier = pareto_frontier(candidates)
    pareto_ids = frozenset(row.architecture_id for row in frontier)
    candidate_rows = tuple(_candidate_row(row, pareto_ids)
                           for row in candidates)
    frontier_rows = tuple(row for row in candidate_rows if int(row["pareto"]))
    catalog_rows = tuple(_catalog_row(point, assumptions) for point in points)

    output.mkdir(parents=True)
    _write_csv(output / "joint_architecture_catalog.csv", catalog_rows)
    _write_csv(output / "joint_layer_results.csv", layer_rows)
    _write_csv(output / "joint_model_summary.csv", model_rows)
    _write_csv(output / "joint_candidate_summary.csv", candidate_rows)
    _write_csv(
        output / "joint_pareto_frontier.csv", frontier_rows,
        fieldnames=tuple(candidate_rows[0]))
    _plot(output / "joint_pareto.png", candidates, frontier)
    compute_path = _reference(Path(config_path), config["compute_dse_config"])
    metadata = {
        "format_version": 1,
        "repository_revision": revision,
        "repository_dirty_before_output": dirty_before_output,
        "llmcompass_revision": LLMCOMPASS_REVISION,
        "cacti_revision": CACTI_REVISION,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": _sha256(Path(config_path).resolve()),
        "config": config,
        "compute_config_path": str(compute_path),
        "compute_config_sha256": _sha256(compute_path),
        "sram_macro_paths": {
            name: str(path) for name, path in macro_paths.items()},
        "sram_macro_sha256": {
            name: _sha256(path) for name, path in macro_paths.items()},
        "input_paths": {
            model: str(path) for model, path in resolved_workloads.items()},
        "input_sha256": {
            model: _sha256(path) for model, path in resolved_workloads.items()},
        "architecture_count": len(points),
        "profile_count": len(config["architecture_profiles"]),
        "array_count": len(compute.array_shapes),
        "model_count": len(MODEL_NAMES),
        "objectives": [
            "minimize_equal_model_geomean_latency_ms",
            "minimize_estimated_total_area_mm2",
            "minimize_equal_model_mean_dynamic_energy_uj",
        ],
        "area_model": {
            "array": "pinned LLMCompass FP16 FPU proxy for BF16 MAC",
            "sram": "pinned CACTI macro area",
            "io_and_fixed_logic": "configured analytical assumptions",
        },
        "energy_model": {
            "mac": "configured BF16 MAC pJ/MAC",
            "sram": "CACTI read/write energy times aggregate full-width accesses",
            "dram": "configured LPDDR pJ/byte times mapped DRAM traffic",
            "excludes": ["leakage", "clock tree", "NoC", "controller scheduling"],
        },
        "execution": {"workers": int(workers)},
        "limitations": [
            "Dense GEMMs only; propagation, sampling, and vector operations are excluded.",
            "No inter-layer activation retention or operator fusion is modelled.",
            "LPDDR bank/row-buffer timing is represented only by sustained efficiency.",
            "SRAM bank conflicts are not address-level simulated.",
            "Array area, LPDDR I/O area, MAC energy, and DRAM energy are analytical proxies, not signoff PPA.",
        ],
        "confidence": {
            "capacity_and_aggregate_bandwidth_feasibility": "MEDIUM",
            "whole_model_latency": "LOW",
            "area_and_dynamic_energy": "LOW",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return output


def _read_csv(path: Path) -> Tuple[Dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def validate_artifacts(output: Path) -> str:
    output = Path(output)
    catalog = _read_csv(output / "joint_architecture_catalog.csv")
    layers = _read_csv(output / "joint_layer_results.csv")
    models = _read_csv(output / "joint_model_summary.csv")
    candidates = _read_csv(output / "joint_candidate_summary.csv")
    frontier = _read_csv(output / "joint_pareto_frontier.csv")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if not all((catalog, layers, models, candidates)):
        raise ValueError("joint DSE artifacts contain an empty required table")
    if (metadata.get("cacti_revision") != CACTI_REVISION or
            metadata.get("llmcompass_revision") != LLMCOMPASS_REVISION):
        raise ValueError("tool provenance does not match pinned revisions")
    if len(catalog) != int(metadata["architecture_count"]):
        raise ValueError("architecture catalog is incomplete")
    catalog_map = {row["architecture_id"]: row for row in catalog}
    if len(catalog_map) != len(catalog):
        raise ValueError("architecture catalog identifiers are not unique")
    layer_groups: Dict[Tuple[str, str], list] = {}
    for row in layers:
        architecture = catalog_map.get(row["architecture_id"])
        if architecture is None:
            raise ValueError("layer references an unknown architecture")
        if int(row["activation_capacity_bytes"]) > \
                int(row["activation_buffer_bytes"]):
            raise ValueError("layer mapping violates activation capacity")
        if int(row["weight_capacity_bytes"]) > \
                int(row["weight_buffer_bytes"]):
            raise ValueError("layer mapping violates weight capacity")
        floors = tuple(int(row[name]) for name in (
            "compute_cycles", "activation_sram_cycles",
            "weight_sram_cycles", "dram_cycles"))
        if int(row["cycles"]) < max(floors):
            raise ValueError("layer schedule violates a resource floor")
        energy = sum(float(row[name]) for name in (
            "mac_energy_uj", "activation_sram_energy_uj",
            "weight_sram_energy_uj", "dram_energy_uj"))
        if not _close(energy, float(row["total_energy_uj"])):
            raise ValueError("layer energy does not match its breakdown")
        key = (row["model"], row["architecture_id"])
        layer_groups.setdefault(key, []).append(row)
    expected_models = int(metadata["architecture_count"]) * len(MODEL_NAMES)
    if len(models) != expected_models:
        raise ValueError("model summaries do not cover every architecture")
    model_map = {}
    for row in models:
        key = (row["model"], row["architecture_id"])
        detail = layer_groups.get(key)
        if not detail:
            raise ValueError("model summary has no layer rows")
        if sum(int(item["cycles"]) for item in detail) != int(row["cycles"]):
            raise ValueError("model cycles do not match layer rows")
        if not _close(
                sum(float(item["total_energy_uj"]) for item in detail),
                float(row["total_energy_uj"])):
            raise ValueError("model energy does not match layer rows")
        model_map[key] = row
    if len(candidates) != int(metadata["architecture_count"]):
        raise ValueError("candidate summary is incomplete")
    best_latency = {
        model: min(float(row["latency_ms"]) for row in models
                   if row["model"] == model)
        for model in MODEL_NAMES}
    for row in candidates:
        architecture_id = row["architecture_id"]
        model_rows = [model_map[(model, architecture_id)]
                      for model in MODEL_NAMES]
        if int(row["model_count"]) != len(MODEL_NAMES):
            raise ValueError("candidate does not weight all models")
        expected_latency = math.exp(sum(
            math.log(float(item["latency_ms"])) for item in model_rows) /
            len(model_rows))
        expected_energy = sum(float(item["total_energy_uj"])
                              for item in model_rows) / len(model_rows)
        expected_worst = max(
            float(item["latency_ms"]) / best_latency[item["model"]]
            for item in model_rows)
        if not _close(expected_latency, float(row["geomean_latency_ms"])):
            raise ValueError("candidate latency objective is inconsistent")
        if not _close(expected_energy, float(row["mean_energy_uj"])):
            raise ValueError("candidate energy objective is inconsistent")
        if not _close(expected_worst, float(row["worst_normalized_latency"])):
            raise ValueError("candidate fairness metric is inconsistent")
        if not _close(
                float(catalog_map[architecture_id]["total_area_mm2"]),
                float(row["total_area_mm2"])):
            raise ValueError("candidate area is inconsistent")
    feasible = tuple(row for row in candidates if int(row["feasible"]))
    expected_frontier = set()
    for row in feasible:
        dominated = any(
            float(other["geomean_latency_ms"]) <=
                float(row["geomean_latency_ms"]) and
            float(other["total_area_mm2"]) <= float(row["total_area_mm2"]) and
            float(other["mean_energy_uj"]) <= float(row["mean_energy_uj"]) and
            (float(other["geomean_latency_ms"]) <
                 float(row["geomean_latency_ms"]) or
             float(other["total_area_mm2"]) < float(row["total_area_mm2"]) or
             float(other["mean_energy_uj"]) < float(row["mean_energy_uj"]))
            for other in feasible if other is not row)
        if not dominated:
            expected_frontier.add(row["architecture_id"])
    observed_frontier = {row["architecture_id"] for row in frontier}
    flagged_frontier = {
        row["architecture_id"] for row in candidates if int(row["pareto"])}
    if observed_frontier != expected_frontier or \
            flagged_frontier != expected_frontier:
        raise ValueError("joint Pareto frontier is inconsistent")
    return "SPN_JOINT_DSE_VALIDATION_OK"


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


def _parse_sram_macros(values: Sequence[str]) -> Any:
    if not values:
        return None
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("SRAM macro override must use name=/path/to/result.json")
        if name in result:
            raise ValueError("duplicate SRAM macro override: %s" % name)
        result[name] = Path(path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sram-macro", action="append", default=[])
    parser.add_argument("--validate-only", type=Path)
    return parser


def main(argv: Any = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only is not None:
        if (args.config is not None or args.workload or args.output is not None or
                args.sram_macro):
            raise ValueError("--validate-only cannot be combined with run inputs")
        print(validate_artifacts(args.validate_only))
        return 0
    if args.config is None or args.output is None:
        raise ValueError("--config and --output are required for a DSE run")
    output = run(
        args.config, _parse_workloads(args.workload), args.output,
        workers=args.workers,
        sram_macro_paths=_parse_sram_macros(args.sram_macro))
    print(validate_artifacts(output))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
