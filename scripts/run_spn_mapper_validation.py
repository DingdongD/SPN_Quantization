#!/usr/bin/env python3
"""Validate LLMCompass, AccelForge/TCM, and FactorFlow on dense SPN GEMMs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_spn_joint_dse import _load_study  # noqa: E402
from spn_quant.hardware.factorflow_adapter import (  # noqa: E402
    FACTORFLOW_REVISION,
    factorflow_request,
    parse_factorflow_result,
)
from spn_quant.hardware.joint_dse import (  # noqa: E402
    ArchitecturePoint,
    CostAssumptions,
    architecture_area,
    estimate_layer,
)
from spn_quant.hardware.llmcompass_adapter import (  # noqa: E402
    LLMCOMPASS_REVISION,
)
from spn_quant.hardware.mapper_validation import (  # noqa: E402
    BackendResult,
    UniqueGemm,
    deduplicate_gemms,
    dense_gemm_instances,
    validate_backend_result,
)
from spn_quant.hardware.memory_dse import search_memory_mapping  # noqa: E402
from spn_quant.hardware.tcm_adapter import (  # noqa: E402
    TCM_REVISION,
    parse_tcm_result,
    tcm_request,
)
from spn_quant.hardware.workload import (  # noqa: E402
    LayerWorkload,
    WorkloadProfile,
)


BACKENDS = ("llmcompass", "tcm", "factorflow")
BACKEND_REVISIONS = {
    "llmcompass": LLMCOMPASS_REVISION,
    "tcm": TCM_REVISION,
    "factorflow": FACTORFLOW_REVISION,
}
Executor = Callable[[UniqueGemm, ArchitecturePoint, CostAssumptions], BackendResult]


def _write_csv(
        path: Path, rows: Sequence[Mapping[str, Any]],
        fieldnames: Sequence[str] | None = None) -> None:
    fields = tuple(rows[0]) if rows else tuple(fieldnames or ())
    if not fields:
        raise ValueError("CSV schema is required for an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)


def _geomean(values: Sequence[float]) -> float:
    rows = tuple(float(value) for value in values)
    if not rows or any(value <= 0 for value in rows):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in rows) / len(rows))


def _coverage_rows(
        profiles: Mapping[str, WorkloadProfile]) -> list[Dict[str, Any]]:
    rows = []
    for model, profile in sorted(profiles.items()):
        included = tuple(layer for layer in profile.layers if layer.array_mappable)
        propagation = tuple(
            layer for layer in profile.layers if layer.kind == "propagation")
        other = tuple(
            layer for layer in profile.layers
            if not layer.array_mappable and layer.kind != "propagation")
        rows.append({
            "model": model,
            "total_layer_count": len(profile.layers),
            "included_layer_count": len(included),
            "included_gemm_count": len(dense_gemm_instances(profile)),
            "excluded_propagation_count": len(propagation),
            "excluded_other_irregular_count": len(other),
            "included_macs": sum(layer.macs for layer in included),
            "excluded_propagation_macs": sum(
                layer.macs for layer in propagation),
        })
    return rows


def _architecture_rows(
        points: Sequence[ArchitecturePoint],
        assumptions: CostAssumptions) -> list[Dict[str, Any]]:
    rows = []
    for point in points:
        memory = point.memory
        area = architecture_area(point, assumptions)
        rows.append({
            "architecture_id": point.architecture_id,
            "architecture_class": point.architecture_class,
            "array_shape": "%dx%d" % (
                point.array.height, point.array.width),
            "array_height": point.array.height,
            "array_width": point.array.width,
            "pe_count": point.array.pe_count,
            "dtype": memory.dtype,
            "clock_hz": memory.clock_hz,
            "activation_buffer_bytes": memory.activation_buffer_bytes,
            "weight_buffer_bytes": memory.weight_buffer_bytes,
            "activation_sram_bytes_per_cycle": (
                memory.activation_sram_bytes_per_cycle),
            "weight_sram_bytes_per_cycle": (
                memory.weight_sram_bytes_per_cycle),
            "lpddr_channels": memory.lpddr_channels,
            "lpddr_peak_bandwidth_bytes_per_second": (
                memory.lpddr_peak_bandwidth_bytes_per_second),
            "primary_dram_efficiency": memory.primary_dram_efficiency,
            "primary_dram_bandwidth_bytes_per_second": (
                memory.primary_scenario.bandwidth_bytes_per_second),
            "activation_sram_area_mm2": area.activation_sram_area_mm2,
            "weight_sram_area_mm2": area.weight_sram_area_mm2,
            "array_area_mm2_proxy": area.array_area_mm2,
            "total_area_mm2_proxy": area.total_area_mm2,
        })
    return rows


def _unique_row(result: BackendResult, unique: UniqueGemm) -> Dict[str, Any]:
    return {
        "backend": result.backend,
        "backend_revision": BACKEND_REVISIONS.get(result.backend, "test"),
        "architecture_id": result.architecture_id,
        "gemm_id": result.gemm_id,
        "m": unique.workload.m,
        "n": unique.workload.n,
        "k": unique.workload.k,
        "repetitions": unique.workload.repetitions,
        "macs": result.macs,
        "cycles": result.cycles,
        "energy_uj": result.energy_uj,
        "edp_uj_cycles": result.edp_uj_cycles,
        "effective_array_utilization": result.utilization,
        "search_seconds": result.search_seconds,
        "native_objective": result.native_objective,
        "original_instance_count": len(unique.instances),
    }


def _layer_rows(
        results: Sequence[tuple[UniqueGemm, BackendResult]],
        points: Mapping[str, ArchitecturePoint]) -> list[Dict[str, Any]]:
    rows = []
    for unique, result in results:
        point = points[result.architecture_id]
        for instance in unique.instances:
            rows.append({
                "backend": result.backend,
                "architecture_id": result.architecture_id,
                "model": instance.model_name,
                "execution_index": instance.execution_index,
                "layer": instance.layer_name,
                "kind": instance.kind,
                "gemm_id": unique.gemm_id,
                "m": unique.workload.m,
                "n": unique.workload.n,
                "k": unique.workload.k,
                "repetitions": unique.workload.repetitions,
                "macs": result.macs,
                "cycles": result.cycles,
                "latency_ms": result.cycles / point.memory.clock_hz * 1e3,
                "energy_uj": result.energy_uj,
                "edp_uj_cycles": result.edp_uj_cycles,
                "effective_array_utilization": result.utilization,
            })
    return sorted(rows, key=lambda row: (
        row["backend"], row["architecture_id"], row["model"],
        int(row["execution_index"]), row["layer"]))


def _model_rows(
        layer_rows: Sequence[Mapping[str, Any]],
        points: Mapping[str, ArchitecturePoint]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in layer_rows:
        key = (str(row["backend"]), str(row["architecture_id"]),
               str(row["model"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (backend, architecture_id, model), rows in sorted(groups.items()):
        cycles = sum(int(row["cycles"]) for row in rows)
        macs = sum(int(row["macs"]) for row in rows)
        energy = sum(float(row["energy_uj"]) for row in rows)
        point = points[architecture_id]
        utilization = macs / (cycles * point.array.pe_count)
        output.append({
            "backend": backend,
            "architecture_id": architecture_id,
            "model": model,
            "dense_layer_count": len(rows),
            "macs": macs,
            "cycles": cycles,
            "latency_ms": cycles / point.memory.clock_hz * 1e3,
            "energy_uj": energy,
            "edp_uj_cycles": energy * cycles,
            "effective_array_utilization": utilization,
        })
    return output


def _ranking_rows(
        model_rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    best: Dict[tuple[str, str, str], float] = {}
    for row in model_rows:
        for metric in ("latency_ms", "energy_uj", "edp_uj_cycles"):
            key = (str(row["backend"]), str(row["model"]), metric)
            value = float(row[metric])
            best[key] = min(best.get(key, value), value)
    groups: Dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in model_rows:
        groups.setdefault(
            (str(row["backend"]), str(row["architecture_id"])),
            []).append(row)
    output = []
    for (backend, architecture_id), rows in sorted(groups.items()):
        latency = [float(row["latency_ms"]) /
                   best[(backend, str(row["model"]), "latency_ms")]
                   for row in rows]
        energy = [float(row["energy_uj"]) /
                  best[(backend, str(row["model"]), "energy_uj")]
                  for row in rows]
        edp = [float(row["edp_uj_cycles"]) /
               best[(backend, str(row["model"]), "edp_uj_cycles")]
               for row in rows]
        output.append({
            "backend": backend,
            "architecture_id": architecture_id,
            "model_count": len(rows),
            "geomean_normalized_latency": _geomean(latency),
            "geomean_normalized_energy": _geomean(energy),
            "geomean_normalized_edp": _geomean(edp),
            "worst_model_normalized_latency": max(latency),
            "mean_effective_array_utilization": sum(
                float(row["effective_array_utilization"]) for row in rows) /
                len(rows),
        })
    by_backend: Dict[str, list[Dict[str, Any]]] = {}
    for row in output:
        by_backend.setdefault(str(row["backend"]), []).append(row)
    for rows in by_backend.values():
        rows.sort(key=lambda row: (
            float(row["geomean_normalized_edp"]),
            float(row["geomean_normalized_latency"]),
            str(row["architecture_id"])))
        for rank, row in enumerate(rows, 1):
            row["backend_rank_by_normalized_edp"] = rank
    return sorted(output, key=lambda row: (
        row["backend"], int(row["backend_rank_by_normalized_edp"])))


def _plot_rankings(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    architectures = sorted({str(row["architecture_id"]) for row in rows})
    figure, axes = plt.subplots(1, len(BACKENDS), figsize=(15, 4), sharey=True)
    for axis, backend in zip(axes, BACKENDS):
        lookup = {
            str(row["architecture_id"]): float(row["geomean_normalized_edp"])
            for row in rows if row["backend"] == backend}
        values = [lookup.get(name, math.nan) for name in architectures]
        axis.bar(range(len(architectures)), values)
        axis.set_title(backend)
        axis.set_xticks(range(len(architectures)))
        axis.set_xticklabels(architectures, rotation=60, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Geomean normalized EDP (native backend)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_validation(
        profiles: Mapping[str, WorkloadProfile],
        points: Sequence[ArchitecturePoint], assumptions: CostAssumptions,
        output_dir: Path, executors: Mapping[str, Executor]) -> Path:
    """Run all dense workloads; propagation is excluded before mapper dispatch."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if set(executors) != set(BACKENDS):
        raise ValueError("validation requires exactly three backends")
    point_rows = tuple(points)
    if not point_rows:
        raise ValueError("validation requires architecture points")
    point_lookup = {point.architecture_id: point for point in point_rows}
    if len(point_lookup) != len(point_rows):
        raise ValueError("architecture identifiers must be unique")

    coverage = _coverage_rows(profiles)
    instances = tuple(
        instance for _, profile in sorted(profiles.items())
        for instance in dense_gemm_instances(profile))
    unique_gemms = deduplicate_gemms(instances)
    if not unique_gemms:
        raise ValueError("validation contains no dense GEMM")

    executed: list[tuple[UniqueGemm, BackendResult]] = []
    statuses = []
    for backend in BACKENDS:
        started = time.perf_counter()
        count_before = len(executed)
        tasks = tuple(
            (unique, point, assumptions)
            for point in point_rows for unique in unique_gemms)
        executor = executors[backend]
        if hasattr(executor, "execute_many"):
            results = executor.execute_many(tasks)  # type: ignore[attr-defined]
        else:
            results = tuple(executor(*task) for task in tasks)
        if len(results) != len(tasks):
            raise ValueError("executor returned an incomplete result batch")
        for (unique, point, _), result in zip(tasks, results):
            validate_backend_result(unique, result)
            if result.backend != backend:
                raise ValueError("executor returned the wrong backend")
            if result.architecture_id != point.architecture_id:
                raise ValueError("executor returned the wrong architecture")
            executed.append((unique, result))
        statuses.append({
            "backend": backend,
            "backend_revision": BACKEND_REVISIONS[backend],
            "status": "passed",
            "unique_result_count": len(executed) - count_before,
            "dense_layer_count": len(instances),
            "unique_gemm_count": len(unique_gemms),
            "architecture_count": len(point_rows),
            "wall_seconds": time.perf_counter() - started,
        })

    unique_rows = [
        _unique_row(result, unique) for unique, result in executed]
    layers = _layer_rows(executed, point_lookup)
    models = _model_rows(layers, point_lookup)
    rankings = _ranking_rows(models)
    _write_csv(output / "backend_status.csv", statuses)
    _write_csv(
        output / "architecture_manifest.csv",
        _architecture_rows(point_rows, assumptions))
    _write_csv(output / "workload_coverage.csv", coverage)
    _write_csv(output / "unique_gemm_results.csv", unique_rows)
    _write_csv(output / "layer_results.csv", layers)
    _write_csv(output / "model_summary.csv", models)
    _write_csv(output / "architecture_ranking.csv", rankings)
    _plot_rankings(rankings, output / "architecture_rank_comparison.png")
    metadata = {
        "format_version": 1,
        "status": "SPN_THREE_MAPPER_VALIDATION_OK",
        "backends": list(BACKENDS),
        "backend_revisions": BACKEND_REVISIONS,
        "architecture_ids": [point.architecture_id for point in point_rows],
        "model_names": sorted(profiles),
        "dense_layer_count": len(instances),
        "unique_gemm_count": len(unique_gemms),
        "excluded_propagation_count": sum(
            int(row["excluded_propagation_count"]) for row in coverage),
        "propagation_dispatched_to_mapper": False,
        "comparison_policy": (
            "absolute metrics remain backend-native; architecture ranks are "
            "normalized within each backend and model"),
        "backend_methodology": {
            "llmcompass": (
                "memory-constrained tiling search using the pinned "
                "LLMCompass cycle primitive; minimum latency objective"),
            "tcm": (
                "pinned AccelForge single-Einsum EDP mapping at the official "
                "O3 default (optimal perfectly-factorized mapspace), with "
                "separate activation/weight SRAMs and max_fused_loops=1"),
            "factorflow": (
                "pinned FactorFlow hybrid WART mapper with padded mappings, "
                "separate activation/weight SRAMs, and physical 2-D fanout"),
        },
        "memory_path": "LPDDR4 -> activation/weight SRAM -> 2-D MAC array",
        "objective_scope": (
            "backend-native latency/energy/EDP mapping at four fixed anchor "
            "architectures; architecture area is reported but is not a "
            "three-mapper optimization objective"),
    }
    (output / "validation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    validate_artifacts(output)
    return output


def _llmcompass_executor(
        unique: UniqueGemm, point: ArchitecturePoint,
        assumptions: CostAssumptions) -> BackendResult:
    started = time.perf_counter()
    result = search_memory_mapping(
        unique.workload, point.array, point.memory,
        point.memory.primary_scenario)
    estimate = estimate_layer("unique", point, result, assumptions)
    runtime = time.perf_counter() - started
    utilization = unique.workload.macs / (
        result.cycles * point.array.pe_count)
    return BackendResult(
        backend="llmcompass", architecture_id=point.architecture_id,
        gemm_id=unique.gemm_id, macs=unique.workload.macs,
        cycles=result.cycles, energy_uj=estimate.total_energy_uj,
        utilization=utilization, search_seconds=runtime,
        native_objective="minimum_latency_constrained_tiling")


class _CachedLocalExecutor:
    """Persist deterministic in-process backend results between resumptions."""

    def __init__(self, backend: str, delegate: Executor, cache_root: Path):
        self.backend = backend
        self.delegate = delegate
        self.cache_root = Path(cache_root)

    def __call__(
            self, unique: UniqueGemm, point: ArchitecturePoint,
            assumptions: CostAssumptions) -> BackendResult:
        request = {
            "backend": self.backend,
            "backend_revision": BACKEND_REVISIONS[self.backend],
            "gemm": asdict(unique.workload),
            "architecture": asdict(point),
            "cost_assumptions": asdict(assumptions),
        }
        key = hashlib.sha256(json.dumps(
            request, sort_keys=True).encode("utf-8")).hexdigest()
        path = self.cache_root / self.backend / (key + ".json")
        if path.is_file():
            return BackendResult(**json.loads(
                path.read_text(encoding="utf-8")))
        result = self.delegate(unique, point, assumptions)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return result


class _WorkerExecutor:
    """Cache-aware batched subprocess bridge to an isolated mapper runtime."""

    def __init__(
            self, backend: str, python: Path, worker: Path,
            extra_args: Sequence[str], cache_root: Path | None,
            batch_size: int = 24, workers: int = 1):
        self.backend = backend
        self.python = Path(python)
        self.worker = Path(worker)
        self.extra_args = tuple(extra_args)
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.batch_size = int(batch_size)
        self.workers = int(workers)
        if self.batch_size < 1:
            raise ValueError("worker batch size must be positive")
        if self.workers < 1:
            raise ValueError("worker concurrency must be positive")
        self.adapter_request = (
            factorflow_request if backend == "factorflow" else tcm_request)
        self.adapter_parse = (
            parse_factorflow_result if backend == "factorflow" else
            parse_tcm_result)

    def _request(self, task: tuple[
            UniqueGemm, ArchitecturePoint, CostAssumptions]
            ) -> tuple[Dict[str, Any], Path | None]:
        unique, point, assumptions = task
        request = self.adapter_request(
            unique.gemm_id, unique.workload, point, assumptions)
        serialized = json.dumps(request, sort_keys=True).encode("utf-8")
        key = hashlib.sha256(serialized).hexdigest()
        path = None if self.cache_root is None else (
            self.cache_root / self.backend / (key + ".json"))
        return request, path

    def _run_batch(self, requests: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        with tempfile.TemporaryDirectory(
                prefix="spn-%s-batch-" % self.backend) as tmp:
            root = Path(tmp)
            input_path = root / "input.json"
            output_path = root / "output.json"
            input_path.write_text(json.dumps(
                {"format_version": 1, "requests": list(requests)},
                indent=2, sort_keys=True) + "\n", encoding="utf-8")
            completed = subprocess.run(
                (str(self.python), str(self.worker),
                 "--input", str(input_path), "--output", str(output_path),
                 *self.extra_args), cwd=str(root), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed.returncode:
                raise RuntimeError(
                    "%s batch worker failed:\n%s" % (
                        self.backend, completed.stderr[-6000:]))
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        results = list(payload.get("results", ()))
        if len(results) != len(requests):
            raise ValueError("worker returned an incomplete batch")
        return results

    def execute_many(self, tasks: Sequence[tuple[
            UniqueGemm, ArchitecturePoint, CostAssumptions]
            ]) -> tuple[BackendResult, ...]:
        task_rows = tuple(tasks)
        raw: list[Dict[str, Any] | None] = [None] * len(task_rows)
        pending: list[tuple[int, Dict[str, Any], Path | None]] = []
        for index, task in enumerate(task_rows):
            request, cache_path = self._request(task)
            if cache_path is not None and cache_path.is_file():
                raw[index] = json.loads(
                    cache_path.read_text(encoding="utf-8"))
            else:
                pending.append((index, request, cache_path))
        chunks = tuple(
            pending[start:start + self.batch_size]
            for start in range(0, len(pending), self.batch_size))
        request_chunks = tuple(
            tuple(row[1] for row in chunk) for chunk in chunks)
        def store_responses(chunk: Sequence[tuple[
                int, Dict[str, Any], Path | None]],
                responses: Sequence[Dict[str, Any]]) -> None:
            if len(responses) != len(chunk):
                raise ValueError("worker returned an incomplete batch")
            for (index, _, cache_path), response in zip(chunk, responses):
                raw[index] = response
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(response, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

        if self.workers == 1:
            for chunk, requests in zip(chunks, request_chunks):
                store_responses(chunk, self._run_batch(requests))
        else:
            pool = ThreadPoolExecutor(max_workers=self.workers)
            future_to_chunk = {
                pool.submit(self._run_batch, requests): chunk
                for chunk, requests in zip(chunks, request_chunks)}
            try:
                for future in as_completed(future_to_chunk):
                    store_responses(
                        future_to_chunk[future], future.result())
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        parsed = []
        for task, response in zip(task_rows, raw):
            if response is None:
                raise RuntimeError("worker response was not populated")
            unique, point, _ = task
            parsed.append(self.adapter_parse(
                point.architecture_id, unique.workload, response))
        return tuple(parsed)

    def __call__(
            self, unique: UniqueGemm, point: ArchitecturePoint,
            assumptions: CostAssumptions) -> BackendResult:
        return self.execute_many(((unique, point, assumptions),))[0]


def _worker_executor(
        backend: str, python: Path, worker: Path,
        extra_args: Sequence[str] = (), cache_root: Path | None = None,
        batch_size: int = 24, workers: int = 1) -> _WorkerExecutor:
    return _WorkerExecutor(
        backend, python, worker, extra_args, cache_root, batch_size, workers)


def validate_artifacts(output_dir: Path) -> None:
    output = Path(output_dir)
    required = (
        "backend_status.csv", "architecture_manifest.csv",
        "workload_coverage.csv",
        "unique_gemm_results.csv", "layer_results.csv",
        "model_summary.csv", "architecture_ranking.csv", "validation.json")
    missing = tuple(name for name in required if not (output / name).is_file())
    if missing:
        raise ValueError("missing validation artifacts: %s" % (missing,))
    metadata = json.loads(
        (output / "validation.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "SPN_THREE_MAPPER_VALIDATION_OK":
        raise ValueError("validation status marker is missing")
    if metadata.get("propagation_dispatched_to_mapper") is not False:
        raise ValueError("propagation entered the mapper result set")

    def read_rows(name: str) -> tuple[Dict[str, str], ...]:
        with (output / name).open(newline="", encoding="utf-8") as handle:
            return tuple(csv.DictReader(handle))

    statuses = read_rows("backend_status.csv")
    if {row["backend"] for row in statuses} != set(BACKENDS):
        raise ValueError("backend coverage is incomplete")
    if any(row["status"] != "passed" for row in statuses):
        raise ValueError("one or more backends did not pass")

    architectures = tuple(str(row) for row in metadata["architecture_ids"])
    models = tuple(str(row) for row in metadata["model_names"])
    dense_count = int(metadata["dense_layer_count"])
    unique_count = int(metadata["unique_gemm_count"])
    expected_unique = len(BACKENDS) * len(architectures) * unique_count
    expected_layers = len(BACKENDS) * len(architectures) * dense_count
    expected_models = len(BACKENDS) * len(architectures) * len(models)
    expected_rankings = len(BACKENDS) * len(architectures)

    unique_rows = read_rows("unique_gemm_results.csv")
    layer_rows = read_rows("layer_results.csv")
    model_rows = read_rows("model_summary.csv")
    ranking_rows = read_rows("architecture_ranking.csv")
    coverage_rows = read_rows("workload_coverage.csv")
    architecture_rows = read_rows("architecture_manifest.csv")
    if len(unique_rows) != expected_unique:
        raise ValueError(
            "unique GEMM result count mismatch: %d != %d" %
            (len(unique_rows), expected_unique))
    if len(layer_rows) != expected_layers:
        raise ValueError(
            "layer result count mismatch: %d != %d" %
            (len(layer_rows), expected_layers))
    if len(model_rows) != expected_models:
        raise ValueError(
            "model summary count mismatch: %d != %d" %
            (len(model_rows), expected_models))
    if len(ranking_rows) != expected_rankings:
        raise ValueError(
            "architecture ranking count mismatch: %d != %d" %
            (len(ranking_rows), expected_rankings))
    if len(coverage_rows) != len(models):
        raise ValueError("workload coverage model count mismatch")
    if len(architecture_rows) != len(architectures):
        raise ValueError("architecture manifest count mismatch")
    if {row["architecture_id"] for row in architecture_rows} != set(
            architectures):
        raise ValueError("architecture manifest identity mismatch")
    if sum(int(row["included_gemm_count"]) for row in coverage_rows) != dense_count:
        raise ValueError("workload coverage dense count mismatch")
    if sum(int(row["excluded_propagation_count"])
           for row in coverage_rows) != int(
               metadata["excluded_propagation_count"]):
        raise ValueError("workload coverage propagation count mismatch")

    expected_backends = set(BACKENDS)
    expected_architectures = set(architectures)
    for table_name, rows in (
            ("unique GEMM", unique_rows), ("layer", layer_rows),
            ("model", model_rows), ("ranking", ranking_rows)):
        if {row["backend"] for row in rows} != expected_backends:
            raise ValueError("%s backend coverage mismatch" % table_name)
        if {row["architecture_id"] for row in rows} != expected_architectures:
            raise ValueError("%s architecture coverage mismatch" % table_name)
    if any(row["kind"] == "propagation" for row in layer_rows):
        raise ValueError("propagation appeared in layer results")
    if {row["model"] for row in model_rows} != set(models):
        raise ValueError("model summary model coverage mismatch")

    for table_name, rows in (
            ("unique GEMM", unique_rows), ("layer", layer_rows),
            ("model", model_rows)):
        for row in rows:
            if int(row["macs"]) <= 0 or int(row["cycles"]) <= 0:
                raise ValueError("%s contains non-positive work" % table_name)
            if float(row["energy_uj"]) <= 0:
                raise ValueError("%s contains non-positive energy" % table_name)
            utilization = float(row["effective_array_utilization"])
            if not math.isfinite(utilization) or not 0 < utilization <= 1:
                raise ValueError("%s contains invalid utilization" % table_name)

    for row in statuses:
        if int(row["unique_result_count"]) != len(architectures) * unique_count:
            raise ValueError("backend status unique result count mismatch")
        if int(row["dense_layer_count"]) != dense_count:
            raise ValueError("backend status dense count mismatch")


def _reference(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_cli(config_path: Path) -> tuple[
        Mapping[str, WorkloadProfile], tuple[ArchitecturePoint, ...],
        CostAssumptions, Dict[str, Executor], Path]:
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    joint_path = _reference(config_path, payload["joint_dse_config"])
    _, _, assumptions, _, all_points, _, _ = _load_study(joint_path)
    requested = tuple(payload["anchor_architectures"])
    point_lookup = {point.architecture_id: point for point in all_points}
    missing = tuple(name for name in requested if name not in point_lookup)
    if missing:
        raise ValueError("unknown anchor architectures: %s" % (missing,))
    points = tuple(point_lookup[name] for name in requested)
    profiles = {
        str(model): WorkloadProfile.read_json(_reference(config_path, path))
        for model, path in payload["workloads"].items()}
    cache = _reference(config_path, payload["cache_dir"])
    factorflow_root = _reference(
        config_path, payload["factorflow"]["root"])
    tcm_python = _reference(config_path, payload["tcm"]["python"])
    executors: Dict[str, Executor] = {
        "llmcompass": _CachedLocalExecutor(
            "llmcompass", _llmcompass_executor, cache),
        "factorflow": _worker_executor(
            "factorflow", Path(sys.executable),
            REPO_ROOT / "scripts/mapper_backends/run_factorflow.py",
            ("--factorflow-root", str(factorflow_root)), cache,
            batch_size=24, workers=4),
        "tcm": _worker_executor(
            "tcm", tcm_python,
            REPO_ROOT / "scripts/mapper_backends/run_tcm.py", (), cache,
            batch_size=1, workers=16),
    }
    output = _reference(config_path, payload["output_dir"])
    return profiles, points, assumptions, executors, output


def _microbench_profiles() -> Dict[str, WorkloadProfile]:
    layers = (
        LayerWorkload.linear(
            "aligned_64", (64, 64), (64, 64), execution_index=0),
        LayerWorkload.linear(
            "skinny_n16", (1024, 64), (1024, 16), execution_index=1),
        LayerWorkload.linear(
            "tail_63x65x31", (63, 31), (63, 65), execution_index=2),
    )
    return {
        "mapper_microbench": WorkloadProfile(
            model_name="mapper_microbench", source_config="generated",
            checkpoint="not_applicable", sample_index=0,
            input_shapes=(("synthetic", (1,)),), layers=layers)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode", choices=("microbench", "anchors"), default="anchors")
    parser.add_argument("--validate", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.validate is not None:
        validate_artifacts(args.validate)
        print("SPN_THREE_MAPPER_VALIDATION_OK")
        return 0
    if args.config is None:
        raise ValueError("--config is required")
    profiles, points, assumptions, executors, default_output = _load_cli(
        args.config)
    output = default_output if args.output is None else args.output.resolve()
    if args.mode == "microbench":
        profiles = _microbench_profiles()
        points = (points[0],)
        if args.output is None:
            output = default_output.parent / "microbench"
    run_validation(profiles, points, assumptions, output, executors)
    print("SPN_THREE_MAPPER_VALIDATION_OK: %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
