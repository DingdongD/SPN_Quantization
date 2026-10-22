"""Fair systolic-array design-space exploration for SPN workloads."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .llmcompass_adapter import (
    ArrayShape,
    GemmSimulationResult,
    simulate_gemm,
)
from .workload import WorkloadProfile, lower_layer_to_gemms


@dataclass(frozen=True)
class ArrayDSEConfig:
    dtype: str
    word_bytes: int
    mac_per_cycle: float
    dataflow: str
    array_shapes: Tuple[ArrayShape, ...]
    iso_pe_groups: Tuple[Tuple[int, Tuple[ArrayShape, ...]], ...]
    low_utilization_threshold: float
    format_version: int = 1

    def __post_init__(self) -> None:
        if int(self.format_version) != 1:
            raise ValueError("unsupported DSE config format")
        if self.dtype != "bf16" or int(self.word_bytes) != 2:
            raise ValueError("this study requires two-byte BF16 operands")
        if float(self.mac_per_cycle) <= 0:
            raise ValueError("MAC rate must be positive")
        if self.dataflow not in ("os", "ws", "is"):
            raise ValueError("unsupported dataflow")
        if not self.array_shapes or len(set(self.array_shapes)) != len(
                self.array_shapes):
            raise ValueError("array shapes must be non-empty and unique")
        threshold = float(self.low_utilization_threshold)
        if not 0 < threshold < 1:
            raise ValueError("utilization threshold must lie between zero and one")
        configured = set(self.array_shapes)
        for pe_count, shapes in self.iso_pe_groups:
            if any(shape.pe_count != int(pe_count) for shape in shapes):
                raise ValueError("iso-PE group contains a mismatched shape")
            if not set(shapes).issubset(configured):
                raise ValueError("iso-PE group contains an unknown shape")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ArrayDSEConfig":
        shapes = tuple(ArrayShape(*row) for row in payload["array_shapes"])
        groups = tuple(
            (int(pe_count), tuple(ArrayShape(*row) for row in rows))
            for pe_count, rows in sorted(
                payload["iso_pe_groups"].items(), key=lambda item: int(item[0])))
        return cls(
            dtype=str(payload["dtype"]), word_bytes=int(payload["word_bytes"]),
            mac_per_cycle=float(payload["mac_per_cycle"]),
            dataflow=str(payload["dataflow"]), array_shapes=shapes,
            iso_pe_groups=groups,
            low_utilization_threshold=float(
                payload["low_utilization_threshold"]),
            format_version=int(payload["format_version"]))

    @classmethod
    def read_json(cls, path: Path) -> "ArrayDSEConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CandidateSummary:
    model_name: str
    array: ArrayShape
    dense_macs: int
    irregular_macs: int
    cycles: int
    ideal_cycles: int
    effective_utilization: float
    ideal_cycle_overhead: float
    cycles_per_mac: float
    low_utilization_mac_fraction: float
    low_utilization_layer_fraction: float
    array_coverage_fraction: float
    analytical_gemm_fraction: float
    gemm_count: int

    @property
    def pe_count(self) -> int:
        return self.array.pe_count


@dataclass(frozen=True)
class SharedCandidateRanking:
    array: ArrayShape
    geometric_mean_utilization: float
    model_count: int
    total_cycles: int
    mean_array_coverage_fraction: float


@dataclass(frozen=True)
class ModelSimulation:
    model_name: str
    array: ArrayShape
    rows: Tuple[GemmSimulationResult, ...]
    summary: CandidateSummary


def aggregate_candidate(
        model_name: str, rows: Sequence[GemmSimulationResult],
        irregular_macs: int, threshold: float) -> CandidateSummary:
    results = tuple(rows)
    if not results:
        raise ValueError("candidate aggregation requires simulated GEMMs")
    shapes = set(row.array for row in results)
    rates = set(row.mac_per_cycle for row in results)
    if len(shapes) != 1 or len(rates) != 1:
        raise ValueError("candidate rows must use one array and MAC rate")
    if int(irregular_macs) < 0:
        raise ValueError("irregular MAC estimate cannot be negative")
    if not 0 < float(threshold) < 1:
        raise ValueError("utilization threshold must lie between zero and one")
    array = results[0].array
    dense_macs = sum(row.macs for row in results)
    cycles = sum(row.cycles for row in results)
    ideal_cycles = sum(row.ideal_cycles for row in results)
    capacity_macs = sum(
        row.cycles * row.array_pe_count * row.mac_per_cycle
        for row in results)
    effective = dense_macs / float(capacity_macs)
    low_rows = tuple(
        row for row in results
        if row.effective_utilization < float(threshold))
    analytical = sum(
        row.simulation_source == "analytical" for row in results)
    total_estimated = dense_macs + int(irregular_macs)
    return CandidateSummary(
        model_name=str(model_name), array=array, dense_macs=dense_macs,
        irregular_macs=int(irregular_macs), cycles=cycles,
        ideal_cycles=ideal_cycles, effective_utilization=effective,
        ideal_cycle_overhead=cycles / float(ideal_cycles),
        cycles_per_mac=cycles / float(dense_macs),
        low_utilization_mac_fraction=(
            sum(row.macs for row in low_rows) / float(dense_macs)),
        low_utilization_layer_fraction=len(low_rows) / float(len(results)),
        array_coverage_fraction=dense_macs / float(total_estimated),
        analytical_gemm_fraction=analytical / float(len(results)),
        gemm_count=len(results))


def pareto_frontier(
        candidates: Sequence[CandidateSummary]) -> Tuple[CandidateSummary, ...]:
    rows = tuple(candidates)
    if not rows:
        return ()
    if len(set(row.model_name for row in rows)) != 1:
        raise ValueError("Pareto candidates must belong to one model")
    frontier = []
    for candidate in rows:
        dominated = any(
            other.pe_count <= candidate.pe_count and
            other.cycles <= candidate.cycles and
            (other.pe_count < candidate.pe_count or
             other.cycles < candidate.cycles)
            for other in rows if other is not candidate)
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(
        frontier,
        key=lambda row: (row.pe_count, row.cycles,
                         row.array.height, row.array.width)))


def rank_shared_candidates(
        summaries: Sequence[CandidateSummary]
        ) -> Tuple[SharedCandidateRanking, ...]:
    rows = tuple(summaries)
    if not rows:
        return ()
    by_shape: Dict[ArrayShape, Dict[str, CandidateSummary]] = {}
    for row in rows:
        model_rows = by_shape.setdefault(row.array, {})
        if row.model_name in model_rows:
            raise ValueError("duplicate model/array candidate summary")
        model_rows[row.model_name] = row
    expected_models = set(next(iter(by_shape.values())))
    if not expected_models or any(
            set(model_rows) != expected_models
            for model_rows in by_shape.values()):
        raise ValueError("every shared candidate must cover the same models")
    ranking = []
    for array, model_rows in by_shape.items():
        ordered = tuple(model_rows[name] for name in sorted(model_rows))
        log_mean = sum(math.log(row.effective_utilization)
                       for row in ordered) / len(ordered)
        ranking.append(SharedCandidateRanking(
            array=array,
            geometric_mean_utilization=math.exp(log_mean),
            model_count=len(ordered),
            total_cycles=sum(row.cycles for row in ordered),
            mean_array_coverage_fraction=sum(
                row.array_coverage_fraction for row in ordered) /
                float(len(ordered))))
    return tuple(sorted(
        ranking,
        key=lambda row: (-row.geometric_mean_utilization,
                         row.array.pe_count, row.total_cycles,
                         row.array.height, row.array.width)))


def simulate_profile(profile: WorkloadProfile, config: ArrayDSEConfig
                     ) -> Tuple[ModelSimulation, ...]:
    gemms = tuple(
        gemm for layer in profile.layers
        for gemm in lower_layer_to_gemms(layer))
    if not gemms:
        raise ValueError("profile has no array-mappable GEMMs")
    output = []
    for array in config.array_shapes:
        results = tuple(simulate_gemm(
            gemm, array, config.mac_per_cycle, config.dataflow)
            for gemm in gemms)
        output.append(ModelSimulation(
            model_name=profile.model_name, array=array, rows=results,
            summary=aggregate_candidate(
                profile.model_name, results, profile.irregular_macs,
                config.low_utilization_threshold)))
    return tuple(output)
