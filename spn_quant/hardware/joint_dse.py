"""Joint latency, area, and dynamic-energy analysis for SPN accelerators."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional, Sequence, Tuple

from .llmcompass_adapter import ArrayShape
from .memory_dse import MemoryArchitecture, MemorySimulationResult


# Pinned LLMCompass cost_model/cost_model.py constants.  Its cost model has no
# BF16 primitive, so the FP16 FPU estimate is deliberately exposed as a proxy.
LLMCOMPASS_FPU64_TRANSISTORS = 685300.0
LLMCOMPASS_FP16_MANTISSA_BITS = 10.0
LLMCOMPASS_FP64_MANTISSA_BITS = 52.0


@dataclass(frozen=True)
class CostAssumptions:
    technology_nm: int
    logic_transistor_density_million_per_mm2: float
    bf16_mac_energy_pj: float
    lpddr_energy_pj_per_byte: float
    lpddr_channel_area_mm2: float
    fixed_logic_area_mm2: float
    implementation_area_margin: float

    def __post_init__(self) -> None:
        positive = (
            "technology_nm", "logic_transistor_density_million_per_mm2",
            "bf16_mac_energy_pj", "lpddr_energy_pj_per_byte")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive" % name)
        nonnegative = (
            "lpddr_channel_area_mm2", "fixed_logic_area_mm2",
            "implementation_area_margin")
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s cannot be negative" % name)
        object.__setattr__(self, "technology_nm", int(self.technology_nm))
        for name in positive[1:] + nonnegative:
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True)
class DesignConstraints:
    max_total_area_mm2: Optional[float] = None
    max_model_average_dynamic_power_w: Optional[float] = None
    max_model_latency_ms: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for name in (
                "max_total_area_mm2",
                "max_model_average_dynamic_power_w"):
            value = getattr(self, name)
            if value is not None and (
                    not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError("%s must be positive when provided" % name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        names = tuple(name for name, _ in self.max_model_latency_ms)
        if len(set(names)) != len(names):
            raise ValueError("model latency constraints must be unique")
        normalized = tuple(
            (str(name), float(value))
            for name, value in self.max_model_latency_ms)
        if any(not name or not math.isfinite(value) or value <= 0
               for name, value in normalized):
            raise ValueError("model latency constraints must be positive")
        object.__setattr__(self, "max_model_latency_ms", normalized)


@dataclass(frozen=True)
class ArchitecturePoint:
    architecture_id: str
    architecture_class: str
    array: ArrayShape
    memory: MemoryArchitecture

    def __post_init__(self) -> None:
        if not self.architecture_id or not self.architecture_class:
            raise ValueError("architecture identity and class are required")


@dataclass(frozen=True)
class ArchitectureArea:
    array_area_mm2: float
    activation_sram_area_mm2: float
    weight_sram_area_mm2: float
    lpddr_io_area_mm2: float
    fixed_logic_area_mm2: float
    margin_area_mm2: float
    total_area_mm2: float


@dataclass(frozen=True)
class LayerEstimate:
    model_name: str
    architecture_id: str
    result: MemorySimulationResult
    mac_energy_uj: float
    activation_sram_energy_uj: float
    weight_sram_energy_uj: float
    dram_energy_uj: float
    total_energy_uj: float
    average_dynamic_power_w: float
    clock_hz: float

    @property
    def latency_ms(self) -> float:
        return self.result.cycles / self.clock_hz * 1e3


@dataclass(frozen=True)
class ModelEstimate:
    model_name: str
    architecture_id: str
    cycles: int
    latency_ms: float
    total_energy_uj: float
    average_dynamic_power_w: float
    layer_count: int


@dataclass(frozen=True)
class CandidateEstimate:
    architecture_id: str
    architecture_class: str
    array: ArrayShape
    model_count: int
    geomean_latency_ms: float
    mean_energy_uj: float
    max_average_dynamic_power_w: float
    total_area_mm2: float
    model_estimates: Tuple[ModelEstimate, ...]
    feasible: bool
    constraint_violations: Tuple[str, ...]
    worst_normalized_latency: float


def systolic_array_area_mm2(
        array: ArrayShape, assumptions: CostAssumptions) -> float:
    """Return the pinned LLMCompass FP16-FPU area proxy for one array."""
    transistors_per_pe = LLMCOMPASS_FPU64_TRANSISTORS * (
        LLMCOMPASS_FP16_MANTISSA_BITS /
        LLMCOMPASS_FP64_MANTISSA_BITS) ** 2
    return (transistors_per_pe * array.pe_count / 1e6 /
            assumptions.logic_transistor_density_million_per_mm2)


def architecture_area(
        point: ArchitecturePoint,
        assumptions: CostAssumptions) -> ArchitectureArea:
    array_area = systolic_array_area_mm2(point.array, assumptions)
    activation_area = point.memory.activation_cacti.area_mm2
    weight_area = point.memory.weight_cacti.area_mm2
    io_area = (point.memory.lpddr_channels *
               assumptions.lpddr_channel_area_mm2)
    subtotal = (array_area + activation_area + weight_area + io_area +
                assumptions.fixed_logic_area_mm2)
    margin = subtotal * assumptions.implementation_area_margin
    return ArchitectureArea(
        array_area_mm2=array_area,
        activation_sram_area_mm2=activation_area,
        weight_sram_area_mm2=weight_area,
        lpddr_io_area_mm2=io_area,
        fixed_logic_area_mm2=assumptions.fixed_logic_area_mm2,
        margin_area_mm2=margin,
        total_area_mm2=subtotal + margin)


def _accesses(byte_count: int, width_bits: int) -> int:
    return int(math.ceil(int(byte_count) / (int(width_bits) / 8.0)))


def estimate_layer(
        model_name: str, point: ArchitecturePoint,
        result: MemorySimulationResult,
        assumptions: CostAssumptions) -> LayerEstimate:
    if result.array != point.array:
        raise ValueError("layer result array does not match architecture point")
    if result.scenario not in point.memory.scenarios:
        raise ValueError("layer result scenario does not match architecture point")
    activation = point.memory.activation_cacti
    weight = point.memory.weight_cacti
    mac_uj = (result.workload.macs * assumptions.bf16_mac_energy_pj /
              1e6)
    activation_uj = (
        _accesses(result.activation_sram_read_bytes,
                  activation.output_width_bits) * activation.read_energy_nj +
        _accesses(result.activation_sram_write_bytes,
                  activation.output_width_bits) * activation.write_energy_nj
    ) / 1e3
    weight_uj = (
        _accesses(result.weight_sram_read_bytes,
                  weight.output_width_bits) * weight.read_energy_nj +
        _accesses(result.weight_sram_write_bytes,
                  weight.output_width_bits) * weight.write_energy_nj
    ) / 1e3
    dram_uj = (result.total_dram_bytes *
               assumptions.lpddr_energy_pj_per_byte / 1e6)
    total_uj = mac_uj + activation_uj + weight_uj + dram_uj
    latency_seconds = result.cycles / point.memory.clock_hz
    estimate = LayerEstimate(
        model_name=str(model_name), architecture_id=point.architecture_id,
        result=result, mac_energy_uj=mac_uj,
        activation_sram_energy_uj=activation_uj,
        weight_sram_energy_uj=weight_uj, dram_energy_uj=dram_uj,
        total_energy_uj=total_uj,
        average_dynamic_power_w=total_uj * 1e-6 / latency_seconds,
        clock_hz=point.memory.clock_hz)
    return estimate


def aggregate_model(
        model_name: str, point: ArchitecturePoint,
        layers: Sequence[LayerEstimate]) -> ModelEstimate:
    rows = tuple(layers)
    if not rows:
        raise ValueError("model aggregation requires at least one layer")
    if any(row.model_name != model_name for row in rows):
        raise ValueError("model aggregation contains another model")
    if any(row.architecture_id != point.architecture_id for row in rows):
        raise ValueError("model aggregation contains another architecture")
    cycles = sum(row.result.cycles for row in rows)
    latency_seconds = cycles / point.memory.clock_hz
    energy_uj = sum(row.total_energy_uj for row in rows)
    return ModelEstimate(
        model_name=str(model_name), architecture_id=point.architecture_id,
        cycles=cycles, latency_ms=latency_seconds * 1e3,
        total_energy_uj=energy_uj,
        average_dynamic_power_w=energy_uj * 1e-6 / latency_seconds,
        layer_count=len(rows))


def aggregate_candidate(
        point: ArchitecturePoint, models: Sequence[ModelEstimate],
        assumptions: CostAssumptions,
        constraints: DesignConstraints = DesignConstraints()
        ) -> CandidateEstimate:
    rows = tuple(models)
    if not rows:
        raise ValueError("candidate aggregation requires model estimates")
    names = tuple(row.model_name for row in rows)
    if len(set(names)) != len(names):
        raise ValueError("candidate aggregation contains duplicate models")
    if any(row.architecture_id != point.architecture_id for row in rows):
        raise ValueError("candidate aggregation contains another architecture")
    area = architecture_area(point, assumptions)
    geomean_latency = math.exp(
        sum(math.log(row.latency_ms) for row in rows) / len(rows))
    mean_energy = sum(row.total_energy_uj for row in rows) / len(rows)
    max_power = max(row.average_dynamic_power_w for row in rows)
    violations = []
    if constraints.max_total_area_mm2 is not None and \
            area.total_area_mm2 > constraints.max_total_area_mm2:
        violations.append("total_area_mm2")
    if constraints.max_model_average_dynamic_power_w is not None and \
            max_power > constraints.max_model_average_dynamic_power_w:
        violations.append("model_average_dynamic_power_w")
    latency_limits = dict(constraints.max_model_latency_ms)
    for row in rows:
        if row.model_name in latency_limits and \
                row.latency_ms > latency_limits[row.model_name]:
            violations.append("latency_ms:%s" % row.model_name)
    return CandidateEstimate(
        architecture_id=point.architecture_id,
        architecture_class=point.architecture_class, array=point.array,
        model_count=len(rows), geomean_latency_ms=geomean_latency,
        mean_energy_uj=mean_energy,
        max_average_dynamic_power_w=max_power,
        total_area_mm2=area.total_area_mm2,
        model_estimates=tuple(sorted(rows, key=lambda row: row.model_name)),
        feasible=not violations, constraint_violations=tuple(violations),
        worst_normalized_latency=math.nan)


def normalize_candidate_fairness(
        candidates: Sequence[CandidateEstimate]
        ) -> Tuple[CandidateEstimate, ...]:
    rows = tuple(candidates)
    if not rows:
        return ()
    model_sets = tuple(
        frozenset(row.model_name for row in candidate.model_estimates)
        for candidate in rows)
    if any(models != model_sets[0] for models in model_sets):
        raise ValueError("all candidates must cover the same model set")
    best = {
        model: min(
            row.latency_ms for candidate in rows
            for row in candidate.model_estimates if row.model_name == model)
        for model in model_sets[0]}
    return tuple(replace(
        candidate,
        worst_normalized_latency=max(
            row.latency_ms / best[row.model_name]
            for row in candidate.model_estimates))
        for candidate in rows)


def pareto_frontier(
        candidates: Sequence[CandidateEstimate]
        ) -> Tuple[CandidateEstimate, ...]:
    feasible = tuple(row for row in candidates if row.feasible)
    frontier = []
    for candidate in feasible:
        dominated = any(
            other.geomean_latency_ms <= candidate.geomean_latency_ms and
            other.total_area_mm2 <= candidate.total_area_mm2 and
            other.mean_energy_uj <= candidate.mean_energy_uj and
            (other.geomean_latency_ms < candidate.geomean_latency_ms or
             other.total_area_mm2 < candidate.total_area_mm2 or
             other.mean_energy_uj < candidate.mean_energy_uj)
            for other in feasible if other is not candidate)
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(
        frontier,
        key=lambda row: (row.total_area_mm2, row.geomean_latency_ms,
                         row.mean_energy_uj, row.architecture_id)))
