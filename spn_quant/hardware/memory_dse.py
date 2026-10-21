"""Capacity- and bandwidth-constrained DSE for SPN GEMM workloads."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Any, Mapping, Optional, Tuple

from .cacti import CactiSramResult
from .llmcompass_adapter import ArrayShape, simulate_gemm
from .workload import GemmWorkload


LOOP_ORDERS = ("mkn", "mnk", "nkm", "nmk", "knm", "kmn")
ACTIVATION_RESIDENCY_MODES = ("none", "a", "c", "both")


@dataclass(frozen=True)
class MemoryScenario:
    efficiency: float
    bandwidth_bytes_per_second: float
    bytes_per_cycle: float
    primary: bool = False

    @property
    def label(self) -> str:
        return "lpddr4_%dpct" % round(self.efficiency * 100)


@dataclass(frozen=True)
class MemoryArchitecture:
    dtype: str
    operand_bytes: int
    output_bytes: int
    accumulator_bytes: int
    clock_hz: float
    activation_buffer_bytes: int
    weight_buffer_bytes: int
    lpddr_transfers_per_second: float
    lpddr_bus_width_bits: int
    lpddr_channels: int
    dram_efficiencies: Tuple[float, ...]
    primary_dram_efficiency: float
    cacti: CactiSramResult
    format_version: int = 1

    def __post_init__(self) -> None:
        if int(self.format_version) != 1:
            raise ValueError("unsupported memory architecture format")
        if self.dtype != "bf16" or int(self.operand_bytes) != 2 or \
                int(self.output_bytes) != 2:
            raise ValueError("the constrained study requires two-byte BF16 data")
        if int(self.accumulator_bytes) != 4:
            raise ValueError("the constrained study requires FP32 partial sums")
        if float(self.clock_hz) <= 0:
            raise ValueError("accelerator clock must be positive")
        if int(self.activation_buffer_bytes) <= 0:
            raise ValueError("activation buffer capacity must be positive")
        if int(self.weight_buffer_bytes) <= 0:
            raise ValueError("weight buffer capacity must be positive")
        if (int(self.activation_buffer_bytes) != self.cacti.capacity_bytes or
                int(self.weight_buffer_bytes) != self.cacti.capacity_bytes):
            raise ValueError(
                "CACTI capacity must match both independent SRAM buffers")
        if (float(self.lpddr_transfers_per_second) <= 0 or
                int(self.lpddr_bus_width_bits) <= 0 or
                int(self.lpddr_channels) <= 0 or
                int(self.lpddr_bus_width_bits) % 8):
            raise ValueError("LPDDR4 fields must define a positive byte-wide bus")
        efficiencies = tuple(float(value) for value in self.dram_efficiencies)
        if (not efficiencies or len(set(efficiencies)) != len(efficiencies) or
                any(not 0 < value <= 1 for value in efficiencies)):
            raise ValueError(
                "DRAM efficiencies must be unique values in the interval (0,1]")
        primary = float(self.primary_dram_efficiency)
        if primary not in efficiencies:
            raise ValueError("primary DRAM efficiency must be present in the sweep")
        object.__setattr__(self, "operand_bytes", int(self.operand_bytes))
        object.__setattr__(self, "output_bytes", int(self.output_bytes))
        object.__setattr__(self, "accumulator_bytes", int(self.accumulator_bytes))
        object.__setattr__(self, "clock_hz", float(self.clock_hz))
        object.__setattr__(
            self, "activation_buffer_bytes", int(self.activation_buffer_bytes))
        object.__setattr__(
            self, "weight_buffer_bytes", int(self.weight_buffer_bytes))
        object.__setattr__(
            self, "lpddr_transfers_per_second",
            float(self.lpddr_transfers_per_second))
        object.__setattr__(
            self, "lpddr_bus_width_bits", int(self.lpddr_bus_width_bits))
        object.__setattr__(self, "lpddr_channels", int(self.lpddr_channels))
        object.__setattr__(self, "dram_efficiencies", efficiencies)
        object.__setattr__(self, "primary_dram_efficiency", primary)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any],
                  cacti: CactiSramResult) -> "MemoryArchitecture":
        lpddr = payload["lpddr4"]
        return cls(
            dtype=str(payload["dtype"]),
            operand_bytes=int(payload["operand_bytes"]),
            output_bytes=int(payload["output_bytes"]),
            accumulator_bytes=int(payload["accumulator_bytes"]),
            clock_hz=float(payload["clock_hz"]),
            activation_buffer_bytes=int(payload["activation_buffer_bytes"]),
            weight_buffer_bytes=int(payload["weight_buffer_bytes"]),
            lpddr_transfers_per_second=float(
                lpddr["transfers_per_second"]),
            lpddr_bus_width_bits=int(lpddr["bus_width_bits"]),
            lpddr_channels=int(lpddr["channels"]),
            dram_efficiencies=tuple(payload["dram_efficiencies"]),
            primary_dram_efficiency=float(
                payload["primary_dram_efficiency"]),
            cacti=cacti, format_version=int(payload["format_version"]))

    @property
    def clock_period_ns(self) -> float:
        return 1e9 / self.clock_hz

    @property
    def sram_bytes_per_cycle(self) -> float:
        port_bytes = self.cacti.output_width_bits / 8.0
        sram_cycles = max(1.0, self.cacti.cycle_time_ns /
                          self.clock_period_ns)
        return port_bytes / sram_cycles

    @property
    def sram_bandwidth_bytes_per_second(self) -> float:
        return self.sram_bytes_per_cycle * self.clock_hz

    @property
    def sram_access_latency_cycles(self) -> int:
        return int(math.ceil(
            self.cacti.access_time_ns / self.clock_period_ns))

    @property
    def lpddr_peak_bandwidth_bytes_per_second(self) -> float:
        return (self.lpddr_transfers_per_second * self.lpddr_bus_width_bits *
                self.lpddr_channels / 8.0)

    @property
    def scenarios(self) -> Tuple[MemoryScenario, ...]:
        return tuple(MemoryScenario(
            efficiency=value,
            bandwidth_bytes_per_second=(
                self.lpddr_peak_bandwidth_bytes_per_second * value),
            bytes_per_cycle=(
                self.lpddr_peak_bandwidth_bytes_per_second * value /
                self.clock_hz),
            primary=value == self.primary_dram_efficiency)
            for value in self.dram_efficiencies)

    @property
    def primary_scenario(self) -> MemoryScenario:
        return next(row for row in self.scenarios if row.primary)


@dataclass(frozen=True)
class TileMapping:
    tile_m: int
    tile_n: int
    tile_k: int
    loop_order: str
    double_buffered: bool
    activation_residency: str
    weight_resident: bool
    activation_capacity_bytes: int
    weight_capacity_bytes: int


@dataclass(frozen=True)
class CapacityFootprint:
    activation_bytes: int
    weight_bytes: int


@dataclass(frozen=True)
class MemoryTraffic:
    activation_dram_bytes: int
    weight_dram_bytes: int
    output_dram_bytes: int
    partial_sum_dram_bytes: int
    activation_sram_bytes: int
    weight_sram_bytes: int

    @property
    def total_dram_bytes(self) -> int:
        return (self.activation_dram_bytes + self.weight_dram_bytes +
                self.output_dram_bytes + self.partial_sum_dram_bytes)


@dataclass(frozen=True)
class MemorySimulationResult:
    workload: GemmWorkload
    array: ArrayShape
    scenario: MemoryScenario
    mapping: TileMapping
    compute_cycles: int
    activation_sram_cycles: int
    weight_sram_cycles: int
    dram_cycles: int
    startup_cycles: int
    drain_cycles: int
    cycles: int
    activation_dram_bytes: int
    weight_dram_bytes: int
    output_dram_bytes: int
    partial_sum_dram_bytes: int
    activation_sram_bytes: int
    weight_sram_bytes: int
    bottleneck: str

    @property
    def total_dram_bytes(self) -> int:
        return (self.activation_dram_bytes + self.weight_dram_bytes +
                self.output_dram_bytes + self.partial_sum_dram_bytes)


def candidate_tile_sizes(dimension: int, quantum: int) -> Tuple[int, ...]:
    """Return aligned powers of two plus the exact problem dimension."""
    size, alignment = int(dimension), int(quantum)
    if size <= 0 or alignment <= 0:
        raise ValueError("tile dimension and quantum must be positive")
    if size <= alignment:
        return (size,)
    values = []
    value = alignment
    while value < size:
        values.append(value)
        value *= 2
    values.append(size)
    return tuple(sorted(set(values)))


def _validate_mapping(workload: GemmWorkload, mapping: TileMapping) -> None:
    if mapping.loop_order not in LOOP_ORDERS:
        raise ValueError("unsupported loop order: %s" % mapping.loop_order)
    if mapping.activation_residency not in ACTIVATION_RESIDENCY_MODES:
        raise ValueError(
            "unsupported activation residency: %s" %
            mapping.activation_residency)
    for tile, dimension, label in (
            (mapping.tile_m, workload.m, "M"),
            (mapping.tile_n, workload.n, "N"),
            (mapping.tile_k, workload.k, "K")):
        if int(tile) <= 0 or int(tile) > int(dimension):
            raise ValueError("%s tile must lie within the GEMM dimension" % label)


def capacity_footprint(
        workload: GemmWorkload, mapping: TileMapping,
        architecture: MemoryArchitecture,
        require_feasible: bool = False) -> CapacityFootprint:
    """Compute independent A/C and B scratchpad footprints for one tile."""
    _validate_mapping(workload, mapping)
    factor = 2 if mapping.double_buffered else 1
    a_tile = mapping.tile_m * mapping.tile_k * architecture.operand_bytes
    b_tile = mapping.tile_k * mapping.tile_n * architecture.operand_bytes
    c_tile = mapping.tile_m * mapping.tile_n * architecture.accumulator_bytes
    full_a = workload.m * workload.k * architecture.operand_bytes
    full_b = workload.k * workload.n * architecture.operand_bytes
    full_c = workload.m * workload.n * architecture.accumulator_bytes
    if mapping.activation_residency == "none":
        activation = factor * (a_tile + c_tile)
    elif mapping.activation_residency == "a":
        activation = full_a + factor * c_tile
    elif mapping.activation_residency == "c":
        activation = factor * a_tile + full_c
    else:
        activation = full_a + full_c
    weight = full_b if mapping.weight_resident else factor * b_tile
    footprint = CapacityFootprint(
        activation_bytes=int(activation), weight_bytes=int(weight))
    if require_feasible:
        if footprint.activation_bytes > architecture.activation_buffer_bytes:
            raise ValueError("mapping exceeds activation buffer capacity")
        if footprint.weight_bytes > architecture.weight_buffer_bytes:
            raise ValueError("mapping exceeds weight buffer capacity")
    return footprint


def mapping_traffic(
        workload: GemmWorkload, array: ArrayShape, mapping: TileMapping,
        architecture: MemoryArchitecture) -> MemoryTraffic:
    """Count exact matrix traffic for a capacity-safe tiled loop nest."""
    capacity_footprint(
        workload, mapping, architecture, require_feasible=True)
    m_tiles = int(math.ceil(workload.m / float(mapping.tile_m)))
    n_tiles = int(math.ceil(workload.n / float(mapping.tile_n)))
    k_tiles = int(math.ceil(workload.k / float(mapping.tile_k)))
    repetitions = workload.repetitions
    full_a = workload.m * workload.k * architecture.operand_bytes
    full_b = workload.k * workload.n * architecture.operand_bytes
    final_c = workload.m * workload.n * architecture.output_bytes
    psum = workload.m * workload.n * architecture.accumulator_bytes
    a_resident = mapping.activation_residency in ("a", "both")
    c_resident = mapping.activation_residency in ("c", "both")
    activation_dram = full_a * (
        1 if a_resident or mapping.loop_order[-1] == "n" else n_tiles)
    weight_dram = full_b * (
        1 if mapping.weight_resident or mapping.loop_order[-1] == "m"
        else m_tiles)
    k_stays_in_array = mapping.loop_order[-1] == "k" or k_tiles == 1
    partial_sum_dram = 0 if c_resident or k_stays_in_array else (
        2 * (k_tiles - 1) * psum)
    a_array_reads = (workload.m * workload.k * architecture.operand_bytes *
                     int(math.ceil(workload.n / float(array.width))))
    b_array_reads = (workload.k * workload.n * architecture.operand_bytes *
                     int(math.ceil(workload.m / float(array.height))))
    c_array_traffic = psum if k_stays_in_array else (2 * k_tiles - 1) * psum
    activation_dram *= repetitions
    weight_dram *= repetitions
    final_c *= repetitions
    partial_sum_dram *= repetitions
    activation_sram = (
        activation_dram + final_c + partial_sum_dram +
        (a_array_reads + c_array_traffic) * repetitions)
    weight_sram = weight_dram + b_array_reads * repetitions
    return MemoryTraffic(
        activation_dram_bytes=int(activation_dram),
        weight_dram_bytes=int(weight_dram),
        output_dram_bytes=int(final_c),
        partial_sum_dram_bytes=int(partial_sum_dram),
        activation_sram_bytes=int(activation_sram),
        weight_sram_bytes=int(weight_sram))


def _compute_cycles(workload: GemmWorkload, array: ArrayShape,
                    tile_k: int) -> int:
    full_tiles, tail = divmod(workload.k, int(tile_k))
    total = 0
    if full_tiles:
        total += simulate_gemm(GemmWorkload(
            name=workload.name, kind=workload.kind,
            m=workload.m, n=workload.n, k=int(tile_k),
            repetitions=workload.repetitions * full_tiles,
            execution_index=workload.execution_index), array, 1, "os").cycles
    if tail:
        total += simulate_gemm(GemmWorkload(
            name=workload.name, kind=workload.kind,
            m=workload.m, n=workload.n, k=tail,
            repetitions=workload.repetitions,
            execution_index=workload.execution_index), array, 1, "os").cycles
    if total <= 0:
        raise RuntimeError("memory DSE produced a non-positive compute floor")
    return total


def _cycles(byte_count: int, bytes_per_cycle: float) -> int:
    if byte_count <= 0 or bytes_per_cycle <= 0:
        raise ValueError("traffic and bandwidth must be positive")
    return int(math.ceil(byte_count / float(bytes_per_cycle)))


def simulate_mapping(
        workload: GemmWorkload, array: ArrayShape, mapping: TileMapping,
        architecture: MemoryArchitecture,
        scenario: MemoryScenario) -> MemorySimulationResult:
    footprint = capacity_footprint(
        workload, mapping, architecture, require_feasible=True)
    traffic = mapping_traffic(workload, array, mapping, architecture)
    compute = _compute_cycles(workload, array, mapping.tile_k)
    activation_sram = _cycles(
        traffic.activation_sram_bytes, architecture.sram_bytes_per_cycle)
    weight_sram = _cycles(
        traffic.weight_sram_bytes, architecture.sram_bytes_per_cycle)
    dram = _cycles(traffic.total_dram_bytes, scenario.bytes_per_cycle)
    a_tile = mapping.tile_m * mapping.tile_k * architecture.operand_bytes
    b_tile = mapping.tile_k * mapping.tile_n * architecture.operand_bytes
    c_tile = mapping.tile_m * mapping.tile_n * architecture.output_bytes
    startup = workload.repetitions * (
        _cycles(a_tile + b_tile, scenario.bytes_per_cycle) +
        architecture.sram_access_latency_cycles)
    drain = workload.repetitions * (
        _cycles(c_tile, scenario.bytes_per_cycle) +
        architecture.sram_access_latency_cycles)
    resource_cycles = {
        "compute": compute,
        "activation_sram": activation_sram,
        "weight_sram": weight_sram,
        "dram": dram,
    }
    bottleneck = max(
        resource_cycles, key=lambda name: (resource_cycles[name], name))
    if mapping.double_buffered:
        total = max(resource_cycles.values()) + startup + drain
    else:
        total = compute + dram + max(activation_sram, weight_sram)
    mapped = replace(
        mapping, activation_capacity_bytes=footprint.activation_bytes,
        weight_capacity_bytes=footprint.weight_bytes)
    return MemorySimulationResult(
        workload=workload, array=array, scenario=scenario, mapping=mapped,
        compute_cycles=compute, activation_sram_cycles=activation_sram,
        weight_sram_cycles=weight_sram, dram_cycles=dram,
        startup_cycles=startup, drain_cycles=drain, cycles=int(total),
        activation_dram_bytes=traffic.activation_dram_bytes,
        weight_dram_bytes=traffic.weight_dram_bytes,
        output_dram_bytes=traffic.output_dram_bytes,
        partial_sum_dram_bytes=traffic.partial_sum_dram_bytes,
        activation_sram_bytes=traffic.activation_sram_bytes,
        weight_sram_bytes=traffic.weight_sram_bytes,
        bottleneck=bottleneck)


def _mapping_key(result: MemorySimulationResult) -> Tuple[Any, ...]:
    mapping = result.mapping
    return (
        result.cycles, result.total_dram_bytes,
        result.activation_sram_bytes + result.weight_sram_bytes,
        -mapping.tile_m * mapping.tile_n * mapping.tile_k,
        mapping.loop_order, not mapping.double_buffered,
        mapping.activation_residency, not mapping.weight_resident)


@lru_cache(maxsize=None)
def search_memory_mapping(
        workload: GemmWorkload, array: ArrayShape,
        architecture: MemoryArchitecture,
        scenario: MemoryScenario) -> MemorySimulationResult:
    best: Optional[MemorySimulationResult] = None
    k_quantum = max(
        1, architecture.cacti.output_width_bits // 8 //
        architecture.operand_bytes)
    for tile_m in candidate_tile_sizes(workload.m, array.height):
        for tile_n in candidate_tile_sizes(workload.n, array.width):
            for tile_k in candidate_tile_sizes(workload.k, k_quantum):
                for double_buffered in (True, False):
                    for activation_residency in ACTIVATION_RESIDENCY_MODES:
                        for weight_resident in (True, False):
                            mapping = TileMapping(
                                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                                loop_order="mnk",
                                double_buffered=double_buffered,
                                activation_residency=activation_residency,
                                weight_resident=weight_resident,
                                activation_capacity_bytes=0,
                                weight_capacity_bytes=0)
                            footprint = capacity_footprint(
                                workload, mapping, architecture)
                            if (footprint.activation_bytes >
                                    architecture.activation_buffer_bytes or
                                    footprint.weight_bytes >
                                    architecture.weight_buffer_bytes):
                                continue
                            for loop_order in LOOP_ORDERS:
                                result = simulate_mapping(
                                    workload, array,
                                    replace(mapping, loop_order=loop_order),
                                    architecture, scenario)
                                if best is None or _mapping_key(result) < \
                                        _mapping_key(best):
                                    best = result
    if best is None:
        raise RuntimeError(
            "no feasible memory mapping for %s on %s" %
            (workload.name, array.label))
    return best

