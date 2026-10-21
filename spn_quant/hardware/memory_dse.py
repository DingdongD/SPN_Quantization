"""Capacity- and bandwidth-constrained DSE for SPN GEMM workloads."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Tuple

from .cacti import CactiSramResult
from .llmcompass_adapter import ArrayShape
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

