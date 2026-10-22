"""Reproducible request/response adapter for the FactorFlow mapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .joint_dse import ArchitecturePoint, CostAssumptions
from .mapper_validation import BackendResult
from .workload import GemmWorkload


FACTORFLOW_REVISION = "40bcf3109377f12a63c8e8ba3a01df2208977bf3"


@dataclass(frozen=True)
class FactorFlowArchitecture:
    architecture_id: str
    array_height: int
    array_width: int
    operand_bits: int
    accumulator_bits: int
    activation_entries: int
    weight_entries: int
    activation_bandwidth_values_per_cycle: float
    weight_bandwidth_values_per_cycle: float
    dram_bandwidth_values_per_cycle: float
    activation_read_energy_pj_per_value: float
    activation_write_energy_pj_per_value: float
    weight_read_energy_pj_per_value: float
    weight_write_energy_pj_per_value: float
    dram_energy_pj_per_value: float
    mac_energy_pj: float


def _energy_per_value(energy_nj: float, width_bits: int,
                      operand_bits: int) -> float:
    values = float(width_bits) / float(operand_bits)
    return float(energy_nj) * 1e3 / values


def factorflow_architecture(
        point: ArchitecturePoint,
        assumptions: CostAssumptions) -> FactorFlowArchitecture:
    """Translate the shared BF16 architecture into FactorFlow units."""
    memory = point.memory
    operand_bits = memory.operand_bytes * 8
    return FactorFlowArchitecture(
        architecture_id=point.architecture_id,
        array_height=point.array.height,
        array_width=point.array.width,
        operand_bits=operand_bits,
        accumulator_bits=memory.accumulator_bytes * 8,
        activation_entries=(
            memory.activation_buffer_bytes // memory.operand_bytes),
        weight_entries=memory.weight_buffer_bytes // memory.operand_bytes,
        activation_bandwidth_values_per_cycle=(
            memory.activation_sram_bytes_per_cycle /
            memory.operand_bytes),
        weight_bandwidth_values_per_cycle=(
            memory.weight_sram_bytes_per_cycle / memory.operand_bytes),
        dram_bandwidth_values_per_cycle=(
            memory.primary_scenario.bytes_per_cycle /
            memory.operand_bytes),
        activation_read_energy_pj_per_value=_energy_per_value(
            memory.activation_cacti.read_energy_nj,
            memory.activation_cacti.output_width_bits, operand_bits),
        activation_write_energy_pj_per_value=_energy_per_value(
            memory.activation_cacti.write_energy_nj,
            memory.activation_cacti.output_width_bits, operand_bits),
        weight_read_energy_pj_per_value=_energy_per_value(
            memory.weight_cacti.read_energy_nj,
            memory.weight_cacti.output_width_bits, operand_bits),
        weight_write_energy_pj_per_value=_energy_per_value(
            memory.weight_cacti.write_energy_nj,
            memory.weight_cacti.output_width_bits, operand_bits),
        dram_energy_pj_per_value=(
            assumptions.lpddr_energy_pj_per_byte * memory.operand_bytes),
        mac_energy_pj=assumptions.bf16_mac_energy_pj,
    )


def factorflow_request(
        gemm_id: str, workload: GemmWorkload, point: ArchitecturePoint,
        assumptions: CostAssumptions) -> Dict[str, Any]:
    """Return the JSON contract consumed by the isolated worker."""
    if gemm_id != workload.name:
        raise ValueError("FactorFlow GEMM identity must match workload name")
    return {
        "format_version": 1,
        "backend": "factorflow",
        "backend_revision": FACTORFLOW_REVISION,
        "workload": {
            "gemm_id": gemm_id,
            "m": workload.m,
            "n": workload.n,
            "k": workload.k,
            "repetitions": workload.repetitions,
            "macs": workload.macs,
        },
        "architecture": asdict(factorflow_architecture(
            point, assumptions)),
        "mapper": {
            "algorithm": "hybrid",
            "multithreaded": False,
            "padded_mappings": True,
            "objective": "wart",
        },
    }


def parse_factorflow_result(
        architecture_id: str, workload: GemmWorkload,
        payload: Mapping[str, Any]) -> BackendResult:
    """Parse and validate one FactorFlow worker response."""
    if str(payload["gemm_id"]) != workload.name:
        raise ValueError("FactorFlow GEMM identity mismatch")
    if int(payload["macs"]) != workload.macs:
        raise ValueError("FactorFlow MAC count mismatch")
    return BackendResult(
        backend="factorflow",
        architecture_id=str(architecture_id),
        gemm_id=workload.name,
        macs=int(payload["macs"]),
        cycles=int(payload["cycles"]),
        energy_uj=float(payload["energy_uj"]),
        utilization=float(payload["utilization"]),
        search_seconds=float(payload["search_seconds"]),
        native_objective=str(payload["native_objective"]),
    )
