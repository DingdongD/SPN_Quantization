"""Reproducible request/response adapter for AccelForge's TCM mapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .joint_dse import ArchitecturePoint, CostAssumptions
from .mapper_validation import BackendResult
from .workload import GemmWorkload


TCM_REVISION = "7312a7d915625552bf6eb1db577941f8317b6ac7"


@dataclass(frozen=True)
class TcmArchitecture:
    """Shared hardware expressed in AccelForge's bit/Joule conventions."""

    architecture_id: str
    array_height: int
    array_width: int
    operand_bits: int
    accumulator_bits: int
    activation_buffer_bits: int
    weight_buffer_bits: int
    activation_bandwidth_bits_per_cycle: float
    weight_bandwidth_bits_per_cycle: float
    dram_bandwidth_bits_per_cycle: float
    activation_read_energy_j_per_value: float
    activation_write_energy_j_per_value: float
    weight_read_energy_j_per_value: float
    weight_write_energy_j_per_value: float
    dram_energy_j_per_value: float
    mac_energy_j: float


def _energy_j_per_value(
        energy_nj: float, width_bits: int, operand_bits: int) -> float:
    values_per_access = float(width_bits) / float(operand_bits)
    return float(energy_nj) * 1e-9 / values_per_access


def tcm_architecture(
        point: ArchitecturePoint,
        assumptions: CostAssumptions) -> TcmArchitecture:
    """Translate one shared BF16 architecture point into TCM units."""
    memory = point.memory
    operand_bits = memory.operand_bytes * 8
    return TcmArchitecture(
        architecture_id=point.architecture_id,
        array_height=point.array.height,
        array_width=point.array.width,
        operand_bits=operand_bits,
        accumulator_bits=memory.accumulator_bytes * 8,
        activation_buffer_bits=memory.activation_buffer_bytes * 8,
        weight_buffer_bits=memory.weight_buffer_bytes * 8,
        activation_bandwidth_bits_per_cycle=(
            memory.activation_sram_bytes_per_cycle * 8),
        weight_bandwidth_bits_per_cycle=(
            memory.weight_sram_bytes_per_cycle * 8),
        dram_bandwidth_bits_per_cycle=(
            memory.primary_scenario.bytes_per_cycle * 8),
        activation_read_energy_j_per_value=_energy_j_per_value(
            memory.activation_cacti.read_energy_nj,
            memory.activation_cacti.output_width_bits, operand_bits),
        activation_write_energy_j_per_value=_energy_j_per_value(
            memory.activation_cacti.write_energy_nj,
            memory.activation_cacti.output_width_bits, operand_bits),
        weight_read_energy_j_per_value=_energy_j_per_value(
            memory.weight_cacti.read_energy_nj,
            memory.weight_cacti.output_width_bits, operand_bits),
        weight_write_energy_j_per_value=_energy_j_per_value(
            memory.weight_cacti.write_energy_nj,
            memory.weight_cacti.output_width_bits, operand_bits),
        dram_energy_j_per_value=(
            assumptions.lpddr_energy_pj_per_byte * memory.operand_bytes *
            1e-12),
        mac_energy_j=assumptions.bf16_mac_energy_pj * 1e-12,
    )


def tcm_workload(gemm_id: str, workload: GemmWorkload) -> Dict[str, Any]:
    """Create one independent, unfused AccelForge Einsum workload."""
    if gemm_id != workload.name:
        raise ValueError("TCM GEMM identity must match workload name")
    return {
        "workload": {
            "rank_sizes": {
                "M": workload.m,
                "N": workload.n,
                "K": workload.k,
            },
            "bits_per_value": {"All": 16},
            "einsums": [{
                "name": gemm_id,
                "tensor_accesses": [
                    {"name": "A", "projection": ["M", "K"]},
                    {"name": "B", "projection": ["K", "N"]},
                    {"name": "C", "projection": ["M", "N"],
                     "output": True},
                ],
                "renames": {
                    "input": "A", "weight": "B", "output": "C"},
            }],
        },
        "repetitions": workload.repetitions,
        "macs": workload.macs,
    }


def tcm_request(
        gemm_id: str, workload: GemmWorkload, point: ArchitecturePoint,
        assumptions: CostAssumptions) -> Dict[str, Any]:
    """Return the JSON contract consumed by the isolated TCM worker."""
    return {
        "format_version": 1,
        "backend": "tcm",
        "backend_revision": TCM_REVISION,
        "workload": tcm_workload(gemm_id, workload),
        "architecture": asdict(tcm_architecture(point, assumptions)),
        "mapper": {
            "algorithm": "accelforge_single_einsum_tcm",
            "objective": "energy_delay_product",
            "max_fused_loops": 1,
            "optimization_level": 3,
            "explore_imperfect_spatial_loops": False,
            "explore_imperfect_temporal_loops": False,
            "tiling_coarseness": 1.0,
        },
    }


def parse_tcm_result(
        architecture_id: str, workload: GemmWorkload,
        payload: Mapping[str, Any]) -> BackendResult:
    """Parse and validate one isolated AccelForge/TCM response."""
    if str(payload["gemm_id"]) != workload.name:
        raise ValueError("TCM GEMM identity mismatch")
    if int(payload["macs"]) != workload.macs:
        raise ValueError("TCM MAC count mismatch")
    return BackendResult(
        backend="tcm",
        architecture_id=str(architecture_id),
        gemm_id=workload.name,
        macs=int(payload["macs"]),
        cycles=int(payload["cycles"]),
        energy_uj=float(payload["energy_uj"]),
        utilization=float(payload["utilization"]),
        search_seconds=float(payload["search_seconds"]),
        native_objective=str(payload["native_objective"]),
    )
