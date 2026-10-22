import pytest

from spn_quant.hardware.cacti import CactiSramResult
from spn_quant.hardware.joint_dse import ArchitecturePoint, CostAssumptions
from spn_quant.hardware.llmcompass_adapter import ArrayShape
from spn_quant.hardware.memory_dse import MemoryArchitecture
from spn_quant.hardware.tcm_adapter import (
    parse_tcm_result,
    tcm_architecture,
    tcm_request,
    tcm_workload,
)
from spn_quant.hardware.workload import GemmWorkload


def _macro(capacity, width, read, write):
    return CactiSramResult(
        capacity_bytes=capacity, access_time_ns=1.0, cycle_time_ns=1.0,
        area_mm2=2.0, read_energy_nj=read, write_energy_nj=write,
        output_width_bits=width, bank_count=8)


def _point():
    activation = _macro(4 * 1024 * 1024, 512, 0.4, 0.6)
    weight = _macro(4 * 1024 * 1024, 512, 0.5, 0.7)
    memory = MemoryArchitecture(
        dtype="bf16", operand_bytes=2, output_bytes=2,
        accumulator_bytes=4, clock_hz=1e9,
        activation_buffer_bytes=activation.capacity_bytes,
        weight_buffer_bytes=weight.capacity_bytes,
        lpddr_transfers_per_second=4.266e9,
        lpddr_bus_width_bits=32, lpddr_channels=1,
        dram_efficiencies=(0.7,), primary_dram_efficiency=0.7,
        activation_cacti=activation, weight_cacti=weight)
    return ArchitecturePoint(
        "balanced_4m_512b_1ch__32x32", "balanced",
        ArrayShape(32, 32), memory)


def _assumptions():
    return CostAssumptions(
        technology_nm=32,
        logic_transistor_density_million_per_mm2=10,
        bf16_mac_energy_pj=1.25,
        lpddr_energy_pj_per_byte=15.0,
        lpddr_channel_area_mm2=1.0,
        fixed_logic_area_mm2=1.0,
        implementation_area_margin=0.2)


def test_tcm_workload_is_one_unfused_bf16_matmul():
    workload = GemmWorkload(
        "gemm_0000", "linear", 32, 64, 16, repetitions=3)

    payload = tcm_workload("gemm_0000", workload)

    assert payload["workload"]["rank_sizes"] == {"M": 32, "N": 64, "K": 16}
    assert payload["workload"]["bits_per_value"] == {"All": 16}
    assert len(payload["workload"]["einsums"]) == 1
    assert payload["workload"]["einsums"][0]["name"] == "gemm_0000"
    assert payload["repetitions"] == 3
    assert payload["macs"] == workload.macs


def test_tcm_architecture_carries_capacity_bandwidth_and_physical_mesh():
    point = _point()

    spec = tcm_architecture(point, _assumptions())

    assert spec.architecture_id == point.architecture_id
    assert spec.array_height == 32
    assert spec.array_width == 32
    assert spec.activation_buffer_bits == 4 * 1024 * 1024 * 8
    assert spec.weight_buffer_bits == 4 * 1024 * 1024 * 8
    assert spec.activation_bandwidth_bits_per_cycle == pytest.approx(512.0)
    assert spec.weight_bandwidth_bits_per_cycle == pytest.approx(512.0)
    assert spec.dram_bandwidth_bits_per_cycle == pytest.approx(
        point.memory.primary_scenario.bytes_per_cycle * 8)
    assert spec.mac_energy_j == pytest.approx(1.25e-12)
    assert spec.dram_energy_j_per_value == pytest.approx(30e-12)


def test_tcm_request_preserves_backend_revision_and_common_gemm():
    workload = GemmWorkload("gemm_0000", "linear", 32, 64, 16)

    request = tcm_request(
        "gemm_0000", workload, _point(), _assumptions())

    assert request["backend"] == "tcm"
    assert request["workload"]["macs"] == workload.macs
    assert request["workload"]["workload"]["bits_per_value"] == {"All": 16}
    assert request["mapper"] == {
        "algorithm": "accelforge_single_einsum_tcm",
        "objective": "energy_delay_product",
        "max_fused_loops": 1,
        "optimization_level": 3,
        "explore_imperfect_spatial_loops": False,
        "explore_imperfect_temporal_loops": False,
        "tiling_coarseness": 1.0,
    }


def test_tcm_result_rejects_wrong_mac_count():
    workload = GemmWorkload("gemm_0000", "linear", 32, 64, 16)

    with pytest.raises(ValueError, match="MAC count"):
        parse_tcm_result(
            "balanced_4m_512b_1ch__32x32", workload, {
                "gemm_id": "gemm_0000",
                "macs": workload.macs - 1,
                "cycles": 64,
                "energy_uj": 2.0,
                "utilization": 0.5,
                "search_seconds": 0.1,
                "native_objective": "edp",
            })


def test_tcm_result_parses_valid_native_metrics():
    workload = GemmWorkload("gemm_0000", "linear", 32, 64, 16)

    result = parse_tcm_result(
        "balanced_4m_512b_1ch__32x32", workload, {
            "gemm_id": "gemm_0000",
            "macs": workload.macs,
            "cycles": 64,
            "energy_uj": 2.0,
            "utilization": 0.5,
            "search_seconds": 0.1,
            "native_objective": "edp",
        })

    assert result.backend == "tcm"
    assert result.edp_uj_cycles == 128.0
