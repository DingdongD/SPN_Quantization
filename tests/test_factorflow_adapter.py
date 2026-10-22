import pytest

from spn_quant.hardware.cacti import CactiSramResult
from spn_quant.hardware.factorflow_adapter import (
    factorflow_architecture,
    factorflow_request,
    parse_factorflow_result,
)
from spn_quant.hardware.joint_dse import ArchitecturePoint, CostAssumptions
from spn_quant.hardware.llmcompass_adapter import ArrayShape
from spn_quant.hardware.memory_dse import MemoryArchitecture
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


def test_factorflow_architecture_uses_bf16_entries_and_physical_mesh():
    point = _point()

    spec = factorflow_architecture(point, _assumptions())

    assert spec.array_height == 32
    assert spec.array_width == 32
    assert spec.activation_entries == point.memory.activation_buffer_bytes // 2
    assert spec.weight_entries == point.memory.weight_buffer_bytes // 2
    assert spec.activation_bandwidth_values_per_cycle == pytest.approx(32.0)
    assert spec.weight_bandwidth_values_per_cycle == pytest.approx(32.0)
    assert spec.dram_bandwidth_values_per_cycle == pytest.approx(
        point.memory.primary_scenario.bytes_per_cycle / 2)


def test_factorflow_request_preserves_common_gemm_and_energy_units():
    workload = GemmWorkload(
        "gemm_0000", "linear", 32, 64, 16, repetitions=3)

    request = factorflow_request(
        "gemm_0000", workload, _point(), _assumptions())

    assert request["workload"] == {
        "gemm_id": "gemm_0000", "m": 32, "n": 64, "k": 16,
        "repetitions": 3, "macs": workload.macs}
    assert request["architecture"]["operand_bits"] == 16
    assert request["architecture"]["accumulator_bits"] == 32
    assert request["architecture"]["mac_energy_pj"] == 1.25
    assert request["architecture"]["dram_energy_pj_per_value"] == 30.0


def test_factorflow_result_rejects_wrong_mac_count():
    workload = GemmWorkload("gemm_0000", "linear", 32, 64, 16)

    with pytest.raises(ValueError, match="MAC count"):
        parse_factorflow_result(
            "balanced_4m_512b_1ch__32x32", workload, {
                "gemm_id": "gemm_0000",
                "macs": workload.macs - 1,
                "cycles": 64,
                "energy_uj": 2.0,
                "utilization": 0.5,
                "search_seconds": 0.1,
                "native_objective": "wart",
            })


def test_factorflow_result_parses_valid_native_metrics():
    workload = GemmWorkload("gemm_0000", "linear", 32, 64, 16)

    result = parse_factorflow_result(
        "balanced_4m_512b_1ch__32x32", workload, {
            "gemm_id": "gemm_0000",
            "macs": workload.macs,
            "cycles": 64,
            "energy_uj": 2.0,
            "utilization": 0.5,
            "search_seconds": 0.1,
            "native_objective": "wart",
        })

    assert result.backend == "factorflow"
    assert result.edp_uj_cycles == 128.0
