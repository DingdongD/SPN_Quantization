import math

import pytest

from spn_quant.hardware.cacti import CactiSramResult
from spn_quant.hardware.memory_dse import MemoryArchitecture


def _cacti(cycle_time_ns=0.995496):
    return CactiSramResult(
        capacity_bytes=4 * 1024 * 1024,
        access_time_ns=2.10289,
        cycle_time_ns=cycle_time_ns,
        area_mm2=7.0543941208,
        read_energy_nj=0.589761,
        write_energy_nj=0.672701,
        output_width_bits=512,
        bank_count=16)


def _payload():
    return {
        "format_version": 1,
        "dtype": "bf16",
        "operand_bytes": 2,
        "output_bytes": 2,
        "accumulator_bytes": 4,
        "clock_hz": 1.0e9,
        "activation_buffer_bytes": 4 * 1024 * 1024,
        "weight_buffer_bytes": 4 * 1024 * 1024,
        "lpddr4": {
            "transfers_per_second": 4.266e9,
            "bus_width_bits": 32,
            "channels": 1,
        },
        "dram_efficiencies": [0.5, 0.7, 0.9],
        "primary_dram_efficiency": 0.7,
    }


def test_memory_architecture_derives_cacti_and_lpddr_bandwidths():
    architecture = MemoryArchitecture.from_dict(_payload(), _cacti())

    assert architecture.activation_buffer_bytes == 4 * 1024 * 1024
    assert architecture.weight_buffer_bytes == 4 * 1024 * 1024
    assert architecture.sram_bytes_per_cycle == 64.0
    assert architecture.sram_bandwidth_bytes_per_second == 64e9
    assert architecture.lpddr_peak_bandwidth_bytes_per_second == 17.064e9
    assert tuple(row.efficiency for row in architecture.scenarios) == (
        0.5, 0.7, 0.9)
    primary = architecture.primary_scenario
    assert primary.efficiency == 0.7
    assert math.isclose(
        primary.bandwidth_bytes_per_second, 11.9448e9, rel_tol=1e-12)
    assert math.isclose(primary.bytes_per_cycle, 11.9448, rel_tol=1e-12)


def test_cacti_cycle_time_limits_sram_bandwidth_when_slower_than_clock():
    architecture = MemoryArchitecture.from_dict(_payload(), _cacti(2.0))

    assert architecture.sram_bytes_per_cycle == 32.0
    assert architecture.sram_bandwidth_bytes_per_second == 32e9


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("dtype", "fp16", "BF16"),
        ("operand_bytes", 4, "BF16"),
        ("accumulator_bytes", 2, "FP32"),
        ("clock_hz", 0, "clock"),
        ("activation_buffer_bytes", 0, "activation"),
        ("weight_buffer_bytes", 0, "weight"),
    ))
def test_memory_architecture_rejects_invalid_fixed_constraints(
        field, value, match):
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        MemoryArchitecture.from_dict(payload, _cacti())


@pytest.mark.parametrize("efficiencies", ([0.0, 0.7], [0.7, 1.1], []))
def test_memory_architecture_rejects_invalid_efficiencies(efficiencies):
    payload = _payload()
    payload["dram_efficiencies"] = efficiencies

    with pytest.raises(ValueError, match="efficien"):
        MemoryArchitecture.from_dict(payload, _cacti())


def test_memory_architecture_requires_primary_scenario_in_sweep():
    payload = _payload()
    payload["primary_dram_efficiency"] = 0.8

    with pytest.raises(ValueError, match="primary"):
        MemoryArchitecture.from_dict(payload, _cacti())


def test_memory_architecture_rejects_cacti_capacity_mismatch():
    payload = _payload()
    payload["weight_buffer_bytes"] = 2 * 1024 * 1024

    with pytest.raises(ValueError, match="CACTI capacity"):
        MemoryArchitecture.from_dict(payload, _cacti())


@pytest.mark.parametrize(
    "field,value",
    (
        ("transfers_per_second", 0),
        ("bus_width_bits", 0),
        ("channels", 0),
    ))
def test_memory_architecture_rejects_invalid_lpddr_fields(field, value):
    payload = _payload()
    payload["lpddr4"][field] = value

    with pytest.raises(ValueError, match="LPDDR4"):
        MemoryArchitecture.from_dict(payload, _cacti())
