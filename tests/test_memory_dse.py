import math

import pytest

from spn_quant.hardware.cacti import CactiSramResult
from spn_quant.hardware.llmcompass_adapter import ArrayShape
from spn_quant.hardware.memory_dse import (
    MemoryArchitecture,
    TileMapping,
    candidate_tile_sizes,
    capacity_footprint,
    mapping_traffic,
    search_memory_mapping,
    search_memory_mappings,
    simulate_mapping,
)
from spn_quant.hardware.workload import GemmWorkload


def _cacti(cycle_time_ns=0.995496, capacity=4 * 1024 * 1024,
           width=512, access_time_ns=2.10289):
    return CactiSramResult(
        capacity_bytes=capacity,
        access_time_ns=access_time_ns,
        cycle_time_ns=cycle_time_ns,
        area_mm2=7.0543941208,
        read_energy_nj=0.589761,
        write_energy_nj=0.672701,
        output_width_bits=width,
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


def _architecture(capacity=4 * 1024 * 1024, cycle_time_ns=0.995496):
    payload = _payload()
    payload["activation_buffer_bytes"] = capacity
    payload["weight_buffer_bytes"] = capacity
    cacti = _cacti(cycle_time_ns)
    object.__setattr__(cacti, "capacity_bytes", capacity)
    return MemoryArchitecture.from_dict(payload, cacti)


def _mapping(**overrides):
    values = {
        "tile_m": 32,
        "tile_n": 32,
        "tile_k": 32,
        "loop_order": "mkn",
        "double_buffered": False,
        "activation_residency": "none",
        "weight_resident": False,
        "activation_capacity_bytes": 0,
        "weight_capacity_bytes": 0,
    }
    values.update(overrides)
    return TileMapping(**values)


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


def test_memory_architecture_supports_independent_sram_macros():
    payload = _payload()
    payload["activation_buffer_bytes"] = 2 * 1024 * 1024
    payload["weight_buffer_bytes"] = 8 * 1024 * 1024
    activation = _cacti(
        capacity=2 * 1024 * 1024, width=256, access_time_ns=1.2)
    weight = _cacti(
        cycle_time_ns=2.0, capacity=8 * 1024 * 1024, width=512,
        access_time_ns=3.1)

    architecture = MemoryArchitecture.from_dict(payload, activation, weight)

    assert architecture.activation_cacti == activation
    assert architecture.weight_cacti == weight
    assert architecture.activation_sram_bytes_per_cycle == 32.0
    assert architecture.weight_sram_bytes_per_cycle == 32.0
    assert architecture.activation_sram_access_latency_cycles == 2
    assert architecture.weight_sram_access_latency_cycles == 4


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


def test_tile_candidates_are_array_aligned_and_include_exact_tail():
    assert candidate_tile_sizes(70, 16) == (16, 32, 64, 70)
    assert candidate_tile_sizes(7, 16) == (7,)


def test_capacity_footprint_separates_activation_weight_and_fp32_psum():
    workload = GemmWorkload("gemm", "linear", 64, 64, 64)
    architecture = _architecture()

    single = capacity_footprint(
        workload, _mapping(), architecture)
    double = capacity_footprint(
        workload, _mapping(double_buffered=True), architecture)

    assert single.activation_bytes == 32 * 32 * 2 + 32 * 32 * 4
    assert single.weight_bytes == 32 * 32 * 2
    assert double.activation_bytes == 2 * single.activation_bytes
    assert double.weight_bytes == 2 * single.weight_bytes


def test_persistent_residency_uses_full_operand_capacity_once():
    workload = GemmWorkload("gemm", "linear", 64, 64, 64)
    architecture = _architecture()

    footprint = capacity_footprint(
        workload,
        _mapping(double_buffered=True, activation_residency="both",
                 weight_resident=True),
        architecture)

    assert footprint.activation_bytes == 64 * 64 * 2 + 64 * 64 * 4
    assert footprint.weight_bytes == 64 * 64 * 2


def test_infeasible_activation_capacity_is_rejected_independently():
    workload = GemmWorkload("gemm", "linear", 64, 64, 64)
    architecture = _architecture(capacity=8 * 1024)

    with pytest.raises(ValueError, match="activation buffer"):
        capacity_footprint(
            workload, _mapping(double_buffered=True), architecture,
            require_feasible=True)


def test_loop_order_accounts_for_exact_tail_reloads_and_psum_spills():
    workload = GemmWorkload("tail", "linear", 70, 10, 18)
    architecture = _architecture()
    traffic = mapping_traffic(
        workload, ArrayShape(16, 8),
        _mapping(tile_m=32, tile_n=8, tile_k=8, loop_order="mkn"),
        architecture)

    assert traffic.activation_dram_bytes == 70 * 18 * 2
    assert traffic.weight_dram_bytes == 18 * 10 * 2 * 3
    assert traffic.output_dram_bytes == 70 * 10 * 2
    assert traffic.partial_sum_dram_bytes == 2 * (3 - 1) * 70 * 10 * 4


def test_sram_read_write_breakdown_conserves_all_modelled_traffic():
    workload = GemmWorkload("directions", "linear", 70, 10, 18)
    traffic = mapping_traffic(
        workload, ArrayShape(16, 8),
        _mapping(tile_m=32, tile_n=8, tile_k=8, loop_order="mkn"),
        _architecture())

    assert traffic.activation_sram_read_bytes + \
        traffic.activation_sram_write_bytes == traffic.activation_sram_bytes
    assert traffic.weight_sram_read_bytes + \
        traffic.weight_sram_write_bytes == traffic.weight_sram_bytes
    assert traffic.weight_sram_write_bytes == traffic.weight_dram_bytes
    assert traffic.activation_sram_write_bytes >= \
        traffic.activation_dram_bytes


def test_loop_orders_expose_a_b_and_c_reuse_choices():
    workload = GemmWorkload("reuse", "linear", 64, 64, 64)
    architecture = _architecture()
    array = ArrayShape(16, 16)
    mkn = mapping_traffic(workload, array, _mapping(loop_order="mkn"), architecture)
    knm = mapping_traffic(workload, array, _mapping(loop_order="knm"), architecture)
    mnk = mapping_traffic(workload, array, _mapping(loop_order="mnk"), architecture)

    assert mkn.activation_dram_bytes < mnk.activation_dram_bytes
    assert knm.weight_dram_bytes < mkn.weight_dram_bytes
    assert mnk.partial_sum_dram_bytes == 0
    assert mkn.partial_sum_dram_bytes > 0


def test_resident_operands_remove_reloads_and_dram_psum_spills():
    workload = GemmWorkload("resident", "linear", 64, 64, 64)
    architecture = _architecture()
    traffic = mapping_traffic(
        workload, ArrayShape(16, 16),
        _mapping(loop_order="mkn", activation_residency="both",
                 weight_resident=True),
        architecture)

    assert traffic.activation_dram_bytes == 64 * 64 * 2
    assert traffic.weight_dram_bytes == 64 * 64 * 2
    assert traffic.partial_sum_dram_bytes == 0


def test_repetitions_scale_all_memory_traffic():
    architecture = _architecture()
    array = ArrayShape(16, 16)
    single = mapping_traffic(
        GemmWorkload("one", "attention_qk", 64, 64, 64, 1), array,
        _mapping(loop_order="mnk"), architecture)
    repeated = mapping_traffic(
        GemmWorkload("four", "attention_qk", 64, 64, 64, 4), array,
        _mapping(loop_order="mnk"), architecture)

    assert repeated.total_dram_bytes == 4 * single.total_dram_bytes
    assert repeated.activation_sram_bytes == 4 * single.activation_sram_bytes
    assert repeated.weight_sram_bytes == 4 * single.weight_sram_bytes


def test_repetitions_pay_pipeline_startup_and_drain_each_time():
    architecture = _architecture()
    array = ArrayShape(16, 16)
    mapping = _mapping(loop_order="mnk", double_buffered=True)
    single = simulate_mapping(
        GemmWorkload("one", "attention_qk", 64, 64, 64, 1), array,
        mapping, architecture, architecture.primary_scenario)
    repeated = simulate_mapping(
        GemmWorkload("four", "attention_qk", 64, 64, 64, 4), array,
        mapping, architecture, architecture.primary_scenario)

    assert repeated.startup_cycles == 4 * single.startup_cycles
    assert repeated.drain_cycles == 4 * single.drain_cycles


def test_scheduled_cycles_respect_every_resource_floor():
    workload = GemmWorkload("scheduled", "linear", 64, 64, 64)
    architecture = _architecture()
    result = simulate_mapping(
        workload, ArrayShape(16, 16), _mapping(double_buffered=True),
        architecture, architecture.primary_scenario)

    assert result.cycles >= result.compute_cycles
    assert result.cycles >= result.activation_sram_cycles
    assert result.cycles >= result.weight_sram_cycles
    assert result.cycles >= result.dram_cycles
    assert result.startup_cycles > 0
    assert result.drain_cycles > 0


def test_search_returns_a_deterministic_capacity_safe_mapping():
    workload = GemmWorkload("search", "linear", 96, 48, 80)
    architecture = _architecture(capacity=32 * 1024)
    array = ArrayShape(16, 16)

    first = search_memory_mapping(
        workload, array, architecture, architecture.primary_scenario)
    second = search_memory_mapping(
        workload, array, architecture, architecture.primary_scenario)

    assert first == second
    assert first.mapping.activation_capacity_bytes <= \
        architecture.activation_buffer_bytes
    assert first.mapping.weight_capacity_bytes <= \
        architecture.weight_buffer_bytes
    assert first.mapping.tile_k in (32, 64, 80)
    assert first.cycles >= first.compute_cycles


def test_search_respects_one_512bit_bf16_burst_in_k_dimension():
    architecture = _architecture()
    result = search_memory_mapping(
        GemmWorkload("burst", "linear", 1024, 64, 64),
        ArrayShape(32, 16), architecture, architecture.primary_scenario)

    assert result.mapping.tile_k >= 32


def test_multi_scenario_search_matches_independent_searches():
    workload = GemmWorkload("multi", "linear", 96, 48, 80)
    architecture = _architecture(capacity=32 * 1024)
    array = ArrayShape(16, 16)

    combined = search_memory_mappings(
        workload, array, architecture, architecture.scenarios)
    independent = tuple(
        search_memory_mapping(workload, array, architecture, scenario)
        for scenario in architecture.scenarios)

    assert combined == independent
    assert tuple(result.scenario for result in combined) == \
        architecture.scenarios


def test_shared_scenario_search_matches_preoptimization_full_run_anchor():
    architecture = _architecture()
    results = search_memory_mappings(
        GemmWorkload("conv1_1", "conv2d", 17328, 64, 196),
        ArrayShape(16, 16), architecture, architecture.scenarios)

    assert tuple(result.cycles for result in results) == (
        1059879, 858261, 858128)
    assert tuple(result.mapping for result in results) == tuple(
        _mapping(
            tile_m=16, tile_n=64, tile_k=32, loop_order="mnk",
            double_buffered=True, weight_resident=True,
            activation_capacity_bytes=10240,
            weight_capacity_bytes=25088)
        for _ in architecture.scenarios)
