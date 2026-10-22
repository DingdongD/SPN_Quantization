import math

import pytest

from spn_quant.hardware.cacti import CactiSramResult
from spn_quant.hardware.joint_dse import (
    ArchitecturePoint,
    CandidateEstimate,
    CostAssumptions,
    DesignConstraints,
    aggregate_candidate,
    aggregate_model,
    architecture_area,
    estimate_layer,
    normalize_candidate_fairness,
    pareto_frontier,
    systolic_array_area_mm2,
)
from spn_quant.hardware.llmcompass_adapter import ArrayShape
from spn_quant.hardware.memory_dse import (
    MemoryArchitecture,
    TileMapping,
    simulate_mapping,
)
from spn_quant.hardware.workload import GemmWorkload


def _macro(capacity, width=512, area=2.0, read_nj=0.5, write_nj=0.75):
    return CactiSramResult(
        capacity_bytes=capacity, access_time_ns=1.0, cycle_time_ns=1.0,
        area_mm2=area, read_energy_nj=read_nj,
        write_energy_nj=write_nj, output_width_bits=width, bank_count=8)


def _memory(activation=None, weight=None, channels=1):
    activation = activation or _macro(64 * 1024)
    weight = weight or _macro(64 * 1024)
    payload = {
        "format_version": 1,
        "dtype": "bf16",
        "operand_bytes": 2,
        "output_bytes": 2,
        "accumulator_bytes": 4,
        "clock_hz": 1.0e9,
        "activation_buffer_bytes": activation.capacity_bytes,
        "weight_buffer_bytes": weight.capacity_bytes,
        "lpddr4": {
            "transfers_per_second": 4.266e9,
            "bus_width_bits": 32,
            "channels": channels,
        },
        "dram_efficiencies": [0.7],
        "primary_dram_efficiency": 0.7,
    }
    return MemoryArchitecture.from_dict(payload, activation, weight)


def _cost():
    return CostAssumptions(
        technology_nm=32,
        logic_transistor_density_million_per_mm2=10.0,
        bf16_mac_energy_pj=1.0,
        lpddr_energy_pj_per_byte=100.0,
        lpddr_channel_area_mm2=3.0,
        fixed_logic_area_mm2=1.0,
        implementation_area_margin=0.2,
    )


def _result(memory=None, array=ArrayShape(8, 8)):
    memory = memory or _memory()
    mapping = TileMapping(
        tile_m=8, tile_n=8, tile_k=32, loop_order="mnk",
        double_buffered=True, activation_residency="none",
        weight_resident=False, activation_capacity_bytes=0,
        weight_capacity_bytes=0)
    return simulate_mapping(
        GemmWorkload("linear", "linear", 16, 16, 32), array, mapping,
        memory, memory.primary_scenario)


def test_llmcompass_fp16_proxy_area_matches_pinned_formula():
    array = ArrayShape(8, 16)
    assumptions = _cost()

    expected_transistors_per_pe = 685300 * (10.0 / 52.0) ** 2
    expected = (expected_transistors_per_pe * array.pe_count / 1e6 /
                assumptions.logic_transistor_density_million_per_mm2)

    assert math.isclose(
        systolic_array_area_mm2(array, assumptions), expected, rel_tol=1e-12)


def test_layer_energy_uses_mac_cacti_read_write_and_dram_terms():
    memory = _memory()
    result = _result(memory)
    point = ArchitecturePoint("balanced", "balanced", result.array, memory)

    estimate = estimate_layer("cspn", point, result, _cost())

    activation = memory.activation_cacti
    weight = memory.weight_cacti
    expected_sram_nj = (
        math.ceil(result.activation_sram_read_bytes /
                  (activation.output_width_bits / 8)) *
        activation.read_energy_nj +
        math.ceil(result.activation_sram_write_bytes /
                  (activation.output_width_bits / 8)) *
        activation.write_energy_nj +
        math.ceil(result.weight_sram_read_bytes /
                  (weight.output_width_bits / 8)) * weight.read_energy_nj +
        math.ceil(result.weight_sram_write_bytes /
                  (weight.output_width_bits / 8)) * weight.write_energy_nj)
    expected_uj = (
        result.workload.macs * 1.0 + result.total_dram_bytes * 100.0) / 1e6 + \
        expected_sram_nj / 1e3

    assert math.isclose(estimate.total_energy_uj, expected_uj, rel_tol=1e-12)
    assert math.isclose(
        estimate.average_dynamic_power_w,
        estimate.total_energy_uj * 1e-6 /
        (result.cycles / memory.clock_hz), rel_tol=1e-12)


def test_architecture_area_includes_array_macros_io_fixed_and_margin():
    memory = _memory(
        activation=_macro(64 * 1024, area=2.0),
        weight=_macro(128 * 1024, area=4.0), channels=2)
    point = ArchitecturePoint("aggressive", "aggressive", ArrayShape(8, 8), memory)

    area = architecture_area(point, _cost())

    subtotal = area.array_area_mm2 + 2.0 + 4.0 + 2 * 3.0 + 1.0
    assert math.isclose(area.total_area_mm2, subtotal * 1.2, rel_tol=1e-12)
    assert math.isclose(area.margin_area_mm2, subtotal * 0.2, rel_tol=1e-12)


def test_candidate_aggregation_weights_models_equally_and_applies_constraints():
    memory = _memory()
    point = ArchitecturePoint("balanced", "balanced", ArrayShape(8, 8), memory)
    layers = tuple(
        estimate_layer(model, point, _result(memory), _cost())
        for model in ("cspn", "dyspn", "nlspn", "completionformer"))
    models = tuple(
        aggregate_model(model, point, tuple(
            row for row in layers if row.model_name == model))
        for model in ("cspn", "dyspn", "nlspn", "completionformer"))

    candidate = aggregate_candidate(
        point, models, _cost(),
        DesignConstraints(max_total_area_mm2=0.01))

    assert candidate.model_count == 4
    assert math.isclose(
        candidate.geomean_latency_ms, models[0].latency_ms, rel_tol=1e-12)
    assert math.isclose(
        candidate.mean_energy_uj, models[0].total_energy_uj, rel_tol=1e-12)
    assert not candidate.feasible
    assert candidate.constraint_violations == ("total_area_mm2",)


def _candidate(name, latency, area, energy, feasible=True):
    return CandidateEstimate(
        architecture_id=name, architecture_class="test",
        array=ArrayShape(8, 8), model_count=4,
        geomean_latency_ms=latency, mean_energy_uj=energy,
        max_average_dynamic_power_w=1.0, total_area_mm2=area,
        model_estimates=(), feasible=feasible,
        constraint_violations=(), worst_normalized_latency=math.nan)


def test_three_objective_pareto_excludes_dominated_and_infeasible_points():
    candidates = (
        _candidate("small", 10.0, 2.0, 4.0),
        _candidate("fast", 5.0, 4.0, 5.0),
        _candidate("dominated", 11.0, 3.0, 6.0),
        _candidate("invalid", 1.0, 1.0, 1.0, feasible=False),
    )

    frontier = pareto_frontier(candidates)

    assert tuple(row.architecture_id for row in frontier) == ("small", "fast")


def test_fairness_normalization_reports_worst_model_slowdown():
    memory = _memory()
    point_a = ArchitecturePoint("a", "test", ArrayShape(8, 8), memory)
    point_b = ArchitecturePoint("b", "test", ArrayShape(8, 8), memory)
    base = _result(memory)
    layer_a = estimate_layer("cspn", point_a, base, _cost())
    layer_b = estimate_layer("cspn", point_b, base, _cost())
    model_a = aggregate_model("cspn", point_a, (layer_a,))
    model_b = aggregate_model("cspn", point_b, (layer_b,))
    candidate_a = _candidate("a", 1.0, 1.0, 1.0)
    candidate_b = _candidate("b", 2.0, 1.0, 1.0)
    object.__setattr__(candidate_a, "model_estimates", (model_a,))
    object.__setattr__(candidate_b, "model_estimates", (model_b,))
    object.__setattr__(model_b, "latency_ms", 2 * model_a.latency_ms)

    normalized = normalize_candidate_fairness((candidate_a, candidate_b))

    assert normalized[0].worst_normalized_latency == 1.0
    assert normalized[1].worst_normalized_latency == 2.0


@pytest.mark.parametrize(
    "field,value",
    (("technology_nm", 0), ("bf16_mac_energy_pj", 0),
     ("lpddr_energy_pj_per_byte", -1),
     ("implementation_area_margin", -0.1)))
def test_cost_assumptions_reject_invalid_values(field, value):
    values = _cost().__dict__.copy()
    values[field] = value

    with pytest.raises(ValueError):
        CostAssumptions(**values)
