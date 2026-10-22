import math

from spn_quant.hardware.array_dse import (
    CandidateSummary,
    aggregate_candidate,
    pareto_frontier,
    rank_shared_candidates,
)
from spn_quant.hardware.llmcompass_adapter import ArrayShape, simulate_gemm
from spn_quant.hardware.workload import GemmWorkload


def _result(name, shape, m, n, k):
    return simulate_gemm(
        GemmWorkload(name, "conv2d", m, n, k),
        ArrayShape(*shape), 1, "os")


def _summary(model, shape, cycles, utilization=0.5, dense_macs=1000):
    return CandidateSummary(
        model_name=model,
        array=ArrayShape(*shape),
        dense_macs=dense_macs,
        irregular_macs=0,
        cycles=cycles,
        ideal_cycles=max(1, cycles // 2),
        effective_utilization=utilization,
        ideal_cycle_overhead=2.0,
        cycles_per_mac=cycles / float(dense_macs),
        low_utilization_mac_fraction=0.0,
        low_utilization_layer_fraction=0.0,
        array_coverage_fraction=1.0,
        analytical_gemm_fraction=1.0,
        gemm_count=1,
    )


def test_aggregate_weights_utilization_by_useful_macs():
    rows = (
        _result("large", (16, 16), 1024, 64, 288),
        _result("tail", (16, 16), 4096, 1, 64),
    )
    summary = aggregate_candidate(
        "cspn", rows, irregular_macs=rows[1].macs, threshold=0.25)
    expected = sum(row.macs for row in rows) / sum(
        row.cycles * row.array_pe_count * row.mac_per_cycle
        for row in rows)
    assert summary.effective_utilization == expected
    assert summary.array_coverage_fraction == (
        sum(row.macs for row in rows) /
        float(sum(row.macs for row in rows) + rows[1].macs))


def test_pareto_rejects_more_pes_and_more_cycles():
    candidates = (
        _summary("cspn", (8, 8), 1000),
        _summary("cspn", (16, 16), 400),
        _summary("cspn", (32, 16), 250),
        _summary("cspn", (16, 32), 300),
        _summary("cspn", (64, 64), 300),
    )
    frontier = pareto_frontier(candidates)
    assert {(row.array.height, row.array.width) for row in frontier} == {
        (8, 8), (16, 16), (32, 16)}


def test_shared_ranking_gives_each_model_equal_weight():
    summaries = (
        _summary("cspn", (16, 16), 100, 0.90, dense_macs=10**12),
        _summary("dyspn", (16, 16), 100, 0.40, dense_macs=1),
        _summary("cspn", (32, 8), 100, 0.80, dense_macs=10**12),
        _summary("dyspn", (32, 8), 100, 0.80, dense_macs=1),
    )
    ranking = rank_shared_candidates(summaries)
    assert ranking[0].array == ArrayShape(32, 8)
    assert math.isclose(ranking[0].geometric_mean_utilization, 0.8)
    assert ranking[0].model_count == 2


def test_low_utilization_fractions_use_macs_and_layer_counts_separately():
    rows = (
        _result("efficient", (16, 16), 1024, 64, 288),
        _result("tail", (16, 16), 4096, 1, 64),
    )
    summary = aggregate_candidate(
        "nlspn", rows, irregular_macs=0, threshold=0.25)
    assert summary.low_utilization_layer_fraction == 0.5
    assert summary.low_utilization_mac_fraction == (
        rows[1].macs / float(rows[0].macs + rows[1].macs))
