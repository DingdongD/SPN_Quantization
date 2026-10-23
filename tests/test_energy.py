import math
import json

import pytest

from spn_quant.energy import (
    energy_per_forward,
    integrate_power_joules,
    normalize_power_sample,
    parse_power_trace,
    summarize_energy_trace,
    summarize_power_window,
    validate_energy_settings,
)


def test_integrate_power_clips_and_interpolates_window_edges():
    samples = [(0.0, 10.0), (1.0, 20.0), (2.0, 10.0)]

    assert integrate_power_joules(samples, 0.5, 1.5) == pytest.approx(17.5)


def test_integrate_power_rejects_missing_window_coverage():
    with pytest.raises(ValueError, match="cover"):
        integrate_power_joules([(1.0, 10.0), (2.0, 10.0)], 0.0, 2.0)


@pytest.mark.parametrize(
    "samples",
    [
        [(0.0, 10.0), (0.0, 11.0)],
        [(0.0, math.nan), (1.0, 10.0)],
    ],
)
def test_integrate_power_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        integrate_power_joules(samples, 0.0, 1.0)


def test_summarize_power_window_reports_time_weighted_values():
    summary = summarize_power_window(
        [(0.0, 10.0), (1.0, 20.0), (2.0, 10.0)], 0.5, 1.5)

    assert summary["duration_seconds"] == pytest.approx(1.0)
    assert summary["energy_joules"] == pytest.approx(17.5)
    assert summary["mean_power_w"] == pytest.approx(17.5)
    assert summary["sample_count"] == 3


def test_energy_per_forward_reports_gross_and_idle_subtracted_values():
    result = energy_per_forward(
        active_energy_j=200.0,
        active_seconds=10.0,
        idle_mean_power_w=5.0,
        completed_forwards=100,
    )

    assert result["gross_j_per_forward"] == pytest.approx(2.0)
    assert result["incremental_j_per_forward"] == pytest.approx(1.5)
    assert result["active_mean_power_w"] == pytest.approx(20.0)


def test_energy_per_forward_rejects_active_power_below_idle():
    with pytest.raises(ValueError, match="below idle"):
        energy_per_forward(
            active_energy_j=40.0,
            active_seconds=10.0,
            idle_mean_power_w=5.0,
            completed_forwards=100,
        )


def test_energy_per_forward_allows_a_physically_inactive_zero_watt_rail():
    result = energy_per_forward(
        active_energy_j=0.0,
        active_seconds=10.0,
        idle_mean_power_w=0.0,
        completed_forwards=100,
    )

    assert result["active_mean_power_w"] == 0.0
    assert result["gross_j_per_forward"] == 0.0
    assert result["incremental_j_per_forward"] == 0.0


def test_power_sample_schema_preserves_source_and_rails():
    sample = normalize_power_sample(
        1_000_000_000, {"gpu_board": 12_000, "memory": 500})

    assert sample == {
        "kind": "sample",
        "monotonic_ns": 1_000_000_000,
        "power_w": {"gpu_board": 12.0, "memory": 0.5},
    }


def test_energy_contract_requires_synchronized_full_forwards():
    assert validate_energy_settings({
        "warmup_iterations": 30,
        "idle_seconds": 15,
        "active_seconds": 60,
        "sample_interval_ms": 50,
        "synchronize_each_forward": True,
    }) is None


def test_energy_contract_rejects_unsynchronized_forwards():
    with pytest.raises(ValueError, match="synchronized"):
        validate_energy_settings({
            "warmup_iterations": 30,
            "idle_seconds": 15,
            "active_seconds": 60,
            "sample_interval_ms": 50,
            "synchronize_each_forward": False,
        })


def test_parse_power_trace_retains_metadata_and_samples(tmp_path):
    trace = tmp_path / "power.jsonl"
    records = [
        {"kind": "metadata", "source": "nvml", "scope": "gpu_board"},
        normalize_power_sample(1_000_000_000, {"gpu_board": 12_000}),
        normalize_power_sample(2_000_000_000, {"gpu_board": 20_000}),
        {"kind": "end", "monotonic_ns": 2_000_000_001},
    ]
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8")

    parsed = parse_power_trace(trace)

    assert parsed["metadata"]["scope"] == "gpu_board"
    assert len(parsed["samples"]) == 2
    assert parsed["samples"][1]["power_w"]["gpu_board"] == 20.0


def test_summarize_energy_trace_integrates_each_rail():
    trace = {
        "metadata": {
            "kind": "metadata",
            "source": "nvml",
            "scope": "gpu_board",
        },
        "samples": [
            normalize_power_sample(0, {"gpu_board": 10_000}),
            normalize_power_sample(1_000_000_000, {"gpu_board": 10_000}),
            normalize_power_sample(2_000_000_000, {"gpu_board": 20_000}),
            normalize_power_sample(3_000_000_000, {"gpu_board": 20_000}),
        ],
    }

    summary = summarize_energy_trace(
        trace,
        idle_start_monotonic_ns=0,
        idle_end_monotonic_ns=1_000_000_000,
        active_start_monotonic_ns=2_000_000_000,
        active_end_monotonic_ns=3_000_000_000,
        completed_forwards=10,
    )

    rail = summary["rails"]["gpu_board"]
    assert rail["idle"]["mean_power_w"] == pytest.approx(10.0)
    assert rail["active"]["mean_power_w"] == pytest.approx(20.0)
    assert rail["gross_j_per_forward"] == pytest.approx(2.0)
    assert rail["incremental_j_per_forward"] == pytest.approx(1.0)


def test_auxiliary_rail_preserves_signed_incremental_energy():
    trace = {
        "metadata": {
            "kind": "metadata",
            "source": "jtop_ina3221",
            "scope": "jetson_module_and_rails",
        },
        "samples": [
            normalize_power_sample(
                0, {"total_module": 10_000, "cv": 2_000}),
            normalize_power_sample(
                1_000_000_000, {"total_module": 10_000, "cv": 2_000}),
            normalize_power_sample(
                2_000_000_000, {"total_module": 20_000, "cv": 1_000}),
            normalize_power_sample(
                3_000_000_000, {"total_module": 20_000, "cv": 1_000}),
        ],
    }

    summary = summarize_energy_trace(
        trace, 0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 10)

    assert summary["rails"]["total_module"][
        "incremental_j_per_forward"] == pytest.approx(1.0)
    assert summary["rails"]["cv"][
        "incremental_j_per_forward"] == pytest.approx(-0.1)
