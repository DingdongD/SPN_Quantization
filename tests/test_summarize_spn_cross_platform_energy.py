import pytest

from scripts.summarize_spn_cross_platform_energy import summarize_runs


def _run(value, scope="gpu_board"):
    return {
        "model": "cspn",
        "power": {
            "source": "nvml",
            "sensor_scope": scope,
            "rails": {
                "board": {
                    "gross_j_per_forward": value + 1.0,
                    "incremental_j_per_forward": value,
                    "idle": {"mean_power_w": 10.0},
                    "active": {"mean_power_w": 20.0},
                },
            },
        },
        "workload": {
            "completed_forwards": 100,
            "latency_ms": {"mean": 10.0},
        },
    }


def test_summary_uses_independent_run_values_for_ci():
    summary = summarize_runs(
        [_run(value) for value in (1.0, 1.1, 0.9, 1.05, 0.95)],
        rail_name="board")

    energy = summary["incremental_j_per_forward"]
    assert energy["process_count"] == 5
    assert energy["mean"] == pytest.approx(1.0)
    assert energy["ci95_high"] > energy["ci95_low"]


def test_summary_rejects_mixed_sensor_boundaries():
    runs = [_run(1.0) for _ in range(4)] + [_run(1.0, "gpu_rail")]

    with pytest.raises(ValueError, match="sensor scope"):
        summarize_runs(runs, rail_name="board")


def test_summary_rejects_missing_rail():
    runs = [_run(1.0) for _ in range(5)]

    with pytest.raises(ValueError, match="rail"):
        summarize_runs(runs, rail_name="total_module")


def test_summary_preserves_negative_auxiliary_rail_increments():
    summary = summarize_runs([_run(-0.1) for _ in range(5)], "board")

    assert summary["incremental_j_per_forward"]["mean"] == pytest.approx(-0.1)
