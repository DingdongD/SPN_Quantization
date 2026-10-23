import pytest

from scripts.summarize_jetson_cpu_spn import summarize_cpu_model


def _main(value):
    return {
        "model": "cspn", "precision": "fp32",
        "checkpoint_sha256": "checkpoint", "config_sha256": "config",
        "fixed_input_artifact": {"sha256": "input"},
        "settings": {"cpu_threads": 4, "interop_threads": 1,
                     "warmup_iterations": 5,
                     "timed_iterations_per_round": 3, "rounds": 3,
                     "python_gc_enabled_during_timing": False},
        "cpu_execution_audit": {"execution_device": "cpu",
                                "cuda_available": False},
        "latency_ms": {"mean": value},
        "raw_latency_ms": [value] * 9,
        "breakdown": {
            "profiled_iterations": 2,
            "category_mean_self_cpu_time_us": {
                "dense_compute": 600.0, "attention": 0.0,
                "sampling": 10.0, "propagation": 200.0,
                "elementwise_norm": 100.0, "layout_memory": 50.0,
                "other_unclassified": 40.0,
            },
            "self_cpu_time_sum_mean_us": 1000.0,
            "negative_self_time_events": {"count": 1,
                                           "total_magnitude_us": 2.0},
        },
    }


def _energy(value):
    def rail(scale):
        return {
            "gross_j_per_forward": value * scale + 1.0,
            "incremental_j_per_forward": value * scale,
            "idle": {"mean_power_w": 2.0, "sample_count": 300},
            "active": {"mean_power_w": 4.0, "sample_count": 1200},
        }
    return {
        "model": "cspn",
        "settings": {"idle_seconds": 15.0, "active_seconds": 60.0,
                     "sample_interval_ms": 50.0,
                     "synchronize_each_forward": True,
                     "warmup_iterations": 5},
        "workload": {"completed_forwards": 20,
                     "active_wall_seconds": 60.1,
                     "latency_ms": {"mean": 10.0}},
        "power": {"source": "jtop_ina3221",
                  "sensor_scope": "jetson_module_and_rails",
                  "rails": {"cpu": rail(1.0), "total_module": rail(2.0)}},
    }


def test_cpu_summary_uses_process_means_and_exclusive_categories():
    values = (10.0, 11.0, 9.0, 10.5, 9.5)

    result = summarize_cpu_model(
        [_main(value) for value in values],
        [_energy(value) for value in values])

    assert result["latency_ms"]["mean"] == pytest.approx(10.0)
    assert result["latency_sample_count"] == 45
    assert result["breakdown"]["category_share_percent"][
        "dense_compute"] == pytest.approx(60.0)
    assert result["energy"]["cpu"]["incremental_j_per_forward"][
        "mean"] == pytest.approx(10.0)


def test_cpu_summary_rejects_cuda_available_process():
    mains = [_main(10.0) for _ in range(5)]
    mains[2]["cpu_execution_audit"]["cuda_available"] = True

    with pytest.raises(ValueError, match="CPU-only"):
        summarize_cpu_model(mains, [_energy(10.0) for _ in range(5)])


def test_cpu_summary_requires_nine_latency_samples_per_process():
    mains = [_main(10.0) for _ in range(5)]
    mains[0]["raw_latency_ms"].pop()

    with pytest.raises(ValueError, match="nine latency"):
        summarize_cpu_model(mains, [_energy(10.0) for _ in range(5)])
