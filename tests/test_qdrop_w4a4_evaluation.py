import math

import pytest

from scripts.run_nyu_qdrop_w4a4 import (
    MODEL_ORDER,
    QDROP_EVALUATION_BACKEND,
    QDROP_EVALUATION_CONFIG,
    aggregate_model_metrics,
    build_execution_waves,
    build_run_matrix,
    validate_aligned_sample_rows,
    validate_qdrop_layer_rows,
)


def test_run_matrix_is_four_models_three_seeds_and_w4a4_only():
    rows = build_run_matrix(
        phase="formal",
        seeds=(1005, 1006, 1007),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )

    assert len(rows) == 12
    assert set(row["model"] for row in rows) == set(MODEL_ORDER)
    assert set(row["seed"] for row in rows) == {1005, 1006, 1007}
    assert all(row["weight_bits"] == 4 for row in rows)
    assert all(row["activation_bits"] == 4 for row in rows)


def test_qdrop_evaluation_uses_propagation_aware_exclusive_ownership():
    assert QDROP_EVALUATION_BACKEND == "propagation"
    assert QDROP_EVALUATION_CONFIG == "PA_Constraint"


def test_run_matrix_rejects_missing_seed_or_device():
    with pytest.raises(ValueError):
        build_run_matrix(
            phase="formal",
            seeds=(1005, 1006),
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        )
    with pytest.raises(ValueError):
        build_run_matrix(
            phase="formal",
            seeds=(1005, 1006, 1007),
            devices=("cuda:0",),
        )


def test_formal_commands_execute_one_four_gpu_wave_per_seed():
    rows = build_run_matrix(
        phase="formal",
        seeds=(1005, 1006, 1007),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )

    waves = build_execution_waves(rows)

    assert len(waves) == 3
    assert all(len(wave) == 4 for wave in waves)
    assert tuple(row["model"] for row in waves[0]) == MODEL_ORDER
    assert all(
        len(set(row["device"] for row in wave)) == 4
        for wave in waves)


def test_aligned_rows_require_every_method_seed_and_sample_once():
    indices = (3, 7)
    rows = []
    for method, seeds in (
            ("fp32", (0,)), ("rtn", (0,)), ("brecq", (0,)),
            ("qdrop", (1005, 1006, 1007))):
        for seed in seeds:
            for index in indices:
                rows.append({
                    "model": "cspn",
                    "method": method,
                    "seed": seed,
                    "sample_index": index,
                    "RMSE": 0.1,
                    "MAE": 0.05,
                    "ABS_REL": 0.01,
                    "nonfinite_pixels": 0,
                })

    validate_aligned_sample_rows(
        rows, "cspn", indices, (1005, 1006, 1007))

    with pytest.raises(ValueError):
        validate_aligned_sample_rows(
            rows[:-1], "cspn", indices, (1005, 1006, 1007))
    broken = [dict(row) for row in rows]
    broken[-1]["RMSE"] = math.inf
    with pytest.raises(ValueError):
        validate_aligned_sample_rows(
            broken, "cspn", indices, (1005, 1006, 1007))


def test_model_aggregation_reports_seed_spread_and_acceptance():
    rows = [
        {"model": "cspn", "method": "fp32", "seed": 0,
         "mean_rmse": 1.0, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "rtn", "seed": 0,
         "mean_rmse": 1.3, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "brecq", "seed": 0,
         "mean_rmse": 1.2, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "qdrop", "seed": 1005,
         "mean_rmse": 1.08, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "qdrop", "seed": 1006,
         "mean_rmse": 1.10, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "qdrop", "seed": 1007,
         "mean_rmse": 1.09, "nonfinite_ratio": 0.0},
    ]

    summary, acceptance = aggregate_model_metrics(
        rows, seeds=(1005, 1006, 1007))

    assert summary[0]["mean_rmse"] == pytest.approx(1.09)
    assert summary[0]["min_rmse"] == pytest.approx(1.08)
    assert summary[0]["max_rmse"] == pytest.approx(1.10)
    assert acceptance[0]["better_than_brecq"] == 1
    assert acceptance[0]["within_fp32_10pct"] == 1


def test_qdrop_layer_rows_require_executed_finite_exact_a4_sites():
    row = {
        "kind": "exact_activation_contract",
        "module": "activation::conv::input",
        "calls": "64",
        "numel": "128",
        "zero_code_rate": "0.25",
        "saturation_rate": "0.05",
        "sqnr_db": "12.0",
    }

    validate_qdrop_layer_rows([row])

    for field, value in (
            ("calls", "0"), ("numel", "0"),
            ("zero_code_rate", "nan"),
            ("saturation_rate", "1.1"), ("sqnr_db", "nan")):
        broken = dict(row)
        broken[field] = value
        with pytest.raises(ValueError):
            validate_qdrop_layer_rows([broken])
