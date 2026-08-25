from argparse import Namespace
import math

import numpy as np
import pytest

from scripts.run_nyu_qdrop_w4a4 import (
    MODEL_ORDER,
    PRECISION_ORDER,
    QDROP_EVALUATION_BACKEND,
    QDROP_EVALUATION_CONFIGS,
    _brecq_command,
    _depth_metrics,
    _prepare_process,
    _qdrop_command,
    _validate_reference_payload,
    aggregate_model_metrics,
    build_execution_waves,
    build_run_matrix,
    validate_aligned_sample_rows,
    validate_qdrop_layer_rows,
)


def test_strict_depth_metrics_separate_nonfinite_and_nonpositive_pixels():
    gt = np.ones((1, 1, 1, 4), dtype=np.float32)
    pred = np.array([[[[1.0, np.nan, 0.0, -0.25]]]], dtype=np.float32)

    metrics = _depth_metrics(gt, pred)

    assert math.isinf(metrics["RMSE"])
    assert metrics["nonfinite_pixels"] == 1
    assert metrics["nonpositive_pixels"] == 2
    assert metrics["invalid_pixels"] == 3
    assert metrics["prediction_min"] == -0.25


def _command_args(tmp_path):
    return Namespace(
        config=str(tmp_path / "qdrop.json"),
        run_dir=str(tmp_path / "run"),
        checkpoint=str(tmp_path / "best.pt"),
        data_root=str(tmp_path / "data"),
        calibration_indices=str(tmp_path / "indices.json"),
        calibration_metadata=str(tmp_path / "calibration.json"),
        evaluation_protocol=str(tmp_path / "evaluation.json"),
        out_dir=str(tmp_path / "output"),
    )


def test_formal_matrix_covers_two_precisions_and_three_seeds():
    rows = build_run_matrix(
        phase="formal",
        seeds=(1005, 1006, 1007),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )

    assert len(rows) == 6
    assert set(row["model"] for row in rows) == set(MODEL_ORDER)
    assert set(row["precision"] for row in rows) == set(PRECISION_ORDER)
    assert set(row["seed"] for row in rows) == {1005, 1006, 1007}
    assert set((row["weight_bits"], row["activation_bits"])
               for row in rows) == {(4, 4), (6, 6)}


def test_brecq_matrix_covers_two_precisions_once():
    rows = build_run_matrix(
        phase="brecq",
        seeds=(1005, 1006, 1007),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )

    assert len(rows) == 2
    assert tuple(row["precision"] for row in rows) == PRECISION_ORDER
    assert all(row["method"] == "brecq" for row in rows)
    assert all(row["seed"] == 0 for row in rows)


def test_reconstruction_commands_bind_precision_and_shared_protocol(tmp_path):
    args = _command_args(tmp_path)
    row = {
        "method": "brecq",
        "precision": "W6A6",
        "seed": 1006,
        "device": "cuda:2",
        "weight_bits": 6,
        "activation_bits": 6,
    }

    brecq = _brecq_command(row, args)
    qdrop = _qdrop_command(dict(row, method="qdrop"), args)

    for command in (brecq, qdrop):
        assert command[command.index("--device") + 1] == "cuda:2"
        assert command[command.index("--calibration-indices") + 1] == \
            str(tmp_path / "indices.json")
        assert command[command.index("--calibration-metadata") + 1] == \
            str(tmp_path / "calibration.json")
        assert command[command.index("--evaluation-protocol") + 1] == \
            str(tmp_path / "evaluation.json")
    assert brecq[1].endswith("run_nyu_qdrop_reconstruction.py")
    assert brecq[brecq.index("--algorithm") + 1] == "brecq"
    assert brecq[brecq.index("--precision") + 1] == "W6A6"
    assert brecq[brecq.index("--seed") + 1] == "20260812"
    assert "--w-bits" not in brecq
    assert "--qdrop-target-plan" not in brecq
    assert qdrop[qdrop.index("--algorithm") + 1] == "qdrop"
    assert qdrop[qdrop.index("--precision") + 1] == "W6A6"
    assert qdrop[qdrop.index("--seed") + 1] == "1006"


def test_qdrop_evaluation_uses_propagation_aware_exclusive_ownership():
    assert QDROP_EVALUATION_BACKEND == "propagation"
    assert QDROP_EVALUATION_CONFIGS == {
        "W4A4": "PA_W4A4_PROP_A8",
        "W6A6": "PA_W6A6_PROP_A8",
    }


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
            devices=(),
        )


def test_formal_commands_fill_four_gpu_waves_without_collision():
    rows = build_run_matrix(
        phase="formal",
        seeds=(1005, 1006, 1007),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )

    waves = build_execution_waves(rows)

    assert tuple(len(wave) for wave in waves) == (4, 2)
    assert all(
        len(set(row["device"] for row in wave)) == len(wave)
        for wave in waves)


def test_cspn_process_runs_from_explicit_data_root(tmp_path):
    row = {
        "device": "cuda:2",
        "command": ["python", "runner.py", "--device", "cuda:2"],
        "working_directory": str(tmp_path / "official_cspn"),
    }

    command, environment, working_directory = _prepare_process(
        row, {"PYTHONPATH": "quantization"})

    assert command[-1] == "cuda:0"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert working_directory == str(tmp_path / "official_cspn")


def test_aligned_rows_require_every_method_seed_and_sample_once():
    indices = (3, 7)
    rows = []
    method_precisions = (
        ("fp32", "FP32", (0,)),
        ("rtn", "W4A4", (0,)),
        ("rtn", "W6A6", (0,)),
        ("brecq", "W4A4", (0,)),
        ("brecq", "W6A6", (0,)),
        ("qdrop", "W4A4", (1005, 1006, 1007)),
        ("qdrop", "W6A6", (1005, 1006, 1007)),
        ("p3_t3", "P3T3", (0,)),
    )
    for method, precision, seeds in method_precisions:
        for seed in seeds:
            for index in indices:
                rows.append({
                    "model": "cspn",
                    "method": method,
                    "precision": precision,
                    "seed": seed,
                    "sample_index": index,
                    "RMSE": 0.1,
                    "MAE": 0.05,
                    "ABS_REL": 0.01,
                    "IRMSE": 0.2,
                    "nonfinite_pixels": 0,
                })

    validate_aligned_sample_rows(
        rows, "cspn", indices, (1005, 1006, 1007), PRECISION_ORDER)

    with pytest.raises(ValueError):
        validate_aligned_sample_rows(
            rows[:-1], "cspn", indices,
            (1005, 1006, 1007), PRECISION_ORDER)


def test_nonpositive_depth_is_recorded_as_invalid_output():
    gt = np.ones((2, 2), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)
    pred[0, 0] = 0.0

    metrics = _depth_metrics(gt, pred)

    assert metrics["nonfinite_pixels"] == 0
    assert metrics["nonpositive_pixels"] == 1
    assert metrics["invalid_pixels"] == 1
    assert np.isinf(metrics["RMSE"])


def test_reference_payload_requires_identical_inputs_and_close_fp32():
    reference = {
        "sample_index": np.asarray(7),
        "gt": np.ones((2, 3), dtype=np.float32),
        "fp32": np.full((2, 3), 2.0, dtype=np.float32),
        "sparse": np.full((2, 3), 3.0, dtype=np.float32),
        "rgb": np.full((2, 3, 3), 4.0, dtype=np.float32),
    }
    current = dict((name, value.copy()) for name, value in reference.items())

    _validate_reference_payload(reference, current)

    current["fp32"] += 1.0e-6
    _validate_reference_payload(reference, current)

    for field in ("sample_index", "gt", "fp32", "sparse", "rgb"):
        broken = dict((name, value.copy())
                      for name, value in current.items())
        broken[field].flat[0] += 1
        with pytest.raises(ValueError, match=field):
            _validate_reference_payload(reference, broken)


def test_model_aggregation_reports_seed_spread_and_acceptance():
    rows = [
        {"model": "cspn", "method": "fp32", "precision": "FP32", "seed": 0,
         "mean_rmse": 1.0, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "rtn", "precision": "W4A4", "seed": 0,
         "mean_rmse": 1.3, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "brecq", "precision": "W4A4", "seed": 0,
         "mean_rmse": 1.2, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "qdrop", "precision": "W4A4", "seed": 1005,
         "mean_rmse": 1.08, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "qdrop", "precision": "W4A4", "seed": 1006,
         "mean_rmse": 1.10, "nonfinite_ratio": 0.0},
        {"model": "cspn", "method": "qdrop", "precision": "W4A4", "seed": 1007,
         "mean_rmse": 1.09, "nonfinite_ratio": 0.0},
    ]

    summary, acceptance = aggregate_model_metrics(
        rows, seeds=(1005, 1006, 1007), precisions=("W4A4",))

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
