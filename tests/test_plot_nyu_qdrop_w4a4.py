from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest

from scripts.plot_nyu_qdrop_w4a4 import (
    PREDICTION_COLUMNS,
    load_prediction_arrays,
    plot_predictions,
    plot_rmse_summary,
    select_visual_samples,
    validate_prediction_sources,
)


def test_plot_script_runs_directly_outside_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / \
        "scripts" / "plot_nyu_qdrop_w4a4.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_prediction_columns_cover_exact_w4_and_w6_comparisons():
    assert PREDICTION_COLUMNS == {
        "W4A4": ("gt", "fp32", "rtn", "brecq", "qdrop"),
        "W6A6": (
            "gt", "fp32", "rtn", "brecq", "qdrop", "p3_t3"),
    }


def test_visual_samples_use_preselected_seed_and_precision():
    rows = [
        {
            "method": "qdrop",
            "precision": precision,
            "seed": seed,
            "sample_index": index,
            "RMSE": value + seed * 1.0e-6,
        }
        for precision in ("W4A4", "W6A6")
        for seed in (1005, 1006, 1007)
        for index, value in ((3, 0.1), (7, 0.3), (9, 0.2))
    ]

    assert select_visual_samples(rows, "W4A4", 1006) == (9, 7)


def _payload(sample_index, pred_value):
    return {
        "sample_index": np.asarray(sample_index),
        "rgb": np.full((2, 3, 3), 0.5, dtype=np.float32),
        "sparse": np.full((2, 3), 1.0, dtype=np.float32),
        "gt": np.full((2, 3), 2.0, dtype=np.float32),
        "fp32": np.full((2, 3), 2.1, dtype=np.float32),
        "pred": np.full((2, 3), pred_value, dtype=np.float32),
    }


def test_prediction_sources_require_exact_inputs_and_close_fp32():
    sources = {
        name: _payload(7, 2.0 + rank * 0.1)
        for rank, name in enumerate(
            ("fp32", "rtn", "brecq", "qdrop", "p3_t3"))
    }

    validate_prediction_sources(sources)

    close_sources = dict(sources)
    close_qdrop = dict((name, value.copy())
                       for name, value in sources["qdrop"].items())
    close_qdrop["fp32"] += 1.0e-6
    close_sources["qdrop"] = close_qdrop
    validate_prediction_sources(close_sources)

    for field in ("sample_index", "rgb", "sparse", "gt", "fp32"):
        broken = dict(sources)
        changed = dict((name, value.copy())
                       for name, value in sources["qdrop"].items())
        changed[field].flat[0] += 1
        broken["qdrop"] = changed
        with pytest.raises(ValueError, match=field):
            validate_prediction_sources(broken)


def _save(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def test_loader_uses_selected_qdrop_seed_without_prediction_averaging(tmp_path):
    index = 7
    config = "PA_W6A6_PROP_A8"
    root = tmp_path / "unified"
    p3_root = tmp_path / "p3"
    filename = "sample_00007.npz"
    _save(
        root / "evaluation" / "rtn" / "cspn" / "predictions" /
        "FP32" / filename,
        _payload(index, 2.1),
    )
    _save(
        root / "evaluation" / "rtn" / "cspn" / "predictions" /
        config / filename,
        _payload(index, 2.2),
    )
    _save(
        root / "evaluation" / "brecq" / "W6A6" / "cspn" /
        "predictions" / config / filename,
        _payload(index, 2.3),
    )
    for seed, value in ((1005, 2.4), (1006, 2.5), (1007, 2.6)):
        _save(
            root / "evaluation" / "qdrop" / "W6A6" /
            ("seed_%d" % seed) / "cspn" / "predictions" /
            config / filename,
            _payload(index, value),
        )
    _save(
        p3_root / "predictions" / "CONTEXT_P3_T3_W8A8" / filename,
        _payload(index, 2.7),
    )

    arrays = load_prediction_arrays(
        root, p3_root, index, "W6A6", 1006)

    assert np.all(arrays["qdrop"] == np.float32(2.5))
    assert np.all(arrays["p3_t3"] == np.float32(2.7))
    assert "qdrop_mean" not in arrays


def test_plotters_emit_png_and_pdf_without_titles(tmp_path):
    arrays = {
        "rgb": np.full((4, 6, 3), 0.5, dtype=np.float32),
        "sparse": np.full((4, 6), 1.0, dtype=np.float32),
        "gt": np.full((4, 6), 2.0, dtype=np.float32),
        "fp32": np.full((4, 6), 2.1, dtype=np.float32),
        "rtn": np.full((4, 6), 2.2, dtype=np.float32),
        "brecq": np.full((4, 6), 2.3, dtype=np.float32),
        "qdrop": np.full((4, 6), 2.4, dtype=np.float32),
        "p3_t3": np.full((4, 6), 2.5, dtype=np.float32),
    }
    seed_rows = []
    for method, precision in (
            ("fp32", "FP32"), ("rtn", "W4A4"),
            ("brecq", "W4A4"), ("rtn", "W6A6"),
            ("brecq", "W6A6"), ("p3_t3", "P3T3")):
        seed_rows.append({
            "method": method,
            "precision": precision,
            "mean_rmse": (
                float("inf") if
                (method, precision) == ("rtn", "W4A4") else 0.2),
            "nonfinite_ratio": (
                1.0 if
                (method, precision) == ("rtn", "W4A4") else 0.0),
        })
    for precision in ("W4A4", "W6A6"):
        for value in (0.18, 0.20, 0.22):
            seed_rows.append({
                "method": "qdrop",
                "precision": precision,
                "mean_rmse": value,
                "nonfinite_ratio": 0.0,
            })

    plot_predictions(
        tmp_path / "w6_predictions", "W6A6", {"Median": arrays})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plot_rmse_summary(tmp_path / "rmse", seed_rows)

    for name in ("w6_predictions.png", "w6_predictions.pdf",
                 "rmse.png", "rmse.pdf"):
        assert (tmp_path / name).stat().st_size > 0
