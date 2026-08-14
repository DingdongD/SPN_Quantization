from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_nyu_cspn_group_a4_qat as evaluator


EXPECTED = (
    "FP32",
    "PTQ_STATIC_G8_W4A4",
    "PTQ_DYNAMIC_G8_W4A4",
    "QAT_STATIC_G8_W4A4",
    "QAT_DYNAMIC_G8_W4A4",
)


def _write_predictions(root: Path, indices):
    for config in EXPECTED:
        directory = root / "predictions" / config
        directory.mkdir(parents=True)
        for index in indices:
            np.savez_compressed(
                directory / ("sample_%05d.npz" % index),
                sample_index=np.array(index), config=np.array(config))


def test_evaluation_matrix_is_fixed():
    assert evaluator.CONFIGURATIONS == EXPECTED


def test_prediction_coverage_requires_same_64_indices(tmp_path):
    indices = tuple(range(64))
    _write_predictions(tmp_path, indices)
    evaluator.validate_prediction_coverage(tmp_path, indices)

    (tmp_path / "predictions" / EXPECTED[-1] /
     "sample_00063.npz").unlink()
    with pytest.raises(RuntimeError, match="prediction coverage"):
        evaluator.validate_prediction_coverage(tmp_path, indices)


def test_configuration_mode_is_explicit():
    assert evaluator.configuration_mode("FP32") is None
    assert evaluator.configuration_mode("PTQ_STATIC_G8_W4A4") == "static"
    assert evaluator.configuration_mode("QAT_DYNAMIC_G8_W4A4") == "dynamic"
    with pytest.raises(KeyError):
        evaluator.configuration_mode("unknown")


def test_canonical_checkpoint_rejects_parametrization_keys():
    with pytest.raises(ValueError, match="parametrization"):
        evaluator.validate_canonical_state({
            "conv.parametrizations.weight.original": np.array([1.0]),
        })


def test_qat_checkpoint_loads_after_source_range_calibration(monkeypatch):
    calls = []

    class Instrumentor:
        def refresh_parameter_sources(self):
            calls.append("refresh")

    def calibrate(*args):
        calls.append("calibrate")

    def load(*args):
        calls.append("load")
        return {"epoch": 1, "validation": {"RMSE": 0.25}}

    monkeypatch.setattr(evaluator.base, "_calibrate", calibrate)
    monkeypatch.setattr(evaluator, "_load_canonical_checkpoint", load)

    source = evaluator.prepare_deployment_state(
        "QAT_STATIC_G8_W4A4", object(), object(), object(),
        (0,), object(), 1, Instrumentor(), object(), object(), Path("qat.pt"))

    assert calls == ["calibrate", "load", "refresh"]
    assert source["epoch"] == 1
