from pathlib import Path

import numpy as np
import pytest
import torch

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


def test_visualization_upgrade_preserves_model_rgb(tmp_path):
    index = 3
    model_rgb = np.full((2, 4, 3), 0.75, dtype=np.float32)
    natural_rgb = np.full((2, 4, 3), 0.25, dtype=np.float32)
    for config in EXPECTED:
        directory = tmp_path / "predictions" / config
        directory.mkdir(parents=True)
        np.savez_compressed(
            directory / ("sample_%05d.npz" % index),
            gt=np.ones((2, 4), dtype=np.float32),
            fp32=np.ones((2, 4), dtype=np.float32),
            pred=np.ones((2, 4), dtype=np.float32),
            abs_err=np.zeros((2, 4), dtype=np.float32),
            valid_gt=np.ones((2, 4), dtype=np.bool_),
            nonfinite=np.zeros((2, 4), dtype=np.bool_),
            sample_index=np.array(index),
            model=np.array("cspn"),
            config=np.array(config),
            sparse=np.zeros((2, 4), dtype=np.float32),
            rgb=model_rgb,
        )

    class VisualizationDataset:
        def __getitem__(self, sample_index):
            assert sample_index == index
            rgb = torch.from_numpy(natural_rgb).permute(2, 0, 1)
            return {"rgbd": torch.cat((rgb, torch.zeros(1, 2, 4)), dim=0)}

    evaluator.upgrade_prediction_visuals(
        tmp_path, (index,), VisualizationDataset())

    for config in EXPECTED:
        path = tmp_path / "predictions" / config / \
            ("sample_%05d.npz" % index)
        with np.load(path, allow_pickle=False) as source:
            assert set(source.files) == evaluator.VISUAL_PREDICTION_FIELDS
            np.testing.assert_array_equal(source["model_rgb"], model_rgb)
            np.testing.assert_array_equal(source["rgb"], natural_rgb)
    assert (tmp_path / "prediction_visualization_manifest.json").is_file()
