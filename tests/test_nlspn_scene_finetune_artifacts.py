import csv

import numpy as np
import pytest
from PIL import Image

from scripts import nlspn_scene_finetune_artifacts as artifacts


def summary(overall, room3, room7):
    return {
        "pooled_rmse": overall,
        "scenes": {"room3": {"rmse": room3}, "room7": {"rmse": room7}},
    }


def minimal_valid_test_rows():
    rows = []
    for variant, scale in (("generic", 1.0), ("specialized", 0.9)):
        for frame_id, scene in enumerate(("room3", "room7"), start=1):
            row = {
                "variant": variant, "scene": scene, "frame_id": frame_id,
                "squared_error_sum": 5.0 * scale,
                "absolute_error_sum": 5.0 * scale,
                "abs_rel_sum": 0.5 * scale, "valid_pixels": 5,
            }
            for band in ("0_2", "2_4", "4_6", "6_8", "8_10"):
                row["band_{}_squared_error_sum".format(band)] = 1.0 * scale
                row["band_{}_absolute_error_sum".format(band)] = 1.0 * scale
                row["band_{}_valid_pixels".format(band)] = 1
            rows.append(row)
    return rows


def test_aggregate_recomputes_pooled_scene_macro_and_bands():
    rows = minimal_valid_test_rows()
    result = artifacts.aggregate_test_rows(rows, exact_geometry=False)
    baseline = result["variants"]["generic"]
    expected = np.sqrt(sum(float(row["squared_error_sum"]) for row in rows
                           if row["variant"] == "generic") /
                       sum(int(row["valid_pixels"]) for row in rows
                           if row["variant"] == "generic"))
    assert baseline["pooled_rmse"] == pytest.approx(expected)
    assert tuple(baseline["depth_bands"]) == artifacts.DEPTH_BAND_NAMES
    assert baseline["scene_macro_rmse"] == pytest.approx(expected)


def test_gate_requires_five_percent_and_no_scene_regression():
    baseline = summary(overall=1.0, room3=1.0, room7=1.0)
    specialized = summary(overall=0.94, room3=0.95, room7=0.93)
    result = artifacts.calculate_success_gate(baseline, specialized)
    assert result["passed"]
    assert result["relative_improvement"] == pytest.approx(0.06)
    specialized = summary(overall=0.94, room3=1.02, room7=0.86)
    assert not artifacts.calculate_success_gate(baseline, specialized)["passed"]


def test_gate_uses_unrounded_values_at_five_percent_boundary():
    baseline = summary(1.0, 1.0, 1.0)
    specialized = summary(0.95000001, 1.0, 0.9)
    result = artifacts.calculate_success_gate(baseline, specialized)
    assert result["relative_improvement"] < 0.05
    assert not result["pooled_improvement_passed"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows + [dict(rows[0])], "duplicate"),
    (lambda rows: [dict(rows[0], scene="room6")] + rows[1:], "scene"),
    (lambda rows: [dict(rows[0], squared_error_sum=float("nan"))] + rows[1:],
     "nonfinite"),
    (lambda rows: [dict(rows[0], band_0_2_valid_pixels=2)] + rows[1:],
     "band.*count"),
])
def test_aggregate_rejects_corrupt_rows(mutation, match):
    with pytest.raises(ValueError, match=match):
        artifacts.aggregate_test_rows(
            mutation(minimal_valid_test_rows()), exact_geometry=False)


def test_exact_geometry_rejects_non_16000_rows():
    with pytest.raises(ValueError, match="16000"):
        artifacts.aggregate_test_rows(
            minimal_valid_test_rows(), exact_geometry=True)


def _window_payload():
    shape = (150, 4, 5)
    gt = np.linspace(0.5, 9.5, np.prod(shape), dtype=np.float32).reshape(shape)
    return {
        "scenes": np.asarray(["room3"] * 75 + ["room7"] * 75),
        "frame_ids": np.concatenate((np.arange(1, 76), np.arange(1, 76))),
        "rgb": np.zeros((150, 3, 4, 5), dtype=np.float32),
        "sparse": np.ones(shape, dtype=np.float32),
        "gt": gt, "valid": np.ones(shape, dtype=bool),
        "generic": gt + 0.2, "specialized": gt + 0.1,
    }


def test_write_window_artifacts_creates_thirty_nonblank_bundles(tmp_path):
    result = artifacts.write_window_artifacts(tmp_path, _window_payload())
    assert result["window_count"] == 30
    assert len(list(tmp_path.rglob("depth_comparison.png"))) == 30
    assert len(list(tmp_path.rglob("error_comparison.png"))) == 30
    assert len(list(tmp_path.rglob("predictions.npz"))) == 30
    for path in tmp_path.rglob("*.png"):
        artifacts.validate_png(path, require_nonblank=True)


def test_validate_png_rejects_blank_panel(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (20, 20), color="white").save(path)
    with pytest.raises(ValueError, match="blank"):
        artifacts.validate_png(path, require_nonblank=True)
