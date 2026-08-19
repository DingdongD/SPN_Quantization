import csv

import numpy as np
import pytest
from PIL import Image

from scripts import export_nlspn_invalid_region_predictions as exporter


def test_depth_to_millimetres_rounds_and_clips():
    depth = np.array(
        [[-1.0, 0.0, 1.234, 1.2346, 10.0, 11.0]], dtype=np.float32)
    actual = exporter.depth_to_millimetres(depth)
    assert actual.dtype == np.uint16
    assert actual.tolist() == [[0, 0, 1234, 1235, 10000, 10000]]


def test_compose_fill_uses_gt_only_where_valid():
    gt = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    prediction = np.array([[8.0, 8.1], [8.2, 8.3]], dtype=np.float32)
    actual = exporter.compose_gt_with_prediction(gt, valid, prediction)
    expected = np.array([[1.0, 8.1], [3.0, 8.3]], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_invalid_mask_is_white_only_for_invalid_gt():
    valid = np.array([[True, False], [False, True]])
    actual = exporter.invalid_mask(valid)
    assert actual.dtype == np.uint8
    assert actual.tolist() == [[0, 255], [255, 0]]


def test_colorize_depth_has_fixed_zero_to_ten_metre_scale():
    depth = np.array([[0.0, 5.0, 10.0, -1.0, 11.0]], dtype=np.float32)
    actual = exporter.colorize_depth(depth)
    assert actual.shape == (1, 5, 3)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[0, 0], actual[0, 3])
    np.testing.assert_array_equal(actual[0, 2], actual[0, 4])
    assert not np.array_equal(actual[0, 0], actual[0, 1])
    assert not np.array_equal(actual[0, 1], actual[0, 2])


@pytest.mark.parametrize(
    "function_name", ["depth_to_millimetres", "colorize_depth"])
def test_prediction_products_reject_nonfinite_values(function_name):
    function = getattr(exporter, function_name)
    with pytest.raises(ValueError, match="finite"):
        function(np.array([[np.nan]], dtype=np.float32))


def make_source(root):
    windows = root / "windows"
    windows.mkdir(parents=True)
    identities = []
    for index in range(30):
        scene = "room3" if index < 15 else "room7"
        start = index * 10 + 1
        name = "{:02d}_{}_{:04d}_{:04d}".format(
            index + 1, scene, start, start + 4)
        directory = windows / name
        directory.mkdir()
        frame_ids = np.arange(start, start + 5, dtype=np.int32)
        gt = np.full((5, 228, 304), 2.0, dtype=np.float32)
        valid = np.ones((5, 228, 304), dtype=bool)
        valid[:, 70:90, 90:120] = False
        gt[~valid] = 0.0
        specialized = np.full((5, 228, 304), 8.25, dtype=np.float32)
        np.savez_compressed(
            directory / "predictions.npz",
            scenes=np.asarray([scene] * 5), frame_ids=frame_ids,
            gt=gt, valid=valid, specialized=specialized)
        identities.extend((scene, int(frame_id)) for frame_id in frame_ids)
    return windows, identities


def test_load_source_frames_accepts_exact_formal_geometry(tmp_path):
    windows, identities = make_source(tmp_path)
    frames, fingerprints = exporter.load_source_frames(windows)
    assert [(row["scene"], row["frame_id"]) for row in frames] == identities
    assert len(frames) == 150
    assert len(fingerprints) == 30
    assert all(len(value) == 64 for value in fingerprints.values())


def test_load_source_frames_rejects_duplicate_identity(tmp_path):
    windows, _ = make_source(tmp_path)
    path = sorted(windows.iterdir())[1] / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    first = sorted(windows.iterdir())[0] / "predictions.npz"
    with np.load(first, allow_pickle=False) as archive:
        payload["scenes"] = archive["scenes"]
        payload["frame_ids"] = archive["frame_ids"]
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="duplicate"):
        exporter.load_source_frames(windows)


def test_load_source_frames_rejects_nonfinite_prediction(tmp_path):
    windows, _ = make_source(tmp_path)
    path = sorted(windows.iterdir())[0] / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["specialized"][0, 0, 0] = np.nan
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="finite"):
        exporter.load_source_frames(windows)


def test_load_source_frames_rejects_nonboolean_validity(tmp_path):
    windows, _ = make_source(tmp_path)
    path = sorted(windows.iterdir())[0] / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["valid"] = payload["valid"].astype(np.uint8)
    payload["valid"][0, 0, 0] = 2
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="boolean-compatible"):
        exporter.load_source_frames(windows)


def test_export_predictions_writes_exact_products_and_manifest(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    result = exporter.export_predictions(windows, target)
    assert result == {"window_count": 30, "frame_count": 150,
                      "png_count": 600}
    with (target / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 150
    first = rows[0]
    frame = target / "windows" / first["window"] / "frame_0001"
    assert {path.name for path in frame.iterdir()} == set(exporter.PNG_NAMES)
    with Image.open(frame / "specialized_depth_mm.png") as image:
        assert image.size == (304, 228)
        depth = np.asarray(image)
    assert depth.dtype in (np.dtype("uint16"), np.dtype("int32"))
    assert np.all(depth == 8250)
    with Image.open(frame / "invalid_mask.png") as image:
        mask = np.asarray(image)
    assert set(np.unique(mask)) == {0, 255}
    exporter.validate_export(target, windows)


def test_export_predictions_refuses_to_overwrite_completed_target(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        exporter.export_predictions(windows, target)
    assert marker.read_text() == "keep"


def test_validate_export_detects_changed_hybrid_pixel(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    exporter.export_predictions(windows, target)
    path = (target / "windows/01_room3_0001_0005/frame_0001/"
            "gt_with_prediction_fill.png")
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    array[0, 0] = 0
    Image.fromarray(array, mode="RGB").save(path)
    with pytest.raises(ValueError, match="hybrid"):
        exporter.validate_export(target, windows)


def test_validate_export_rejects_extra_frame_file(tmp_path):
    windows, _ = make_source(tmp_path / "source")
    target = tmp_path / "export"
    exporter.export_predictions(windows, target)
    path = target / "windows/01_room3_0001_0005/frame_0001/extra.txt"
    path.write_text("unexpected")
    with pytest.raises(ValueError, match="exact contract"):
        exporter.validate_export(target, windows)
