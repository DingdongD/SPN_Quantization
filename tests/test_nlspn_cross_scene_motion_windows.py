from pathlib import Path
import csv

import numpy as np
from PIL import Image
import pytest

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_cross_scene_motion_windows as motion


def test_select_motion_window_scores_four_pairs_and_chooses_maximum():
    ids = (1, 2, 3, 4, 5, 6)
    thumbnails = {
        frame_id: np.full((2, 3), value, dtype=np.float32)
        for frame_id, value in zip(ids, (0, 0, 0, 0, 1, 1))
    }

    result = motion.select_motion_window(ids, thumbnails)

    assert result["frame_ids"] == [1, 2, 3, 4, 5]
    assert result["pair_scores"] == [0.0, 0.0, 0.0, 1.0]
    assert result["motion_score"] == pytest.approx(0.25)


def test_select_motion_window_rejects_gaps_and_breaks_ties_early():
    ids = (1, 2, 3, 4, 5, 10, 11, 12, 13, 14)
    thumbnails = {
        frame_id: np.zeros((2, 3), dtype=np.float32)
        for frame_id in ids
    }

    result = motion.select_motion_window(ids, thumbnails)

    assert result["frame_ids"] == [1, 2, 3, 4, 5]
    assert result["motion_score"] == 0.0


@pytest.mark.parametrize(
    "ids, thumbnails, message",
    [
        ((1, 1, 2, 3, 4), {}, "unique"),
        ((1, 2, 3, 4, 5), {i: np.zeros((2, 3), dtype=np.float64)
                            for i in range(1, 6)}, "float32"),
        ((1, 2, 3, 4, 5), {i: np.zeros((2, 3), dtype=np.float32)
                            for i in range(1, 5)}, "missing"),
    ],
)
def test_select_motion_window_validates_inputs(ids, thumbnails, message):
    with pytest.raises(ValueError, match=message):
        motion.select_motion_window(ids, thumbnails)


def _write_scene(root: Path, frame_ids, missing_depth=()):
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    missing_depth = set(missing_depth)
    for frame_id in frame_ids:
        pixels = np.full((12, 16, 3), frame_id * 20, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(
            rgb_dir / "{:04d}.jpg".format(frame_id))
        if frame_id not in missing_depth:
            (depth_dir / "Image{:04d}.exr".format(frame_id)).write_bytes(b"exr")
    return root


def test_scan_scene_intersects_rgb_and_depth_and_loads_76x57(tmp_path):
    scene = _write_scene(tmp_path / "room3", range(1, 7), missing_depth={6})

    result = motion.scan_scene(scene)

    assert result["scene"] == "room3"
    assert result["frame_ids"] == [1, 2, 3, 4, 5]
    assert result["start_frame"] == 1
    assert result["end_frame"] == 5
    assert len(result["pair_scores"]) == 4


def test_scan_scene_requires_five_consecutive_complete_frames(tmp_path):
    scene = _write_scene(tmp_path / "room4", (1, 2, 3, 5, 6, 7, 8))

    with pytest.raises(ValueError, match="five consecutive"):
        motion.scan_scene(scene)


def test_require_fixed_configs_accepts_only_formal_values():
    configs = {
        variant: cache.CacheConfig(variant, 2.0 / 255.0, 8)
        for variant in ("rgb_diff", "global_diff")
    }

    result = motion.require_fixed_configs(configs)

    assert result["rgb_diff"]["threshold"] == pytest.approx(2.0 / 255.0)
    assert result["global_diff"]["dilation_radius"] == 8
    configs["rgb_diff"] = cache.CacheConfig("rgb_diff", 4.0 / 255.0, 8)
    with pytest.raises(ValueError, match="2/255"):
        motion.require_fixed_configs(configs)


def _fake_frame_metrics(scene="room3", methods=None):
    methods = motion.METHOD_ORDER if methods is None else tuple(methods)
    rows = []
    rmse_by_method = {
        "full": 1.0,
        "zero_flow": 1.02,
        "rgb_diff": 0.99,
        "global_diff": 1.01,
        "raft_gop2": 1.005,
    }
    for method in methods:
        for frame_id in range(11, 16):
            rows.append({
                "scene": scene,
                "method": method,
                "frame_id": frame_id,
                "rmse": rmse_by_method[method],
                "mae": rmse_by_method[method] / 2.0,
                "valid_pixels": 100,
                "latency_ms": 2.0,
            })
    return rows


def test_build_scene_summary_emits_four_pooled_rows_and_ratios():
    summary = motion.build_scene_summary("room3", _fake_frame_metrics())

    assert len(summary) == 4
    zero = next(row for row in summary if row["method"] == "zero_flow")
    assert zero["rmse"] == pytest.approx(1.02)
    assert zero["rmse_ratio"] == pytest.approx(1.02)
    assert zero["passes_1pct"] is False
    assert zero["valid_pixels"] == 500
    assert zero["latency_ms"] == pytest.approx(10.0)


def test_build_scene_summary_accepts_exact_raft_method_order():
    summary = motion.build_scene_summary(
        "room3", _fake_frame_metrics(methods=motion.RAFT_METHOD_ORDER))

    assert [row["method"] for row in summary] == \
        list(motion.RAFT_METHOD_ORDER)
    raft = summary[-1]
    assert raft["rmse_ratio"] == pytest.approx(1.005)
    assert raft["passes_1pct"] is True


def _six_windows():
    return [
        {
            "scene": scene,
            "start_frame": index * 10 + 1,
            "end_frame": index * 10 + 5,
            "frame_ids": list(range(index * 10 + 1, index * 10 + 6)),
            "pair_scores": [0.1, 0.2, 0.3, 0.4],
            "motion_score": 0.25,
        }
        for index, scene in enumerate(motion.SCENES)
    ]


def _twenty_four_summary_rows():
    rows = []
    for scene in motion.SCENES:
        rows.extend(motion.build_scene_summary(
            scene, _fake_frame_metrics(scene)))
    return rows


def _thirty_summary_rows():
    rows = []
    for scene in motion.SCENES:
        rows.extend(motion.build_scene_summary(
            scene, _fake_frame_metrics(
                scene, methods=motion.RAFT_METHOD_ORDER)))
    return rows


def test_write_root_artifacts_creates_four_files_and_twenty_four_rows(tmp_path):
    completed = motion.write_root_artifacts(
        tmp_path, _six_windows(), _twenty_four_summary_rows(),
        {"checkpoint_sha256": "checkpoint"})

    assert completed["complete"] is True
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == \
        sorted(motion.ROOT_ARTIFACTS)
    with (tmp_path / "selected_windows.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 6
    with (tmp_path / "cross_scene_summary.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 24
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "All-scene pooled metrics" in report
    assert "room7" in report


def test_write_root_artifacts_accepts_thirty_rows(tmp_path):
    completed = motion.write_root_artifacts(
        tmp_path, _six_windows(), _thirty_summary_rows(),
        {"method_order": list(motion.RAFT_METHOD_ORDER)})

    assert completed["summary_row_count"] == 30
    assert completed["method_order"] == list(motion.RAFT_METHOD_ORDER)
    assert completed["all_scene_summary"][-1]["method"] == "raft_gop2"
    with (tmp_path / "cross_scene_summary.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 30


def test_write_root_artifacts_rejects_incomplete_raft_rows(tmp_path):
    with pytest.raises(ValueError, match="approved method schema|unique"):
        motion.write_root_artifacts(
            tmp_path, _six_windows(), _thirty_summary_rows()[:-1],
            {"method_order": list(motion.RAFT_METHOD_ORDER)})
