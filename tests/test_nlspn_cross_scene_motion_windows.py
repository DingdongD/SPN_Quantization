from pathlib import Path

import numpy as np
from PIL import Image
import pytest

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
