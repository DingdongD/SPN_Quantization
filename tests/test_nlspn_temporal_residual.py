import numpy as np
import pytest

from scripts import nlspn_temporal_residual as residual


def test_pilot_clips_cover_256_frames_without_cross_clip_pairs():
    assert residual.PILOT_CLIPS == (
        (1, 32), (282, 313), (563, 594), (844, 875),
        (1126, 1157), (1407, 1438), (1688, 1719), (1969, 2000),
    )
    frames = residual.clip_frame_ids(residual.PILOT_CLIPS)
    pairs = residual.clip_pairs(residual.PILOT_CLIPS)
    assert len(frames) == 256
    assert len(set(frames)) == 256
    assert len(pairs) == 248
    assert (32, 282) not in pairs
    assert pairs[0] == (1, 2)
    assert pairs[-1] == (1999, 2000)


def test_validate_clip_payload_requires_fixed_shapes_and_500_points():
    payload = {
        "frame_ids": np.arange(1, 33, dtype=np.int32),
        "rgb": np.zeros((32, 3, 228, 304), dtype=np.float32),
        "sparse": np.zeros((32, 228, 304), dtype=np.float32),
        "gt": np.ones((32, 228, 304), dtype=np.float32),
        "valid": np.ones((32, 228, 304), dtype=bool),
    }
    locations = np.arange(500)
    payload["sparse"].reshape(32, -1)[:, locations] = 1.0
    residual.validate_clip_payload(payload)
    payload["sparse"][0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="500 sparse points"):
        residual.validate_clip_payload(payload)
