import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_nlspn_temporal_residual_validation as runner


def test_prepare_clip_uses_common_fixed_sparse_mask(monkeypatch, tmp_path):
    frame_ids = (1, 2)
    rgb = np.zeros((3, 228, 304), dtype=np.float32)
    gt = np.ones((228, 304), dtype=np.float32)
    valid = np.ones((228, 304), dtype=bool)
    monkeypatch.setattr(
        runner,
        "load_preprocessed_frame",
        lambda data_root, scene, frame_id: (rgb, gt, valid),
    )
    output = tmp_path / "clip_0001_0002.npz"
    payload = runner.prepare_clip(
        Path("/data"), "scene", frame_ids, output, seed=2026)
    assert output.is_file()
    assert payload["rgb"].shape == (2, 3, 228, 304)
    masks = payload["sparse"] > 0
    np.testing.assert_array_equal(masks[0], masks[1])
    assert masks[0].sum() == 500
