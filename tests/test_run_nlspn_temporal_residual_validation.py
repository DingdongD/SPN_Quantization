import json
from pathlib import Path

import numpy as np
import pytest
import torch

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


class FakeRaft(torch.nn.Module):
    def forward(self, image1, image2):
        batch, _, height, width = image1.shape
        flow = torch.zeros((batch, 2, height, width), device=image1.device)
        flow[:, 0] = 2.0
        return [flow]


def test_predict_backward_flow_is_current_to_previous_and_crops_padding():
    rgb = np.zeros((3, 3, 228, 304), dtype=np.float32)
    flow, seconds = runner.predict_backward_flow(
        FakeRaft(), lambda current, previous: (current, previous),
        rgb, torch.device("cpu"), batch_size=2)
    assert seconds >= 0.0
    assert flow.shape == (2, 2, 228, 304)
    np.testing.assert_allclose(flow[:, 0], 2.0)
