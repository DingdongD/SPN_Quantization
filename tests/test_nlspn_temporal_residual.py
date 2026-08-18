import numpy as np
import pytest
import torch

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


def test_backward_warp_uses_target_to_source_pixel_flow_and_border_fill():
    source = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    flow = torch.zeros((1, 2, 1, 4))
    flow[:, 0] = 1.0
    warped, in_bounds = residual.backward_warp(source, flow)
    assert torch.equal(warped, torch.tensor([[[[2.0, 3.0, 4.0, 4.0]]]]))
    assert torch.equal(
        in_bounds, torch.tensor([[[[True, True, True, False]]]]))


def test_backward_warp_falls_back_for_old_meshgrid(monkeypatch):
    original = torch.meshgrid
    calls = []

    def old_meshgrid(*args, **kwargs):
        calls.append(kwargs)
        if kwargs:
            raise TypeError("unexpected keyword argument 'indexing'")
        return original(*args)

    monkeypatch.setattr(torch, "meshgrid", old_meshgrid)
    source = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    flow = torch.zeros((1, 2, 2, 2))
    warped, _ = residual.backward_warp(source, flow)
    assert torch.equal(warped, source)
    assert calls == [{"indexing": "ij"}, {}]


def test_backward_warp_does_not_suppress_unrelated_meshgrid_error(monkeypatch):
    def broken_meshgrid(*args, **kwargs):
        raise TypeError("unrelated meshgrid failure")

    monkeypatch.setattr(torch, "meshgrid", broken_meshgrid)
    source = torch.zeros((1, 1, 2, 2))
    flow = torch.zeros((1, 2, 2, 2))
    with pytest.raises(TypeError, match="unrelated"):
        residual.backward_warp(source, flow)


def test_sparse_residual_seed_keeps_signed_values_only_at_measurements():
    base = np.full((2, 2), 2.0, dtype=np.float32)
    sparse = np.array([[0.0, 1.5], [3.0, 0.0]], dtype=np.float32)
    seed, mask = residual.sparse_residual_seed(sparse, base)
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])
    np.testing.assert_allclose(seed, [[0.0, -0.5], [1.0, 0.0]])


def test_pooled_quality_ratio_uses_pixels_not_mean_of_frame_rmse():
    gt = np.zeros((2, 1, 2), dtype=np.float32)
    full = np.array([[[1.0, 1.0]], [[2.0, 2.0]]], dtype=np.float32)
    reconstructed = full * 1.01
    valid = np.ones_like(gt, dtype=bool)
    result = residual.pooled_quality(full, reconstructed, gt, valid)
    assert result["rmse_full"] == pytest.approx(np.sqrt(2.5))
    assert result["quality_ratio"] == pytest.approx(1.01)
    assert result["passes"] is True


def test_pooled_quality_rejects_zero_error_baseline():
    zeros = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        residual.pooled_quality(zeros, zeros, zeros, np.ones_like(zeros, bool))


def test_residual_statistics_reports_thresholds_and_finite_values():
    values = np.array([-0.10, -0.02, 0.0, 0.01, 0.04], dtype=np.float32)
    stats = residual.residual_statistics(values)
    assert stats["count"] == 5
    assert stats["rmse"] == pytest.approx(
        np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    assert stats["fraction_below_1cm"] == pytest.approx(0.4)
    assert stats["fraction_below_5cm"] == pytest.approx(0.8)
    assert all(np.isfinite(value) for value in stats.values())
