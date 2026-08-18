import pytest
import torch

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_temporal_residual as residual


def synthetic_texture(height=228, width=304):
    generator = torch.Generator().manual_seed(17)
    return torch.rand(1, 3, height, width, generator=generator)


def coordinate_image(height=228, width=304):
    values = torch.arange(height * width, dtype=torch.float32)
    return values.reshape(1, 1, height, width)


def test_candidate_grid_is_fixed_and_complete():
    configs = cache.candidate_configs("rgb_diff")
    assert len(configs) == 12
    assert configs[0] == cache.CacheConfig(
        variant="rgb_diff", threshold=2.0 / 255.0,
        dilation_radius=2)
    assert configs[-1] == cache.CacheConfig(
        variant="rgb_diff", threshold=16.0 / 255.0,
        dilation_radius=8)
    assert cache.candidate_configs("global_diff")[0].variant == "global_diff"


def test_zero_flow_has_one_parameter_free_config():
    assert cache.candidate_configs("zero_flow") == (
        cache.CacheConfig(
            variant="zero_flow", threshold=None,
            dilation_radius=None),)


def test_rgb_delta_uses_three_by_three_blur_and_max_channel():
    previous = torch.zeros(1, 3, 5, 5)
    current = previous.clone()
    current[:, 1, 2, 2] = 0.9
    delta = cache.blurred_rgb_delta(current, previous)
    assert delta.shape == (1, 1, 5, 5)
    assert delta[0, 0, 2, 2] == pytest.approx(0.1)


def test_threshold_is_strict_and_change_mask_is_dilated():
    delta = torch.zeros(1, 1, 9, 9)
    delta[0, 0, 4, 4] = 4.0 / 255.0
    unchanged = cache.photometric_changed(
        delta, threshold=4.0 / 255.0, radius=2)
    assert not unchanged.any()
    delta[0, 0, 4, 4] += 1e-5
    changed = cache.photometric_changed(
        delta, threshold=4.0 / 255.0, radius=2)
    assert changed.sum().item() == 25


def test_sparse_inconsistency_uses_two_centimeter_gate():
    sparse = torch.zeros(1, 1, 9, 9)
    base = torch.ones_like(sparse)
    sparse[0, 0, 4, 4] = 1.02
    at_gate = cache.sparse_changed(sparse, base, radius=2)
    assert not at_gate.any()
    sparse[0, 0, 4, 4] = 1.021
    above_gate = cache.sparse_changed(sparse, base, radius=2)
    assert above_gate.sum().item() == 25


def test_stable_mask_and_blend_use_previous_only_when_stable():
    photo = torch.tensor([[[[False, True]]]])
    depth = torch.tensor([[[[False, False]]]])
    stable = cache.compose_stable_mask(photo, depth)
    previous = torch.tensor([[[[2.0, 2.0]]]])
    candidate = torch.tensor([[[[3.0, 3.0]]]])
    result = cache.blend_cached_depth(previous, candidate, stable)
    torch.testing.assert_close(
        result, torch.tensor([[[[2.0, 3.0]]]]))


def test_phase_translation_returns_current_to_previous_displacement():
    previous = synthetic_texture()
    current = torch.roll(previous, shifts=(4, -8), dims=(-2, -1))
    dx, dy = cache.estimate_backward_translation(
        current, previous, downsample=4)
    assert (dx, dy) == (8.0, -4.0)


def test_translation_flow_warps_previous_and_marks_boundaries():
    source = coordinate_image()
    flow = cache.constant_backward_flow(
        dx=8.0, dy=-4.0, height=228, width=304,
        device=source.device, dtype=source.dtype)
    warped, in_bounds = residual.backward_warp(source, flow)
    assert flow.shape == (1, 2, 228, 304)
    assert not in_bounds[:, :, :4].any()
    assert not in_bounds[:, :, :, -8:].any()
    torch.testing.assert_close(warped[0, 0, 4, 0], source[0, 0, 0, 8])
