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


def make_online_inputs(rgb_value=0.0, sparse_value=3.0):
    rgb = torch.full((3, residual.HEIGHT, residual.WIDTH), rgb_value)
    sparse = torch.zeros(residual.HEIGHT, residual.WIDTH)
    sparse.reshape(-1)[:residual.SPARSE_COUNT] = sparse_value
    return rgb, sparse


class FakePropLayer(torch.nn.Module):
    def forward(self, seed, guidance, confidence, fixed, rgb):
        assert fixed is None
        dense = seed + 0.5
        return dense, [dense], None, None, torch.tensor(1.0)


class FakeNLSPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prop_layer = FakePropLayer()
        self.full_calls = 0

    def forward(self, sample):
        self.full_calls += 1
        batch, _, height, width = sample["dep"].shape
        prediction = torch.full(
            (batch, 1, height, width), 2.0,
            device=sample["dep"].device)
        return {
            "pred": prediction,
            "guidance": torch.zeros(
                batch, 8, height, width, device=sample["dep"].device),
            "confidence": torch.ones(
                batch, 1, height, width, device=sample["dep"].device),
        }


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


def test_zero_flow_p_frame_uses_previous_state_without_raft():
    model = FakeNLSPN()
    engine = cache.FrameDifferenceGOP2Engine(model, device="cpu")
    rgb0, sparse0 = make_online_inputs()
    rgb1, sparse1 = make_online_inputs(rgb_value=0.25)
    engine.infer_i(rgb0, sparse0, local_index=0)
    result = engine.infer_p(
        rgb1, sparse1, local_index=1,
        config=cache.candidate_configs("zero_flow")[0])
    assert result.kind == "P"
    assert result.variant == "zero_flow"
    assert result.prediction.device.type == "cpu"
    assert result.mask_metrics["stable_fraction"] == 0.0
    assert result.mask_metrics["changed_fraction"] == 1.0
    assert model.full_calls == 1
    assert not hasattr(engine, "raft")


def test_rgb_diff_p_frame_blends_independently_computed_stable_mask():
    engine = cache.FrameDifferenceGOP2Engine(FakeNLSPN(), device="cpu")
    rgb0, sparse0 = make_online_inputs(sparse_value=2.0)
    engine.infer_i(rgb0, sparse0, local_index=0)

    rgb1, sparse1 = make_online_inputs(sparse_value=2.0)
    rgb1[1, 100, 100] = 0.9
    sparse1.reshape(-1)[0] = 2.1
    config = cache.CacheConfig("rgb_diff", 4.0 / 255.0, 2)

    current_rgb = rgb1[None]
    current_sparse = sparse1[None, None]
    base = torch.full_like(current_sparse, 2.0)
    photo = cache.photometric_changed(
        cache.blurred_rgb_delta(current_rgb, rgb0[None]),
        config.threshold, config.dilation_radius)
    depth = cache.sparse_changed(
        current_sparse, base, config.dilation_radius)
    stable = cache.compose_stable_mask(photo, depth)
    seed = torch.zeros_like(base)
    sparse_mask = current_sparse > 0
    seed[sparse_mask] = current_sparse[sparse_mask] - base[sparse_mask]
    candidate = torch.clamp(base + seed + 0.5, 0.0, residual.MAX_DEPTH)
    expected = torch.where(stable, base, candidate)[0, 0]

    result = engine.infer_p(rgb1, sparse1, local_index=1, config=config)
    torch.testing.assert_close(result.prediction, expected)
    assert result.variant == "rgb_diff"
    assert result.mask_metrics["stable_fraction"] == pytest.approx(
        stable.float().mean().item())
    assert result.mask_metrics["photometric_changed_fraction"] == pytest.approx(
        photo.float().mean().item())
    assert result.mask_metrics["sparse_changed_fraction"] == pytest.approx(
        depth.float().mean().item())


def test_global_diff_warps_state_and_forces_boundaries_changed(monkeypatch):
    engine = cache.FrameDifferenceGOP2Engine(FakeNLSPN(), device="cpu")
    rgb0, sparse0 = make_online_inputs(sparse_value=2.0)
    engine.infer_i(rgb0, sparse0, local_index=0)
    rgb1, sparse1 = make_online_inputs(sparse_value=2.0)
    monkeypatch.setattr(
        cache, "estimate_backward_translation",
        lambda current, previous, downsample=4: (8.0, -4.0))

    result = engine.infer_p(
        rgb1, sparse1, local_index=1,
        config=cache.CacheConfig("global_diff", 4.0 / 255.0, 2))

    expected_oob = 1.0 - ((228 - 4) * (304 - 8)) / float(228 * 304)
    assert result.mask_metrics["dx"] == 8.0
    assert result.mask_metrics["dy"] == -4.0
    assert result.mask_metrics["out_of_bounds_fraction"] == pytest.approx(
        expected_oob)
    assert torch.all(result.prediction[:4] == 2.5)
    assert torch.all(result.prediction[4:, :-8] == 2.0)
    assert torch.all(result.prediction[:, -8:] == 2.5)
    assert torch.isfinite(engine.state.previous_depth).all()
    assert torch.isfinite(engine.state.previous_guidance).all()
    assert torch.isfinite(engine.state.previous_confidence).all()


def sweep_row(config, rmse, valid=10):
    return {
        "variant": config.variant,
        "threshold": config.threshold,
        "dilation_radius": config.dilation_radius,
        "sse": (rmse ** 2) * valid,
        "valid_pixels": valid,
    }


def complete_sweep(variant, rmse=2.0):
    return [sweep_row(config, rmse)
            for config in cache.candidate_configs(variant)]


def set_sweep_rmse(rows, threshold, radius, rmse):
    for row in rows:
        if (row["threshold"] == threshold and
                row["dilation_radius"] == radius):
            row["sse"] = (rmse ** 2) * row["valid_pixels"]
            return
    raise AssertionError("test configuration not found")


def test_select_config_uses_calibration_rmse():
    rows = complete_sweep("rgb_diff")
    set_sweep_rmse(rows, 4.0 / 255.0, 8, 0.9)
    selected = cache.select_calibration_config(rows, "rgb_diff")
    assert selected.threshold == pytest.approx(4.0 / 255.0)
    assert selected.dilation_radius == 8


def test_selection_tie_prefers_lower_threshold_then_larger_radius():
    rows = complete_sweep("rgb_diff")
    set_sweep_rmse(rows, 4.0 / 255.0, 8, 1.0)
    set_sweep_rmse(rows, 2.0 / 255.0, 2, 1.0 + 5e-10)
    set_sweep_rmse(rows, 2.0 / 255.0, 8, 1.0 + 5e-10)
    selected = cache.select_calibration_config(rows, "rgb_diff")
    assert selected.threshold == pytest.approx(2.0 / 255.0)
    assert selected.dilation_radius == 8


def test_selection_rejects_incomplete_grid():
    rows = complete_sweep("global_diff")[:-1]
    with pytest.raises(ValueError, match="12"):
        cache.select_calibration_config(rows, "global_diff")
