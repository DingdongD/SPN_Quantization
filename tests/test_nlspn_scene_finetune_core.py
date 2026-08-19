import math

import numpy as np
import pytest
import torch

from scripts import nlspn_scene_finetune_core as core


class ToyNLSPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in (
                "conv1_rgb", "conv1_dep", "conv2", "conv3", "conv4",
                "conv5", "conv6", "dec5", "dec4", "dec3", "dec2",
                "id_dec1", "id_dec0", "gd_dec1", "gd_dec0",
                "cf_dec1", "cf_dec0", "prop_layer"):
            setattr(self, name, torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, 1), torch.nn.BatchNorm2d(1)))
        self.prop_layer.register_parameter(
            "fixed_dummy", torch.nn.Parameter(torch.ones(1), requires_grad=False))


@pytest.fixture
def toy_nlspn():
    return ToyNLSPN()


def test_stage_one_freezes_encoder_and_batch_norm(toy_nlspn):
    groups = core.configure_stage(toy_nlspn, stage=1)
    names = core.trainable_parameter_names(toy_nlspn)
    assert all(not name.startswith((
        "conv1_rgb.", "conv2.", "conv3.", "conv4.", "conv5.", "conv6."))
        for name in names)
    assert any(name.startswith("conv1_dep.") for name in names)
    assert {group["lr"] for group in groups} == {1e-4}
    assert toy_nlspn.prop_layer.fixed_dummy.requires_grad is False
    assert all(not parameter.requires_grad
               for module in toy_nlspn.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for parameter in module.parameters(recurse=False))


def test_stage_two_uses_differential_rates(toy_nlspn):
    groups = core.configure_stage(toy_nlspn, stage=2)
    assert [group["name"] for group in groups] == ["encoder", "adaptation"]
    assert [group["lr"] for group in groups] == [5e-6, 2e-5]
    grouped = {name for group in groups for name in group["parameter_names"]}
    assert grouped == set(core.trainable_parameter_names(toy_nlspn))


def test_stage_two_restores_original_encoder_eligibility_after_stage_one(toy_nlspn):
    core.configure_stage(toy_nlspn, stage=1)
    core.configure_stage(toy_nlspn, stage=2)
    assert toy_nlspn.conv1_rgb[0].weight.requires_grad


def test_configure_stage_rejects_unknown_trainable_prefix(toy_nlspn):
    toy_nlspn.mystery = torch.nn.Conv2d(1, 1, 1)
    with pytest.raises(ValueError, match="unknown trainable.*mystery"):
        core.configure_stage(toy_nlspn, stage=1)


def test_configure_stage_rejects_empty_declared_group():
    model = torch.nn.Module()
    model.conv1_rgb = torch.nn.Conv2d(1, 1, 1)
    with pytest.raises(ValueError, match="adaptation.*empty"):
        core.configure_stage(model, stage=1)


def test_assert_trainable_set_rejects_declared_actual_mismatch(toy_nlspn):
    groups = core.configure_stage(toy_nlspn, stage=2)
    toy_nlspn.conv1_rgb[0].weight.requires_grad = False
    with pytest.raises(ValueError, match="trainable.*mismatch"):
        core.assert_trainable_set(toy_nlspn, groups)


def test_keep_batch_norm_frozen_after_model_train(toy_nlspn):
    core.configure_stage(toy_nlspn, stage=2)
    toy_nlspn.train()
    assert any(module.training for module in toy_nlspn.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    core.keep_batch_norm_frozen(toy_nlspn)
    assert all(not module.training for module in toy_nlspn.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))


def test_configure_stage_rejects_unknown_stage(toy_nlspn):
    with pytest.raises(ValueError, match="stage must be 1 or 2"):
        core.configure_stage(toy_nlspn, stage=3)


def test_l1_l2_objective_uses_only_valid_target_pixels():
    pred = torch.tensor([[[[1.0, 4.0], [9.0, 7.0]]]])
    gt = torch.tensor([[[[2.0, 2.0], [0.0, 0.0]]]])
    assert core.masked_l1_l2(pred, gt, 10.0).item() == pytest.approx(4.0)


def test_l1_l2_objective_clamps_prediction_to_depth_range():
    pred = torch.tensor([[[[-5.0, 12.0]]]])
    gt = torch.tensor([[[[1.0, 9.0]]]])
    assert core.masked_l1_l2(pred, gt, 10.0).item() == pytest.approx(2.0)


def test_l1_l2_objective_rejects_empty_mask_and_nonfinite_input():
    with pytest.raises(ValueError, match="no valid"):
        core.masked_l1_l2(torch.ones(1), torch.zeros(1), 10.0)
    with pytest.raises(ValueError, match="nonfinite"):
        core.masked_l1_l2(
            torch.tensor([float("nan")]), torch.ones(1), 10.0)


def test_frame_error_sums_cover_all_depth_bands_exactly():
    gt = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]).reshape(1, 1, 1, 5)
    pred = gt + 1.0
    result = core.frame_error_sums(pred, gt, 10.0)
    assert result["squared_error_sum"] == pytest.approx(5.0)
    assert result["absolute_error_sum"] == pytest.approx(5.0)
    assert result["valid_pixel_count"] == 5
    assert result["abs_rel_sum"] == pytest.approx(sum(1.0 / x for x in (1, 3, 5, 7, 9)))
    for name in core.DEPTH_BAND_NAMES:
        assert result[name + "_squared_error_sum"] == pytest.approx(1.0)
        assert result[name + "_absolute_error_sum"] == pytest.approx(1.0)
        assert result[name + "_valid_pixel_count"] == 1
    assert sum(result[name + "_squared_error_sum"]
               for name in core.DEPTH_BAND_NAMES) == result["squared_error_sum"]
    assert sum(result[name + "_absolute_error_sum"]
               for name in core.DEPTH_BAND_NAMES) == result["absolute_error_sum"]
    assert sum(result[name + "_valid_pixel_count"]
               for name in core.DEPTH_BAND_NAMES) == result["valid_pixel_count"]


def test_frame_error_sums_rejects_empty_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="no valid"):
        core.frame_error_sums(torch.ones(1), torch.zeros(1), 10.0)
    with pytest.raises(ValueError, match="nonfinite"):
        core.frame_error_sums(
            torch.tensor([float("inf")]), torch.ones(1), 10.0)


def test_metric_accumulator_recomputes_pooled_and_scene_macro_metrics():
    acc = core.MetricAccumulator()
    acc.add("room3", 8.0, 4.0, 2.0, 2)
    acc.add("room7", 9.0, 3.0, 1.0, 1)
    result = acc.finalize()
    assert result["pooled_rmse"] == pytest.approx((17.0 / 3.0) ** 0.5)
    assert result["pooled_mae"] == pytest.approx(7.0 / 3.0)
    assert result["pooled_abs_rel"] == pytest.approx(1.0)
    assert result["scene_macro_rmse"] == pytest.approx(
        (math.sqrt(8.0 / 2.0) + math.sqrt(9.0)) / 2.0)
    assert set(result["scenes"]) == {"room3", "room7"}


def test_metric_accumulator_keeps_float64_raw_sums():
    acc = core.MetricAccumulator()
    acc.add("room3", np.float32(1.25), np.float32(0.5), np.float32(0.25), 1)
    result = acc.finalize()
    for name in (
            "squared_error_sum", "absolute_error_sum", "abs_rel_sum"):
        assert isinstance(result[name], np.float64)


def test_metric_accumulator_rejects_invalid_updates_and_empty_finalize():
    with pytest.raises(ValueError, match="no metrics"):
        core.MetricAccumulator().finalize()
    acc = core.MetricAccumulator()
    with pytest.raises(ValueError, match="nonfinite"):
        acc.add("room3", float("nan"), 1.0, 1.0, 1)
    with pytest.raises(ValueError, match="positive"):
        acc.add("room3", 1.0, 1.0, 1.0, 0)
