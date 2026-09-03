import torch
import torch.nn.functional as F

from spn_quant.selective_channel_smoothing import (
    channel_smoothing_scales,
    smooth_conv_weight,
)


def test_channel_smoothing_preserves_conv2d_function():
    torch.manual_seed(7)
    inputs = torch.randn(2, 3, 5, 5)
    weight = torch.randn(4, 3, 3, 3)
    activation_maximum = inputs.abs().amax(dim=(0, 2, 3))
    scales = channel_smoothing_scales(
        activation_maximum, weight, input_channel_dim=1,
        alpha=0.5, epsilon=1e-8)
    reference = F.conv2d(inputs, weight, padding=1)
    candidate = F.conv2d(
        inputs / scales.reshape(1, -1, 1, 1),
        smooth_conv_weight(weight, scales, input_channel_dim=1),
        padding=1)
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)


def test_channel_smoothing_reduces_activation_channel_imbalance():
    activation_maximum = torch.tensor([1.0, 10.0])
    weight = torch.ones(3, 2, 1, 1)
    scales = channel_smoothing_scales(
        activation_maximum, weight, input_channel_dim=1,
        alpha=0.5, epsilon=1e-8)
    before = activation_maximum.max() / activation_maximum.median()
    transformed = activation_maximum / scales
    after = transformed.max() / transformed.median()
    assert float(after) < float(before)


def test_channel_smoothing_preserves_conv_transpose2d_function():
    torch.manual_seed(11)
    inputs = torch.randn(2, 3, 4, 4)
    weight = torch.randn(3, 5, 3, 3)
    activation_maximum = inputs.abs().amax(dim=(0, 2, 3))
    scales = channel_smoothing_scales(
        activation_maximum, weight, input_channel_dim=0,
        alpha=0.75, epsilon=1e-8)
    reference = F.conv_transpose2d(inputs, weight, padding=1)
    candidate = F.conv_transpose2d(
        inputs / scales.reshape(1, -1, 1, 1),
        smooth_conv_weight(weight, scales, input_channel_dim=0),
        padding=1)
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)
