import pytest
import torch
import torch.nn as nn

from spn_quant.rotation import (
    absorb_input_rotation,
    hadamard_rotation_matrix,
    random_orthogonal_matrix,
    rotate_channels,
)


def test_random_rotation_is_deterministic_and_orthogonal():
    first = random_orthogonal_matrix(8, seed=17)
    second = random_orthogonal_matrix(8, seed=17)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first @ first.t(), torch.eye(8), rtol=1e-5, atol=1e-6)


def test_hadamard_rotation_is_orthogonal():
    rotation = hadamard_rotation_matrix(8, seed=17)

    torch.testing.assert_close(
        rotation @ rotation.t(), torch.eye(8), rtol=1e-5, atol=1e-6)


def test_hadamard_requires_power_of_two_channels():
    with pytest.raises(ValueError, match="power of two"):
        hadamard_rotation_matrix(6, seed=17)


def test_absorbed_conv_matches_rotated_input():
    torch.manual_seed(3)
    conv = nn.Conv2d(8, 5, 3, padding=1, bias=True)
    value = torch.randn(2, 8, 7, 9)
    rotation = random_orthogonal_matrix(8, seed=3)
    reference = conv(value)

    absorb_input_rotation(conv, rotation)
    actual = conv(rotate_channels(value, rotation))

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


def test_absorb_rotation_updates_only_concat_slice():
    torch.manual_seed(5)
    conv = nn.Conv2d(12, 5, 1, bias=False)
    left = torch.rand(2, 4, 3, 3)
    right = torch.randn(2, 8, 3, 3)
    rotation = random_orthogonal_matrix(8, seed=5)
    reference = conv(torch.cat((left, right), dim=1))

    absorb_input_rotation(
        conv, rotation, channel_start=4, channel_count=8)
    actual = conv(torch.cat(
        (left, rotate_channels(right, rotation)), dim=1))

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
