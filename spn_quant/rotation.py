"""Orthogonal channel rotation for low-bit convolution activations."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def random_orthogonal_matrix(channels: int, seed: int) -> torch.Tensor:
    channels = int(channels)
    if channels <= 0:
        raise ValueError("rotation channels must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        channels, channels, generator=generator, dtype=torch.float64)
    orthogonal, triangular = torch.linalg.qr(matrix)
    signs = torch.where(
        torch.diag(triangular) < 0.0,
        torch.full((channels,), -1.0, dtype=torch.float64),
        torch.ones(channels, dtype=torch.float64),
    )
    return (orthogonal * signs.unsqueeze(0)).to(torch.float32)


def hadamard_rotation_matrix(channels: int, seed: int) -> torch.Tensor:
    channels = int(channels)
    if channels <= 0 or channels & (channels - 1):
        raise ValueError("Hadamard channels must be a power of two")
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < channels:
        matrix = torch.cat((
            torch.cat((matrix, matrix), dim=1),
            torch.cat((matrix, -matrix), dim=1),
        ), dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (channels,), generator=generator,
        dtype=torch.int64).to(torch.float32).mul_(2.0).sub_(1.0)
    return matrix.mul(signs.unsqueeze(0)).div(math.sqrt(channels))


def rotate_channels(tensor: torch.Tensor,
                    rotation: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("channel rotation requires an NCHW tensor")
    if tensor.shape[1] != rotation.shape[1]:
        raise ValueError("rotation channel count does not match the tensor")
    return torch.einsum("oc,nchw->nohw", rotation.to(tensor), tensor)


def absorb_input_rotation(
        module: nn.Conv2d, rotation: torch.Tensor,
        channel_start: int = 0,
        channel_count: Optional[int] = None) -> None:
    if module.groups != 1:
        raise ValueError("rotation requires groups=1")
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("rotation matrix must be square")
    count = int(rotation.shape[0]) \
        if channel_count is None else int(channel_count)
    start = int(channel_start)
    stop = start + count
    if count != rotation.shape[0] or start < 0 or stop > module.in_channels:
        raise ValueError("rotation slice does not match convolution channels")
    weight = module.weight.data[:, start:stop]
    transformed = torch.einsum(
        "oihw,ji->ojhw", weight, rotation.to(weight))
    module.weight.data[:, start:stop].copy_(transformed)
