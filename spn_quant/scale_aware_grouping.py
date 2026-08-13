"""RMS-ranked channel grouping and consumer-side permutation primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ScaleAwareGrouping:
    channel_rms: torch.Tensor
    permutation: torch.Tensor
    inverse: torch.Tensor
    group_rms: torch.Tensor
    contiguous_dispersion: torch.Tensor
    scale_aware_dispersion: torch.Tensor
    group_size: int
    epsilon: float


class PermutedGroupedActivationQuantizer(object):
    """Group quantizer with consumer-order execution and source-order metrics."""

    def __init__(self, quantizer, permutation, channel_dim):
        self.quantizer = quantizer
        self.permutation = _validated_permutation(permutation)
        self.inverse = inverse_permutation(self.permutation)
        self.channel_dim = int(channel_dim)
        if self.permutation.numel() != int(quantizer.channels):
            raise ValueError(
                "channel permutation count does not match quantizer")
        if self.channel_dim != int(quantizer.channel_dim):
            raise ValueError(
                "channel permutation dimension does not match quantizer")
        self.format = quantizer.format
        self.bits = int(quantizer.bits)
        self.unsigned = bool(quantizer.unsigned)
        self.granularity = quantizer.granularity
        self.group_size = int(quantizer.group_size)
        self.channels = int(quantizer.channels)
        self.qmin = int(quantizer.qmin)
        self.qmax = int(quantizer.qmax)
        self.zero_point = int(quantizer.zero_point)
        self.scale = quantizer.scale
        self.scale_count = int(quantizer.scale_count)

    def quantize_for_consumer(self, tensor):
        permuted = permute_activation(
            tensor, self.permutation, self.channel_dim)
        execution, execution_codes = self.quantizer.quantize_with_codes(
            permuted)
        comparable = permute_activation(
            execution, self.inverse, self.channel_dim)
        comparable_codes = permute_activation(
            execution_codes, self.inverse, self.channel_dim)
        return execution, comparable, comparable_codes

    def quantize_with_codes(self, tensor):
        execution, comparable, comparable_codes = \
            self.quantize_for_consumer(tensor)
        del execution
        return comparable, comparable_codes

    def scale_for(self, tensor):
        permuted = permute_activation(
            tensor, self.permutation, self.channel_dim)
        execution_scale = self.quantizer.scale_for(permuted)
        return permute_activation(
            execution_scale, self.inverse, self.channel_dim)

    def __call__(self, tensor):
        return self.quantize_for_consumer(tensor)[0]


def _validated_permutation(permutation):
    indices = torch.as_tensor(permutation, dtype=torch.long).reshape(-1).cpu()
    if indices.numel() == 0:
        raise ValueError("channel permutation must be nonempty")
    expected = torch.arange(indices.numel(), dtype=torch.long)
    if not torch.equal(torch.sort(indices).values, expected):
        raise ValueError("channel permutation must be a bijection")
    return indices


def inverse_permutation(permutation):
    indices = _validated_permutation(permutation)
    inverse = torch.empty_like(indices)
    inverse[indices] = torch.arange(indices.numel(), dtype=torch.long)
    return inverse


def _dispersion(groups, epsilon):
    return groups.amax(dim=1) / (groups.amin(dim=1) + float(epsilon))


def build_scale_aware_grouping(channel_rms, group_size, epsilon):
    rms = torch.as_tensor(
        channel_rms, dtype=torch.float32).reshape(-1).cpu()
    group_size = int(group_size)
    epsilon = float(epsilon)
    if rms.numel() == 0:
        raise ValueError("channel RMS must be nonempty")
    if not bool(torch.isfinite(rms).all().item()):
        raise ValueError("channel RMS must be finite")
    if bool((rms < 0.0).any().item()):
        raise ValueError("channel RMS must be nonnegative")
    if group_size <= 0 or rms.numel() % group_size != 0:
        raise ValueError("group size must divide the channel count")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("dispersion epsilon must be finite and positive")

    permutation = torch.argsort(rms, stable=True)
    inverse = inverse_permutation(permutation)
    contiguous = rms.reshape(-1, group_size)
    grouped = rms.index_select(0, permutation).reshape(-1, group_size)
    return ScaleAwareGrouping(
        channel_rms=rms,
        permutation=permutation,
        inverse=inverse,
        group_rms=grouped,
        contiguous_dispersion=_dispersion(contiguous, epsilon),
        scale_aware_dispersion=_dispersion(grouped, epsilon),
        group_size=group_size,
        epsilon=epsilon)


def permute_activation(tensor, permutation, channel_dim):
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        raise ValueError("activation must be a nonempty tensor")
    axis = int(channel_dim)
    if axis < 0:
        axis += tensor.ndim
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError("activation channel dimension is outside tensor rank")
    indices = _validated_permutation(permutation)
    if int(tensor.shape[axis]) != indices.numel():
        raise ValueError("channel permutation count does not match activation")
    return tensor.index_select(axis, indices.to(tensor.device))


def permute_input_weight(module, weight, permutation):
    if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError("input weight permutation requires Conv2d or ConvTranspose2d")
    if module.groups != 1:
        raise ValueError("input weight permutation requires groups=1")
    if not torch.is_tensor(weight) or weight.ndim != 4:
        raise ValueError("input weight permutation requires a rank-four weight")
    indices = _validated_permutation(permutation)
    axis = 0 if isinstance(module, nn.ConvTranspose2d) else 1
    if int(weight.shape[axis]) != indices.numel():
        raise ValueError("channel permutation count does not match weight input")
    return weight.index_select(axis, indices.to(weight.device))


def grouped_channel_maximum(channel_maximum, permutation, group_size):
    maximum = torch.as_tensor(
        channel_maximum, dtype=torch.float32).reshape(-1).cpu()
    indices = _validated_permutation(permutation)
    group_size = int(group_size)
    if maximum.numel() != indices.numel():
        raise ValueError("channel maximum count does not match permutation")
    if not bool(torch.isfinite(maximum).all().item()) or \
            bool((maximum < 0.0).any().item()):
        raise ValueError("channel maxima must be finite and nonnegative")
    if group_size <= 0 or maximum.numel() % group_size != 0:
        raise ValueError("group size must divide the channel count")
    return maximum.index_select(0, indices).reshape(
        -1, group_size).amax(dim=1)
