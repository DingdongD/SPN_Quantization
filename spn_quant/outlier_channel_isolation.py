"""Contiguous Group-8 outlier channel isolation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OutlierCandidate:
    group_index: int
    outlier_channel: int
    victim_channels: tuple[int, ...]
    outlier_maximum: float
    remaining_maximum: float
    original_zero_threshold: float
    isolated_zero_threshold: float


def _validated_maximum(channel_maximum, group_size):
    maximum = torch.as_tensor(
        channel_maximum, dtype=torch.float32).reshape(-1).cpu()
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("group size must be positive")
    if maximum.numel() == 0 or maximum.numel() % group_size != 0:
        raise ValueError("group size must divide activation channels")
    if not bool(torch.isfinite(maximum).all().item()):
        raise ValueError("channel maxima must be finite")
    if bool((maximum < 0.0).any().item()):
        raise ValueError("channel maxima must be nonnegative")
    return maximum, group_size


def build_outlier_candidates(channel_maximum, group_size):
    maximum, group_size = _validated_maximum(
        channel_maximum, group_size)
    candidates = []
    for start in range(0, int(maximum.numel()), group_size):
        members = torch.arange(start, start + group_size, dtype=torch.long)
        values = maximum[members]
        local_outlier = int(torch.argmax(values).item())
        outlier_channel = start + local_outlier
        victims = tuple(
            int(channel) for channel in members.tolist()
            if int(channel) != outlier_channel)
        remaining = float(maximum[list(victims)].max().item())
        outlier = float(maximum[outlier_channel].item())
        candidates.append(OutlierCandidate(
            group_index=start // group_size,
            outlier_channel=outlier_channel,
            victim_channels=victims,
            outlier_maximum=outlier,
            remaining_maximum=remaining,
            original_zero_threshold=outlier / 30.0,
            isolated_zero_threshold=remaining / 30.0,
        ))
    return tuple(candidates)


def isolated_channel_scales(
        channel_maximum, bits, group_size, isolated_channels):
    maximum, group_size = _validated_maximum(
        channel_maximum, group_size)
    bits = int(bits)
    if bits < 1:
        raise ValueError("activation bits must be positive")
    isolated = tuple(int(channel) for channel in isolated_channels)
    if len(set(isolated)) != len(isolated):
        raise ValueError("one isolated channel is allowed per group")
    if any(channel < 0 or channel >= maximum.numel()
           for channel in isolated):
        raise ValueError("isolated channel is outside activation channels")
    groups = [channel // group_size for channel in isolated]
    if len(set(groups)) != len(groups):
        raise ValueError("one isolated channel is allowed per group")

    candidates = build_outlier_candidates(maximum, group_size)
    expected = dict(
        (candidate.group_index, candidate.outlier_channel)
        for candidate in candidates)
    for channel, group in zip(isolated, groups):
        if expected[group] != channel:
            raise ValueError(
                "isolated channel must be the contiguous group maximum channel")

    qmax = 2 ** bits - 1
    scales = torch.empty_like(maximum)
    isolated_set = set(isolated)
    for candidate in candidates:
        members = (candidate.outlier_channel,) + candidate.victim_channels
        if candidate.outlier_channel in isolated_set:
            for channel in candidate.victim_channels:
                scales[channel] = candidate.remaining_maximum / float(qmax)
            scales[candidate.outlier_channel] = \
                candidate.outlier_maximum / float(qmax)
        else:
            for channel in members:
                scales[channel] = candidate.outlier_maximum / float(qmax)
    return torch.where(scales > 0.0, scales, torch.ones_like(scales))
