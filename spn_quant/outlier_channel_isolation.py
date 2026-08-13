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


class OutlierHarmAccumulator:
    """Accumulate activation values rescued by contiguous-group isolation."""

    def __init__(self, candidates, channel_dim):
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ValueError("outlier harm requires candidates")
        self.channel_dim = int(channel_dim)
        channels = max(
            max((candidate.outlier_channel,) + candidate.victim_channels)
            for candidate in self.candidates) + 1
        self.channels = int(channels)
        self.elements = torch.zeros(channels, dtype=torch.int64)
        self.nonzero = torch.zeros(channels, dtype=torch.int64)
        self.rescued = torch.zeros(channels, dtype=torch.int64)
        self.rescued_energy = torch.zeros(channels, dtype=torch.float64)
        self.lower = torch.zeros(channels, dtype=torch.float64)
        self.upper = torch.zeros(channels, dtype=torch.float64)
        self.victim = torch.zeros(channels, dtype=torch.bool)
        for candidate in self.candidates:
            for channel in candidate.victim_channels:
                self.lower[channel] = candidate.isolated_zero_threshold
                self.upper[channel] = candidate.original_zero_threshold
                self.victim[channel] = True

    def update(self, tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            raise ValueError("activation must be a nonempty tensor")
        axis = self.channel_dim \
            if self.channel_dim >= 0 else tensor.ndim + self.channel_dim
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError("activation channel dimension is outside tensor rank")
        if int(tensor.shape[axis]) != self.channels:
            raise ValueError("activation channel count changed")
        values = tensor.detach().movedim(axis, 0).reshape(
            self.channels, -1).double()
        lower = self.lower.to(values.device).reshape(-1, 1)
        upper = self.upper.to(values.device).reshape(-1, 1)
        victim = self.victim.to(values.device).reshape(-1, 1)
        nonzero = values > 0.0
        rescued = victim & (values >= lower) & (values < upper)
        self.elements += torch.full(
            (self.channels,), values.shape[1], dtype=torch.int64)
        self.nonzero += nonzero.sum(dim=1).to(torch.int64).cpu()
        self.rescued += rescued.sum(dim=1).to(torch.int64).cpu()
        self.rescued_energy += (
            values.square() * rescued).sum(dim=1).cpu()

    def rows(self, module, group):
        candidate_rows = []
        victim_rows = []
        for candidate in self.candidates:
            rescued_elements = sum(
                int(self.rescued[channel].item())
                for channel in candidate.victim_channels)
            rescued_energy = sum(
                float(self.rescued_energy[channel].item())
                for channel in candidate.victim_channels)
            affected = sum(
                int(self.rescued[channel].item()) > 0
                for channel in candidate.victim_channels)
            harm_score = sum(
                int(self.rescued[channel].item()) /
                float(self.elements[channel].item())
                for channel in candidate.victim_channels)
            victim_nonzero = sum(
                int(self.nonzero[channel].item())
                for channel in candidate.victim_channels)
            candidate_rows.append({
                "module": str(module),
                "group": str(group),
                "group_index": candidate.group_index,
                "outlier_channel": candidate.outlier_channel,
                "outlier_maximum": candidate.outlier_maximum,
                "remaining_maximum": candidate.remaining_maximum,
                "maximum_ratio": candidate.outlier_maximum /
                max(candidate.remaining_maximum, 1e-30),
                "original_zero_threshold":
                    candidate.original_zero_threshold,
                "isolated_zero_threshold":
                    candidate.isolated_zero_threshold,
                "harm_score": harm_score,
                "harm_probability_mean": harm_score /
                float(len(candidate.victim_channels)),
                "rescued_elements": rescued_elements,
                "rescued_energy": rescued_energy,
                "affected_victim_channels": affected,
                "victim_nonzero_elements": victim_nonzero,
                "rescued_rate_victim_nonzero": rescued_elements /
                float(victim_nonzero) if victim_nonzero else 0.0,
            })
            for channel in candidate.victim_channels:
                count = int(self.rescued[channel].item())
                if count == 0:
                    continue
                elements = int(self.elements[channel].item())
                nonzero_elements = int(self.nonzero[channel].item())
                victim_rows.append({
                    "module": str(module),
                    "group": str(group),
                    "group_index": candidate.group_index,
                    "outlier_channel": candidate.outlier_channel,
                    "channel": channel,
                    "elements": elements,
                    "nonzero_elements": nonzero_elements,
                    "rescued_elements": count,
                    "rescued_energy": float(
                        self.rescued_energy[channel].item()),
                    "harm_probability": count / float(elements),
                    "rescued_rate_nonzero": count /
                    float(nonzero_elements) if nonzero_elements else 0.0,
                })
        return candidate_rows, victim_rows


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
