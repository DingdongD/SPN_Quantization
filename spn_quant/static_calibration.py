"""Static grouped activation histogram calibration."""

from __future__ import annotations

import torch


class GroupedHistogramObserver(object):
    def __init__(self, maximum, channels, channel_dim, group_size,
                 bins, signed):
        self.maximum = torch.as_tensor(
            maximum, dtype=torch.float32).reshape(-1).cpu()
        self.channels = int(channels)
        self.channel_dim = int(channel_dim)
        self.group_size = int(group_size)
        self.bins = int(bins)
        self.signed = bool(signed)
        if self.channels <= 0 or self.group_size <= 0 or \
                self.channels % self.group_size != 0:
            raise ValueError("group size must divide positive channels")
        if self.bins < 2:
            raise ValueError("histogram bins must be at least two")
        self.groups = self.channels // self.group_size
        if self.maximum.numel() != self.groups:
            raise ValueError("maximum count must match activation groups")
        if not bool(torch.isfinite(self.maximum).all().item()) or \
                bool((self.maximum < 0.0).any().item()):
            raise ValueError("histogram maxima must be finite and nonnegative")
        self.counts = torch.zeros(
            self.groups, self.bins, dtype=torch.int64)
        self.updates = 0

    def _axis(self, rank):
        axis = self.channel_dim \
            if self.channel_dim >= 0 else rank + self.channel_dim
        if axis < 0 or axis >= rank:
            raise ValueError("activation channel dimension is outside rank")
        return axis

    def update(self, tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            raise ValueError("histogram input must be a nonempty tensor")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("histogram input must be finite")
        axis = self._axis(tensor.ndim)
        if int(tensor.shape[axis]) != self.channels:
            raise ValueError("histogram activation channels changed")
        values = tensor.detach().movedim(axis, 0).reshape(
            self.groups, self.group_size, -1).reshape(self.groups, -1)
        if self.signed:
            values = values.abs()
        elif bool((values < 0.0).any().item()):
            raise ValueError("unsigned histogram input must be nonnegative")

        maximum = self.maximum.to(device=values.device, dtype=values.dtype)
        zero_range = maximum == 0.0
        if bool(zero_range.any().item()):
            zero_values = values[zero_range]
            if bool((zero_values != 0.0).any().item()):
                raise ValueError("zero-range histogram received nonzero values")
        positive = ~zero_range
        if bool(positive.any().item()):
            tolerance = maximum[positive].abs() * 1e-6 + 1e-12
            observed = values[positive].amax(dim=1)
            if bool((observed > maximum[positive] + tolerance).any().item()):
                raise ValueError("histogram input exceeds calibrated range")

        safe_maximum = torch.where(
            maximum > 0.0, maximum, torch.ones_like(maximum))
        normalized = (values / safe_maximum[:, None]).clamp(0.0, 1.0)
        bin_index = torch.floor(normalized * float(self.bins)).to(torch.int64)
        bin_index.clamp_(max=self.bins - 1)
        offsets = torch.arange(
            self.groups, device=values.device,
            dtype=torch.int64)[:, None] * self.bins
        combined = bin_index + offsets
        update = torch.bincount(
            combined.reshape(-1), minlength=self.groups * self.bins)
        self.counts += update.reshape(self.groups, self.bins).cpu()
        self.updates += 1

    def qmax(self, bits):
        bits = int(bits)
        if bits < 2:
            raise ValueError("activation bits must be at least two")
        return 2 ** (bits - 1) - 1 if self.signed else 2 ** bits - 1

    def _require_observed(self):
        if self.updates <= 0 or bool((self.counts.sum(dim=1) == 0).any().item()):
            raise RuntimeError("histogram observer has incomplete observations")

    def _percentile_threshold(self, percentile):
        percentile = float(percentile)
        if percentile <= 0.0 or percentile > 1.0:
            raise ValueError("percentile must be in (0, 1]")
        cumulative = self.counts.cumsum(dim=1)
        totals = cumulative[:, -1]
        targets = torch.ceil(
            totals.to(torch.float64) * percentile).to(torch.int64)
        indices = (cumulative >= targets[:, None]).to(
            torch.int64).argmax(dim=1)
        ratios = (indices.to(torch.float32) + 1.0) / float(self.bins)
        return self.maximum * ratios

    def _histogram_mse_threshold(self, bits):
        qmax = self.qmax(bits)
        centers = (
            torch.arange(self.bins, dtype=torch.float64) + 0.5
        ) / float(self.bins)
        candidates = (
            torch.arange(self.bins, dtype=torch.float64) + 1.0
        ) / float(self.bins)
        clipped = torch.minimum(centers[None, :], candidates[:, None])
        scales = candidates[:, None] / float(qmax)
        quantized = torch.round(clipped / scales).clamp(0, qmax) * scales
        squared_error = (quantized - centers[None, :]).square()
        errors = self.counts.to(torch.float64).matmul(squared_error.t())
        minimum = errors.min(dim=1).values
        tolerance = torch.maximum(
            minimum.abs(), torch.ones_like(minimum)
        ) * torch.finfo(torch.float64).eps * 32.0
        tied = errors <= minimum[:, None] + tolerance[:, None]
        reversed_index = torch.flip(tied, dims=(1,)).to(
            torch.int64).argmax(dim=1)
        indices = self.bins - 1 - reversed_index
        ratios = (indices.to(torch.float32) + 1.0) / float(self.bins)
        return self.maximum * ratios

    def thresholds(self, method, percentile=None, bits=4):
        self._require_observed()
        if method == "minmax":
            return self.maximum.clone()
        if method == "percentile":
            if percentile is None:
                raise ValueError("percentile calibration requires a percentile")
            return self._percentile_threshold(percentile)
        if method == "hist_mse":
            return self._histogram_mse_threshold(bits)
        raise ValueError("unknown histogram calibration method: %s" % method)
