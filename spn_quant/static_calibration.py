"""Static grouped activation histogram calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HistogramSite:
    module: str
    kind: str
    group: str
    channels: int
    channel_dim: int
    group_size: int
    maximum: torch.Tensor
    signed: bool


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


class StaticCalibrationRecorder(object):
    def __init__(self, sites, bins):
        self.sites = {}
        self.observers = {}
        for site in sites:
            if not isinstance(site, HistogramSite):
                raise TypeError("calibration site must be HistogramSite")
            key = (site.module, site.kind)
            if key in self.sites:
                raise ValueError("duplicate calibration owner: %s" % (key,))
            self.sites[key] = site
            self.observers[key] = GroupedHistogramObserver(
                site.maximum, site.channels, site.channel_dim,
                site.group_size, bins, site.signed)
        if not self.sites:
            raise ValueError("calibration recorder requires declared sites")
        self.bins = int(bins)

    def record_reference(self, module, kind, group, tensor, channel_dim):
        key = (str(module), str(kind))
        if key not in self.sites:
            raise ValueError("undeclared calibration owner: %s" % (key,))
        site = self.sites[key]
        if str(group) != site.group:
            raise ValueError("calibration owner group changed: %s" % (key,))
        if int(channel_dim) != site.channel_dim:
            raise ValueError(
                "calibration owner channel dimension changed: %s" % (key,))
        self.observers[key].update(tensor)

    def validate_coverage(self):
        missing = [
            key for key in self.observers
            if self.observers[key].updates == 0
        ]
        if missing:
            raise RuntimeError(
                "static calibration coverage is incomplete: %s" % missing)

    def thresholds(self, method, bits):
        self.validate_coverage()
        output = {}
        for key in self.observers:
            observer = self.observers[key]
            if method == "minmax":
                output[key] = observer.thresholds("minmax", bits=bits)
            elif method == "percentile_p999":
                output[key] = observer.thresholds(
                    "percentile", percentile=0.999, bits=bits)
            elif method == "percentile_p9999":
                output[key] = observer.thresholds(
                    "percentile", percentile=0.9999, bits=bits)
            elif method == "hist_mse":
                output[key] = observer.thresholds("hist_mse", bits=bits)
            else:
                raise ValueError(
                    "unknown static calibration method: %s" % method)
        return output

    def rows(self, method, thresholds):
        if set(thresholds) != set(self.sites):
            raise ValueError("threshold owners do not match calibration sites")
        rows = []
        for key in sorted(self.sites):
            site = self.sites[key]
            observer = self.observers[key]
            threshold = torch.as_tensor(
                thresholds[key], dtype=torch.float32).reshape(-1)
            if threshold.shape != observer.maximum.shape:
                raise ValueError("threshold shape changed: %s" % (key,))
            positive = observer.maximum > 0.0
            ratios = torch.ones_like(threshold)
            ratios[positive] = threshold[positive] / observer.maximum[positive]
            rows.append({
                "method": method,
                "module": site.module,
                "kind": site.kind,
                "group": site.group,
                "signed": int(site.signed),
                "channels": site.channels,
                "group_size": site.group_size,
                "scale_count": int(threshold.numel()),
                "threshold_min": float(threshold.min().item()),
                "threshold_max": float(threshold.max().item()),
                "threshold_ratio_min": float(ratios.min().item()),
                "threshold_ratio_mean": float(ratios.mean().item()),
                "threshold_ratio_max": float(ratios.max().item()),
                "zero_range_scales": int((~positive).sum().item()),
            })
        return rows

    def scale_rows(self, method, thresholds):
        if set(thresholds) != set(self.sites):
            raise ValueError("threshold owners do not match calibration sites")
        rows = []
        for key in sorted(self.sites):
            site = self.sites[key]
            observer = self.observers[key]
            threshold = torch.as_tensor(
                thresholds[key], dtype=torch.float32).reshape(-1)
            for index in range(threshold.numel()):
                maximum = float(observer.maximum[index].item())
                selected = float(threshold[index].item())
                rows.append({
                    "method": method,
                    "module": site.module,
                    "kind": site.kind,
                    "group": site.group,
                    "scale_index": index,
                    "signed": int(site.signed),
                    "group_size": site.group_size,
                    "maximum": maximum,
                    "threshold": selected,
                    "threshold_ratio": 1.0 if maximum == 0.0 else
                    selected / maximum,
                    "histogram_elements": int(
                        observer.counts[index].sum().item()),
                })
        return rows
