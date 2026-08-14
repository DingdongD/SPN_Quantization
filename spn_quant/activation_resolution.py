"""Activation resolution diagnostics for uniform integer QDQ."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch


_PERCENTILES = (0.75, 0.99, 0.999, 0.9999)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else float("inf")


def _sqnr(signal: float, error: float) -> float:
    if error == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal / error)


class BoundedChannelSampler(object):
    """Keep a deterministic, equally bounded magnitude sample per channel."""

    def __init__(self, channels: int, capacity: int) -> None:
        self.channels = int(channels)
        self.capacity = int(capacity)
        if self.channels <= 0:
            raise ValueError("sampler channels must be positive")
        if self.capacity <= 0:
            raise ValueError("sampler capacity must be positive")
        self._samples = torch.empty(
            self.channels, 0, dtype=torch.float32)
        self._global_samples = torch.empty(0, dtype=torch.float32)

    @property
    def sample_count(self) -> int:
        return int(self._samples.shape[1])

    @property
    def global_sample_count(self) -> int:
        return int(self._global_samples.numel())

    @staticmethod
    def _indices(length: int, count: int) -> torch.Tensor:
        if count == length:
            return torch.arange(length, dtype=torch.long)
        return torch.linspace(
            0, length - 1, steps=count, dtype=torch.float64
        ).round().to(torch.long)

    def update(self, values: torch.Tensor) -> None:
        if not torch.is_tensor(values) or values.ndim != 2:
            raise ValueError("sampler values must be channel-first 2D")
        if int(values.shape[0]) != self.channels:
            raise ValueError("sampler channel count changed")
        detached = values.detach().to(
            device="cpu", dtype=torch.float32).abs()
        if not bool(torch.isfinite(detached).all().item()):
            raise ValueError("sampler values must be finite")
        combined = torch.cat((self._samples, detached), dim=1)
        retained = min(int(combined.shape[1]), self.capacity)
        self._samples = combined.index_select(
            1, self._indices(int(combined.shape[1]), retained))
        global_values = torch.cat((
            self._global_samples, detached.reshape(-1)))
        global_retained = min(int(global_values.numel()), self.capacity)
        self._global_samples = global_values.index_select(
            0, self._indices(int(global_values.numel()), global_retained))

    def percentiles(self, values: Sequence[float]) -> torch.Tensor:
        if self.sample_count == 0:
            raise RuntimeError("cannot summarize an empty sampler")
        probabilities = torch.tensor(tuple(values), dtype=torch.float32)
        if bool(((probabilities < 0.0) | (probabilities > 1.0)).any().item()):
            raise ValueError("percentiles must be in [0, 1]")
        return torch.quantile(
            self._samples, probabilities, dim=1).transpose(0, 1)

    def global_percentiles(self, values: Sequence[float]) -> torch.Tensor:
        if self.global_sample_count == 0:
            raise RuntimeError("cannot summarize an empty sampler")
        probabilities = torch.tensor(tuple(values), dtype=torch.float32)
        return torch.quantile(self._global_samples, probabilities)


class ActivationResolutionAccumulator(object):
    """Accumulate exact QDQ error partitions and bounded percentiles."""

    def __init__(self, channel_dim: int, capacity: int) -> None:
        self.channel_dim = int(channel_dim)
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("activation sample capacity must be positive")
        self.channels = None
        self.sampler = None
        self.elements = 0
        self.reference_zeros = 0
        self.quantized_zeros = 0
        self.nonzero_elements = 0
        self.new_zero_elements = 0
        self.saturated = 0
        self.signal_energy = 0.0
        self.total_error_energy = 0.0
        self.zero_collapse_error_energy = 0.0
        self.rounding_error_energy = 0.0
        self.clipping_error_energy = 0.0
        self.maximum_abs = 0.0
        self.code_counts = {}  # type: Dict[int, int]
        self.channel_elements = None
        self.channel_reference_zeros = None
        self.channel_nonzero_elements = None
        self.channel_new_zero_elements = None
        self.channel_signal_energy = None
        self.channel_total_error_energy = None
        self.channel_zero_collapse_error_energy = None
        self.channel_rounding_error_energy = None
        self.channel_clipping_error_energy = None
        self.channel_maximum_abs = None

    def _channel_axis(self, rank: int) -> int:
        axis = self.channel_dim if self.channel_dim >= 0 else rank + self.channel_dim
        if axis < 0 or axis >= rank:
            raise ValueError("activation channel dimension is outside tensor rank")
        return axis

    def _initialize(self, channels: int) -> None:
        self.channels = int(channels)
        self.sampler = BoundedChannelSampler(self.channels, self.capacity)
        self.channel_elements = torch.zeros(self.channels, dtype=torch.int64)
        self.channel_reference_zeros = torch.zeros(
            self.channels, dtype=torch.int64)
        self.channel_nonzero_elements = torch.zeros(
            self.channels, dtype=torch.int64)
        self.channel_new_zero_elements = torch.zeros(
            self.channels, dtype=torch.int64)
        self.channel_signal_energy = torch.zeros(
            self.channels, dtype=torch.float64)
        self.channel_total_error_energy = torch.zeros(
            self.channels, dtype=torch.float64)
        self.channel_zero_collapse_error_energy = torch.zeros(
            self.channels, dtype=torch.float64)
        self.channel_rounding_error_energy = torch.zeros(
            self.channels, dtype=torch.float64)
        self.channel_clipping_error_energy = torch.zeros(
            self.channels, dtype=torch.float64)
        self.channel_maximum_abs = torch.zeros(
            self.channels, dtype=torch.float64)

    @staticmethod
    def _channel_values(tensor: torch.Tensor, axis: int) -> torch.Tensor:
        return tensor.movedim(axis, 0).reshape(tensor.shape[axis], -1)

    @staticmethod
    def _sum_channels(values: torch.Tensor) -> torch.Tensor:
        return values.to(torch.float64).sum(dim=1).cpu()

    def update(self, reference: torch.Tensor, quantized: torch.Tensor,
               codes: torch.Tensor, quantizer: object) -> None:
        if reference.shape != quantized.shape or reference.shape != codes.shape:
            raise ValueError("reference, quantized, and codes must share shape")
        if reference.numel() == 0:
            raise ValueError("activation tensor must be nonempty")
        if not bool(torch.isfinite(reference).all().item()) or \
                not bool(torch.isfinite(quantized).all().item()):
            raise ValueError("activation diagnostics require finite tensors")
        if quantizer.format != "uniform":
            raise ValueError("activation resolution requires uniform QDQ")
        if int(quantizer.zero_point) != 0:
            raise ValueError("activation resolution requires zero-point zero")

        axis = self._channel_axis(reference.ndim)
        channels = int(reference.shape[axis])
        if self.channels is None:
            self._initialize(channels)
        elif channels != self.channels:
            raise ValueError("activation channel count changed")

        scale = quantizer.scale_for(reference)
        scale_tensor = torch.as_tensor(
            scale, device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(scale_tensor).all().item()) or \
                bool((scale_tensor <= 0).any().item()):
            raise ValueError("activation quantization scale must be positive")
        unrounded = reference / scale_tensor
        zero_mask = (reference != 0) & (codes == 0)
        outside = (unrounded < int(quantizer.qmin)) | \
            (unrounded > int(quantizer.qmax))
        clipping_mask = outside & ~zero_mask
        rounding_mask = ~(zero_mask | clipping_mask)
        error = (quantized - reference).to(torch.float64).square()
        signal = reference.to(torch.float64).square()

        self.elements += int(reference.numel())
        self.reference_zeros += int((reference == 0).sum().item())
        self.quantized_zeros += int((codes == 0).sum().item())
        self.nonzero_elements += int((reference != 0).sum().item())
        self.new_zero_elements += int(zero_mask.sum().item())
        self.saturated += int(
            ((codes == int(quantizer.qmin)) |
             (codes == int(quantizer.qmax))).sum().item())
        self.signal_energy += float(signal.sum().item())
        self.total_error_energy += float(error.sum().item())
        self.zero_collapse_error_energy += float(
            error[zero_mask].sum().item())
        self.clipping_error_energy += float(
            error[clipping_mask].sum().item())
        self.rounding_error_energy += float(
            error[rounding_mask].sum().item())
        self.maximum_abs = max(
            self.maximum_abs, float(reference.abs().max().item()))

        code_counts = torch.bincount(
            (codes.detach().to(torch.int64).reshape(-1) -
             int(quantizer.qmin)).cpu(),
            minlength=int(quantizer.qmax) - int(quantizer.qmin) + 1)
        for offset, count in enumerate(code_counts.tolist()):
            if count == 0:
                continue
            code = offset + int(quantizer.qmin)
            if code in self.code_counts:
                self.code_counts[code] += int(count)
            else:
                self.code_counts[code] = int(count)

        reference_channels = self._channel_values(reference, axis)
        error_channels = self._channel_values(error, axis)
        zero_channels = self._channel_values(zero_mask, axis)
        clipping_channels = self._channel_values(clipping_mask, axis)
        rounding_channels = self._channel_values(rounding_mask, axis)
        channel_count = int(reference_channels.shape[1])
        self.channel_elements += channel_count
        self.channel_reference_zeros += self._channel_values(
            reference == 0, axis).sum(dim=1).cpu()
        self.channel_nonzero_elements += self._channel_values(
            reference != 0, axis).sum(dim=1).cpu()
        self.channel_new_zero_elements += zero_channels.sum(dim=1).cpu()
        self.channel_signal_energy += self._sum_channels(
            reference_channels.to(torch.float64).square())
        self.channel_total_error_energy += self._sum_channels(error_channels)
        self.channel_zero_collapse_error_energy += self._sum_channels(
            error_channels * zero_channels)
        self.channel_clipping_error_energy += self._sum_channels(
            error_channels * clipping_channels)
        self.channel_rounding_error_energy += self._sum_channels(
            error_channels * rounding_channels)
        self.channel_maximum_abs = torch.maximum(
            self.channel_maximum_abs,
            reference_channels.abs().max(dim=1).values.to(
                torch.float64).cpu())
        self.sampler.update(reference_channels)

    def _effective_code_count(self) -> float:
        counts = torch.tensor(
            tuple(self.code_counts.values()), dtype=torch.float64)
        probabilities = counts / counts.sum()
        entropy = -(probabilities * probabilities.log()).sum()
        return float(entropy.exp().item())

    def tensor_summary(self) -> Dict[str, object]:
        if self.elements == 0 or self.sampler is None:
            raise RuntimeError("cannot summarize empty activation diagnostics")
        percentiles = self.sampler.global_percentiles(_PERCENTILES)
        channel_rms = torch.sqrt(
            self.channel_signal_energy /
            self.channel_elements.to(torch.float64))
        p99 = float(percentiles[1].item())
        return {
            "elements": self.elements,
            "reference_zero_rate": self.reference_zeros / float(self.elements),
            "quantized_zero_rate": self.quantized_zeros / float(self.elements),
            "new_zero_elements": self.new_zero_elements,
            "nonzero_elements": self.nonzero_elements,
            "new_zero_rate": _ratio(
                float(self.new_zero_elements), float(self.nonzero_elements)),
            "zero_collapse_error_energy": self.zero_collapse_error_energy,
            "rounding_error_energy": self.rounding_error_energy,
            "clipping_error_energy": self.clipping_error_energy,
            "signal_energy": self.signal_energy,
            "total_error_energy": self.total_error_energy,
            "sqnr_db": _sqnr(self.signal_energy, self.total_error_energy),
            "p75": float(percentiles[0].item()),
            "p99": p99,
            "p99_9": float(percentiles[2].item()),
            "p99_99": float(percentiles[3].item()),
            "maximum_abs": self.maximum_abs,
            "tail_ratio_p99_99_over_p99": _ratio(
                float(percentiles[3].item()), p99),
            "channel_rms_imbalance": _ratio(
                float(channel_rms.max().item()),
                float(channel_rms.mean().item())),
            "effective_code_count": self._effective_code_count(),
            "saturation_rate": self.saturated / float(self.elements),
        }

    def channel_summaries(self) -> List[Dict[str, object]]:
        if self.elements == 0 or self.sampler is None:
            raise RuntimeError("cannot summarize empty activation diagnostics")
        percentiles = self.sampler.percentiles(_PERCENTILES)
        rows = []
        for channel in range(int(self.channels)):
            elements = int(self.channel_elements[channel].item())
            signal = float(self.channel_signal_energy[channel].item())
            error = float(self.channel_total_error_energy[channel].item())
            rows.append({
                "channel": channel,
                "elements": elements,
                "rms": math.sqrt(signal / float(elements)),
                "maximum_abs": float(
                    self.channel_maximum_abs[channel].item()),
                "p99": float(percentiles[channel, 1].item()),
                "p99_9": float(percentiles[channel, 2].item()),
                "p99_99": float(percentiles[channel, 3].item()),
                "reference_zero_rate": float(
                    self.channel_reference_zeros[channel].item()) /
                float(elements),
                "new_zero_rate": _ratio(
                    float(self.channel_new_zero_elements[channel].item()),
                    float(self.channel_nonzero_elements[channel].item())),
                "zero_collapse_error_energy": float(
                    self.channel_zero_collapse_error_energy[channel].item()),
                "rounding_error_energy": float(
                    self.channel_rounding_error_energy[channel].item()),
                "clipping_error_energy": float(
                    self.channel_clipping_error_energy[channel].item()),
                "total_error_energy": error,
                "sqnr_db": _sqnr(signal, error),
                "error_energy_share": _ratio(
                    error, self.total_error_energy),
            })
        return rows


class ActivationResolutionRecorder(object):
    """Aggregate instrumentor QDQ events by semantic call-indexed site."""

    def __init__(self, split: str, capacity: int) -> None:
        if split not in ("calibration", "evaluation"):
            raise ValueError("activation split must be calibration or evaluation")
        self.split = split
        self.capacity = int(capacity)
        self.accumulators = {}  # type: Dict[str, ActivationResolutionAccumulator]
        self.metadata = {}  # type: Dict[str, Dict[str, object]]

    @staticmethod
    def _site(module: str, kind: str, call_index: int) -> str:
        return "%s#%d:%s" % (module, int(call_index), kind)

    def record(self, module: str, kind: str, call_index: int, group: str,
               reference: torch.Tensor, quantized: torch.Tensor,
               codes: torch.Tensor, quantizer: object,
               channel_dim: int) -> None:
        site = self._site(module, kind, call_index)
        if site not in self.accumulators:
            self.accumulators[site] = ActivationResolutionAccumulator(
                channel_dim, self.capacity)
            self.metadata[site] = {
                "site": site,
                "module": module,
                "kind": kind,
                "call_index": int(call_index),
                "group": group,
            }
        elif self.metadata[site]["group"] != group:
            raise ValueError("activation site group changed")
        self.accumulators[site].update(
            reference, quantized, codes, quantizer)

    def tensor_rows(self) -> List[Dict[str, object]]:
        rows = []
        for site in sorted(self.accumulators):
            row = dict(self.metadata[site])
            row["split"] = self.split
            row.update(self.accumulators[site].tensor_summary())
            rows.append(row)
        return rows

    def channel_rows(self) -> List[Dict[str, object]]:
        rows = []
        for site in sorted(self.accumulators):
            for channel_row in self.accumulators[site].channel_summaries():
                row = dict(self.metadata[site])
                row["split"] = self.split
                row.update(channel_row)
                rows.append(row)
        return rows
