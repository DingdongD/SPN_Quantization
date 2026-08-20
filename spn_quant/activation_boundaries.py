"""Signed activation quantization at CSPN structural boundaries."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adapters.cspn import ActivationBoundary


class SignedActivationQuantizer:
    def __init__(self, bits: int, channel_maximum: torch.Tensor,
                 group_size: Optional[int]) -> None:
        self.format = "uniform"
        self.zero_point = 0
        self.bits = int(bits)
        if self.bits < 2:
            raise ValueError("signed activation bits must be at least two")
        if channel_maximum.ndim != 1 or channel_maximum.numel() == 0:
            raise ValueError("channel maxima must be a nonempty vector")
        self.channels = int(channel_maximum.numel())
        self.group_size = self.channels \
            if group_size is None else int(group_size)
        if self.group_size <= 0 or self.channels % self.group_size:
            raise ValueError("group size must divide the channel count")
        self.qmax = 2 ** (self.bits - 1) - 1
        self.qmin = -self.qmax
        maxima = []
        for start in range(0, self.channels, self.group_size):
            maxima.append(channel_maximum[
                start:start + self.group_size].max())
        grouped = torch.stack(maxima).to(torch.float32)
        self.scales = torch.where(
            grouped > 0.0, grouped / float(self.qmax),
            torch.ones_like(grouped))

    def scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError("quantizer requires the calibrated NCHW channels")
        scales = self.scales.repeat_interleave(self.group_size).to(tensor)
        return scales.reshape(1, self.channels, 1, 1)

    def quantize_with_codes(
            self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError("quantizer requires the calibrated NCHW channels")
        values = []
        codes = []
        for group_index, start in enumerate(
                range(0, self.channels, self.group_size)):
            stop = start + self.group_size
            scale = self.scales[group_index].to(tensor)
            current_codes = torch.round(
                tensor[:, start:stop] / scale).clamp(
                    -self.qmax, self.qmax)
            values.append(current_codes * scale)
            codes.append(current_codes.to(torch.int32))
        return torch.cat(values, dim=1), torch.cat(codes, dim=1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize_with_codes(tensor)[0]


class ActivationBoundaryObserver:
    def __init__(self, channels: int, capacity: int = 1000000,
                 per_update: int = 8192) -> None:
        self.channels = int(channels)
        self.capacity = int(capacity)
        self.per_update = int(per_update)
        if self.channels <= 0 or self.capacity <= 0 or self.per_update <= 0:
            raise ValueError("observer dimensions must be positive")
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.channel_absmax = torch.zeros(self.channels, dtype=torch.float32)
        self.channel_square_sum = torch.zeros(
            self.channels, dtype=torch.float64)
        self.scalar_count = 0
        self.scalar_sum = 0.0
        self.scalar_square_sum = 0.0
        self.scalar_cube_sum = 0.0
        self.scalar_fourth_sum = 0.0
        self.sample_chunks = []
        self.sample_values = 0

    @property
    def observed(self) -> bool:
        return self.scalar_count > 0

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError(
                "boundary observer requires calibrated NCHW channels")
        detached = tensor.detach()
        if not torch.isfinite(detached).all():
            raise ValueError("boundary calibration activation must be finite")
        values = detached.to(torch.float64)
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))
        moved = values.movedim(1, 0).reshape(self.channels, -1)
        absmax = moved.abs().amax(dim=1).to(torch.float32).cpu()
        self.channel_absmax = torch.maximum(self.channel_absmax, absmax)
        self.channel_square_sum += moved.square().sum(dim=1).cpu()
        self.scalar_count += int(values.numel())
        self.scalar_sum += float(values.sum().item())
        self.scalar_square_sum += float(values.square().sum().item())
        self.scalar_cube_sum += float(values.pow(3).sum().item())
        self.scalar_fourth_sum += float(values.pow(4).sum().item())

        remaining = self.capacity - self.sample_values
        positions = min(
            moved.shape[1], self.per_update // self.channels,
            remaining // self.channels)
        if positions > 0:
            indices = torch.linspace(
                0, moved.shape[1] - 1, positions,
                device=moved.device).round().to(torch.long)
            sample = moved.index_select(1, indices).to(torch.float32).cpu()
            self.sample_chunks.append(sample)
            self.sample_values += int(sample.numel())

    def quantizer(self, bits: int, group_size: Optional[int],
                  scale_factor: float = 1.0,
                  maximum: Optional[torch.Tensor] = None
                  ) -> SignedActivationQuantizer:
        if not self.observed:
            raise RuntimeError("activation boundary was not observed")
        if self.minimum >= 0.0:
            raise RuntimeError("activation boundary is not signed")
        scale_factor = float(scale_factor)
        if not math.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("boundary scale factor must be finite and positive")
        if maximum is None:
            channel_maximum = self.channel_absmax
        else:
            current_group_size = self.channels \
                if group_size is None else int(group_size)
            if current_group_size <= 0 or self.channels % current_group_size:
                raise ValueError("group size must divide the channel count")
            maximum = torch.as_tensor(
                maximum, dtype=torch.float32).reshape(-1)
            groups = self.channels // current_group_size
            if maximum.numel() != groups:
                raise ValueError("boundary maximum count does not match groups")
            if not bool(torch.isfinite(maximum).all().item()) or \
                    bool((maximum < 0.0).any().item()):
                raise ValueError(
                    "boundary maxima must be finite and nonnegative")
            channel_maximum = maximum.repeat_interleave(current_group_size)
        return SignedActivationQuantizer(
            bits, channel_maximum * scale_factor, group_size)

    def _samples(self) -> torch.Tensor:
        if not self.sample_chunks:
            raise RuntimeError("boundary observer has no retained samples")
        return torch.cat(self.sample_chunks, dim=1).unsqueeze(0).unsqueeze(2)

    def statistics(self, quantizer: SignedActivationQuantizer
                   ) -> Dict[str, float]:
        samples = self._samples()
        quantized, codes = quantizer.quantize_with_codes(samples)
        absolute = samples.abs().reshape(-1)
        percentiles = torch.quantile(
            absolute, torch.tensor(
                [0.75, 0.99, 0.999, 0.9999], dtype=torch.float32))
        signal = float(samples.to(torch.float64).square().sum().item())
        error = float((quantized - samples).to(
            torch.float64).square().sum().item())
        sqnr = float("inf") if error == 0.0 else \
            10.0 * math.log10(signal / error)

        count = float(self.scalar_count)
        mean = self.scalar_sum / count
        second = self.scalar_square_sum / count
        third = self.scalar_cube_sum / count
        fourth = self.scalar_fourth_sum / count
        variance = second - mean * mean
        central_fourth = fourth - 4.0 * mean * third + \
            6.0 * mean * mean * second - 3.0 * mean ** 4
        kurtosis = central_fourth / (variance * variance) \
            if variance > 0.0 else 0.0
        channel_norm = self.channel_square_sum.sqrt()
        imbalance = float(
            channel_norm.max().item() / channel_norm.mean().item())
        return {
            "minimum": self.minimum,
            "maximum": max(abs(self.minimum), abs(self.maximum)),
            "p75": float(percentiles[0].item()),
            "p99": float(percentiles[1].item()),
            "p99_9": float(percentiles[2].item()),
            "p99_99": float(percentiles[3].item()),
            "kurtosis": kurtosis,
            "channel_imbalance": imbalance,
            "sqnr": sqnr,
            "zero_code_ratio": float((codes == 0).float().mean().item()),
            "saturation_ratio": float(
                (codes.abs() == quantizer.qmax).float().mean().item()),
        }


class CSPNActivationBoundaryController:
    def __init__(self, model: nn.Module,
                 boundaries: Sequence[ActivationBoundary]) -> None:
        self.model = model
        self.boundaries = tuple(boundaries)
        self.modules = dict(model.named_modules())
        self.mode = "bypass"
        self.frozen = False
        self.quantize_enabled = False
        self.channels = {}
        self.observers = {}
        self.active_quantizers = {}
        self.activation_recorder = None
        self.calibration_recorder = None
        self.handles = []
        for boundary in self.boundaries:
            consumer = boundary.consumers[0]
            conv = self.modules[consumer.module]
            channels = conv.in_channels \
                if consumer.channel_count is None else consumer.channel_count
            self.channels[boundary.name] = int(channels)
            self.observers[boundary.name] = \
                ActivationBoundaryObserver(channels)
            module = self.modules[boundary.module]
            self.handles.append(module.register_forward_pre_hook(
                self._make_hook(boundary)))

    def _make_hook(self, boundary: ActivationBoundary):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            del module
            current = inputs[boundary.argument_index]
            if self.mode == "observe":
                self.observers[boundary.name].update(current)
                if self.calibration_recorder is not None:
                    self.calibration_recorder.record_reference(
                        "boundary.%s" % boundary.name, "boundary",
                        "decoder", current, 1)
                return None
            if self.mode != "quantize" or not self.quantize_enabled:
                return None
            quantizer = self.active_quantizers[boundary.name]
            quantized, codes = quantizer.quantize_with_codes(current)
            if self.activation_recorder is not None:
                self.activation_recorder.record(
                    "boundary.%s" % boundary.name,
                    "boundary", 0, "decoder", current,
                    quantized, codes, quantizer, 1)
            updated = list(inputs)
            updated[boundary.argument_index] = quantized
            return tuple(updated)
        return hook

    def observe(self) -> None:
        self.mode = "observe"

    def freeze(self) -> None:
        for boundary in self.boundaries:
            self.observers[boundary.name].quantizer(
                bits=4, group_size=None)
        self.frozen = True
        self.mode = "bypass"

    def _validate_specs(
            self, bit_widths: Mapping[str, int],
            group_sizes: Mapping[str, Optional[int]],
            scale_factors: Mapping[str, float]) -> None:
        expected = set(boundary.name for boundary in self.boundaries)
        if set(bit_widths) != expected:
            raise ValueError("bit widths must name every boundary")
        if set(group_sizes) != expected:
            raise ValueError("group sizes must name every boundary")
        if set(scale_factors) != expected:
            raise ValueError("scale factors must name every boundary")

    def configure_specs(
            self, bit_widths: Mapping[str, int],
            group_sizes: Mapping[str, Optional[int]],
            scale_factors: Mapping[str, float],
            quantize: bool = True) -> None:
        self._configure_specs(
            bit_widths, group_sizes, scale_factors, None, quantize)

    def configure_specs_with_ranges(
            self, bit_widths: Mapping[str, int],
            group_sizes: Mapping[str, Optional[int]],
            scale_factors: Mapping[str, float],
            maximum_overrides: Mapping[str, torch.Tensor],
            quantize: bool = True) -> None:
        expected = set(boundary.name for boundary in self.boundaries)
        if set(maximum_overrides) != expected:
            raise ValueError("ranges must name every boundary")
        self._configure_specs(
            bit_widths, group_sizes, scale_factors,
            maximum_overrides, quantize)

    def _configure_specs(
            self, bit_widths: Mapping[str, int],
            group_sizes: Mapping[str, Optional[int]],
            scale_factors: Mapping[str, float],
            maximum_overrides: Optional[Mapping[str, torch.Tensor]],
            quantize: bool) -> None:
        self._validate_specs(bit_widths, group_sizes, scale_factors)
        if quantize and not self.frozen:
            raise RuntimeError("activation boundary calibration must be frozen")
        quantizers = {}
        if quantize:
            for boundary in self.boundaries:
                quantizers[boundary.name] = self.observers[
                    boundary.name].quantizer(
                        int(bit_widths[boundary.name]),
                        group_sizes[boundary.name],
                        float(scale_factors[boundary.name]),
                        None if maximum_overrides is None else
                        maximum_overrides[boundary.name])
        self.active_quantizers = quantizers
        self.quantize_enabled = bool(quantize)
        self.mode = "quantize"

    def statistics(
            self, bit_widths: Mapping[str, int],
            group_sizes: Mapping[str, Optional[int]]) \
            -> Sequence[Dict[str, float]]:
        expected = set(boundary.name for boundary in self.boundaries)
        if set(bit_widths) != expected or set(group_sizes) != expected:
            raise ValueError("statistics must name every boundary")
        rows = []
        for boundary in self.boundaries:
            observer = self.observers[boundary.name]
            quantizer = observer.quantizer(
                int(bit_widths[boundary.name]), group_sizes[boundary.name])
            row = observer.statistics(quantizer)
            row.update({
                "boundary": boundary.name,
                "method": "identity",
                "bits": int(bit_widths[boundary.name]),
                "group_size": quantizer.group_size,
            })
            rows.append(row)
        return rows

    def manifest(self) -> Sequence[Dict[str, object]]:
        rows = []
        for boundary in self.boundaries:
            row = {
                "module": "boundary.%s" % boundary.name,
                "kind": "boundary",
                "method": "identity",
                "channels": self.channels[boundary.name],
                "quantized": boundary.name in self.active_quantizers,
            }
            if boundary.name in self.active_quantizers:
                quantizer = self.active_quantizers[boundary.name]
                row["bits"] = quantizer.bits
                row["group_size"] = quantizer.group_size
            rows.append(row)
        return rows

    def set_activation_recorder(self, recorder) -> None:
        if recorder is None or not callable(recorder.record):
            raise TypeError("activation recorder must define record")
        self.activation_recorder = recorder

    def clear_activation_recorder(self) -> None:
        self.activation_recorder = None

    def set_calibration_recorder(self, recorder) -> None:
        if recorder is None or not callable(recorder.record_reference):
            raise TypeError(
                "calibration recorder must define record_reference")
        self.calibration_recorder = recorder

    def clear_calibration_recorder(self) -> None:
        self.calibration_recorder = None

    def disable(self) -> None:
        self.mode = "bypass"
        self.active_quantizers = {}
        self.quantize_enabled = False

    def close(self) -> None:
        self.disable()
        self.clear_activation_recorder()
        self.clear_calibration_recorder()
        for handle in self.handles:
            handle.remove()
        self.handles = []
