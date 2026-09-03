"""Selective channel smoothing primitives for convolution quantization studies."""

from __future__ import annotations

import math

import torch

from spn_quant.fp_formats import make_quantizer


class ActivationDamageAccumulator(object):
    """Separate native sparsity from quantization-created zero codes."""

    CHUNK_ELEMENTS = 1 << 20

    def __init__(self):
        self.total = 0
        self.native_zero = 0
        self.reference_nonzero = 0
        self.new_zero = 0
        self.saturated = 0
        self.nonzero_signal = 0.0
        self.nonzero_error = 0.0

    def update(self, reference, quantized, codes, quantizer):
        if reference.shape != quantized.shape or reference.shape != codes.shape:
            raise ValueError("activation damage tensor shapes differ")
        source = reference.reshape(-1)
        result = quantized.reshape(-1)
        code_values = codes.reshape(-1)
        scales = quantizer.scale_for(reference).expand_as(reference).reshape(-1)
        for start in range(0, source.numel(), self.CHUNK_ELEMENTS):
            stop = start + self.CHUNK_ELEMENTS
            source_chunk = source[start:stop]
            result_chunk = result[start:stop]
            code_chunk = code_values[start:stop]
            scale_chunk = scales[start:stop]
            nonzero = source_chunk != 0
            self.total += int(source_chunk.numel())
            self.native_zero += int((~nonzero).sum().item())
            self.reference_nonzero += int(nonzero.sum().item())
            self.new_zero += int(
                (nonzero & (code_chunk == 0)).sum().item())
            self.saturated += int((
                source_chunk.abs() >
                float(quantizer.spec.maximum) * scale_chunk).sum().item())
            source_nonzero = torch.where(
                nonzero, source_chunk, torch.zeros_like(source_chunk))
            result_nonzero = torch.where(
                nonzero, result_chunk, torch.zeros_like(result_chunk))
            self.nonzero_signal += float(
                source_nonzero.double().square().sum().item())
            difference = source_nonzero.double() - result_nonzero.double()
            self.nonzero_error += float(difference.square().sum().item())

    def diagnostics(self):
        if self.total <= 0 or self.reference_nonzero <= 0 or \
                self.nonzero_signal <= 0.0:
            raise RuntimeError("activation damage accumulator is empty")
        sqnr = float("inf") if self.nonzero_error == 0.0 else \
            10.0 * math.log10(self.nonzero_signal / self.nonzero_error)
        return {
            "total_count": self.total,
            "native_zero_count": self.native_zero,
            "reference_nonzero_count": self.reference_nonzero,
            "new_zero_count": self.new_zero,
            "saturation_count": self.saturated,
            "native_zero_ratio": self.native_zero / float(self.total),
            "new_zero_ratio": self.new_zero / float(self.reference_nonzero),
            "saturation_ratio": self.saturated / float(self.total),
            "nonzero_sqnr_db": sqnr,
        }


class TrackedFPQuantizer(object):
    """Record activation damage without changing one FP quantizer."""

    def __init__(self, quantizer):
        self.base = quantizer
        self.format = quantizer.format
        self.bits = quantizer.bits
        self.unsigned = quantizer.unsigned
        self.qmin = quantizer.qmin
        self.qmax = quantizer.qmax
        self.scale = quantizer.scale
        self.spec = quantizer.spec
        self.calls = 0
        self.damage = ActivationDamageAccumulator()

    def scale_for(self, tensor):
        return self.base.scale_for(tensor)

    def quantize_with_codes(self, tensor):
        quantized, codes = self.base.quantize_with_codes(tensor)
        self.damage.update(tensor, quantized, codes, self.base)
        self.calls += 1
        return quantized, codes

    def diagnostics(self):
        values = self.damage.diagnostics()
        values["calls"] = self.calls
        return values


def channel_smoothing_scales(activation_maximum: torch.Tensor,
                             weight: torch.Tensor,
                             input_channel_dim: int,
                             alpha: float,
                             epsilon: float) -> torch.Tensor:
    activation = torch.as_tensor(
        activation_maximum, dtype=torch.float32).reshape(-1)
    source = torch.as_tensor(weight, dtype=torch.float32)
    channel_dim = int(input_channel_dim)
    exponent = float(alpha)
    stabilizer = float(epsilon)
    if source.ndim != 4 or channel_dim not in (0, 1):
        raise ValueError("channel smoothing requires Conv weight layout")
    if activation.numel() != source.shape[channel_dim]:
        raise ValueError("activation and weight input channels differ")
    if not bool(torch.isfinite(activation).all().item()) or \
            bool((activation < 0.0).any().item()):
        raise ValueError("activation channel maxima must be finite and nonnegative")
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError("channel smoothing weight must be finite")
    if not math.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("channel smoothing alpha must be in [0, 1]")
    if not math.isfinite(stabilizer) or stabilizer <= 0.0:
        raise ValueError("channel smoothing epsilon must be positive")
    dimensions = tuple(index for index in range(source.ndim)
                       if index != channel_dim)
    weight_maximum = source.abs().amax(dim=dimensions)
    scales = activation.clamp_min(stabilizer).pow(exponent) / \
        weight_maximum.clamp_min(stabilizer).pow(1.0 - exponent)
    if not bool(torch.isfinite(scales).all().item()) or \
            bool((scales <= 0.0).any().item()):
        raise RuntimeError("channel smoothing produced invalid scales")
    return scales


def smooth_conv_weight(weight: torch.Tensor, scales: torch.Tensor,
                       input_channel_dim: int) -> torch.Tensor:
    source = torch.as_tensor(weight)
    values = torch.as_tensor(scales, device=source.device,
                             dtype=source.dtype).reshape(-1)
    channel_dim = int(input_channel_dim)
    if source.ndim != 4 or channel_dim not in (0, 1):
        raise ValueError("channel smoothing requires Conv weight layout")
    if values.numel() != source.shape[channel_dim]:
        raise ValueError("channel smoothing scale count differs from weight")
    shape = [1] * source.ndim
    shape[channel_dim] = values.numel()
    return source * values.reshape(shape)


class ChannelSmoothedFPQuantizer(object):
    """Divide input channels before scaled floating-point quantization."""

    def __init__(self, format_name: str, transformed_maximum: float,
                 channel_scales: torch.Tensor, channel_dim: int):
        self.channel_scales = torch.as_tensor(
            channel_scales, dtype=torch.float32).reshape(-1)
        self.channel_dim = int(channel_dim)
        if self.channel_dim < 0:
            raise ValueError("channel smoothing dimension must be nonnegative")
        if not bool(torch.isfinite(self.channel_scales).all().item()) or \
                bool((self.channel_scales <= 0.0).any().item()):
            raise ValueError("channel smoothing scales must be finite and positive")
        self.base = make_quantizer(
            str(format_name), torch.tensor(float(transformed_maximum)))
        self.format = self.base.format
        self.bits = self.base.bits
        self.unsigned = self.base.unsigned
        self.qmin = self.base.qmin
        self.qmax = self.base.qmax
        self.scale = self.base.scale
        self.spec = self.base.spec
        self.signal_energy = 0.0
        self.error_energy = 0.0
        self.calls = 0

    def _channel_scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim <= self.channel_dim or \
                tensor.shape[self.channel_dim] != self.channel_scales.numel():
            raise ValueError("channel smoothing activation shape differs")
        shape = [1] * tensor.ndim
        shape[self.channel_dim] = self.channel_scales.numel()
        return self.channel_scales.to(
            device=tensor.device, dtype=tensor.dtype).reshape(shape)

    def quantize_with_codes(self, tensor: torch.Tensor):
        transformed = tensor / self._channel_scale_for(tensor)
        quantized, codes = self.base.quantize_with_codes(transformed)
        difference = transformed.double() - quantized.double()
        self.signal_energy += float(transformed.double().square().sum().item())
        self.error_energy += float(difference.square().sum().item())
        self.calls += 1
        return quantized, codes

    def diagnostics(self):
        if self.calls <= 0 or self.signal_energy <= 0.0:
            raise RuntimeError("channel smoothing quantizer has no observations")
        sqnr = float("inf") if self.error_energy == 0.0 else \
            10.0 * math.log10(self.signal_energy / self.error_energy)
        return {
            "calls": self.calls,
            "sqnr_db": sqnr,
            "zero_code_ratio": self.base.zero_codes /
            float(self.base.numel),
            "saturation_ratio": self.base.saturated /
            float(self.base.numel),
        }


class BranchIndependentFPQuantizer(object):
    """Quantize contiguous activation branches with explicit FP scales."""

    HISTOGRAM_BINS = 2048

    def __init__(self, format_name, branch_maxima: torch.Tensor,
                 branch_channels, channel_dim: int):
        maxima = torch.as_tensor(branch_maxima, dtype=torch.float32).reshape(-1)
        channels = tuple(int(value) for value in branch_channels)
        if maxima.numel() != len(channels):
            raise ValueError("branch maximum count differs from branch count")
        if not channels or any(value <= 0 for value in channels):
            raise ValueError("branch channel counts must be positive")
        if not bool(torch.isfinite(maxima).all().item()) or \
                bool((maxima <= 0.0).any().item()):
            raise ValueError("branch maxima must be finite and positive")
        if isinstance(format_name, str):
            format_names = (format_name,) * len(channels)
        elif isinstance(format_name, (tuple, list)):
            format_names = tuple(str(name) for name in format_name)
        else:
            raise TypeError("branch formats must be a string or sequence")
        if len(format_names) != len(channels):
            raise ValueError("branch format count differs from branch count")
        self.branch_channels = channels
        self.branch_maxima = maxima
        self.formats = format_names
        self.channel_dim = int(channel_dim)
        if self.channel_dim < 0:
            raise ValueError("branch channel dimension must be nonnegative")
        self.quantizers = tuple(
            make_quantizer(format_names[index], maximum)
            for index, maximum in enumerate(maxima))
        first = self.quantizers[0]
        self.format = first.format if len(set(format_names)) == 1 else "mixed"
        self.bits = first.bits if len(set(format_names)) == 1 else tuple(
            quantizer.bits for quantizer in self.quantizers)
        self.unsigned = first.unsigned
        self.qmin = first.qmin
        self.qmax = first.qmax
        self.spec = first.spec
        self.scale = torch.cat(tuple(
            quantizer.scale.reshape(1).repeat(channels[index])
            for index, quantizer in enumerate(self.quantizers)))
        count = len(channels)
        self.signal_energy = [0.0] * count
        self.error_energy = [0.0] * count
        self.reference_maximum = [0.0] * count
        self.reference_count = [0] * count
        self.histogram_maximum = [None] * count
        self.histograms = [
            torch.zeros(self.HISTOGRAM_BINS, dtype=torch.int64)
            for _ in channels
        ]
        self.damage = [ActivationDamageAccumulator() for _ in channels]
        self.calls = 0

    def _validate_shape(self, tensor: torch.Tensor) -> None:
        if tensor.ndim <= self.channel_dim or \
                tensor.shape[self.channel_dim] != sum(self.branch_channels):
            raise ValueError("branch activation shape differs from calibration")

    def scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        self._validate_shape(tensor)
        shape = [1] * tensor.ndim
        shape[self.channel_dim] = self.scale.numel()
        return self.scale.to(
            device=tensor.device, dtype=tensor.dtype).reshape(shape)

    def quantize_with_codes(self, tensor: torch.Tensor):
        self._validate_shape(tensor)
        references = torch.split(
            tensor, self.branch_channels, dim=self.channel_dim)
        quantized_branches = []
        code_branches = []
        for index, reference in enumerate(references):
            quantized, codes = self.quantizers[index].quantize_with_codes(
                reference)
            self.damage[index].update(
                reference, quantized, codes, self.quantizers[index])
            self.signal_energy[index] = self.damage[index].nonzero_signal
            self.error_energy[index] = self.damage[index].nonzero_error
            self.reference_maximum[index] = max(
                self.reference_maximum[index],
                float(reference.detach().abs().max().item()))
            self.reference_count[index] += int(reference.numel())
            maximum = float(reference.detach().abs().max().item())
            if maximum <= 0.0:
                raise ValueError("branch reference maximum must be positive")
            if self.histogram_maximum[index] is None:
                self.histogram_maximum[index] = maximum
            elif maximum != self.histogram_maximum[index]:
                raise RuntimeError(
                    "branch reference maximum changed between repeated forwards")
            flat = reference.detach().reshape(-1)
            for start in range(
                    0, flat.numel(), ActivationDamageAccumulator.CHUNK_ELEMENTS):
                chunk = flat[start:start +
                             ActivationDamageAccumulator.CHUNK_ELEMENTS]
                histogram = torch.histc(
                    chunk.abs().float(), bins=self.HISTOGRAM_BINS,
                    min=0.0, max=maximum)
                self.histograms[index] += histogram.to(
                    device="cpu", dtype=torch.int64)
            quantized_branches.append(quantized)
            code_branches.append(codes)
        self.calls += 1
        return torch.cat(quantized_branches, dim=self.channel_dim), \
            torch.cat(code_branches, dim=self.channel_dim)

    def _percentile(self, branch: int, percentile: float) -> float:
        cumulative = torch.cumsum(self.histograms[branch], dim=0)
        count = int(cumulative[-1].item())
        if count <= 0:
            raise RuntimeError("branch quantizer has no observations")
        target = int(math.ceil(float(percentile) * float(count)))
        index = int(torch.searchsorted(
            cumulative, torch.tensor(target, dtype=torch.int64)).item())
        index = min(index, self.HISTOGRAM_BINS - 1)
        maximum = self.histogram_maximum[branch]
        if maximum is None:
            raise RuntimeError("branch histogram maximum is missing")
        return float(maximum) * \
            float(index + 1) / float(self.HISTOGRAM_BINS)

    def diagnostics(self):
        values = {"calls": self.calls}
        for index, quantizer in enumerate(self.quantizers):
            count = self.reference_count[index]
            if count <= 0:
                raise RuntimeError("branch quantizer has no observations")
            error = self.error_energy[index]
            signal = self.signal_energy[index]
            sqnr = float("inf") if error == 0.0 else \
                10.0 * math.log10(signal / error)
            prefix = "branch_%d_" % index
            values[prefix + "format"] = quantizer.format
            values[prefix + "bits"] = quantizer.bits
            values[prefix + "calibration_maximum"] = float(
                self.branch_maxima[index].item())
            values[prefix + "reference_maximum"] = \
                self.reference_maximum[index]
            values[prefix + "reference_p99"] = self._percentile(index, 0.99)
            values[prefix + "reference_rms"] = math.sqrt(
                signal / float(count))
            values[prefix + "histogram_maximum"] = float(
                self.histogram_maximum[index])
            values[prefix + "histogram_count"] = int(
                self.histograms[index].sum().item())
            values[prefix + "sqnr_db"] = sqnr
            values[prefix + "zero_code_ratio"] = quantizer.zero_codes / \
                float(quantizer.numel)
            values[prefix + "saturation_ratio"] = quantizer.saturated / \
                float(quantizer.numel)
            for name, value in self.damage[index].diagnostics().items():
                values[prefix + name] = value
        return values
