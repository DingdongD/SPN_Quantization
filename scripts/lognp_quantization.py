#!/usr/bin/env python3
"""Reference LogNP companding quantization primitives."""

from __future__ import division

import math

import torch


def _validate_alpha(alpha):
    alpha = torch.as_tensor(alpha)
    if not bool(torch.isfinite(alpha).all()) or bool((alpha <= 0).any()):
        raise ValueError("alpha must be finite and strictly positive")
    return alpha


def _validate_max_z(max_z):
    max_z = float(max_z)
    if not math.isfinite(max_z) or max_z <= 0.0:
        raise ValueError("max_z must be finite and strictly positive")
    return max_z


def _channel_view(tensor, values):
    values = torch.as_tensor(values, device=tensor.device, dtype=tensor.dtype)
    if values.numel() == 1:
        return values.reshape([1] * tensor.ndim)
    if tensor.ndim < 2 or tensor.shape[1] != values.numel():
        raise ValueError("channel parameters do not match tensor channel dimension")
    return values.reshape([1, values.numel()] + [1] * (tensor.ndim - 2))


def lognp_transform(tensor, alpha, max_z=24.0):
    """Apply sign(x) * log2(1 + abs(x) / alpha) with a bounded result."""
    max_z = _validate_max_z(max_z)
    tensor = torch.as_tensor(tensor)
    alpha = _validate_alpha(alpha).to(device=tensor.device, dtype=tensor.dtype)
    alpha = _channel_view(tensor, alpha)
    magnitude = torch.log1p(tensor.abs() / alpha) / math.log(2.0)
    return tensor.sign() * magnitude.clamp(max=max_z)


def lognp_inverse(transformed, alpha, max_z=24.0):
    """Apply the bounded inverse of :func:`lognp_transform`."""
    max_z = _validate_max_z(max_z)
    transformed = torch.as_tensor(transformed)
    alpha = _validate_alpha(alpha).to(
        device=transformed.device, dtype=transformed.dtype)
    alpha = _channel_view(transformed, alpha)
    magnitude = transformed.abs().clamp(max=max_z)
    magnitude = torch.expm1(magnitude * math.log(2.0)) * alpha
    return transformed.sign() * magnitude


class LogNPActivationQuantizer(object):
    """Uniform quantization in the LogNP transformed domain."""

    def __init__(self, bits, alpha, scale, unsigned=False, max_z=24.0):
        bits = int(bits)
        if bits < 2:
            raise ValueError("activation bits must be at least 2")
        alpha = _validate_alpha(alpha).detach().float().cpu().reshape(-1)
        scale = _validate_alpha(scale).detach().float().cpu().reshape(-1)
        if alpha.numel() != scale.numel():
            raise ValueError("alpha and scale must have the same channel count")
        self.bits = bits
        self.unsigned = bool(unsigned)
        self.qmin = 0 if self.unsigned else -(2 ** (bits - 1) - 1)
        self.qmax = (2 ** bits - 1) if self.unsigned else 2 ** (bits - 1) - 1
        self.alpha = alpha
        self.scale = scale
        self.max_z = _validate_max_z(max_z)
        self.zero_point = 0

    def quantize_with_codes(self, tensor):
        tensor = torch.as_tensor(tensor)
        source = tensor.clamp_min(0.0) if self.unsigned else tensor
        transformed = lognp_transform(
            source, self.alpha.to(tensor), max_z=self.max_z)
        scale = _channel_view(tensor, self.scale.to(tensor))
        codes = torch.round(transformed / scale).clamp(
            self.qmin, self.qmax).to(torch.int32)
        reconstructed = lognp_inverse(
            codes.to(tensor.dtype) * scale,
            self.alpha.to(tensor), max_z=self.max_z)
        return reconstructed, codes

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]

    def manifest(self):
        return {
            "bits": self.bits,
            "unsigned": self.unsigned,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "alpha": self.alpha.tolist(),
            "scale": self.scale.tolist(),
            "max_z": self.max_z,
            "zero_point": self.zero_point,
        }


class ChannelLogNPObserver(object):
    """Bounded deterministic per-channel calibration for LogNP QDQ."""

    def __init__(self, sample_limit=4096, epsilon=1e-8):
        sample_limit = int(sample_limit)
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        self.sample_limit = sample_limit
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self._samples = None
        self._maximum = None
        self._minimum = None
        self._sample_count = 0
        self._quantizer = None
        self.alpha = None
        self.scale = None
        self.p50 = None
        self.p99 = None
        self.zmax = None
        self.calibrated_maximum = None
        self.bits = None
        self.unsigned = None
        self.max_z = None

    @property
    def observed(self):
        return self._samples is not None and self._sample_count > 0

    @property
    def sample_count(self):
        return self._sample_count

    @property
    def minimum(self):
        if self._minimum is None:
            return float("inf")
        return float(self._minimum.min().item())

    @property
    def samples(self):
        return [] if self._samples is None else list(self._samples.unbind(0))

    def _select_evenly(self, values, limit):
        if values.numel() <= limit:
            return values
        indices = torch.linspace(
            0, values.numel() - 1, steps=limit, dtype=torch.long)
        return values.index_select(0, indices)

    def update(self, tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        if tensor.ndim < 2:
            raise ValueError("LogNP channel observer requires a channel dimension")
        detached = tensor.detach().float()
        channels = int(detached.shape[1])
        flattened = detached.reshape(detached.shape[0], channels, -1)
        flattened = flattened.permute(1, 0, 2).reshape(channels, -1)
        if self._samples is None:
            self._samples = torch.empty(channels, 0)
            self._maximum = torch.zeros(channels)
            self._minimum = torch.full((channels,), float("inf"))
        elif self._samples.shape[0] != channels:
            raise ValueError("channel count changed during LogNP calibration")
        self._sample_count += int(flattened.shape[1])
        finite = torch.isfinite(flattened)
        safe_abs = torch.where(finite, flattened.abs(), torch.zeros_like(flattened))
        safe_min = torch.where(
            finite, flattened, torch.full_like(flattened, float("inf")))
        self._maximum = torch.maximum(self._maximum, safe_abs.amax(dim=1).cpu())
        batch_minimum = safe_min.amin(dim=1).cpu()
        has_finite = finite.any(dim=1).cpu()
        self._minimum = torch.where(
            has_finite, torch.minimum(self._minimum, batch_minimum),
            self._minimum)
        count = min(int(safe_abs.shape[1]), self.sample_limit)
        indices = torch.linspace(
            0, safe_abs.shape[1] - 1, steps=count, device=safe_abs.device,
            dtype=torch.long)
        selected = safe_abs.index_select(1, indices).cpu()
        combined = torch.cat((self._samples, selected), dim=1)
        count = min(int(combined.shape[1]), self.sample_limit)
        indices = torch.linspace(
            0, combined.shape[1] - 1, steps=count, dtype=torch.long)
        self._samples = combined.index_select(1, indices)

    def freeze(self, bits, unsigned, alpha_factor=1.0, max_z=24.0,
               per_channel=True):
        if not self.observed:
            raise RuntimeError("cannot freeze an unobserved LogNP observer")
        alpha_factor = float(alpha_factor)
        if not math.isfinite(alpha_factor) or alpha_factor <= 0.0:
            raise ValueError("alpha_factor must be finite and positive")
        max_z = _validate_max_z(max_z)
        self.bits = int(bits)
        self.unsigned = bool(unsigned)
        self.max_z = max_z
        qmax = (2 ** self.bits - 1 if self.unsigned
                else 2 ** (self.bits - 1) - 1)
        sample_groups = list(self._samples.unbind(0)) if per_channel else [
            self._samples.reshape(-1)]
        maximum = self._maximum.clone() if per_channel else torch.tensor([
            float(self._maximum.max())])
        self.calibrated_maximum = maximum.clone()
        p50 = []
        p99 = []
        for values in sample_groups:
            if values.numel() == 0:
                p50.append(0.0)
                p99.append(0.0)
            else:
                p50.append(float(torch.quantile(values, 0.50)))
                p99.append(float(torch.quantile(values, 0.99)))
        self.p50 = torch.tensor(p50)
        self.p99 = torch.tensor(p99)
        base = torch.maximum(self.p50, self.p99 * 0.01)
        base = base.clamp_min(self.epsilon)
        nonzero = maximum > 0.0
        self.alpha = (base * alpha_factor).clamp_min(self.epsilon)
        self.zmax = (torch.log1p(maximum / self.alpha) / math.log(2.0))
        self.zmax = self.zmax.clamp(min=self.epsilon, max=max_z)
        self.alpha = torch.where(nonzero, self.alpha, torch.ones_like(self.alpha))
        self.zmax = torch.where(nonzero, self.zmax, torch.ones_like(self.zmax))
        self.scale = self.zmax / float(qmax)
        self.scale = torch.where(nonzero, self.scale, torch.ones_like(self.scale))
        self._quantizer = LogNPActivationQuantizer(
            self.bits, self.alpha, self.scale, self.unsigned, self.max_z)

    def quantizer(self):
        if self._quantizer is None:
            raise RuntimeError("LogNP observer has not been frozen")
        return self._quantizer

    def manifest(self):
        quantizer = self.quantizer()
        row = quantizer.manifest()
        row.update({
            "channels": int(self.alpha.numel()),
            "p50": self.p50.tolist(),
            "p99": self.p99.tolist(),
            "maximum": self.calibrated_maximum.tolist(),
            "zmax": self.zmax.tolist(),
        })
        return row


class LogNPQuantizationStats(object):
    """Original and transformed-domain error accounting for LogNP QDQ."""

    def __init__(self, sample_limit=8192):
        self.sample_limit = int(sample_limit)
        if self.sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        self.numel = 0
        self.saturated = 0
        self.zero_codes = 0
        self.nonfinite = 0
        self.sign_flips = 0
        self.signal_sq = 0.0
        self.error_sq = 0.0
        self.transformed_signal_sq = 0.0
        self.transformed_error_sq = 0.0
        self._error_samples = torch.empty(0)

    def _append_errors(self, errors):
        errors = errors.detach().float().cpu().reshape(-1)
        if errors.numel() == 0:
            return
        combined = torch.cat((self._error_samples, errors))
        if combined.numel() > self.sample_limit:
            indices = torch.linspace(
                0, combined.numel() - 1, steps=self.sample_limit,
                dtype=torch.long)
            combined = combined.index_select(0, indices)
        self._error_samples = combined

    def update(self, reference, quantized, codes=None, qmin=None, qmax=None,
               transformed_reference=None, transformed_quantized=None):
        reference = reference.detach().float()
        quantized = quantized.detach().float()
        finite = torch.isfinite(reference) & torch.isfinite(quantized)
        difference = reference - quantized
        finite_difference = difference[finite]
        self.numel += int(reference.numel())
        self.nonfinite += int((~torch.isfinite(quantized)).sum().item())
        self.signal_sq += float((reference[finite] ** 2).sum().item())
        self.error_sq += float((finite_difference ** 2).sum().item())
        self.sign_flips += int((reference[finite] * quantized[finite] < 0).sum().item())
        self._append_errors(finite_difference.abs())
        if codes is not None:
            codes = codes.detach()
            self.zero_codes += int((codes == 0).sum().item())
            if qmin is not None:
                self.saturated += int((codes == qmin).sum().item())
            if qmax is not None:
                self.saturated += int((codes == qmax).sum().item())
        if transformed_reference is not None and transformed_quantized is not None:
            transformed_reference = transformed_reference.detach().float()
            transformed_quantized = transformed_quantized.detach().float()
            transformed_finite = torch.isfinite(transformed_reference) & \
                torch.isfinite(transformed_quantized)
            transformed_difference = (
                transformed_reference - transformed_quantized)[transformed_finite]
            self.transformed_signal_sq += float(
                (transformed_reference[transformed_finite] ** 2).sum().item())
            self.transformed_error_sq += float(
                (transformed_difference ** 2).sum().item())

    @property
    def mse(self):
        return self.error_sq / float(max(self.numel - self.nonfinite, 1))

    @property
    def sqnr_db(self):
        if self.error_sq == 0.0:
            return float("inf")
        if self.signal_sq == 0.0:
            return float("-inf")
        return 10.0 * math.log10(self.signal_sq / self.error_sq)

    @property
    def transformed_sqnr_db(self):
        if self.transformed_error_sq == 0.0:
            return float("inf")
        if self.transformed_signal_sq == 0.0:
            return float("-inf")
        return 10.0 * math.log10(
            self.transformed_signal_sq / self.transformed_error_sq)

    @property
    def saturation_rate(self):
        return self.saturated / float(max(self.numel, 1))

    @property
    def zero_code_rate(self):
        return self.zero_codes / float(max(self.numel, 1))

    @property
    def nonfinite_rate(self):
        return self.nonfinite / float(max(self.numel, 1))

    @property
    def sign_flip_rate(self):
        return self.sign_flips / float(max(self.numel, 1))

    def _percentile(self, percentile):
        if self._error_samples.numel() == 0:
            return 0.0
        return float(torch.quantile(
            self._error_samples, float(percentile) / 100.0).item())

    @property
    def p50(self):
        return self._percentile(50.0)

    @property
    def p75(self):
        return self._percentile(75.0)

    @property
    def p99(self):
        return self._percentile(99.0)

    @property
    def p99_9(self):
        return self._percentile(99.9)


def fit_bias_correction(target, reconstructed, bias):
    """Return a bias corrected by the mean calibration output residual."""
    target = torch.as_tensor(target)
    reconstructed = torch.as_tensor(reconstructed, device=target.device)
    bias = torch.as_tensor(bias, device=target.device, dtype=target.dtype)
    if target.shape != reconstructed.shape or target.ndim < 2:
        raise ValueError("target and reconstructed must have matching batch shapes")
    if target.shape[-1] != bias.numel():
        raise ValueError("bias must match the final output dimension")
    residual = (target - reconstructed).reshape(-1, bias.numel()).mean(dim=0)
    return bias + residual


def fit_weight_correction(inputs, target, ridge=1e-4):
    """Fit output-channel weights for reconstructed activation inputs."""
    inputs = torch.as_tensor(inputs)
    target = torch.as_tensor(target, device=inputs.device, dtype=inputs.dtype)
    ridge = float(ridge)
    if inputs.ndim != 2 or target.ndim != 2:
        raise ValueError("inputs and target must be rank-2 matrices")
    if inputs.shape[0] != target.shape[0]:
        raise ValueError("inputs and target must have the same row count")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    gram = inputs.transpose(0, 1).matmul(inputs)
    rhs = target.transpose(0, 1).matmul(inputs)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + ridge * eye, rhs.transpose(0, 1)).transpose(0, 1)
