"""Strict scaled FP4/FP6/FP8 fake quantizers for ordinary model tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FPFormatSpec:
    name: str
    bits: int
    maximum: float


FORMAT_SPECS = {
    "fp4_e2m1": FPFormatSpec("fp4_e2m1", 4, 6.0),
    "fp6_e3m2": FPFormatSpec("fp6_e3m2", 6, 28.0),
    "fp8_e4m3fn": FPFormatSpec("fp8_e4m3fn", 8, 448.0),
    "fp16_ieee": FPFormatSpec("fp16_ieee", 16, 65504.0),
    "bf16": FPFormatSpec("bf16", 16, float(torch.finfo(torch.bfloat16).max)),
}


def _spec(name: str) -> FPFormatSpec:
    return FORMAT_SPECS[str(name)]


def _calibrated_scale(maximum: torch.Tensor, format_maximum: float):
    value = torch.as_tensor(maximum, dtype=torch.float32)
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError("floating-point calibration maximum must be finite")
    if bool((value <= 0.0).any().item()):
        raise ValueError("floating-point calibration maximum must be positive")
    return value / float(format_maximum)


def _fp8_e4m3fn_quantize(normalized: torch.Tensor):
    chunks = []
    code_chunks = []
    flat = normalized.reshape(-1)
    for start in range(0, flat.numel(), 1 << 20):
        chunk = flat[start:start + (1 << 20)]
        magnitude = chunk.abs().clamp(0.0, 448.0)
        normal = magnitude >= 2.0 ** -6
        exponent = torch.floor(torch.log2(torch.where(
            normal, magnitude, torch.ones_like(magnitude)))).to(torch.int64)
        base = torch.pow(2.0, exponent.to(chunk.dtype))
        mantissa = torch.round((magnitude / base - 1.0) * 8.0).to(torch.int64)
        carry = mantissa == 8
        exponent = exponent + carry.to(torch.int64)
        mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
        normal_magnitude = (1.0 + mantissa.to(chunk.dtype) / 8.0) * \
            torch.pow(2.0, exponent.to(chunk.dtype))
        subnormal_mantissa = torch.round(
            magnitude / (2.0 ** -9)).to(torch.int64).clamp(0, 7)
        subnormal_magnitude = subnormal_mantissa.to(chunk.dtype) * (2.0 ** -9)
        reconstructed_magnitude = torch.where(
            normal, normal_magnitude, subnormal_magnitude)
        sign = (chunk < 0).to(torch.uint8)
        exponent_field = (exponent + 7).clamp(1, 15).to(torch.uint8)
        normal_codes = (exponent_field << 3) | mantissa.clamp(0, 6).to(torch.uint8)
        magnitude_codes = torch.where(
            normal, normal_codes, subnormal_mantissa.to(torch.uint8))
        codes = magnitude_codes | (sign << 7)
        codes = torch.where(magnitude == 0.0, torch.zeros_like(codes), codes)
        chunks.append(reconstructed_magnitude * torch.sign(chunk))
        code_chunks.append(codes)
    return torch.cat(chunks).reshape(normalized.shape), torch.cat(code_chunks).reshape(
        normalized.shape)


def _fp6_e3m2_quantize(normalized: torch.Tensor):
    values = torch.tensor(
        (0.0, 0.0625, 0.125, 0.1875,
         0.25, 0.3125, 0.375, 0.4375,
         0.5, 0.625, 0.75, 0.875,
         1.0, 1.25, 1.5, 1.75,
         2.0, 2.5, 3.0, 3.5,
         4.0, 5.0, 6.0, 7.0,
         8.0, 10.0, 12.0, 14.0,
         16.0, 20.0, 24.0, 28.0),
        device=normalized.device, dtype=normalized.dtype)
    boundaries = (values[:-1] + values[1:]) * 0.5
    clipped = normalized.clamp(-28.0, 28.0)
    indices = torch.bucketize(clipped.abs(), boundaries)
    sign = torch.sign(clipped).to(torch.int8)
    codes = indices.to(torch.int8) * sign
    reconstructed = values[indices] * sign.to(normalized.dtype)
    return reconstructed, codes


class ScaledFloatingPointQuantizer(object):
    """Apply one calibrated scale tensor and one exact floating format."""

    def __init__(self, format_name: str, maximum: torch.Tensor,
                 broadcast_shape=None):
        self.spec = _spec(format_name)
        self.format = self.spec.name
        self.bits = self.spec.bits
        self.unsigned = False
        if self.bits == 4:
            self.qmin, self.qmax = -7, 7
        elif self.bits == 6:
            self.qmin, self.qmax = -31, 31
        elif self.bits == 8:
            self.qmin, self.qmax = 0, 255
        else:
            raise ValueError("unsupported floating-point bit width")
        self.scale = _calibrated_scale(maximum, self.spec.maximum)
        self.broadcast_shape = None if broadcast_shape is None else tuple(
            int(value) for value in broadcast_shape)
        if self.broadcast_shape is not None and self.scale.numel() != 1:
            non_singleton = 1
            for value in self.broadcast_shape:
                if value != 1:
                    non_singleton *= value
            if len(self.broadcast_shape) == 0 or \
                    self.scale.numel() != non_singleton:
                raise ValueError("floating-point broadcast shape mismatch")
        self.zero_codes = 0
        self.saturated = 0
        self.numel = 0
        self.calls = 0

    def _scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        if scale.numel() == 1:
            return scale
        if self.broadcast_shape is not None:
            if len(self.broadcast_shape) != tensor.ndim or \
                    any(size not in (1, actual) for size, actual in zip(
                        self.broadcast_shape, tensor.shape)):
                raise ValueError("floating-point broadcast tensor shape mismatch")
            return scale.reshape(self.broadcast_shape)
        if tensor.numel() == 0:
            raise ValueError("floating-point quantization tensor is empty")
        if tensor.shape[0] != scale.numel():
            raise ValueError("floating-point scale channel shape mismatch")
        shape = [scale.numel()] + [1] * (tensor.ndim - 1)
        return scale.reshape(shape)

    def quantize_with_codes(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            raise ValueError("floating-point quantization tensor is invalid")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("floating-point quantization tensor is nonfinite")
        scale = self._scale_for(tensor)
        normalized = tensor / scale
        if self.bits == 4:
            values = torch.tensor(
                (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                device=tensor.device, dtype=tensor.dtype)
            clipped = normalized.clamp(
                -float(self.spec.maximum), float(self.spec.maximum))
            boundaries = torch.tensor(
                (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
                device=tensor.device, dtype=tensor.dtype)
            indices = torch.bucketize(clipped.abs(), boundaries)
            codes = indices.to(torch.int8) * torch.sign(clipped).to(torch.int8)
            reconstructed = values[indices] * torch.sign(clipped) * scale
            saturation = normalized.abs() > float(self.spec.maximum)
            zero = codes == 0
        elif self.bits == 6:
            reconstructed, codes = _fp6_e3m2_quantize(normalized)
            reconstructed = reconstructed * scale
            saturation = normalized.abs() > float(self.spec.maximum)
            zero = codes == 0
        elif self.bits == 8:
            clipped = normalized.clamp(
                -float(self.spec.maximum), float(self.spec.maximum))
            reconstructed, codes = _fp8_e4m3fn_quantize(clipped)
            reconstructed = reconstructed * scale
            saturation = normalized.abs() > float(self.spec.maximum)
            zero = codes == 0
        else:
            raise RuntimeError("floating-point format dispatch is incomplete")
        self.numel += int(tensor.numel())
        self.calls += 1
        self.zero_codes += int(zero.sum().item())
        self.saturated += int(saturation.sum().item())
        return reconstructed, codes

    def scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._scale_for(tensor)


class FP4E2M1Quantizer(ScaledFloatingPointQuantizer):
    def __init__(self, maximum: torch.Tensor, broadcast_shape=None):
        super().__init__("fp4_e2m1", maximum, broadcast_shape)


class FP6E3M2Quantizer(ScaledFloatingPointQuantizer):
    def __init__(self, maximum: torch.Tensor, broadcast_shape=None):
        super().__init__("fp6_e3m2", maximum, broadcast_shape)


class FP8E4M3FNQuantizer(ScaledFloatingPointQuantizer):
    def __init__(self, maximum: torch.Tensor, broadcast_shape=None):
        super().__init__("fp8_e4m3fn", maximum, broadcast_shape)


class IEEEFP16Quantizer(object):
    """Apply finite IEEE FP16 QDQ without block scaling."""

    def __init__(self, maximum: torch.Tensor, broadcast_shape=None):
        calibration = torch.as_tensor(maximum, dtype=torch.float32)
        if calibration.numel() == 0 or \
                not bool(torch.isfinite(calibration).all().item()):
            raise ValueError("floating-point calibration maximum must be finite")
        if bool((calibration <= 0.0).any().item()):
            raise ValueError("floating-point calibration maximum must be positive")
        self.spec = FORMAT_SPECS["fp16_ieee"]
        self.format = self.spec.name
        self.bits = self.spec.bits
        self.unsigned = False
        self.qmin = -65504
        self.qmax = 65504
        self.scale = torch.ones_like(calibration)
        self.broadcast_shape = None if broadcast_shape is None else tuple(
            int(value) for value in broadcast_shape)
        if self.broadcast_shape is not None and self.scale.numel() != 1:
            non_singleton = 1
            for value in self.broadcast_shape:
                if value != 1:
                    non_singleton *= value
            if len(self.broadcast_shape) == 0 or \
                    self.scale.numel() != non_singleton:
                raise ValueError("floating-point broadcast shape mismatch")
        self.zero_codes = 0
        self.saturated = 0
        self.numel = 0
        self.calls = 0

    def scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        if scale.numel() == 1:
            return scale
        if self.broadcast_shape is None or \
                len(self.broadcast_shape) != tensor.ndim or \
                any(size not in (1, actual) for size, actual in zip(
                    self.broadcast_shape, tensor.shape)):
            raise ValueError("floating-point broadcast tensor shape mismatch")
        return scale.reshape(self.broadcast_shape)

    def quantize_with_codes(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            raise ValueError("floating-point quantization tensor is invalid")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("floating-point quantization tensor is nonfinite")
        clipped = tensor.clamp(-self.spec.maximum, self.spec.maximum)
        reconstructed = clipped.to(torch.float16).to(tensor.dtype)
        codes = torch.sign(reconstructed).to(torch.int8)
        self.numel += int(tensor.numel())
        self.calls += 1
        self.zero_codes += int((reconstructed == 0).sum().item())
        self.saturated += int(
            (tensor.abs() > self.spec.maximum).sum().item())
        return reconstructed, codes


class BFloat16Quantizer(object):
    """Apply finite BF16 cast-and-restore QDQ without block scaling."""

    def __init__(self, maximum: torch.Tensor, broadcast_shape=None):
        calibration = torch.as_tensor(maximum, dtype=torch.float32)
        if calibration.numel() == 0 or \
                not bool(torch.isfinite(calibration).all().item()):
            raise ValueError("floating-point calibration maximum must be finite")
        if bool((calibration <= 0.0).any().item()):
            raise ValueError("floating-point calibration maximum must be positive")
        self.spec = FORMAT_SPECS["bf16"]
        self.format = self.spec.name
        self.bits = self.spec.bits
        self.unsigned = False
        self.qmin = -self.spec.maximum
        self.qmax = self.spec.maximum
        self.scale = torch.ones_like(calibration)
        self.broadcast_shape = None if broadcast_shape is None else tuple(
            int(value) for value in broadcast_shape)
        self.zero_codes = 0
        self.saturated = 0
        self.numel = 0
        self.calls = 0

    def scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones((), device=tensor.device, dtype=tensor.dtype)

    def quantize_with_codes(self, tensor: torch.Tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            raise ValueError("floating-point quantization tensor is invalid")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("floating-point quantization tensor is nonfinite")
        clipped = tensor.clamp(-self.spec.maximum, self.spec.maximum)
        reconstructed = clipped.to(torch.bfloat16).to(tensor.dtype)
        codes = torch.sign(reconstructed).to(torch.int8)
        self.numel += int(tensor.numel())
        self.calls += 1
        self.zero_codes += int((reconstructed == 0).sum().item())
        self.saturated += int(
            (tensor.abs() > self.spec.maximum).sum().item())
        return reconstructed, codes


def make_quantizer(format_name: str, maximum: torch.Tensor,
                   broadcast_shape=None):
    if str(format_name) == "fp4_e2m1":
        return FP4E2M1Quantizer(maximum, broadcast_shape)
    if str(format_name) == "fp6_e3m2":
        return FP6E3M2Quantizer(maximum, broadcast_shape)
    if str(format_name) == "fp8_e4m3fn":
        return FP8E4M3FNQuantizer(maximum, broadcast_shape)
    if str(format_name) == "fp16_ieee":
        return IEEEFP16Quantizer(maximum, broadcast_shape)
    if str(format_name) == "bf16":
        return BFloat16Quantizer(maximum, broadcast_shape)
    raise KeyError("unsupported floating-point format: %s" % format_name)
