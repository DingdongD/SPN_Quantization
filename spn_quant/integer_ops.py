"""Strict integer primitives for quantized Attention and concat Conv."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def _finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s must be finite" % name)


def _maximum(name: str, maximum: float) -> float:
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("%s maximum must be finite and nonnegative" % name)
    return maximum


def quantize_signed(tensor: torch.Tensor, bits: int,
                    maximum: float) -> Tuple[torch.Tensor, float]:
    if not torch.is_tensor(tensor):
        raise TypeError("signed quantization expects a tensor")
    _finite_tensor("signed quantization input", tensor)
    bits = int(bits)
    if bits < 2 or bits > 8:
        raise ValueError("signed integer storage supports 2 to 8 bits")
    maximum = _maximum("signed quantization", maximum)
    qmax = (1 << (bits - 1)) - 1
    scale = maximum / float(qmax) if maximum > 0.0 else 1.0
    codes = torch.round(tensor.to(torch.float64) / scale).clamp(
        -qmax, qmax).to(torch.int8)
    return codes, scale


def quantize_unsigned(tensor: torch.Tensor, bits: int,
                      maximum: float) -> Tuple[torch.Tensor, float]:
    if not torch.is_tensor(tensor):
        raise TypeError("unsigned quantization expects a tensor")
    _finite_tensor("unsigned quantization input", tensor)
    bits = int(bits)
    if bits < 1 or bits > 8:
        raise ValueError("unsigned integer storage supports 1 to 8 bits")
    maximum = _maximum("unsigned quantization", maximum)
    qmax = (1 << bits) - 1
    scale = maximum / float(qmax) if maximum > 0.0 else 1.0
    codes = torch.round(tensor.to(torch.float64) / scale).clamp(
        0, qmax).to(torch.uint8)
    return codes, scale


def requantize_int32(values: torch.Tensor, source_scale: torch.Tensor,
                     target_scale: torch.Tensor, qmin: int, qmax: int,
                     fraction_bits: int = 31) -> torch.Tensor:
    if not torch.is_tensor(values) or values.dtype != torch.int32:
        raise TypeError("requantization expects INT32 values")
    if int(qmin) >= int(qmax):
        raise ValueError("requantization code range is invalid")
    fraction_bits = int(fraction_bits)
    if fraction_bits < 1 or fraction_bits > 31:
        raise ValueError("requantization fraction bits must be in [1, 31]")

    source = torch.as_tensor(
        source_scale, device=values.device, dtype=torch.float64)
    target = torch.as_tensor(
        target_scale, device=values.device, dtype=torch.float64)
    _finite_tensor("source scale", source)
    _finite_tensor("target scale", target)
    if bool(torch.any(source <= 0.0).item()) or \
            bool(torch.any(target <= 0.0).item()):
        raise ValueError("requantization scales must be positive")

    multiplier = torch.round(
        source / target * float(1 << fraction_bits)).to(torch.int64)
    maximum_value = int(values.to(torch.int64).abs().max().item()) \
        if values.numel() else 0
    maximum_multiplier = int(multiplier.abs().max().item()) \
        if multiplier.numel() else 0
    if maximum_value and maximum_multiplier > \
            torch.iinfo(torch.int64).max // maximum_value:
        raise OverflowError("requantization INT64 product overflows")

    product = values.to(torch.int64) * multiplier
    offset = 1 << (fraction_bits - 1)
    magnitude = torch.bitwise_right_shift(
        product.abs() + offset, fraction_bits)
    rounded = torch.where(product < 0, -magnitude, magnitude)
    return rounded.clamp(int(qmin), int(qmax)).to(torch.int32)


def int8_mm_int32(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.dtype != torch.int8 or right.dtype != torch.int8:
        raise TypeError("integer matrix multiplication requires INT8 operands")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("integer matrix multiplication requires rank-2 operands")
    if left.shape[1] != right.shape[0]:
        raise ValueError("integer matrix multiplication dimensions do not align")
    if left.device != right.device:
        raise ValueError("integer matrix multiplication devices must match")
    rows, reduction = left.shape
    columns = right.shape[1]
    if rows == 0 or reduction == 0 or columns == 0:
        raise ValueError("integer matrix multiplication dimensions must be positive")
    padded_rows = max(int(rows), 17)
    padded_reduction = ((int(reduction) + 7) // 8) * 8
    padded_columns = ((int(columns) + 7) // 8) * 8
    left_padded = F.pad(
        left, (0, padded_reduction - reduction, 0, padded_rows - rows))
    right_padded = F.pad(
        right, (0, padded_columns - columns,
                0, padded_reduction - reduction))
    output = torch._int_mm(
        left_padded.contiguous(), right_padded.contiguous())
    return output[:rows, :columns].contiguous()


def _checked_int32(values: torch.Tensor, name: str) -> torch.Tensor:
    limits = torch.iinfo(torch.int32)
    if values.numel() and (int(values.min().item()) < limits.min or
                           int(values.max().item()) > limits.max):
        raise OverflowError("%s exceeds INT32" % name)
    return values.to(torch.int32)


def uint8_int8_mm_int32(left: torch.Tensor,
                        right: torch.Tensor) -> torch.Tensor:
    if left.dtype != torch.uint8 or right.dtype != torch.int8:
        raise TypeError("unsigned integer matrix multiplication expects U8 x I8")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("integer matrix multiplication requires rank-2 operands")
    if left.shape[1] != right.shape[0]:
        raise ValueError("integer matrix multiplication dimensions do not align")
    if left.device != right.device:
        raise ValueError("integer matrix multiplication devices must match")

    shifted = (left.to(torch.int16) - 128).to(torch.int8)
    product = int8_mm_int32(shifted, right).to(torch.int64)
    correction = 128 * right.to(torch.int64).sum(dim=0, keepdim=True)
    return _checked_int32(product + correction, "unsigned matrix product")


def _batched_shape(left: torch.Tensor,
                   right: torch.Tensor) -> Tuple[int, ...]:
    if left.ndim < 3 or right.ndim < 3:
        raise ValueError("batched integer matrix multiplication requires rank >= 3")
    if left.shape[:-2] != right.shape[:-2]:
        raise ValueError("batched integer matrix dimensions do not match")
    if left.shape[-1] != right.shape[-2]:
        raise ValueError("batched integer reduction dimensions do not align")
    if left.device != right.device:
        raise ValueError("batched integer matrix devices must match")
    return tuple(left.shape[:-2])


def batched_int8_mm_int32(left: torch.Tensor,
                          right: torch.Tensor) -> torch.Tensor:
    leading = _batched_shape(left, right)
    if left.dtype != torch.int8 or right.dtype != torch.int8:
        raise TypeError("batched integer matrix multiplication requires INT8")
    matrices = math.prod(leading)
    left_flat = left.reshape(matrices, left.shape[-2], left.shape[-1])
    right_flat = right.reshape(matrices, right.shape[-2], right.shape[-1])
    output = torch.stack([
        int8_mm_int32(left_flat[index], right_flat[index])
        for index in range(matrices)
    ])
    return output.reshape(*leading, left.shape[-2], right.shape[-1])


def batched_uint8_int8_mm_int32(left: torch.Tensor,
                                right: torch.Tensor) -> torch.Tensor:
    leading = _batched_shape(left, right)
    if left.dtype != torch.uint8 or right.dtype != torch.int8:
        raise TypeError("batched unsigned matrix multiplication expects U8 x I8")
    matrices = math.prod(leading)
    left_flat = left.reshape(matrices, left.shape[-2], left.shape[-1])
    right_flat = right.reshape(matrices, right.shape[-2], right.shape[-1])
    output = torch.stack([
        uint8_int8_mm_int32(left_flat[index], right_flat[index])
        for index in range(matrices)
    ])
    return output.reshape(*leading, left.shape[-2], right.shape[-1])


def _pair(name: str, value: Sequence[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError("%s must contain two values" % name)
    pair = (int(value[0]), int(value[1]))
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError("%s values must be nonnegative" % name)
    return pair


def integer_im2col(tensor: torch.Tensor, kernel_size: Sequence[int],
                   stride: Sequence[int], padding: Sequence[int],
                   dilation: Sequence[int]) -> Tuple[torch.Tensor,
                                                     Tuple[int, int]]:
    if tensor.dtype != torch.int8 or tensor.ndim != 4:
        raise TypeError("integer im2col expects a rank-4 INT8 tensor")
    kernel = _pair("kernel_size", kernel_size)
    step = _pair("stride", stride)
    border = _pair("padding", padding)
    spacing = _pair("dilation", dilation)
    if kernel[0] == 0 or kernel[1] == 0 or step[0] == 0 or step[1] == 0:
        raise ValueError("kernel and stride values must be positive")
    if spacing != (1, 1):
        raise ValueError("integer im2col supports dilation=1 only")

    padded = F.pad(
        tensor, (border[1], border[1], border[0], border[0]))
    output_height = (padded.shape[2] - kernel[0]) // step[0] + 1
    output_width = (padded.shape[3] - kernel[1]) // step[1] + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("integer im2col output shape is invalid")
    patches = padded.unfold(2, kernel[0], step[0]).unfold(
        3, kernel[1], step[1])
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(
        tensor.shape[0] * output_height * output_width,
        tensor.shape[1] * kernel[0] * kernel[1])
    return patches, (output_height, output_width)
