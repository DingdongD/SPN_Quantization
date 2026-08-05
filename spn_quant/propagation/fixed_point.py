"""Integer-domain primitives for propagation-aware SPN quantization."""

from __future__ import annotations

import math
from typing import Tuple

import torch


Q13_FRACTION_BITS = 13
Q13_ONE = 1 << Q13_FRACTION_BITS
_INT16_MIN = -(1 << 15)
_INT16_MAX = (1 << 15) - 1


def symmetric_qdq(tensor: torch.Tensor, bits: int, maximum: float
                  ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Per-tensor signed symmetric QDQ with an integer zero code."""
    bits = int(bits)
    if bits < 2 or bits > 16:
        raise ValueError("symmetric quantization supports 2 to 16 bits")
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("quantization maximum must be finite and nonnegative")
    qmax = (1 << (bits - 1)) - 1
    scale = maximum / float(qmax) if maximum > 0.0 else 1.0
    codes = torch.clamp(
        torch.round(tensor / scale), -qmax, qmax).to(torch.int32)
    return codes.to(tensor.dtype) * scale, codes, scale


def _positive_scale(scale: float) -> float:
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("quantization scale must be finite and positive")
    return scale


def _round_divide_signed(numerator: torch.Tensor,
                         denominator: torch.Tensor) -> torch.Tensor:
    numerator = numerator.to(torch.int64)
    denominator = denominator.to(torch.int64)
    if bool(torch.any(denominator <= 0)):
        raise ValueError("normalization denominator must be positive")
    magnitude = (numerator.abs() + denominator // 2) // denominator
    return torch.where(numerator < 0, -magnitude, magnitude)


def _checked_int16(codes: torch.Tensor, name: str) -> torch.Tensor:
    if codes.numel() and (int(codes.min()) < _INT16_MIN or
                          int(codes.max()) > _INT16_MAX):
        raise OverflowError("%s exceeds INT16" % name)
    return codes.to(torch.int16)


def normalize_signed_codes_q13(
        codes: torch.Tensor, scale: float, denominator_floor: bool,
        eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize signed affinity codes and derive the center residual.

    The absolute sum, epsilon, and optional denominator floor are represented
    in the input code domain. Only the reciprocal multiply uses an INT64
    intermediate; normalized coefficients are checked and stored as INT16.
    """
    if not torch.is_tensor(codes) or codes.ndim < 2:
        raise ValueError("affinity codes require a neighbor dimension")
    scale = _positive_scale(scale)
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("normalization epsilon must be finite and nonnegative")

    codes32 = codes.to(torch.int32)
    denominator = codes32.abs().sum(dim=1, keepdim=True, dtype=torch.int32)
    epsilon_codes = int(math.ceil(eps / scale)) if eps else 0
    if epsilon_codes:
        denominator = denominator + epsilon_codes
    if denominator_floor:
        floor_codes = int(math.ceil(1.0 / scale))
        denominator = torch.maximum(
            denominator, torch.full_like(denominator, floor_codes))
    denominator = torch.clamp(denominator, min=1)

    numerator = codes32.to(torch.int64) * Q13_ONE
    normalized32 = _round_divide_signed(numerator, denominator).to(torch.int32)
    excess = torch.clamp(
        normalized32.abs().sum(dim=1, keepdim=True, dtype=torch.int32) -
        Q13_ONE, min=0)
    if bool(torch.any(excess > 0)):
        winner = normalized32.abs().argmax(dim=1, keepdim=True)
        winner_values = normalized32.gather(1, winner)
        correction = -torch.sign(winner_values) * excess
        normalized32.scatter_add_(1, winner, correction)
    center32 = Q13_ONE - normalized32.sum(
        dim=1, keepdim=True, dtype=torch.int32)
    normalized_codes = _checked_int16(
        normalized32, "normalized affinity coefficient")
    center_codes = _checked_int16(center32, "center affinity coefficient")
    values = normalized_codes.to(torch.float32) / float(Q13_ONE)
    return values, center_codes, normalized_codes


def softmax_codes_q13(codes: torch.Tensor, scale: float,
                      dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an exponential LUT to signed logits and return exact-sum Q13."""
    if not torch.is_tensor(codes) or codes.numel() == 0:
        raise ValueError("softmax codes must be a nonempty tensor")
    scale = _positive_scale(scale)
    codes16 = codes.to(torch.int16)
    differences = codes16 - codes16.max(dim=dim, keepdim=True).values
    minimum = int(differences.min())
    if minimum > 0:
        raise RuntimeError("softmax difference must be nonpositive")

    exp_one = 1 << 20
    lut_domain = torch.arange(
        minimum, 1, device=codes.device, dtype=torch.float32)
    lut = torch.round(torch.exp(lut_domain * scale) * exp_one)
    lut = torch.clamp(lut, min=1, max=exp_one).to(torch.int32)
    weights = lut[(differences - minimum).long()]
    weight_sum = weights.sum(dim=dim, keepdim=True, dtype=torch.int32)
    reciprocal_q30 = torch.round(
        float(1 << 30) / weight_sum.to(torch.float64)).to(torch.int64)
    normalized32 = ((weights.to(torch.int64) * reciprocal_q30 * Q13_ONE +
                     (1 << 29)) >> 30).to(torch.int32)
    residual = Q13_ONE - normalized32.sum(
        dim=dim, keepdim=True, dtype=torch.int32)
    winner = weights.argmax(dim=dim, keepdim=True)
    normalized32.scatter_add_(dim, winner, residual)
    normalized_codes = _checked_int16(
        normalized32, "softmax affinity coefficient")
    if bool(torch.any(normalized_codes < 0)):
        raise RuntimeError("softmax coefficient must be nonnegative")
    values = normalized_codes.to(torch.float32) / float(Q13_ONE)
    return values, normalized_codes


def unsigned_unit_qdq(tensor: torch.Tensor, bits: int = 8
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a probability/gate while preserving exact zero and one."""
    bits = int(bits)
    if bits < 2 or bits > 16:
        raise ValueError("unsigned unit quantization supports 2 to 16 bits")
    qmax = (1 << bits) - 1
    codes = torch.clamp(torch.round(tensor * qmax), 0, qmax).to(torch.int32)
    values = codes.to(tensor.dtype) / float(qmax)
    return values, codes
