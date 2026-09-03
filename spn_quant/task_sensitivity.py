"""Task-gradient and propagation sensitivity primitives."""

from __future__ import annotations

import math
from typing import Dict

import torch


def _validate_triplet(gradient, reference, quantized) -> None:
    for name, tensor in (("gradient", gradient),
                         ("reference", reference),
                         ("quantized", quantized)):
        if not torch.is_tensor(tensor):
            raise TypeError("%s must be a tensor" % name)
        if tensor.numel() == 0:
            raise ValueError("%s must be nonempty" % name)
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("%s must be finite" % name)
    if gradient.shape != reference.shape or reference.shape != quantized.shape:
        raise ValueError("gradient, reference and quantized shapes differ")


def gradient_weighted_error(gradient, reference, quantized) -> float:
    """Return sum(abs(gradient * (reference - quantized)))."""
    _validate_triplet(gradient, reference, quantized)
    value = (gradient.detach().float() *
             (reference.detach().float() - quantized.detach().float())).abs().sum()
    result = float(value.item())
    if not math.isfinite(result):
        raise FloatingPointError("gradient-weighted error is non-finite")
    return result


def normalized_gradient_weighted_error(gradient, reference, quantized) -> float:
    """Normalize gradient-weighted error by reference tensor L1."""
    score = gradient_weighted_error(gradient, reference, quantized)
    denominator = float(reference.detach().float().abs().sum().item())
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("reference L1 must be finite and positive")
    return score / denominator


def candidate_sensitivity_scores(gradient, reference, quantized_by_bits):
    """Return task-gradient scores for exactly the 4/6/8 bit candidates."""
    levels = (4, 6, 8)
    if tuple(sorted(int(bits) for bits in quantized_by_bits)) != levels:
        raise ValueError("candidate score levels must be exactly 4, 6, and 8")
    scores = {}
    for bits in levels:
        scores[bits] = gradient_weighted_error(
            gradient, reference, quantized_by_bits[bits])
    return scores


def marginal_score_per_saved_bit(
        current_score: float, lower_precision_score: float,
        memory_saved: float) -> float:
    """Return marginal sensitivity divided by saved memory."""
    current_score = float(current_score)
    lower_precision_score = float(lower_precision_score)
    memory_saved = float(memory_saved)
    if not math.isfinite(current_score) or not math.isfinite(
            lower_precision_score):
        raise ValueError("sensitivity scores must be finite")
    if not math.isfinite(memory_saved) or memory_saved <= 0.0:
        raise ValueError("memory saving must be finite and positive")
    degradation = lower_precision_score - current_score
    if degradation < 0.0:
        raise ValueError("lower precision score must not improve sensitivity")
    return degradation / memory_saved


def propagation_metric_row(model: str, sample: str, signal: str,
                           iteration: int, quantized, reference) -> Dict[str, float]:
    """Return paired propagation MSE and SQNR metrics for one signal."""
    if not isinstance(model, str) or not model:
        raise ValueError("model name is required")
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample name is required")
    if not isinstance(signal, str) or not signal:
        raise ValueError("signal name is required")
    if int(iteration) < 0:
        raise ValueError("propagation iteration must be nonnegative")
    if not torch.is_tensor(quantized) or not torch.is_tensor(reference):
        raise TypeError("propagation metrics require tensors")
    if quantized.shape != reference.shape or quantized.numel() == 0:
        raise ValueError("propagation metric tensor shapes are invalid")
    if not bool(torch.isfinite(quantized).all().item()) or not bool(
            torch.isfinite(reference).all().item()):
        raise ValueError("propagation metric tensors must be finite")
    q = quantized.detach().float()
    r = reference.detach().float()
    error = q - r
    mse = float(error.pow(2).mean().item())
    signal_power = float(r.pow(2).mean().item())
    noise_power = float(error.pow(2).mean().item())
    if noise_power == 0.0:
        sqnr_db = float("inf")
    elif signal_power == 0.0:
        sqnr_db = float("-inf")
    else:
        sqnr_db = 10.0 * math.log10(signal_power / noise_power)
    return {
        "model": model,
        "sample": sample,
        "signal": signal,
        "iteration": int(iteration),
        "mse": mse,
        "sqnr_db": sqnr_db,
        "zero_ratio": float((q == 0).float().mean().item()),
        "nonpositive_ratio": float((q <= 0).float().mean().item()),
    }


__all__ = (
    "gradient_weighted_error",
    "normalized_gradient_weighted_error",
    "candidate_sensitivity_scores",
    "marginal_score_per_saved_bit",
    "propagation_metric_row",
)
