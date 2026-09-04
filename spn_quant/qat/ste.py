"""Exact hard-forward autograd primitives for quantization-aware training."""

from __future__ import annotations

import torch


class _HardForwardProxy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hard: torch.Tensor,
                proxy: torch.Tensor) -> torch.Tensor:
        del ctx
        if hard.shape != proxy.shape:
            raise ValueError(
                "hard and proxy tensors require matching shapes")
        return hard

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return None, gradient


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.round(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return gradient


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("%s contains non-finite values" % name)


def hard_forward_proxy(hard: torch.Tensor,
                       proxy: torch.Tensor) -> torch.Tensor:
    _require_finite(hard, "hard-forward tensor")
    _require_finite(proxy, "proxy tensor")
    return _HardForwardProxy.apply(hard, proxy)


def hard_forward_proxy_unchecked(
        hard: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
    return _HardForwardProxy.apply(hard, proxy)


def round_ste(value: torch.Tensor) -> torch.Tensor:
    _require_finite(value, "rounded tensor")
    return _RoundSTE.apply(value)
