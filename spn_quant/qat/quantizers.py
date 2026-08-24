"""Hard-forward fake quantizers for strict CSPN QAT."""

from __future__ import annotations

import torch
import torch.nn as nn

from spn_quant.qat.ste import hard_forward_proxy
from spn_quant.activation_boundaries import SignedActivationQuantizer


class ActivationSTEQuantizer(nn.Module):
    """Route gradients through an existing hard activation quantizer."""

    def __init__(self, hard_quantizer) -> None:
        super().__init__()
        self.hard_quantizer = hard_quantizer
        self.bits = hard_quantizer.bits
        self.format = hard_quantizer.format
        if isinstance(hard_quantizer, SignedActivationQuantizer):
            if hard_quantizer.group_size == hard_quantizer.channels:
                self.granularity = "tensor"
            elif hard_quantizer.group_size == 1:
                self.granularity = "channel"
            else:
                self.granularity = "group"
            self.unsigned = False
            self.scale_count = int(hard_quantizer.scales.numel())
        else:
            self.granularity = hard_quantizer.granularity
            self.unsigned = hard_quantizer.unsigned
            self.scale_count = hard_quantizer.scale_count
        self.qmin = hard_quantizer.qmin
        self.qmax = hard_quantizer.qmax
        self.group_size = hard_quantizer.group_size
        self.zero_point = hard_quantizer.zero_point

    def scale_for(self, tensor: torch.Tensor):
        return self.hard_quantizer.scale_for(tensor)

    def quantize_with_codes(self, tensor: torch.Tensor):
        hard, codes = self.hard_quantizer.quantize_with_codes(tensor)
        return hard_forward_proxy(hard, tensor), codes

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize_with_codes(tensor)[0]


class PerOutputChannelWeightFakeQuantizer(nn.Module):
    """Signed symmetric weight QDQ with an FP32 master-weight gradient path."""

    def __init__(self, bits: int, channel_dim: int) -> None:
        super().__init__()
        self.bits = int(bits)
        self.channel_dim = int(channel_dim)
        if self.bits not in (4, 6, 8):
            raise ValueError("CSPN QAT weight bits must be 4, 6, or 8")
        self.register_buffer(
            "scale", torch.empty(0, dtype=torch.float32), persistent=False)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        detached = weight.detach()
        flat = detached.movedim(self.channel_dim, 0).reshape(
            detached.shape[self.channel_dim], -1)
        maximum = flat.abs().amax(dim=1)
        safe_maximum = torch.where(
            maximum > 0.0, maximum, torch.ones_like(maximum))
        shape = [1] * detached.ndim
        shape[self.channel_dim] = detached.shape[self.channel_dim]
        qmax = 2 ** (self.bits - 1) - 1
        scale = safe_maximum.to(torch.float64).div(float(qmax)).to(
            detached.dtype).reshape(shape)
        codes = torch.round(detached / scale).clamp(-qmax, qmax)
        hard = codes * scale
        self.scale = scale.detach()
        return hard_forward_proxy(hard, weight)
