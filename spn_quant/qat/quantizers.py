"""Hard-forward fake quantizers for strict CSPN QAT."""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.hardware_aligned_quantization import symmetric_weight_qdq
from spn_quant.qat.ste import hard_forward_proxy
from spn_quant.rotation import SignedActivationQuantizer


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
    """Signed symmetric W4 with an FP32 master-weight gradient path."""

    def __init__(self, bits: int, channel_dim: int) -> None:
        super().__init__()
        self.bits = int(bits)
        self.channel_dim = int(channel_dim)
        if self.bits != 4:
            raise ValueError("CSPN QAT weight quantization requires W4")
        self.register_buffer(
            "scale", torch.empty(0, dtype=torch.float32), persistent=False)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        hard, scale = symmetric_weight_qdq(
            weight, bits=self.bits, channel_dim=self.channel_dim)
        self.scale = scale.detach()
        return hard_forward_proxy(hard, weight)
