#!/usr/bin/env python3
"""Scaled E2M1 activation QDQ primitives."""

from __future__ import division

import torch


E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class E2M1ActivationQuantizer(object):
    bits = 4
    format = "e2m1"
    unsigned = False
    qmin = 0
    qmax = 15
    zero_point = 0

    def __init__(self, maximum, channel_dim=None):
        maximum = torch.as_tensor(maximum, dtype=torch.float32).abs()
        if not bool(torch.isfinite(maximum).all().item()):
            raise ValueError("E2M1 calibration maximum must be finite")
        self.maximum = maximum
        self.channel_dim = channel_dim
        self.scale = torch.where(
            maximum > 0.0, maximum / 6.0, torch.ones_like(maximum))

    def _scale_for(self, tensor):
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        if scale.ndim == 0:
            return scale
        if self.channel_dim is None:
            raise ValueError("vector E2M1 scale requires channel_dim")
        shape = [1] * tensor.ndim
        shape[self.channel_dim] = scale.numel()
        return scale.reshape(shape)

    @staticmethod
    def _positive_codes(values):
        codebook = values.new_tensor(E2M1_POSITIVE_VALUES)
        distances = (values.unsqueeze(-1) - codebook).abs()
        minimum = distances.min(dim=-1, keepdim=True).values
        candidates = distances == minimum
        code_ids = torch.arange(8, device=values.device)
        even = (code_ids.remainder(2) == 0).reshape(
            *([1] * values.ndim), 8)
        rank = torch.where(
            candidates & even,
            torch.zeros_like(distances, dtype=torch.int8),
            torch.where(
                candidates,
                torch.ones_like(distances, dtype=torch.int8),
                torch.full_like(distances, 2, dtype=torch.int8)))
        return rank.argmin(dim=-1).to(torch.int32)

    def quantize_with_codes(self, tensor):
        scale = self._scale_for(tensor)
        normalized = tensor / scale
        magnitude = normalized.abs().clamp(max=6.0)
        positive_codes = self._positive_codes(magnitude)
        sign_codes = (normalized < 0).to(torch.int32) * 8
        encoded = positive_codes | sign_codes
        values = tensor.new_tensor(
            E2M1_POSITIVE_VALUES)[positive_codes.long()]
        quantized = torch.where(normalized < 0, -values, values) * scale
        return quantized, encoded

    def saturated_count(self, tensor):
        normalized = tensor.detach().abs() / self._scale_for(tensor)
        return int((normalized > 6.0).sum().item())

    @staticmethod
    def zero_code_count(codes):
        return int(((codes & 7) == 0).sum().item())

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]

    def manifest(self):
        return {
            "format": self.format,
            "bits": self.bits,
            "unsigned": self.unsigned,
            "codebook": ";".join(
                str(value) for value in E2M1_POSITIVE_VALUES),
            "scale": self.scale,
            "channel_dim": "" if self.channel_dim is None
            else int(self.channel_dim),
        }
