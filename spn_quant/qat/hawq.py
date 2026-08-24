"""HAWQ activation quantization for CSPN semantic owners."""

from __future__ import annotations

import torch
import torch.nn as nn

from spn_quant.qat.ste import hard_forward_proxy


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        raise ValueError("%s must be a nonempty tensor" % name)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s must be finite" % name)


class HAWQActivationQuantizer(nn.Module):
    """Affine QDQ with an explicit running range for one activation owner."""

    def __init__(self, bits: int, unsigned: bool,
                 range_momentum: float) -> None:
        super().__init__()
        self.bits = int(bits)
        self.unsigned = bool(unsigned)
        self.range_momentum = float(range_momentum)
        if self.bits not in (4, 6, 8):
            raise ValueError("HAWQ activation bits must be 4, 6, or 8")
        if not 0.0 <= self.range_momentum < 1.0:
            raise ValueError("HAWQ range momentum must lie in [0, 1)")
        self.qmin = 0
        self.qmax = (1 << self.bits) - 1
        self.format = "int"
        self.granularity = "tensor"
        self.group_size = 1
        self.scale_count = 1
        self.register_buffer("minimum", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("maximum", torch.zeros(1, dtype=torch.float32))
        self.register_buffer(
            "range_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "running_range_state", torch.tensor(True, dtype=torch.bool))

    @property
    def running_range(self) -> bool:
        return bool(self.running_range_state.item())

    def _measured_range(self, tensor: torch.Tensor):
        _require_finite("HAWQ activation", tensor)
        minimum = tensor.detach().amin().to(
            device=self.minimum.device, dtype=self.minimum.dtype)
        maximum = tensor.detach().amax().to(
            device=self.maximum.device, dtype=self.maximum.dtype)
        if self.unsigned:
            if float(minimum.item()) < 0.0:
                raise ValueError("unsigned HAWQ activation contains negatives")
            minimum = torch.zeros_like(minimum)
        return minimum.reshape_as(self.minimum), maximum.reshape_as(self.maximum)

    def initialize_range(self, tensor: torch.Tensor) -> None:
        minimum, maximum = self._measured_range(tensor)
        if float(maximum.item()) <= float(minimum.item()):
            raise ValueError("HAWQ activation range must be nonzero")
        self.minimum.copy_(minimum)
        self.maximum.copy_(maximum)
        self.range_initialized.fill_(True)

    def _update_range(self, tensor: torch.Tensor) -> None:
        minimum, maximum = self._measured_range(tensor)
        if not bool(self.range_initialized.item()):
            if float(maximum.item()) <= float(minimum.item()):
                raise ValueError("HAWQ activation range must be nonzero")
            self.minimum.copy_(minimum)
            self.maximum.copy_(maximum)
            self.range_initialized.fill_(True)
            return
        momentum = self.range_momentum
        self.minimum.mul_(momentum).add_(minimum * (1.0 - momentum))
        self.maximum.mul_(momentum).add_(maximum * (1.0 - momentum))

    def freeze_range(self) -> None:
        if not bool(self.range_initialized.item()):
            raise RuntimeError("HAWQ activation range is not initialized")
        self.running_range_state.fill_(False)

    def _parameters_for(self, tensor: torch.Tensor):
        _require_finite("HAWQ activation", tensor)
        if not bool(self.range_initialized.item()):
            raise RuntimeError("HAWQ activation range is not initialized")
        _require_finite("HAWQ activation minimum", self.minimum)
        _require_finite("HAWQ activation maximum", self.maximum)
        if not bool((self.maximum > self.minimum).all().item()):
            raise ValueError("HAWQ activation range is invalid")
        scale = (self.maximum - self.minimum) / float(self.qmax)
        zero_point = torch.round(-self.minimum / scale).clamp(
            self.qmin, self.qmax)
        return (
            scale.to(device=tensor.device, dtype=tensor.dtype),
            zero_point.to(device=tensor.device, dtype=tensor.dtype),
        )

    @property
    def zero_point(self) -> torch.Tensor:
        scale, zero_point = self._parameters_for(self.minimum)
        del scale
        return zero_point.detach()

    def scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._parameters_for(tensor)[0]

    def quantize_with_codes(self, tensor: torch.Tensor):
        if self.training and self.running_range:
            self._update_range(tensor)
        scale, zero_point = self._parameters_for(tensor)
        codes = torch.round(tensor.detach() / scale + zero_point).clamp(
            self.qmin, self.qmax).to(torch.int64)
        hard = (codes.to(tensor.dtype) - zero_point) * scale
        return hard_forward_proxy(hard, tensor), codes

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize_with_codes(tensor)[0]
