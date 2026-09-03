"""LSQ+ quantizers adapted to CSPN semantic activation owners."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def grad_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    scaled = value * float(scale)
    return (value - scaled).detach() + scaled


def round_pass(value: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(value)
    return (rounded - value).detach() + value


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        raise ValueError("%s must be a nonempty tensor" % name)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s must be finite" % name)


class LSQPlusActivationQuantizer(nn.Module):
    """Learn one affine LSQ+ grid for one semantic activation owner."""

    def __init__(self, bits: int, unsigned: bool) -> None:
        super().__init__()
        self.bits = int(bits)
        self.unsigned = bool(unsigned)
        if self.bits not in (4, 6, 8):
            raise ValueError("LSQ+ activation bits must be 4, 6, or 8")
        self.qmin = 0 if self.unsigned else -(1 << (self.bits - 1))
        self.qmax = (1 << self.bits) - 1 if self.unsigned else \
            (1 << (self.bits - 1)) - 1
        self.format = "int"
        self.granularity = "tensor"
        self.group_size = 1
        self.scale_count = 1
        self.step = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.offset = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.register_buffer(
            "initialized", torch.tensor(False, dtype=torch.bool))

    @property
    def zero_point(self) -> torch.Tensor:
        self._validate_parameters()
        return torch.round(
            -self.offset.detach() / self.step.detach().abs())

    def _validate_parameters(self) -> None:
        _require_finite("LSQ+ activation step", self.step)
        _require_finite("LSQ+ activation offset", self.offset)
        if bool((self.step == 0.0).any().item()):
            raise ValueError("LSQ+ activation raw step must be nonzero")

    def initialize(self, tensor: torch.Tensor) -> None:
        _require_finite("LSQ+ initialization activation", tensor)
        minimum = tensor.detach().amin().to(
            device=self.step.device, dtype=self.step.dtype)
        maximum = tensor.detach().amax().to(
            device=self.step.device, dtype=self.step.dtype)
        step = (maximum - minimum) / float(self.qmax - self.qmin)
        if not bool(torch.isfinite(step).item()) or float(step.item()) <= 0.0:
            raise ValueError("LSQ+ activation initialization step is invalid")
        offset = minimum - step * float(self.qmin)
        self.step.data.copy_(step.reshape_as(self.step))
        self.offset.data.copy_(offset.reshape_as(self.offset))
        self.initialized.fill_(True)

    def _parameters_for(self, tensor: torch.Tensor):
        if not bool(self.initialized.item()):
            raise RuntimeError("LSQ+ activation quantizer is not initialized")
        _require_finite("LSQ+ activation", tensor)
        self._validate_parameters()
        gradient_scale = 1.0 / math.sqrt(
            float(tensor.numel() * self.qmax))
        step = grad_scale(self.step, gradient_scale).abs().to(
            device=tensor.device, dtype=tensor.dtype)
        offset = grad_scale(self.offset, gradient_scale).to(
            device=tensor.device, dtype=tensor.dtype)
        return step, offset

    def scale_for(self, tensor: torch.Tensor) -> torch.Tensor:
        step, _ = self._parameters_for(tensor)
        return step

    def quantize_with_codes(self, tensor: torch.Tensor):
        step, offset = self._parameters_for(tensor)
        normalized = (tensor - offset) / step
        quantized_codes = round_pass(normalized).clamp(self.qmin, self.qmax)
        hard_codes = torch.round(normalized.detach()).clamp(
            self.qmin, self.qmax).to(torch.int64)
        return quantized_codes * step + offset, hard_codes

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize_with_codes(tensor)[0]


class LSQPlusWeightParametrization(nn.Module):
    """Learn one symmetric LSQ+ step per logical output channel."""

    def __init__(self, bits: int, channel_dim: int,
                 initial_weight: torch.Tensor) -> None:
        super().__init__()
        self.bits = int(bits)
        if self.bits not in (4, 6, 8):
            raise ValueError("LSQ+ weight bits must be 4, 6, or 8")
        _require_finite("LSQ+ initial weight", initial_weight)
        self.channel_dim = int(channel_dim)
        if self.channel_dim < 0 or self.channel_dim >= initial_weight.ndim:
            raise ValueError("LSQ+ weight channel dimension is invalid")
        self.qmin = -(1 << (self.bits - 1))
        self.qmax = (1 << (self.bits - 1)) - 1
        self.channel_count = int(initial_weight.shape[self.channel_dim])
        shape = [1] * initial_weight.ndim
        shape[self.channel_dim] = self.channel_count
        flat = initial_weight.detach().movedim(
            self.channel_dim, 0).reshape(self.channel_count, -1)
        mean = flat.mean(dim=1)
        standard_deviation = flat.std(dim=1, unbiased=False)
        maximum = torch.maximum(
            (mean - 3.0 * standard_deviation).abs(),
            (mean + 3.0 * standard_deviation).abs())
        step = maximum / float((1 << self.bits) - 1)
        if not bool(torch.isfinite(step).all().item()) or \
                not bool((step > 0.0).all().item()):
            raise ValueError("LSQ+ weight initialization step is invalid")
        self.step = nn.Parameter(step.reshape(shape).to(torch.float32))

    def _step_for(self, weight: torch.Tensor) -> torch.Tensor:
        _require_finite("LSQ+ weight", weight)
        if int(weight.shape[self.channel_dim]) != self.channel_count:
            raise ValueError("LSQ+ weight output channels changed")
        _require_finite("LSQ+ weight step", self.step)
        if bool((self.step == 0.0).any().item()):
            raise ValueError("LSQ+ weight raw step must be nonzero")
        gradient_scale = 1.0 / math.sqrt(
            float(weight.numel() * self.qmax))
        return grad_scale(self.step, gradient_scale).abs().to(
            device=weight.device, dtype=weight.dtype)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        step = self._step_for(weight)
        codes = round_pass(weight / step).clamp(self.qmin, self.qmax)
        return codes * step
