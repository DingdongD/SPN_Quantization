"""Adaptive weight rounding for post-training quantization.

The implementation follows the core AdaRound idea: keep a fixed uniform weight
scale and learn one binary up/down rounding decision per weight. PyTorch
parametrizations preserve the original module type during reconstruction and
can be removed after hardening without changing inference operators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


SUPPORTED_WEIGHT_MODULES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


@dataclass(frozen=True)
class AdaptiveRoundingConfig:
    bits: int = 4
    clip_ratio: float = 1.0
    gamma: float = -0.1
    zeta: float = 1.1
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if int(self.bits) < 2:
            raise ValueError("weight bits must be at least 2")
        if not 0.0 < float(self.clip_ratio) <= 1.0:
            raise ValueError("clip_ratio must be in (0, 1]")
        if not float(self.gamma) < 0.0 < 1.0 < float(self.zeta):
            raise ValueError("expected gamma < 0 < 1 < zeta")
        if not 0.0 < float(self.eps) < 0.5:
            raise ValueError("eps must be in (0, 0.5)")

    def manifest(self) -> Dict[str, Any]:
        return asdict(self)


def is_supported_weight_module(module: nn.Module) -> bool:
    return isinstance(module, SUPPORTED_WEIGHT_MODULES)


def _quant_limits(bits: int) -> Tuple[int, int]:
    qmax = 2 ** (int(bits) - 1) - 1
    return -qmax, qmax


def _module_layout(module: nn.Module) -> Tuple[str, int]:
    if isinstance(module, nn.ConvTranspose2d):
        return "conv_transpose", int(module.groups)
    if isinstance(module, nn.Conv2d):
        return "output_axis0", int(module.groups)
    if isinstance(module, nn.Linear):
        return "output_axis0", 1
    raise TypeError("unsupported adaptive-rounding module: %s" % type(module).__name__)


def _compact_scale(module: nn.Module, weight: torch.Tensor,
                   qmax: int, clip_ratio: float) -> torch.Tensor:
    layout, groups = _module_layout(module)
    detached = weight.detach().abs()
    if layout == "output_axis0":
        dimensions = tuple(range(1, weight.ndim))
        maximum = detached.amax(dim=dimensions, keepdim=True)
    elif groups == 1:
        dimensions = (0,) + tuple(range(2, weight.ndim))
        maximum = detached.amax(dim=dimensions, keepdim=True)
    else:
        in_per_group = int(module.in_channels // groups)
        out_per_group = int(module.out_channels // groups)
        reshaped = detached.reshape(
            groups, in_per_group, out_per_group, *weight.shape[2:])
        dimensions = (1,) + tuple(range(3, reshaped.ndim))
        maximum = reshaped.amax(dim=dimensions, keepdim=True)
    maximum = maximum * float(clip_ratio)
    safe = torch.where(maximum > 0.0, maximum, torch.ones_like(maximum))
    return safe / float(qmax)


def _expand_scale(module: nn.Module, weight: torch.Tensor,
                  compact: torch.Tensor) -> torch.Tensor:
    layout, groups = _module_layout(module)
    if layout == "output_axis0" or groups == 1:
        return compact
    in_per_group = int(module.in_channels // groups)
    out_per_group = int(module.out_channels // groups)
    expanded = compact.expand(
        groups, in_per_group, out_per_group, *weight.shape[2:])
    return expanded.reshape_as(weight)


class AdaptiveRoundingParametrization(nn.Module):
    """Differentiable up/down rounding for one weight tensor."""

    def __init__(self, module: nn.Module, weight: torch.Tensor,
                 config: AdaptiveRoundingConfig) -> None:
        super(AdaptiveRoundingParametrization, self).__init__()
        if not is_supported_weight_module(module):
            raise TypeError("unsupported adaptive-rounding module")
        self.config = config
        self.module_type = type(module).__name__
        self.layout, self.groups = _module_layout(module)
        self.in_channels = int(getattr(module, "in_channels", 0))
        self.out_channels = int(getattr(module, "out_channels", weight.shape[0]))
        self.qmin, self.qmax = _quant_limits(config.bits)
        compact = _compact_scale(
            module, weight, self.qmax, config.clip_ratio)
        self.register_buffer("scale", compact.detach().clone())
        self.soft_targets = True
        alpha = self._initialize_alpha(module, weight)
        self.alpha = nn.Parameter(alpha)

    def _expanded_scale(self, weight: torch.Tensor) -> torch.Tensor:
        if self.layout == "output_axis0" or self.groups == 1:
            return self.scale
        in_per_group = int(self.in_channels // self.groups)
        out_per_group = int(self.out_channels // self.groups)
        expanded = self.scale.expand(
            self.groups, in_per_group, out_per_group, *weight.shape[2:])
        return expanded.reshape_as(weight)

    def _initialize_alpha(self, module: nn.Module,
                          weight: torch.Tensor) -> torch.Tensor:
        scale = _expand_scale(module, weight, self.scale)
        scaled = weight.detach() / scale
        residual = scaled - torch.floor(scaled)
        probability = (
            (residual - self.config.gamma) /
            (self.config.zeta - self.config.gamma)
        ).clamp(self.config.eps, 1.0 - self.config.eps)
        return torch.log(probability / (1.0 - probability))

    def soft_rounding(self) -> torch.Tensor:
        stretched = (
            torch.sigmoid(self.alpha) *
            (self.config.zeta - self.config.gamma) +
            self.config.gamma
        )
        return stretched.clamp(0.0, 1.0)

    def hard_rounding(self) -> torch.Tensor:
        return (self.alpha >= 0.0).to(dtype=self.alpha.dtype)

    def rounding(self) -> torch.Tensor:
        return self.soft_rounding() if self.soft_targets else self.hard_rounding()

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        scale = self._expanded_scale(weight)
        scaled = weight / scale
        floor = torch.floor(scaled)
        codes = (floor + self.rounding()).clamp(self.qmin, self.qmax)
        return codes * scale

    def regularization(self, beta: float = 2.0) -> torch.Tensor:
        probability = self.soft_rounding()
        return torch.mean(1.0 - torch.abs(2.0 * probability - 1.0).pow(float(beta)))

    def manifest(self, module_name: str = "") -> Dict[str, Any]:
        with torch.no_grad():
            soft = self.soft_rounding()
            row = {
                "module": module_name,
                "module_type": self.module_type,
                "bits": int(self.config.bits),
                "clip_ratio": float(self.config.clip_ratio),
                "scale_min": float(self.scale.min().item()),
                "scale_max": float(self.scale.max().item()),
                "round_up_ratio": float((soft >= 0.5).float().mean().item()),
                "soft_distance": float(torch.minimum(soft, 1.0 - soft).mean().item()),
                "hardened": int(not self.soft_targets),
            }
        return row


class AdaptiveRoundingController(object):
    """Install, optimize, harden, or remove AdaRound parametrizations."""

    def __init__(self, model: nn.Module,
                 config: AdaptiveRoundingConfig = AdaptiveRoundingConfig()) -> None:
        self.model = model
        self.config = config
        self.parametrizations = {}  # type: Dict[str, AdaptiveRoundingParametrization]
        self.modules = {}  # type: Dict[str, nn.Module]
        self._hardened_manifest = []  # type: List[Dict[str, Any]]

    def install(self, module_names: Iterable[str]) -> None:
        named = dict(self.model.named_modules())
        requested = [str(name) for name in module_names]
        if not requested:
            raise ValueError("at least one module name is required")
        for name in requested:
            if name in self.parametrizations:
                raise ValueError("adaptive rounding already installed: %s" % name)
            try:
                module = named[name]
            except KeyError:
                raise KeyError("unknown weight module: %s" % name)
            if not is_supported_weight_module(module):
                raise TypeError("unsupported weight module %s: %s" % (
                    name, type(module).__name__))
            if parametrize.is_parametrized(module, "weight"):
                raise RuntimeError("weight is already parametrized: %s" % name)
            parametrization = AdaptiveRoundingParametrization(
                module, module.weight, self.config)
            parametrize.register_parametrization(
                module, "weight", parametrization, unsafe=True)
            self.parametrizations[name] = parametrization
            self.modules[name] = module

    def install_matching(self, patterns: Sequence[str]) -> List[str]:
        names = select_weight_modules(self.model, patterns)
        self.install(names)
        return names

    def parameters(self) -> Iterator[nn.Parameter]:
        for item in self.parametrizations.values():
            yield item.alpha

    def set_soft_targets(self, enabled: bool) -> None:
        for item in self.parametrizations.values():
            item.soft_targets = bool(enabled)

    def regularization(self, beta: float = 2.0) -> torch.Tensor:
        losses = [item.regularization(beta) for item in self.parametrizations.values()]
        if not losses:
            raise RuntimeError("no adaptive-rounding parametrizations installed")
        return torch.stack(losses).mean()

    def manifest(self) -> List[Dict[str, Any]]:
        if self._hardened_manifest:
            return list(self._hardened_manifest)
        return [self.parametrizations[name].manifest(name)
                for name in sorted(self.parametrizations)]

    def harden(self) -> List[Dict[str, Any]]:
        self.set_soft_targets(False)
        rows = [self.parametrizations[name].manifest(name)
                for name in sorted(self.parametrizations)]
        for row in rows:
            row["hardened"] = 1
        for name in sorted(self.modules):
            module = self.modules[name]
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True)
        self._hardened_manifest = rows
        self.parametrizations = {}
        self.modules = {}
        return list(rows)

    def remove(self) -> None:
        for name in sorted(self.modules):
            module = self.modules[name]
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=False)
        self.parametrizations = {}
        self.modules = {}
        self._hardened_manifest = []


def select_weight_modules(model: nn.Module, patterns: Sequence[str] = (),
                          include_all: bool = False) -> List[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    output = []
    for name, module in model.named_modules():
        if not name or not is_supported_weight_module(module):
            continue
        if include_all or any(pattern.search(name) for pattern in compiled):
            output.append(name)
    if not output:
        raise ValueError("no supported weight modules matched")
    return sorted(output)


class LinearTemperatureDecay(object):
    """BRECQ/AdaRound-style beta schedule after an optional warm-up."""

    def __init__(self, total_steps: int, warmup_fraction: float = 0.2,
                 beta_start: float = 20.0, beta_end: float = 2.0) -> None:
        if int(total_steps) <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= float(warmup_fraction) < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        self.total_steps = int(total_steps)
        self.warmup_steps = int(round(total_steps * warmup_fraction))
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

    def __call__(self, step: int) -> Optional[float]:
        step = int(step)
        if step < self.warmup_steps:
            return None
        denominator = max(self.total_steps - self.warmup_steps - 1, 1)
        progress = min(max((step - self.warmup_steps) / float(denominator), 0.0), 1.0)
        return self.beta_start + progress * (self.beta_end - self.beta_start)
