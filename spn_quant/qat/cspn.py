"""Strict quantization-aware training controllers for official CSPN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from spn_quant.propagation.adapters import (
    _crop_cspn,
    _pad_cspn_channels,
    _pad_cspn_state,
)
from spn_quant.propagation.fixed_point import symmetric_qdq
from spn_quant.propagation.controller import PropagationQuantConfig
from spn_quant.qat.quantizers import (
    ActivationSTEQuantizer,
    PerOutputChannelWeightFakeQuantizer,
)
from spn_quant.qat.ste import hard_forward_proxy


Owner = Tuple[str, str]


def _canonical_weight_bits(
        rows: Sequence[Tuple[str, int]]) -> Tuple[Tuple[str, int], ...]:
    values = tuple((str(name), int(bits)) for name, bits in rows)
    names = tuple(name for name, bits in values)
    if not values:
        raise ValueError("CSPN QAT requires weight modules")
    if len(names) != len(set(names)):
        raise ValueError("CSPN QAT weight bit assignment contains duplicates")
    if not set(bits for name, bits in values) <= {4, 8}:
        raise ValueError("CSPN QAT weight bits must be 4 or 8")
    return tuple(sorted(values))


def _canonical_activation_bits(
        rows: Sequence[Tuple[Owner, int]]) -> Tuple[Tuple[Owner, int], ...]:
    values = tuple(
        ((str(owner[0]), str(owner[1])), int(bits))
        for owner, bits in rows)
    owners = tuple(owner for owner, bits in values)
    if not values:
        raise ValueError("CSPN QAT requires activation owners")
    if len(owners) != len(set(owners)):
        raise ValueError(
            "CSPN QAT activation bit assignment contains duplicates")
    if not set(bits for owner, bits in values) <= {4, 6, 8}:
        raise ValueError("CSPN QAT activation bits must be 4, 6, or 8")
    return tuple(sorted(values))


def _hard_activation_quantizers(instrumentor, boundary_controller):
    quantizers = {}
    for owner, quantizer in instrumentor.quantizers.items():
        canonical = (str(owner[0]), str(owner[1]))
        if canonical in quantizers:
            raise ValueError("CSPN hard activation owners contain duplicates")
        quantizers[canonical] = quantizer
    for owner, quantizer in instrumentor.relu_quantizers.items():
        canonical = (str(owner), "relu_output")
        if canonical in quantizers:
            raise ValueError("CSPN hard activation owners contain duplicates")
        quantizers[canonical] = quantizer
    for owner, quantizer in boundary_controller.active_quantizers.items():
        canonical = ("boundary_controller.%s" % owner, "boundary")
        if canonical in quantizers:
            raise ValueError("CSPN hard activation owners contain duplicates")
        quantizers[canonical] = quantizer
    return quantizers


def cspn_hard_activation_bits(
        instrumentor, boundary_controller) -> Tuple[Tuple[Owner, int], ...]:
    quantizers = _hard_activation_quantizers(
        instrumentor, boundary_controller)
    return _canonical_activation_bits(tuple(
        (owner, quantizer.bits) for owner, quantizer in quantizers.items()))


class CSPNWeightQATController:
    """Install mixed W4/W8 QDQ while retaining FP32 master weights."""

    def __init__(self, model: nn.Module,
                 module_bits: Sequence[Tuple[str, int]]) -> None:
        self.model = model
        self.module_bits = _canonical_weight_bits(module_bits)
        self.module_names = tuple(name for name, bits in self.module_bits)
        self.modules = {}  # type: Dict[str, nn.Module]
        self.installed = False

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN weight QAT is already installed")
        named = dict(self.model.named_modules())
        for name in self.module_names:
            module = named[name]
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                raise TypeError(
                    "unsupported CSPN weight module: %s" % name)
            if parametrize.is_parametrized(module, "weight"):
                raise RuntimeError(
                    "weight is already parametrized: %s" % name)
            channel_dim = 1 if isinstance(
                module, nn.ConvTranspose2d) else 0
            bits = dict(self.module_bits)[name]
            parametrize.register_parametrization(
                module,
                "weight",
                PerOutputChannelWeightFakeQuantizer(bits, channel_dim),
                unsafe=True,
            )
            self.modules[name] = module
        self.installed = True

    def qat_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN weight QAT is not installed")
        return self.model.state_dict()

    def canonical_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN weight QAT is not installed")
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        for name in self.module_names:
            source = "%s.parametrizations.weight.original" % name
            state["%s.weight" % name] = state[source]
            del state[source]
        return state

    def remove(self) -> None:
        if not self.installed:
            raise RuntimeError("CSPN weight QAT is not installed")
        for name in self.module_names:
            parametrize.remove_parametrizations(
                self.modules[name], "weight", leave_parametrized=False)
        self.modules = {}
        self.installed = False


class CSPNActivationQATController:
    """Wrap calibrated ordinary and structural CSPN activation quantizers."""

    def __init__(self, instrumentor, boundary_controller,
                 activation_bits: Sequence[Tuple[Owner, int]]) -> None:
        self.instrumentor = instrumentor
        self.boundary_controller = boundary_controller
        self.activation_bits = _canonical_activation_bits(activation_bits)
        self.original_quantizers = {}
        self.original_relu_quantizers = {}
        self.original_structural = {}
        self.ordinary_owners = set()
        self.structural_owners = set()
        self.installed = False

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN activation QAT is already installed")
        hard_quantizers = _hard_activation_quantizers(
            self.instrumentor, self.boundary_controller)
        declared = dict(self.activation_bits)
        if set(declared) != set(hard_quantizers):
            raise ValueError(
                "CSPN QAT activation owner coverage does not match hard path")
        actual = dict(
            (owner, int(quantizer.bits))
            for owner, quantizer in hard_quantizers.items())
        if actual != declared:
            raise ValueError(
                "CSPN QAT activation bits do not match hard quantizers")
        self.original_quantizers = dict(self.instrumentor.quantizers)
        self.original_relu_quantizers = dict(
            self.instrumentor.relu_quantizers)
        self.original_structural = dict(self.boundary_controller.active_quantizers)
        self.ordinary_owners = set(
            (str(owner[0]), str(owner[1]))
            for owner in self.original_quantizers)
        self.ordinary_owners.update(
            (str(owner), "relu_output")
            for owner in self.original_relu_quantizers)
        if any("gud_up_proj_layer6" in owner[0]
               for owner in self.ordinary_owners):
            raise RuntimeError("guidance cannot be an ordinary QAT owner")
        self.instrumentor.quantizers = {
            key: ActivationSTEQuantizer(value)
            for key, value in self.original_quantizers.items()
        }
        self.instrumentor.relu_quantizers = {
            key: ActivationSTEQuantizer(value)
            for key, value in self.original_relu_quantizers.items()
        }
        self.boundary_controller.active_quantizers = {
            key: ActivationSTEQuantizer(value)
            for key, value in self.original_structural.items()
        }
        self.structural_owners = set(self.original_structural)
        self.installed = True

    def remove(self) -> None:
        if not self.installed:
            raise RuntimeError("CSPN activation QAT is not installed")
        self.instrumentor.quantizers = self.original_quantizers
        self.instrumentor.relu_quantizers = self.original_relu_quantizers
        self.boundary_controller.active_quantizers = self.original_structural
        self.installed = False


class CSPNQATPropagationController:
    """Keep the integer CSPN forward and route gradients through a proxy."""

    def __init__(self, hard_adapter) -> None:
        self.hard_adapter = hard_adapter
        self.module = hard_adapter.module
        self.hard_forward = hard_adapter.patched_forward
        self._proxy_states = []
        self.installed = False

    def _proxy(self, guidance: torch.Tensor, initial: torch.Tensor,
               sparse: torch.Tensor) -> torch.Tensor:
        self._proxy_states = []
        controller = self.hard_adapter.controller
        config = controller.config
        raw = _pad_cspn_channels(guidance)
        if "abs" in self.module.norm_type:
            raw = raw.abs()
        raw_hard = symmetric_qdq(
            raw,
            config.affinity_bits,
            controller.maximum["affinity_raw"],
        )[0]
        raw_qat = hard_forward_proxy(raw_hard, raw)
        denominator = raw_qat.abs().sum(dim=1, keepdim=True)
        affinity_qmax = 2 ** (config.affinity_bits - 1) - 1
        affinity_maximum = controller.maximum["affinity_raw"]
        affinity_scale = affinity_maximum / float(affinity_qmax) \
            if affinity_maximum > 0.0 else 1.0
        neighbor = raw_qat / denominator.clamp_min(affinity_scale)
        center = 1.0 - neighbor.sum(dim=1, keepdim=True)

        state = initial
        mask = torch.zeros_like(initial, dtype=torch.bool) \
            if sparse is None else sparse != 0
        for _ in range(1, int(self.module.prop_time) + 1):
            propagated = _crop_cspn(
                (neighbor * _pad_cspn_state(state)).sum(
                    dim=1, keepdim=True))
            propagated = propagated + _crop_cspn(center) * initial
            state_hard = symmetric_qdq(
                propagated,
                config.state_bits,
                controller.maximum["state"],
            )[0]
            state = hard_forward_proxy(state_hard, propagated)
            state = torch.where(mask, initial, state)
            self._proxy_states.append(state)
        return state

    def _forward(self, guidance: torch.Tensor, initial: torch.Tensor,
                 sparse: torch.Tensor = None) -> torch.Tensor:
        hard = self.hard_forward(guidance, initial, sparse)
        proxy = self._proxy(guidance, initial, sparse)
        return hard_forward_proxy(hard, proxy)

    def hard_result(self, guidance: torch.Tensor, initial: torch.Tensor,
                    sparse: torch.Tensor = None) -> torch.Tensor:
        return self.hard_forward(guidance, initial, sparse)

    def proxy_states(self) -> Tuple[torch.Tensor, ...]:
        if len(self._proxy_states) != int(self.module.prop_time):
            raise RuntimeError("CSPN QAT proxy states are incomplete")
        return tuple(self._proxy_states)

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN propagation QAT is already installed")
        if self.hard_adapter.controller.mode != "quantize":
            raise RuntimeError(
                "hard CSPN propagation must be configured before QAT")
        self.module.forward = self._forward
        self.installed = True

    def remove(self) -> None:
        if not self.installed:
            raise RuntimeError("CSPN propagation QAT is not installed")
        self.module.forward = self.hard_adapter.patched_forward
        self._proxy_states = []
        self.installed = False


@dataclass(frozen=True)
class CSPNQATConfig:
    mode: str
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[Owner, int], ...]
    group_size: int
    propagation: PropagationQuantConfig

    def __post_init__(self) -> None:
        if self.mode not in ("static", "dynamic", "mixed_static"):
            raise ValueError(
                "CSPN QAT mode must be static, dynamic, or mixed_static")
        object.__setattr__(
            self, "weight_bits", _canonical_weight_bits(self.weight_bits))
        object.__setattr__(
            self, "activation_bits",
            _canonical_activation_bits(self.activation_bits))
        if self.group_size != 8:
            raise ValueError("CSPN QAT requires Group-8 activations")
        if not isinstance(self.propagation, PropagationQuantConfig):
            raise TypeError("CSPN QAT propagation config is invalid")


class CSPNQATController:
    """Compose strict weight, activation, and propagation QAT."""

    def __init__(self, model: nn.Module, instrumentor, boundary_controller,
                 hard_propagation, config: CSPNQATConfig) -> None:
        if not isinstance(config, CSPNQATConfig):
            raise TypeError("config must be CSPNQATConfig")
        self.model = model
        self.config = config
        self.weight = CSPNWeightQATController(model, config.weight_bits)
        self.activation = CSPNActivationQATController(
            instrumentor, boundary_controller, config.activation_bits)
        self.propagation = CSPNQATPropagationController(hard_propagation)
        self.installed = False

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN QAT controller is already installed")
        self.weight.install()
        self.activation.install()
        self.propagation.install()
        self.installed = True

    def canonical_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN QAT controller is not installed")
        return self.weight.canonical_state_dict()

    def qat_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN QAT controller is not installed")
        return self.weight.qat_state_dict()

    def assert_finite_gradients(self) -> float:
        if not self.installed:
            raise RuntimeError("CSPN QAT controller is not installed")
        total = None
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            finite = torch.isfinite(parameter.grad)
            if not bool(finite.all().item()):
                raise FloatingPointError(
                    "QAT gradient contains non-finite values: %s "
                    "nan=%d inf=%d" % (
                        name,
                        int(torch.isnan(parameter.grad).sum().item()),
                        int((~finite & ~torch.isnan(
                            parameter.grad)).sum().item())))
            value = parameter.grad.detach().to(
                torch.float64).square().sum()
            total = value if total is None else total + value
        if total is None or float(total.item()) <= 0.0:
            raise RuntimeError("QAT gradient norm is zero")
        return float(torch.sqrt(total).item())

    def manifest(self):
        propagation = self.config.propagation
        return {
            "mode": self.config.mode,
            "weight_bits": self.config.weight_bits,
            "activation_bits": self.config.activation_bits,
            "group_size": self.config.group_size,
            "weight_modules": list(self.weight.module_names),
            "ordinary_owners": [
                str(owner)
                for owner in sorted(
                    self.activation.ordinary_owners, key=str)
            ],
            "structural_owners": sorted(
                self.activation.structural_owners),
            "guidance": "fp32",
            "bias": "fp32",
            "propagation": {
                "affinity_bits": propagation.affinity_bits,
                "confidence_bits": propagation.confidence_bits,
                "offset_bits": propagation.offset_bits,
                "state_bits": propagation.state_bits,
                "coefficient_fraction_bits":
                    propagation.coefficient_fraction_bits,
            },
        }

    def set_runtime_statistics(self, enabled: bool) -> None:
        if not self.installed:
            raise RuntimeError("CSPN QAT controller is not installed")
        self.activation.instrumentor.set_runtime_statistics(enabled)
        self.propagation.hard_adapter.set_runtime_statistics(enabled)

    def remove(self) -> None:
        if not self.installed:
            raise RuntimeError("CSPN QAT controller is not installed")
        self.propagation.remove()
        self.activation.remove()
        self.weight.remove()
        self.installed = False
