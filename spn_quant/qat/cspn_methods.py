"""LSQ+ and HAWQ controllers for the official CSPN quantization graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from spn_quant.activation_boundaries import SignedActivationQuantizer
from spn_quant.propagation.controller import PropagationQuantConfig
from spn_quant.qat.cspn import (
    CSPNQATPropagationController,
    Owner,
    _hard_activation_quantizers,
)
from spn_quant.qat.hawq import HAWQActivationQuantizer
from spn_quant.qat.lsqplus import (
    LSQPlusActivationQuantizer,
    LSQPlusWeightParametrization,
)
from spn_quant.qat.quantizers import PerOutputChannelWeightFakeQuantizer


METHODS = ("lsqplus", "hawq")


def _canonical_weight_bits(rows, allowed_bits):
    values = tuple((str(name), int(bits)) for name, bits in rows)
    names = tuple(name for name, bits in values)
    if not values:
        raise ValueError("CSPN method QAT requires weight modules")
    if len(names) != len(set(names)):
        raise ValueError("CSPN method weight assignment contains duplicates")
    if not set(bits for name, bits in values) <= set(allowed_bits):
        raise ValueError("CSPN method weight assignment has unsupported bits")
    return tuple(sorted(values))


def _canonical_activation_bits(rows, allowed_bits):
    values = tuple(
        ((str(owner[0]), str(owner[1])), int(bits))
        for owner, bits in rows)
    owners = tuple(owner for owner, bits in values)
    if not values:
        raise ValueError("CSPN method QAT requires activation owners")
    if len(owners) != len(set(owners)):
        raise ValueError(
            "CSPN method activation assignment contains duplicates")
    if not set(bits for owner, bits in values) <= set(allowed_bits):
        raise ValueError(
            "CSPN method activation assignment has unsupported bits")
    return tuple(sorted(values))


@dataclass(frozen=True)
class CSPNMethodQATConfig:
    method: str
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[Owner, int], ...]
    propagation: PropagationQuantConfig
    hawq_range_momentum: float

    def __post_init__(self) -> None:
        method = str(self.method)
        if method not in METHODS:
            raise ValueError("CSPN method must be lsqplus or hawq")
        allowed_bits = (4, 6, 8)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self, "weight_bits",
            _canonical_weight_bits(self.weight_bits, allowed_bits))
        object.__setattr__(
            self, "activation_bits",
            _canonical_activation_bits(self.activation_bits, allowed_bits))
        if method == "lsqplus" and (
                any(bits == 8 for name, bits in self.weight_bits) or
                any(bits == 8 for owner, bits in self.activation_bits)):
            raise ValueError("LSQ+ supports only W4A4 and W6A6")
        if not isinstance(self.propagation, PropagationQuantConfig):
            raise TypeError("CSPN method propagation config is invalid")
        momentum = float(self.hawq_range_momentum)
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("HAWQ range momentum must lie in [0, 1)")
        object.__setattr__(self, "hawq_range_momentum", momentum)


def _activation_unsigned(quantizer) -> bool:
    if isinstance(quantizer, SignedActivationQuantizer):
        return False
    return bool(quantizer.unsigned)


class CSPNMethodQATController(nn.Module):
    """Compose learned CNN QDQ with the propagation-aware CSPN path."""

    def __init__(self, model: nn.Module, instrumentor, boundary_controller,
                 hard_propagation,
                 config: CSPNMethodQATConfig) -> None:
        super().__init__()
        if not isinstance(config, CSPNMethodQATConfig):
            raise TypeError("config must be CSPNMethodQATConfig")
        if any("gud_up_proj_layer6" in name
               for name, bits in config.weight_bits):
            raise ValueError("guidance weights must remain FP32")
        if any("gud_up_proj_layer6" in owner[0]
               for owner, bits in config.activation_bits):
            raise ValueError("guidance activations must remain FP32")
        self.model = model
        self.instrumentor = instrumentor
        self.boundary_controller = boundary_controller
        self.hard_propagation = hard_propagation
        self.config = config
        self.propagation = CSPNQATPropagationController(hard_propagation)
        self.original_quantizers = dict(instrumentor.quantizers)
        self.original_relu_quantizers = dict(instrumentor.relu_quantizers)
        self.original_structural = dict(
            boundary_controller.active_quantizers)
        hard_quantizers = _hard_activation_quantizers(
            instrumentor, boundary_controller)
        declared = dict(config.activation_bits)
        if set(declared) != set(hard_quantizers):
            raise ValueError(
                "CSPN method activation owner coverage does not match hard path")
        self.activation_owners = tuple(
            owner for owner, bits in config.activation_bits)
        method_quantizers = []
        for owner, bits in config.activation_bits:
            unsigned = _activation_unsigned(hard_quantizers[owner])
            if config.method == "lsqplus":
                quantizer = LSQPlusActivationQuantizer(bits, unsigned)
            elif config.method == "hawq":
                quantizer = HAWQActivationQuantizer(
                    bits, unsigned, config.hawq_range_momentum)
            else:
                raise ValueError("unsupported CSPN QAT method")
            method_quantizers.append(quantizer)
        self.activation_modules = nn.ModuleList(method_quantizers)
        device = next(model.parameters()).device
        self.activation_modules.to(device)
        self.activation_by_owner = dict(zip(
            self.activation_owners, self.activation_modules))
        self.weight_quantizers = {}  # type: Dict[str, nn.Module]
        self.weight_modules = {}  # type: Dict[str, nn.Module]
        self.activations_initialized = False
        self.installed = False

    def initialize_activations(
            self, rows: Sequence[Tuple[Owner, torch.Tensor]]) -> None:
        if self.activations_initialized:
            raise RuntimeError("CSPN method activations are already initialized")
        values = tuple(
            ((str(owner[0]), str(owner[1])), tensor)
            for owner, tensor in rows)
        owners = tuple(owner for owner, tensor in values)
        if len(owners) != len(set(owners)) or \
                set(owners) != set(self.activation_owners):
            raise ValueError(
                "CSPN method activation initialization coverage mismatch")
        tensors = dict(values)
        for owner in self.activation_owners:
            quantizer = self.activation_by_owner[owner]
            tensor = tensors[owner]
            if self.config.method == "lsqplus":
                quantizer.initialize(tensor)
            elif self.config.method == "hawq":
                quantizer.initialize_range(tensor)
            else:
                raise ValueError("unsupported CSPN QAT method")
        self.activations_initialized = True

    def _install_weights(self) -> None:
        named = dict(self.model.named_modules())
        for name, bits in self.config.weight_bits:
            module = named[name]
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                raise TypeError("unsupported CSPN weight module: %s" % name)
            if parametrize.is_parametrized(module, "weight"):
                raise RuntimeError("weight is already parametrized: %s" % name)
            channel_dim = 1 if isinstance(
                module, nn.ConvTranspose2d) else 0
            if self.config.method == "lsqplus":
                quantizer = LSQPlusWeightParametrization(
                    bits, channel_dim, module.weight.detach())
            elif self.config.method == "hawq":
                quantizer = PerOutputChannelWeightFakeQuantizer(
                    bits, channel_dim)
            else:
                raise ValueError("unsupported CSPN QAT method")
            parametrize.register_parametrization(
                module, "weight", quantizer, unsafe=True)
            self.weight_modules[name] = module
            self.weight_quantizers[name] = quantizer

    def _install_activations(self) -> None:
        self.instrumentor.quantizers = dict(
            (key, self.activation_by_owner[(str(key[0]), str(key[1]))])
            for key in self.original_quantizers)
        self.instrumentor.relu_quantizers = dict(
            (key, self.activation_by_owner[(str(key), "relu_output")])
            for key in self.original_relu_quantizers)
        self.boundary_controller.active_quantizers = dict(
            (key, self.activation_by_owner[
                ("boundary_controller.%s" % key, "boundary")])
            for key in self.original_structural)

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN method QAT is already installed")
        if not self.activations_initialized:
            raise RuntimeError("CSPN method activations are not initialized")
        if self.hard_propagation.controller.config != self.config.propagation:
            raise ValueError("CSPN hard propagation config does not match QAT")
        self._install_weights()
        self._install_activations()
        self.propagation.install()
        self.installed = True

    def activation_quantizers(self) -> Tuple[nn.Module, ...]:
        return tuple(self.activation_modules)

    def freeze_activation_ranges(self) -> None:
        if self.config.method != "hawq":
            raise RuntimeError("only HAWQ activation ranges can be frozen")
        for quantizer in self.activation_modules:
            quantizer.freeze_range()

    def _method_state_tensors(self):
        state = {}
        if self.config.method == "lsqplus":
            for name, quantizer in self.weight_quantizers.items():
                state["weight.%s.step" % name] = quantizer.step
            for owner in self.activation_owners:
                quantizer = self.activation_by_owner[owner]
                prefix = "activation.%s" % (owner,)
                state["%s.step" % prefix] = quantizer.step
                state["%s.offset" % prefix] = quantizer.offset
                state["%s.initialized" % prefix] = quantizer.initialized
        elif self.config.method == "hawq":
            for owner in self.activation_owners:
                quantizer = self.activation_by_owner[owner]
                prefix = "activation.%s" % (owner,)
                state["%s.minimum" % prefix] = quantizer.minimum
                state["%s.maximum" % prefix] = quantizer.maximum
                state["%s.range_initialized" % prefix] = \
                    quantizer.range_initialized
                state["%s.running_range_state" % prefix] = \
                    quantizer.running_range_state
        else:
            raise ValueError("unsupported CSPN QAT method")
        return state

    def method_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        return dict(
            (key, value.detach().cpu().clone())
            for key, value in self._method_state_tensors().items())

    def load_method_state_dict(self, state) -> None:
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        targets = self._method_state_tensors()
        if set(state) != set(targets):
            raise ValueError("CSPN method state contract mismatch")
        with torch.no_grad():
            for key in targets:
                targets[key].copy_(state[key].to(
                    device=targets[key].device, dtype=targets[key].dtype))

    def canonical_model_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        state = dict(
            (key, value.detach().cpu().clone())
            for key, value in self.model.state_dict().items())
        for name, bits in self.config.weight_bits:
            prefix = "%s.parametrizations.weight." % name
            source = "%soriginal" % prefix
            master = state[source]
            for key in tuple(state):
                if key.startswith(prefix):
                    del state[key]
            state["%s.weight" % name] = master
        return state

    def load_canonical_model_state_dict(self, state) -> None:
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        expected = self.canonical_model_state_dict()
        if set(state) != set(expected):
            raise ValueError("canonical CSPN model state contract mismatch")
        current = self.model.state_dict()
        parametrized_weights = dict(
            ("%s.weight" % name,
             self.weight_modules[name].parametrizations.weight.original)
            for name, bits in self.config.weight_bits)
        with torch.no_grad():
            for key in state:
                target = parametrized_weights[key] \
                    if key in parametrized_weights else current[key]
                target.copy_(state[key].to(
                    device=target.device, dtype=target.dtype))

    def assert_finite_gradients(self) -> float:
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        total = None
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(
                    "QAT gradient contains non-finite values: %s" % name)
            value = parameter.grad.detach().to(torch.float64).square().sum()
            total = value if total is None else total + value
        if total is None or float(total.item()) <= 0.0:
            raise RuntimeError("QAT gradient norm is zero")
        return float(torch.sqrt(total).item())

    def manifest(self):
        propagation = self.config.propagation
        return {
            "method": self.config.method,
            "weight_bits": self.config.weight_bits,
            "activation_bits": self.config.activation_bits,
            "guidance": "fp32",
            "bias": "fp32",
            "propagation": {
                "affinity_bits": propagation.affinity_bits,
                "confidence_bits": propagation.confidence_bits,
                "offset_bits": propagation.offset_bits,
                "state_bits": propagation.state_bits,
                "coefficient_fraction_bits":
                    propagation.coefficient_fraction_bits,
                "proxy_states": int(self.propagation.module.prop_time),
            },
        }

    def set_runtime_statistics(self, enabled: bool) -> None:
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        self.instrumentor.set_runtime_statistics(enabled)
        self.hard_propagation.set_runtime_statistics(enabled)

    def remove(self) -> None:
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        self.propagation.remove()
        self.instrumentor.quantizers = self.original_quantizers
        self.instrumentor.relu_quantizers = self.original_relu_quantizers
        self.boundary_controller.active_quantizers = self.original_structural
        for name, bits in self.config.weight_bits:
            parametrize.remove_parametrizations(
                self.weight_modules[name], "weight", leave_parametrized=False)
        self.weight_quantizers = {}
        self.weight_modules = {}
        self.installed = False
