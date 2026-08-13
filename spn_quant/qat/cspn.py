"""Strict quantization-aware training controllers for official CSPN."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from spn_quant.qat.quantizers import (
    ActivationSTEQuantizer,
    PerOutputChannelWeightFakeQuantizer,
)


class CSPNWeightQATController:
    """Install W4 parametrizations while retaining FP32 master weights."""

    def __init__(self, model: nn.Module,
                 module_names: Iterable[str]) -> None:
        self.model = model
        self.module_names = tuple(str(name) for name in module_names)
        if not self.module_names:
            raise ValueError("CSPN W4 QAT requires weight modules")
        self.modules = {}  # type: Dict[str, nn.Module]
        self.installed = False

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN W4 QAT is already installed")
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
            parametrize.register_parametrization(
                module,
                "weight",
                PerOutputChannelWeightFakeQuantizer(4, channel_dim),
                unsafe=True,
            )
            self.modules[name] = module
        self.installed = True

    def qat_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN W4 QAT is not installed")
        return self.model.state_dict()

    def canonical_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN W4 QAT is not installed")
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
            raise RuntimeError("CSPN W4 QAT is not installed")
        for name in self.module_names:
            parametrize.remove_parametrizations(
                self.modules[name], "weight", leave_parametrized=False)
        self.modules = {}
        self.installed = False


class CSPNActivationQATController:
    """Wrap calibrated ordinary and structural CSPN activation quantizers."""

    def __init__(self, instrumentor, rotation) -> None:
        self.instrumentor = instrumentor
        self.rotation = rotation
        self.original_quantizers = {}
        self.original_relu_quantizers = {}
        self.original_structural = {}
        self.ordinary_owners = set()
        self.structural_owners = set()
        self.installed = False

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN activation QAT is already installed")
        self.original_quantizers = dict(self.instrumentor.quantizers)
        self.original_relu_quantizers = dict(
            self.instrumentor.relu_quantizers)
        self.original_structural = dict(self.rotation.active_quantizers)
        self.ordinary_owners = set(self.original_quantizers) | set(
            self.original_relu_quantizers)
        if any("gud_up_proj_layer6" in str(owner)
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
        self.rotation.active_quantizers = {
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
        self.rotation.active_quantizers = self.original_structural
        self.installed = False
