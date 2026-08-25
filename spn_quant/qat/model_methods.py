"""Contract-driven LSQ++, HAWQ, and mixed task-aware QAT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from spn_quant.activation_boundaries import SignedActivationQuantizer
from spn_quant.model_contracts import QuantizationModelContract
from spn_quant.propagation.controller import PropagationQuantConfig
from spn_quant.qdrop_targets import QDropTargetPlan
from spn_quant.qat.hawq import HAWQActivationQuantizer
from spn_quant.qat.lsqplus import (
    LSQPlusActivationQuantizer,
    LSQPlusWeightParametrization,
)
from spn_quant.qat.quantizers import PerOutputChannelWeightFakeQuantizer


METHODS = ("lsqplus", "hawq", "mixed_task_aware")
PROTECTED_LEARNED_SCALE_ROLES = frozenset((
    "guidance",
    "guidance_logits",
    "confidence",
    "confidence_logits",
    "offset",
    "offset_logits",
    "affinity",
    "affinity_logits",
    "normalization",
    "normalization_denominator",
    "anchor",
    "sparse_anchor",
    "sparse_anchor_mask",
    "propagation",
    "propagation_state",
))
PROTECTED_LEARNED_SCALE_NAMES = (
    "guidance",
    "confidence",
    "offset",
    "affinity",
    "normalization",
    "anchor",
    "propagation",
    "initial_depth",
    "pred_init",
)

Owner = Tuple[str, str]


def _canonical_weight_bits(rows, allowed_bits):
    values = tuple((str(name), int(bits)) for name, bits in rows)
    names = tuple(name for name, bits in values)
    if not values:
        raise ValueError("method QAT requires weight modules")
    if len(names) != len(set(names)):
        raise ValueError("method weight assignment contains duplicates")
    if not set(bits for name, bits in values) <= set(allowed_bits):
        raise ValueError("method weight assignment has unsupported bits")
    return tuple(sorted(values))


def _canonical_activation_bits(rows, allowed_bits):
    values = tuple(
        ((str(owner[0]), str(owner[1])), int(bits))
        for owner, bits in rows)
    owners = tuple(owner for owner, bits in values)
    if not values:
        raise ValueError("method QAT requires activation owners")
    if len(owners) != len(set(owners)):
        raise ValueError("method activation assignment contains duplicates")
    if not set(bits for owner, bits in values) <= set(allowed_bits):
        raise ValueError("method activation assignment has unsupported bits")
    return tuple(sorted(values))


@dataclass(frozen=True)
class ModelMethodQATConfig:
    method: str
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[Owner, int], ...]
    propagation: PropagationQuantConfig
    hawq_range_momentum: float

    def __post_init__(self) -> None:
        method = str(self.method)
        if method not in METHODS:
            raise ValueError(
                "method must be lsqplus, hawq, or mixed_task_aware")
        if method == "lsqplus":
            weight_allowed = activation_allowed = (4, 6)
            requested = tuple(
                int(bits) for name, bits in self.weight_bits) + tuple(
                    int(bits) for owner, bits in self.activation_bits)
            if not set(requested) <= {4, 6}:
                raise ValueError("LSQ+ supports only W4A4 and W6A6")
        elif method == "hawq":
            weight_allowed = activation_allowed = (4, 6, 8)
        else:
            weight_allowed = (4, 8)
            activation_allowed = (4, 6, 8)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self, "weight_bits",
            _canonical_weight_bits(self.weight_bits, weight_allowed))
        object.__setattr__(
            self, "activation_bits",
            _canonical_activation_bits(
                self.activation_bits, activation_allowed))
        if not isinstance(self.propagation, PropagationQuantConfig):
            raise TypeError("method propagation config is invalid")
        momentum = float(self.hawq_range_momentum)
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("HAWQ range momentum must lie in [0, 1)")
        object.__setattr__(self, "hawq_range_momentum", momentum)


def _activation_unsigned(quantizer) -> bool:
    if isinstance(quantizer, SignedActivationQuantizer):
        return False
    return bool(quantizer.unsigned)


def _protected_activation_owner(owner: Owner) -> bool:
    site, role = owner
    lowered = str(site).lower()
    return str(role) in PROTECTED_LEARNED_SCALE_ROLES or any(
        name in lowered for name in PROTECTED_LEARNED_SCALE_NAMES)


class _MethodQATControllerBase(nn.Module):
    """Shared quantizer, master-weight, and checkpoint state machinery."""

    def __init__(self, model: nn.Module, config: ModelMethodQATConfig,
                 unsigned_by_owner: Mapping[Owner, bool]) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("method QAT model must be an nn.Module")
        if not isinstance(config, ModelMethodQATConfig):
            raise TypeError("config must be ModelMethodQATConfig")
        expected_owners = tuple(
            owner for owner, bits in config.activation_bits)
        if set(unsigned_by_owner) != set(expected_owners) or \
                len(unsigned_by_owner) != len(expected_owners):
            raise ValueError(
                "method activation signedness coverage mismatch")
        self.model = model
        self.config = config
        self.activation_owners = expected_owners
        quantizers = []
        for owner, bits in config.activation_bits:
            unsigned = bool(unsigned_by_owner[owner])
            if config.method == "lsqplus":
                quantizer = LSQPlusActivationQuantizer(bits, unsigned)
            else:
                quantizer = HAWQActivationQuantizer(
                    bits, unsigned, config.hawq_range_momentum)
            quantizer.site = owner[0]
            quantizers.append(quantizer)
        self.activation_modules = nn.ModuleList(quantizers)
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
            raise RuntimeError("method activations are already initialized")
        values = tuple(
            ((str(owner[0]), str(owner[1])), tensor)
            for owner, tensor in rows)
        owners = tuple(owner for owner, tensor in values)
        if len(owners) != len(set(owners)) or \
                set(owners) != set(self.activation_owners):
            raise ValueError(
                "method activation initialization coverage mismatch")
        tensors = dict(values)
        for owner in self.activation_owners:
            quantizer = self.activation_by_owner[owner]
            tensor = tensors[owner]
            if self.config.method == "lsqplus":
                quantizer.initialize(tensor)
            else:
                quantizer.initialize_range(tensor)
                if self.config.method == "mixed_task_aware":
                    quantizer.freeze_range()
        self.activations_initialized = True

    def _install_weights(self) -> None:
        named = dict(self.model.named_modules())
        for name, bits in self.config.weight_bits:
            module = named[name]
            if not isinstance(
                    module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                raise TypeError("unsupported method weight module: %s" % name)
            if parametrize.is_parametrized(module, "weight"):
                raise RuntimeError("weight is already parametrized: %s" % name)
            channel_dim = 1 if isinstance(
                module, nn.ConvTranspose2d) else 0
            if self.config.method == "lsqplus":
                quantizer = LSQPlusWeightParametrization(
                    bits, channel_dim, module.weight.detach())
            else:
                quantizer = PerOutputChannelWeightFakeQuantizer(
                    bits, channel_dim)
            parametrize.register_parametrization(
                module, "weight", quantizer, unsafe=True)
            self.weight_modules[name] = module
            self.weight_quantizers[name] = quantizer

    def _remove_weights(self) -> None:
        for name, bits in self.config.weight_bits:
            del bits
            parametrize.remove_parametrizations(
                self.weight_modules[name], "weight", leave_parametrized=False)
        self.weight_quantizers = {}
        self.weight_modules = {}

    def activation_quantizers(self) -> Tuple[nn.Module, ...]:
        return tuple(self.activation_modules)

    def activation_owner_manifest(self) -> Tuple[Owner, ...]:
        return tuple(self.activation_owners)

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
        else:
            for owner in self.activation_owners:
                quantizer = self.activation_by_owner[owner]
                prefix = "activation.%s" % (owner,)
                state["%s.minimum" % prefix] = quantizer.minimum
                state["%s.maximum" % prefix] = quantizer.maximum
                state["%s.range_initialized" % prefix] = \
                    quantizer.range_initialized
                state["%s.running_range_state" % prefix] = \
                    quantizer.running_range_state
        return state

    def method_state_dict(self):
        if not self.installed:
            raise RuntimeError("method QAT is not installed")
        return dict(
            (key, value.detach().cpu().clone())
            for key, value in self._method_state_tensors().items())

    def load_method_state_dict(self, state) -> None:
        if not self.installed:
            raise RuntimeError("method QAT is not installed")
        targets = self._method_state_tensors()
        if set(state) != set(targets):
            raise ValueError("method state contract mismatch")
        with torch.no_grad():
            for key in targets:
                targets[key].copy_(state[key].to(
                    device=targets[key].device, dtype=targets[key].dtype))

    def _model_state_with_weights(self, hard: bool):
        state = dict(
            (key, value.detach().cpu().clone())
            for key, value in self.model.state_dict().items())
        for name, bits in self.config.weight_bits:
            del bits
            prefix = "%s.parametrizations.weight." % name
            source = "%soriginal" % prefix
            weight = self.weight_modules[name].weight.detach() if hard else \
                state[source]
            for key in tuple(state):
                if key.startswith(prefix):
                    del state[key]
            state["%s.weight" % name] = weight.detach().cpu().clone()
        return state

    def canonical_model_state_dict(self):
        if not self.installed:
            raise RuntimeError("method QAT is not installed")
        return self._model_state_with_weights(False)

    def hard_model_state_dict(self):
        if not self.installed:
            raise RuntimeError("method QAT is not installed")
        return self._model_state_with_weights(True)

    def load_canonical_model_state_dict(self, state) -> None:
        if not self.installed:
            raise RuntimeError("method QAT is not installed")
        expected = self.canonical_model_state_dict()
        if set(state) != set(expected):
            raise ValueError("canonical model state contract mismatch")
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

    def hard_deployment_manifest(self):
        hard_state = self.hard_model_state_dict()
        if any("parametrizations" in name for name in hard_state):
            raise RuntimeError("hard deployment contains soft weights")
        if any(not bool(torch.isfinite(value).all().item())
               for value in hard_state.values()
               if value.is_floating_point()):
            raise FloatingPointError(
                "hard deployment model state contains non-finite values")
        activation_qparams = []
        for owner in self.activation_owners:
            quantizer = self.activation_by_owner[owner]
            if self.config.method == "lsqplus":
                quantizer._validate_parameters()
                if not bool(quantizer.initialized.item()):
                    raise RuntimeError(
                        "hard deployment activation is uninitialized")
                qparams = {
                    "owner": owner,
                    "bits": int(quantizer.bits),
                    "unsigned": int(quantizer.unsigned),
                    "step": float(quantizer.step.detach().abs().item()),
                    "offset": float(quantizer.offset.detach().item()),
                    "qmin": int(quantizer.qmin),
                    "qmax": int(quantizer.qmax),
                }
            else:
                scale, zero_point = quantizer._parameters_for(
                    quantizer.minimum)
                qparams = {
                    "owner": owner,
                    "bits": int(quantizer.bits),
                    "unsigned": int(quantizer.unsigned),
                    "scale": float(scale.detach().item()),
                    "zero_point": float(zero_point.detach().item()),
                    "qmin": int(quantizer.qmin),
                    "qmax": int(quantizer.qmax),
                }
            activation_qparams.append(qparams)
        protected_excluded = not any(
            _protected_activation_owner(owner)
            for owner in self.activation_owners)
        if not protected_excluded:
            raise RuntimeError(
                "hard deployment contains protected generic scales")
        return {
            "validated": 1,
            "method": self.config.method,
            "materialized_weight_count": len(self.config.weight_bits),
            "activation_owner_count": len(self.activation_owners),
            "canonical_master_weights": 1,
            "weight_bits": self.config.weight_bits,
            "activation_bits": self.config.activation_bits,
            "activation_qparams": tuple(activation_qparams),
            "protected_scale_roles_excluded": 1,
        }

    def assert_finite_gradients(self) -> float:
        if not self.installed:
            raise RuntimeError("method QAT is not installed")
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


class ModelMethodQATController(_MethodQATControllerBase):
    """Attach method quantizers only to explicit model-contract owners."""

    def __init__(self, model: nn.Module,
                 contract: QuantizationModelContract,
                 target_plan: QDropTargetPlan,
                 config: ModelMethodQATConfig,
                 joint_adapter=None,
                 propagation_adapter=None) -> None:
        if not isinstance(config, ModelMethodQATConfig):
            raise TypeError("config must be ModelMethodQATConfig")
        if not isinstance(contract, QuantizationModelContract):
            raise TypeError(
                "model QAT requires QuantizationModelContract")
        if not isinstance(target_plan, QDropTargetPlan):
            raise TypeError("model QAT requires QDropTargetPlan")
        if target_plan.model != contract.model_name:
            raise ValueError("model QAT target plan model differs")
        expected_weights = tuple(contract.weight_modules)
        declared_weights = tuple(name for name, bits in config.weight_bits)
        protected_weights = set(declared_weights).intersection(
            contract.protected_modules)
        if protected_weights:
            raise ValueError(
                "method weight assignment contains protected modules: %s" %
                sorted(protected_weights))
        if set(declared_weights) != set(expected_weights) or \
                len(declared_weights) != len(expected_weights):
            raise ValueError(
                "model method weight assignment coverage mismatch")
        expected_owners = tuple(
            owner for block in contract.blocks
            for owner in block.activation_owners)
        declared_owners = tuple(
            owner for owner, bits in config.activation_bits)
        protected_roles = set(contract.protected_roles) | \
            set(PROTECTED_LEARNED_SCALE_ROLES)
        invalid_roles = sorted(
            owner for owner in declared_owners
            if owner[1] in protected_roles or
            _protected_activation_owner(owner))
        if invalid_roles:
            raise ValueError(
                "method activation assignment contains protected roles: %s" %
                invalid_roles)
        if set(declared_owners) != set(expected_owners) or \
                len(declared_owners) != len(expected_owners):
            raise ValueError(
                "model method activation assignment coverage mismatch")
        sites = dict(
            ((site.site, site.role), site)
            for site in target_plan.activation_sites)
        if set(sites) != set(expected_owners):
            raise ValueError(
                "model method activation sites differ from contract")
        modules = dict(model.named_modules())
        if set(expected_weights) - set(modules):
            raise KeyError("model method weight modules are missing")
        unsigned = dict(
            (owner, not sites[owner].signed) for owner in expected_owners)
        super().__init__(model, config, unsigned)
        self.contract = contract
        self.target_plan = target_plan
        self.sites_by_owner = sites
        self.joint_adapter = joint_adapter
        self.propagation_adapter = propagation_adapter
        self._activation_handles = []
        self._joint_bound = False

    def activation_owner_manifest(self) -> Tuple[Owner, ...]:
        return tuple(
            owner for block in self.contract.blocks
            for owner in block.activation_owners)

    def block_manifest(self):
        weight_bits = dict(self.config.weight_bits)
        activation_bits = dict(self.config.activation_bits)
        return tuple({
            "block": block.name,
            "weight_bits": tuple(
                (name, weight_bits[name]) for name in block.weight_modules),
            "activation_bits": tuple(
                (owner, activation_bits[owner])
                for owner in block.activation_owners),
        } for block in self.contract.blocks)

    @staticmethod
    def _module_boundary(site):
        parts = site.site.split("::")
        if len(parts) != 3 or parts[0] != "activation" or \
                parts[2] not in ("input", "output"):
            raise ValueError(
                "model QAT module activation site is invalid: %s" %
                site.site)
        expected_kind = "module_%s" % parts[2]
        if site.owner_kind != expected_kind:
            raise ValueError(
                "model QAT module activation owner kind differs")
        return parts[1], parts[2]

    def _install_module_site(self, owner, site, module) -> None:
        quantizer = self.activation_by_owner[owner]
        module_name, boundary = self._module_boundary(site)
        del module_name
        if boundary == "input":
            def pre_hook(current, inputs, target=quantizer):
                del current
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError(
                        "model QAT module input must start with a tensor")
                values = list(inputs)
                values[0] = target(values[0])
                return tuple(values)
            handle = module.register_forward_pre_hook(pre_hook)
        else:
            def post_hook(current, inputs, output, target=quantizer):
                del current, inputs
                if not torch.is_tensor(output):
                    raise TypeError("model QAT module output must be a tensor")
                return target(output)
            handle = module.register_forward_hook(post_hook)
        self._activation_handles.append(handle)

    def _install_activations(self) -> None:
        modules = dict(self.model.named_modules())
        joint_sites = []
        joint_quantizers = {}
        for owner in self.activation_owner_manifest():
            site = self.sites_by_owner[owner]
            if site.owner_kind in ("module_input", "module_output"):
                module_name, boundary = self._module_boundary(site)
                del boundary
                if module_name not in modules:
                    raise KeyError(
                        "model QAT activation module is missing: %s" %
                        module_name)
                self._install_module_site(owner, site, modules[module_name])
            elif site.owner_kind in ("attention_qkv", "concat_input"):
                joint_sites.append(site)
                joint_quantizers[site.site] = self.activation_by_owner[owner]
            else:
                raise ValueError(
                    "unsupported model QAT activation owner kind: %s" %
                    site.owner_kind)
        if joint_sites:
            if self.joint_adapter is None:
                raise RuntimeError(
                    "joint activation owners require CompletionFormer adapter")
            self.joint_adapter.bind_qdrop_sites(
                tuple(joint_sites), joint_quantizers)
            self._joint_bound = True
        elif self.joint_adapter is not None and (
                self.contract.attention_edges or self.contract.concat_edges):
            raise ValueError("joint model contract has no bound QAT sites")

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("model method QAT is already installed")
        if not self.activations_initialized:
            raise RuntimeError("model method activations are not initialized")
        if self.propagation_adapter is not None and \
                self.propagation_adapter.controller.config != \
                self.config.propagation:
            raise ValueError(
                "hard propagation config does not match model QAT")
        self._install_weights()
        self._install_activations()
        self.installed = True

    def manifest(self):
        propagation = self.config.propagation
        return {
            "model": self.contract.model_name,
            "method": self.config.method,
            "weight_bits": self.config.weight_bits,
            "activation_bits": self.config.activation_bits,
            "activation_owners": self.activation_owner_manifest(),
            "blocks": self.block_manifest(),
            "protected_roles": self.contract.protected_roles,
            "protected_modules": self.contract.protected_modules,
            "propagation": {
                "affinity_bits": propagation.affinity_bits,
                "confidence_bits": propagation.confidence_bits,
                "offset_bits": propagation.offset_bits,
                "state_bits": propagation.state_bits,
                "coefficient_fraction_bits":
                    propagation.coefficient_fraction_bits,
            },
        }

    def remove(self) -> None:
        if not self.installed:
            raise RuntimeError("model method QAT is not installed")
        if self._joint_bound:
            self.joint_adapter.unbind_qdrop_sites()
            self._joint_bound = False
        for handle in self._activation_handles:
            handle.remove()
        self._activation_handles = []
        self._remove_weights()
        self.installed = False


__all__ = (
    "METHODS",
    "PROTECTED_LEARNED_SCALE_ROLES",
    "ModelMethodQATConfig",
    "ModelMethodQATController",
)
