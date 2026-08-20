"""CSPN semantic adapter and structural merge instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adapters.base import (
    ModelSemanticAdapter,
    ModuleRoleRule,
    SignalRule,
)
from spn_quant.merge import MergeSiteController
from spn_quant.propagation.adapters import cspn_float_affinity
from spn_quant.runtime import EdgeQDQRuntime


@dataclass(frozen=True)
class ActivationConsumer:
    module: str
    channel_start: int
    channel_count: Optional[int]


@dataclass(frozen=True)
class ActivationBoundary:
    name: str
    module: str
    argument_index: int
    consumers: Tuple[ActivationConsumer, ...]


class CSPNStructuralMergeAdapter(object):
    """Expose CSPN ResNet/decoder Add and Cat sites without source edits."""

    _SUPPORTED = frozenset((
        "BasicBlock", "Bottleneck", "UpProj_Block",
        "Gudi_UpProj_Block", "Gudi_UpProj_Block_Cat",
    ))

    def __init__(self, model: nn.Module, policy: str,
                 group_size: Optional[int], runtime: EdgeQDQRuntime,
                 site_policies: Optional[Mapping[str, str]] = None) -> None:
        self.model = model
        self.policy = policy
        self.group_size = group_size
        self.runtime = runtime
        self.site_policies = {} if site_policies is None else \
            dict(site_policies)
        self.mode = "bypass"
        self.controllers = {}  # type: Dict[str, MergeSiteController]
        self.originals = {}  # type: Dict[str, Tuple[nn.Module, Any]]
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            if class_name not in self._SUPPORTED:
                continue
            original = module.forward
            self.originals[name] = (module, original)
            module.forward = types.MethodType(
                self._wrapper(name, class_name, original), module)

    def _controller(self, name: str, operation: str) -> MergeSiteController:
        key = "%s::%s#0" % (name, operation)
        if key not in self.controllers:
            policy = self.site_policies[key] \
                if key in self.site_policies else self.policy
            self.controllers[key] = MergeSiteController(
                key, operation=operation, policy=policy,
                axis=1, group_size=self.group_size, runtime=self.runtime)
        return self.controllers[key]

    def _merge(self, name: str, operation: str,
               branches: Sequence[torch.Tensor]) -> torch.Tensor:
        controller = self._controller(name, operation)
        if operation == "concat":
            plain = torch.cat(tuple(branches), dim=1)
        else:
            plain = branches[0]
            for branch in branches[1:]:
                plain = plain + branch
        if self.mode == "observe":
            controller.observe(branches, merged=plain)
            return plain
        if self.mode == "quantize":
            if self.site_policies and controller.name not in self.site_policies:
                return plain
            return controller.merge(branches)
        return plain

    def _basic(self, module: nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = module.relu(module.bn1(module.conv1(x)))
        out = module.bn2(module.conv2(out))
        if module.downsample is not None:
            residual = module.downsample(x)
        out = self._merge(name, "add", (out, residual))
        return module.relu(out)

    def _bottleneck(self, module: nn.Module, name: str,
                    x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = module.relu(module.bn1(module.conv1(x)))
        out = module.relu(module.bn2(module.conv2(out)))
        out = module.bn3(module.conv3(out))
        if module.downsample is not None:
            residual = module.downsample(x)
        out = self._merge(name, "add", (out, residual))
        return module.relu(out)

    def _up_proj(self, module: nn.Module, name: str,
                 x: torch.Tensor) -> torch.Tensor:
        x = module._up_pooling(x, 2)
        out = module.relu(module.bn1(module.conv1(x)))
        out = module.bn2(module.conv2(out))
        shortcut = module.sc_bn1(module.sc_conv1(x))
        out = self._merge(name, "add", (out, shortcut))
        return module.relu(out)

    def _up_proj_cat(self, module: nn.Module, name: str,
                     x: torch.Tensor, side_input: torch.Tensor) -> torch.Tensor:
        x = module._up_pooling(x, 2)
        out = module.relu(module.bn1(module.conv1(x)))
        out = self._merge(name, "concat", (out, side_input))
        out = module.relu(module.bn1_1(module.conv1_1(out)))
        out = module.bn2(module.conv2(out))
        shortcut = module.sc_bn1(module.sc_conv1(x))
        out = self._merge(name, "add", (out, shortcut))
        return module.relu(out)

    def _wrapper(self, name: str, class_name: str, original: Any):
        def wrapper(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
            if self.mode == "bypass":
                return original(*args, **kwargs)
            if kwargs:
                raise ValueError(
                    "CSPN structural quantization requires positional calls")
            if class_name == "BasicBlock":
                return self._basic(module, name, args[0])
            if class_name == "Bottleneck":
                return self._bottleneck(module, name, args[0])
            if class_name in ("UpProj_Block", "Gudi_UpProj_Block"):
                return self._up_proj(module, name, args[0])
            if class_name == "Gudi_UpProj_Block_Cat":
                return self._up_proj_cat(module, name, args[0], args[1])
            return original(*args, **kwargs)
        return wrapper

    def observe(self) -> None:
        self.mode = "observe"

    def freeze(self, bits: int) -> None:
        if not self.controllers:
            raise RuntimeError("CSPN structural merge sites were not observed")
        unknown = set(self.site_policies) - set(self.controllers)
        if unknown:
            raise ValueError("unknown CSPN merge site policies: %s" %
                             sorted(unknown))
        for controller in self.controllers.values():
            controller.freeze(bits)
        self.mode = "bypass"

    def quantize(self) -> None:
        if not self.controllers:
            raise RuntimeError("CSPN structural merge calibration is missing")
        self.mode = "quantize"

    def disable(self) -> None:
        self.mode = "bypass"

    def manifest(self) -> List[Dict[str, Any]]:
        return [controller.qparams()
                for _, controller in sorted(self.controllers.items())]

    def reset_statistics(self) -> None:
        for controller in self.controllers.values():
            controller.reset_statistics()

    def close(self) -> None:
        self.disable()
        for _, (module, original) in self.originals.items():
            module.forward = original
        self.originals = {}


class CSPNSemanticAdapter(ModelSemanticAdapter):
    MODEL_NAME = "cspn"
    MODULE_RULES = (
        ModuleRoleRule(r"^gud_up_proj_layer6(?:\.|$)", "guidance_logits", True, 40),
        ModuleRoleRule(r"^gud_up_proj_layer5(?:\.|$)", "depth_head_activation", True, 40),
        ModuleRoleRule(r"^gud_up_proj_layer[1-4](?:\.|$)", "decoder_activation", True, 30),
        ModuleRoleRule(r"^(?:conv1_1|bn1|layer[1-4]|conv2)(?:\.|$)", "encoder_activation", True, 20),
    )
    SIGNAL_RULES = (
        SignalRule("signal::initial_depth", "initial_depth", "prop_input",
                   "initial_depth", "gud_up_proj_layer5", True),
        SignalRule("signal::guidance", "guidance_logits", "prop_input",
                   "guidance", "gud_up_proj_layer6", True),
        SignalRule("signal::affinity", "affinity", "prop_input",
                   "affinity", "post_process_layer", True),
        SignalRule("signal::propagation_state", "propagation_state", "prop_output",
                   "propagation_state", "post_process_layer", True),
        SignalRule("signal::prediction", "prediction", "model_output",
                   "prediction", "model_output", True),
    )
    REQUIRED_ROLES = (
        "rgb_input", "sparse_depth_value", "encoder_activation",
        "decoder_activation", "initial_depth", "guidance_logits",
        "affinity", "propagation_state", "prediction",
    )
    PROPAGATION_PATHS = ("post_process_layer",)
    ALLOWED_CONCAT_CALLS = (3,)

    def _input_signals(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        tensor = inputs[0]
        return {
            "rgb": tensor[:, :3],
            "sparse_depth": tensor[:, 3:4],
        }

    def activation_boundaries(self) -> Tuple[ActivationBoundary, ...]:
        modules = dict(self.model.named_modules())
        skip_channels = int(
            modules["gud_up_proj_layer4.conv1"].out_channels)
        return (
            ActivationBoundary(
                "decoder_entry", "gud_up_proj_layer1", 0, (
                    ActivationConsumer(
                        "gud_up_proj_layer1.conv1", 0, None),
                    ActivationConsumer(
                        "gud_up_proj_layer1.sc_conv1", 0, None),
                )),
            ActivationBoundary(
                "layer4_signed_skip", "gud_up_proj_layer4", 1, (
                    ActivationConsumer(
                        "gud_up_proj_layer4.conv1_1",
                        skip_channels, skip_channels),
                )),
        )

    def _propagation_inputs(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        propagation = self._propagation_module()
        values = {
            "guidance": inputs[0],
            "initial_depth": inputs[1],
            "affinity": cspn_float_affinity(
                inputs[0], propagation.norm_type),
        }
        if len(inputs) > 2 and inputs[2] is not None:
            values["sparse_depth"] = inputs[2]
        return values

    def _propagation_outputs(self, output: Any) -> Mapping[str, Any]:
        return {"propagation_state": output}

    def _build_merge_adapters(self) -> Sequence[Any]:
        return (CSPNStructuralMergeAdapter(
            self.model, self.merge_policy, self.group_size, self.runtime),)

    def declared_merge_sites(self) -> Sequence[Tuple[str, str, bool]]:
        # Exact CSPN Add/Concat sites are registered from runtime controllers
        # after calibration, including only blocks that actually execute.
        return ()
