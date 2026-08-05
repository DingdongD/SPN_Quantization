"""CSPN semantic adapter and structural merge instrumentation."""

from __future__ import annotations

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
from spn_quant.runtime import EdgeQDQRuntime


class CSPNStructuralMergeAdapter(object):
    """Expose CSPN ResNet/decoder Add and Cat sites without source edits."""

    _SUPPORTED = frozenset((
        "BasicBlock", "Bottleneck", "UpProj_Block",
        "Gudi_UpProj_Block", "Gudi_UpProj_Block_Cat",
    ))

    def __init__(self, model: nn.Module, policy: str,
                 group_size: Optional[int], runtime: EdgeQDQRuntime) -> None:
        self.model = model
        self.policy = policy
        self.group_size = group_size
        self.runtime = runtime
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
        controller = self.controllers.get(key)
        if controller is None:
            controller = MergeSiteController(
                key, operation=operation, policy=self.policy,
                axis=1, group_size=self.group_size, runtime=self.runtime)
            self.controllers[key] = controller
        return controller

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
                return original(*args, **kwargs)
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
        SignalRule("signal::affinity", "affinity", "method_output",
                   "affinity", "post_process_layer.affinity_normalization", True),
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._affinity_original = None
        super(CSPNSemanticAdapter, self).__init__(*args, **kwargs)

    def _input_signals(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        tensor = inputs[0]
        return {
            "rgb": tensor[:, :3],
            "sparse_depth": tensor[:, 3:4],
        }

    def _propagation_inputs(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        values = {"guidance": inputs[0], "initial_depth": inputs[1]}
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

    def _install_extra_hooks(self) -> None:
        module = getattr(self.model, "post_process_layer", None)
        method = getattr(module, "affinity_normalization", None)
        if method is None:
            if self.strict:
                raise RuntimeError("CSPN affinity_normalization is missing")
            return
        self._affinity_original = method

        def wrapped(current: Any, guidance: torch.Tensor) -> Any:
            output = method(guidance)
            affinity = output[0] if isinstance(output, (list, tuple)) else output
            self._record("signal::affinity", affinity)
            return output

        module.affinity_normalization = types.MethodType(wrapped, module)

    def close(self) -> None:
        module = getattr(self.model, "post_process_layer", None)
        if module is not None and self._affinity_original is not None:
            module.affinity_normalization = self._affinity_original
            self._affinity_original = None
        super(CSPNSemanticAdapter, self).close()
