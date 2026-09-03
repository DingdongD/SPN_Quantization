"""DySPN semantic adapter."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adapters.base import (
    ModelSemanticAdapter,
    ModuleRoleRule,
    PatternSignalRule,
    SignalRule,
)


class DySPNSemanticAdapter(ModelSemanticAdapter):
    MODEL_NAME = "dyspn"
    MODULE_RULES = (
        ModuleRoleRule(r"^base\.gd_dec0", "guidance_logits", True, 50),
        ModuleRoleRule(r"^base\.gd_dec1", "decoder_activation", True, 45),
        ModuleRoleRule(r"^base\.(?:dec|conv6)", "decoder_activation", True, 40),
        ModuleRoleRule(r"^base\.conv1_(?:rgb|dep)", "encoder_activation", True, 35),
        ModuleRoleRule(r"^base\.conv[2-5](?:\.|$)", "encoder_activation", True, 30),
        ModuleRoleRule(r"^dyspn_\d+_\d+\.conv_offset_aff", "offset_logits", True, 60),
    )
    PATTERN_SIGNAL_RULES = (
        PatternSignalRule(r"^base\.conv[2-5].*\.se\.fc\.3$",
                          "se_gate", required=False),
    )
    SIGNAL_RULES = (
        SignalRule("signal::offset_logits", "offset_logits", "manual",
                   "offset_logits", "dyspn.conv_offset_aff", True),
        SignalRule("signal::affinity_logits", "affinity_logits", "manual",
                   "affinity_logits", "dyspn.conv_offset_aff", True),
        SignalRule("signal::initial_depth", "initial_depth", "prop_input",
                   "initial_depth", "base.guide_slice", True),
        SignalRule("signal::guidance", "guidance_logits", "prop_input",
                   "guidance", "base.guide_slice", True),
        SignalRule("signal::confidence_logits", "confidence_logits", "prop_input",
                   "confidence_logits", "base.guide_slice", True),
        SignalRule("signal::confidence", "confidence", "prop_input",
                   "confidence", "dyspn.sigmoid_mask", True),
        SignalRule("signal::offset", "offset", "prop_output",
                   "offset", "dyspn.get_refgrid", True),
        SignalRule("signal::affinity", "affinity", "prop_output",
                   "affinity", "dyspn.softmax", True),
        SignalRule("signal::propagation_state", "propagation_state", "prop_output",
                   "propagation_state", "dyspn.iteration", True),
        SignalRule("signal::prediction", "prediction", "prop_output",
                   "prediction", "dyspn.output", True),
    )
    REQUIRED_ROLES = (
        "rgb_input", "sparse_depth_value", "encoder_activation",
        "decoder_activation", "initial_depth", "guidance_logits",
        "confidence_logits", "confidence", "offset_logits",
        "affinity_logits", "offset", "affinity",
        "propagation_state", "prediction",
    )
    ALLOWED_CONCAT_CALLS = (4, 5)
    CONTRACT_PROTECTED_ROLES = (
        "affinity", "affinity_logits", "confidence", "confidence_logits",
        "guidance_logits", "initial_depth", "offset", "offset_logits",
        "propagation_state",
        "sparse_depth_value", "sparse_mask",
    )
    CONTRACT_PREFIX_GROUP_PATTERNS = (
        (r"^base\.conv1_(?:rgb|dep)$",),
        (r"^base\.conv2\.",),
        (r"^base\.conv3\.",),
        (r"^base\.conv4\.",),
        (r"^base\.conv5\.",),
        (r"^base\.conv6$",),
    )
    CONTRACT_TAIL_GROUP_PATTERNS = (
        (r"^base\.dec5$",),
        (r"^base\.dec4$",),
        (r"^base\.dec3$",),
        (r"^base\.dec2$",),
        (r"^base\.gd_dec1_$",),
    )

    def _resolve_propagation_module(self) -> Optional[nn.Module]:
        for name, module in self.model.named_modules():
            if name.startswith("dyspn_") and name.count("_") >= 2:
                return module
        return None

    def _input_signals(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        return {"rgb": inputs[0], "sparse_depth": inputs[1]}

    def _propagation_inputs(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        confidence_logits = inputs[3]
        confidence = torch.sigmoid(confidence_logits) * inputs[2].sign()
        return {
            "initial_depth": inputs[0],
            "guidance": inputs[1],
            "sparse_depth": inputs[2],
            "confidence_logits": confidence_logits,
            "confidence": confidence,
        }

    def _propagation_outputs(self, output: Any) -> Mapping[str, Any]:
        if not isinstance(output, Mapping):
            return {"prediction": output}
        return {
            "prediction": output.get("pred"),
            "propagation_state": output.get("list_feat"),
            "offset": output.get("offset"),
            "affinity": output.get("aff"),
        }

    def _install_extra_hooks(self) -> None:
        propagation = self._resolve_propagation_module()
        projection = getattr(propagation, "conv_offset_aff", None)
        if projection is None:
            if self.strict:
                raise RuntimeError("DySPN conv_offset_aff is missing")
            return

        def hook(module: nn.Module, inputs: Tuple[Any, ...],
                 output: torch.Tensor) -> None:
            del module, inputs
            channels = int(getattr(propagation, "ch", output.shape[1] // 3))
            offset_logits, affinity_logits = torch.split(
                output, [2 * channels, channels], dim=1)
            self._record("signal::offset_logits", offset_logits)
            self._record("signal::affinity_logits", affinity_logits)

        self._handles.append(projection.register_forward_hook(hook))

    def declared_merge_sites(self) -> Sequence[Tuple[str, str, bool]]:
        rows = [("merge::base::rgb_depth_stem",
                 "concat_merge", False)]
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            if class_name.startswith("StoDepth_") or (
                    name.startswith("base.conv") and
                    class_name in ("BasicBlock", "Bottleneck")):
                rows.append(("merge::%s::residual" % name,
                             "residual_merge", False))
        return rows
