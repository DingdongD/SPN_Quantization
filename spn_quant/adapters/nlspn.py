"""NLSPN semantic adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adapters.base import (
    ModelSemanticAdapter,
    ModuleRoleRule,
    SignalRule,
)


class NLSPNSemanticAdapter(ModelSemanticAdapter):
    MODEL_NAME = "nlspn"
    MODULE_RULES = (
        ModuleRoleRule(r"^prop_layer\.conv_offset_aff", "offset_logits", True, 60),
        ModuleRoleRule(r"^cf_dec", "confidence", True, 55),
        ModuleRoleRule(r"^gd_dec0", "guidance_logits", True, 55),
        ModuleRoleRule(r"^gd_dec1", "decoder_activation", True, 50),
        ModuleRoleRule(r"^id_dec0", "depth_head_activation", True, 55),
        ModuleRoleRule(r"^id_dec1", "decoder_activation", True, 50),
        ModuleRoleRule(r"^dec[2-5](?:\.|$)", "decoder_activation", True, 40),
        ModuleRoleRule(r"^conv1_(?:rgb|dep)", "encoder_activation", True, 35),
        ModuleRoleRule(r"^conv[2-6](?:\.|$)", "encoder_activation", True, 30),
    )
    SIGNAL_RULES = (
        SignalRule("signal::offset_logits", "offset_logits", "manual",
                   "offset_logits", "prop_layer.conv_offset_aff", True),
        SignalRule("signal::affinity_logits", "affinity_logits", "manual",
                   "affinity_logits", "prop_layer.conv_offset_aff", True),
        SignalRule("signal::initial_depth", "initial_depth", "prop_input",
                   "initial_depth", "id_dec0", True),
        SignalRule("signal::guidance", "guidance_logits", "prop_input",
                   "guidance", "gd_dec0", True),
        SignalRule("signal::confidence", "confidence", "prop_input",
                   "confidence", "cf_dec0", True),
        SignalRule("signal::offset", "offset", "prop_output",
                   "offset", "prop_layer._get_offset_affinity", True),
        SignalRule("signal::affinity", "affinity", "prop_output",
                   "affinity", "prop_layer._get_offset_affinity", True),
        SignalRule("signal::propagation_state", "propagation_state", "prop_output",
                   "propagation_state", "prop_layer._propagate_once", True),
        SignalRule("signal::prediction", "prediction", "prop_output",
                   "prediction", "prop_layer", True),
    )
    REQUIRED_ROLES = (
        "rgb_input", "sparse_depth_value", "encoder_activation",
        "decoder_activation", "initial_depth", "guidance_logits",
        "confidence", "offset_logits", "affinity_logits", "offset",
        "affinity", "propagation_state", "prediction",
    )
    PROPAGATION_PATHS = ("prop_layer",)
    ALLOWED_CONCAT_CALLS = (7, 9)
    CONTRACT_PROTECTED_ROLES = (
        "affinity", "affinity_logits", "confidence", "guidance_logits",
        "initial_depth",
        "offset", "offset_logits", "propagation_state",
        "sparse_depth_value", "sparse_mask",
    )
    CONTRACT_PREFIX_GROUP_PATTERNS = (
        (r"^conv1_(?:rgb|dep)$",),
        (r"^conv2\.",),
        (r"^conv3\.",),
        (r"^conv4\.",),
        (r"^conv5\.",),
        (r"^conv6$",),
    )
    CONTRACT_TAIL_GROUP_PATTERNS = (
        (r"^dec5$",),
        (r"^dec4$",),
        (r"^dec3$",),
        (r"^dec2$",),
        (r"^id_dec1$", r"^id_dec0$"),
        (r"^gd_dec1$",),
    )
    CONTRACT_SEARCH_UNIT_RULES = (
        ("stem", (r"^conv1_(?:rgb|dep)\.",), "stem", 4, 4, False,
         "branch_independent"),
        ("early_boundary", (
            r"^conv2\.0\.conv1$", r"^conv2\.0\.conv2$",
            r"^conv3\.0\.downsample\.0$"), "early_boundary", 4, 4, True,
         "static_tensor"),
        ("encoder_stage2_remaining", (r"^conv2\.",), "encoder", 4, 4,
         False, "static_tensor"),
        ("encoder_stage3", (r"^conv3\.",), "encoder", 4, 4, False,
         "static_tensor"),
        ("encoder_stage4", (r"^conv4\.",), "encoder", 4, 4, False,
         "static_tensor"),
        ("encoder_stage5", (r"^conv5\.",), "encoder", 4, 4, False,
         "static_tensor"),
        ("encoder_tail", (r"^conv6\.",), "encoder", 4, 4, False,
         "static_tensor"),
        ("decoder_stage5", (r"^dec5\.",), "decoder", 4, 4, False,
         "static_tensor"),
        ("decoder_stage4", (r"^dec4\.",), "decoder", 4, 4, False,
         "static_tensor"),
        ("decoder_stage3", (r"^dec3\.",), "decoder", 4, 4, False,
         "static_tensor"),
        ("decoder_stage2", (r"^dec2\.",), "fusion", 4, 4, True,
         "branch_independent"),
        ("guidance_decoder", (r"^gd_dec1\.",), "guidance_decoder", 4, 4,
         True, "static_tensor"),
        ("initial_depth", (r"^id_dec[01]\.",), "initial_depth", 4, 4, True,
         "branch_independent"),
    )

    def _input_signals(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        sample = inputs[0]
        return {"rgb": sample["rgb"], "sparse_depth": sample["dep"]}

    def _propagation_inputs(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        values = {
            "initial_depth": inputs[0],
            "guidance": inputs[1],
            "confidence": inputs[2] if len(inputs) > 2 else None,
            "sparse_depth": inputs[3] if len(inputs) > 3 else None,
        }
        return values

    def _propagation_outputs(self, output: Any) -> Mapping[str, Any]:
        if not isinstance(output, (list, tuple)):
            return {"prediction": output}
        return {
            "prediction": output[0],
            "propagation_state": output[1],
            "offset": output[2],
            "affinity": output[3],
        }

    def _install_extra_hooks(self) -> None:
        propagation = self._resolve_propagation_module()
        projection = getattr(propagation, "conv_offset_aff", None)
        if projection is None:
            if self.strict:
                raise RuntimeError("NLSPN conv_offset_aff is missing")
            return

        def hook(module: nn.Module, inputs: Tuple[Any, ...],
                 output: torch.Tensor) -> None:
            del module, inputs
            o1, o2, affinity_logits = torch.chunk(output, 3, dim=1)
            self._record("signal::offset_logits",
                         torch.cat((o1, o2), dim=1))
            self._record("signal::affinity_logits", affinity_logits)

        self._handles.append(projection.register_forward_hook(hook))

    def declared_merge_sites(self) -> Sequence[Tuple[str, str, bool]]:
        rows = [("merge::model::rgb_depth_stem",
                 "concat_merge", False)]
        for name, module in self.model.named_modules():
            if name.startswith(("conv2", "conv3", "conv4", "conv5")) and \
                    module.__class__.__name__ in ("BasicBlock", "Bottleneck"):
                rows.append(("merge::%s::residual" % name,
                             "residual_merge", False))
        return rows
