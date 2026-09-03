"""CompletionFormer semantic adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adapters.base import (
    ModelSemanticAdapter,
    ModuleRoleRule,
    PatternSignalRule,
    SignalRule,
)


class CompletionFormerSemanticAdapter(ModelSemanticAdapter):
    MODEL_NAME = "completionformer"
    MODULE_RULES = (
        ModuleRoleRule(r"^prop_layer\.conv_offset_aff", "offset_logits", True, 80),
        ModuleRoleRule(r"^backbone\.cf_dec", "confidence", True, 75),
        ModuleRoleRule(r"^backbone\.gd_dec0", "guidance_logits", True, 75),
        ModuleRoleRule(r"^backbone\.gd_dec1", "decoder_activation", True, 70),
        ModuleRoleRule(r"^backbone\.dep_dec0", "depth_head_activation", True, 75),
        ModuleRoleRule(r"^backbone\.dep_dec1", "decoder_activation", True, 70),
        ModuleRoleRule(r"^backbone\.dec[2-6](?:\.|$)", "decoder_activation", True, 60),
        ModuleRoleRule(r"^backbone\.former.*(?:attn|mlp|\.q|\.kv|\.proj|\.fc)",
                       "attention_activation", True, 65),
        ModuleRoleRule(r"^backbone\.former", "encoder_activation", True, 50),
        ModuleRoleRule(r"^backbone\.conv1", "encoder_activation", True, 45),
    )
    PATTERN_SIGNAL_RULES = (
        PatternSignalRule(r"(?:^|\.)ca\.sigmoid$",
                          "channel_attention_gate", required=False),
        PatternSignalRule(r"(?:^|\.)sa\.sigmoid$",
                          "spatial_attention_gate", required=False),
    )
    SIGNAL_RULES = (
        SignalRule("signal::offset_logits", "offset_logits", "manual",
                   "offset_logits", "prop_layer.conv_offset_aff", True),
        SignalRule("signal::affinity_logits", "affinity_logits", "manual",
                   "affinity_logits", "prop_layer.conv_offset_aff", True),
        SignalRule("signal::depth_residual", "depth_residual", "module_output",
                   "depth_residual", "backbone.dep_dec0", True),
        SignalRule("signal::initial_depth", "initial_depth", "prop_input",
                   "initial_depth", "model.pred_init_plus_sparse", True),
        SignalRule("signal::guidance", "guidance_logits", "prop_input",
                   "guidance", "backbone.gd_dec0", True),
        SignalRule("signal::confidence", "confidence", "prop_input",
                   "confidence", "backbone.cf_dec0", True),
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
        "attention_activation", "decoder_activation", "depth_residual",
        "initial_depth", "guidance_logits", "confidence",
        "offset_logits", "affinity_logits", "offset",
        "affinity", "propagation_state", "prediction",
    )
    PROPAGATION_PATHS = ("prop_layer",)
    ALLOWED_CONCAT_CALLS = (8, 10)
    CONTRACT_PROTECTED_ROLES = (
        "affinity", "affinity_logits", "confidence", "guidance_logits",
        "initial_depth",
        "offset", "offset_logits", "propagation_state",
        "sparse_depth_value", "sparse_mask",
    )
    CONTRACT_PREFIX_GROUP_PATTERNS = (
        (r"^backbone\.conv1_(?:rgb|dep)$", r"^backbone\.conv1$"),
        (r"^backbone\.former\.embed_layer1\.",),
        (r"^backbone\.former\.embed_layer2\.",),
        (r"^backbone\.former\.patch_embed1$",),
        (r"^backbone\.former\.block1\.",),
        (r"^backbone\.former\.patch_embed2$",),
        (r"^backbone\.former\.block2\.",),
        (r"^backbone\.former\.patch_embed3$",),
        (r"^backbone\.former\.block3\.",),
        (r"^backbone\.former\.patch_embed4$",),
        (r"^backbone\.former\.block4\.",),
    )
    CONTRACT_TAIL_GROUP_PATTERNS = (
        (r"^backbone\.former\.block[1-4]\.",),
        (r"^backbone\.dec6$",),
        (r"^backbone\.dec5$",),
        (r"^backbone\.dec4$",),
        (r"^backbone\.dec3$",),
        (r"^backbone\.dec2$",),
        (r"^backbone\.dep_dec1$", r"^backbone\.dep_dec0$"),
        (r"^backbone\.gd_dec1$",),
    )
    CONTRACT_SEARCH_UNIT_RULES = (
        ("stem", (r"^backbone\.conv1(?:_rgb|_dep)?\.",), "stem", 4, 4,
         False, "branch_independent"),
        ("cnn_encoder", (r"^backbone\.former\.embed_layer[12]\.",),
         "encoder", 4, 4, False, "static_tensor"),
        ("transformer_embed", (r"^backbone\.former\.patch_embed[1-4]\.",),
         "transformer_embed", 4, 4, False, "static_tensor"),
        ("attention_qkv", (r"^backbone\.former\.block[1-4]\.[0-9]+\.attn\.(?:q|kv)$",),
         "attention_qkv", 4, 8, False, "static_tensor"),
        ("attention_output", (r"^backbone\.former\.block[1-4]\.[0-9]+\.attn\.",),
         "attention_output", 4, 4, False, "static_tensor"),
        ("transformer_mlp", (r"^backbone\.former\.block[1-4]\.[0-9]+\.mlp\.",),
         "transformer_mlp", 4, 4, False, "static_tensor"),
        ("transformer_fusion", (
            r"^backbone\.former\.block[1-4]\.[0-9]+\.(?:resblock|concat_conv)(?:\.|$)",),
         "fusion", 4, 4, True, "branch_independent"),
        ("decoder_stage6", (r"^backbone\.dec6\.",), "decoder", 4, 4,
         False, "static_tensor"),
        ("decoder_stage5", (r"^backbone\.dec5\.",), "decoder", 4, 4,
         False, "static_tensor"),
        ("decoder_stage4", (r"^backbone\.dec4\.",), "decoder", 4, 4,
         False, "static_tensor"),
        ("decoder_stage3", (r"^backbone\.dec3\.",), "decoder", 4, 4,
         False, "static_tensor"),
        ("decoder_stage2", (r"^backbone\.dec2\.",), "fusion", 4, 4, True,
         "branch_independent"),
        ("initial_depth", (r"^backbone\.dep_dec[01]\.",), "initial_depth",
         4, 4, True, "branch_independent"),
        ("guidance_decoder", (r"^backbone\.gd_dec1\.",),
         "guidance_decoder", 4, 4, True, "static_tensor"),
    )

    def _input_signals(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        sample = inputs[0]
        return {"rgb": sample["rgb"], "sparse_depth": sample["dep"]}

    def _propagation_inputs(self, inputs: Tuple[Any, ...]) -> Mapping[str, Any]:
        return {
            "initial_depth": inputs[0],
            "guidance": inputs[1],
            "confidence": inputs[2] if len(inputs) > 2 else None,
            "sparse_depth": inputs[3] if len(inputs) > 3 else None,
        }

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
                raise RuntimeError("CompletionFormer conv_offset_aff is missing")
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
        rows = [
            ("merge::backbone::rgb_depth_stem", "concat_merge", False),
            ("merge::model::initial_depth_plus_sparse",
             "residual_merge", False),
        ]
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            if class_name in ("BasicBlock", "Bottleneck"):
                rows.append(("merge::%s::residual" % name,
                             "residual_merge", False))
            if class_name == "Attention" and name.startswith("backbone.former"):
                rows.append(("signal::%s::attention_probability" % name,
                             "attention_probability", False))
            if class_name == "Block" and name.startswith("backbone.former"):
                rows.extend([
                    ("merge::%s::attention_residual" % name,
                     "residual_merge", False),
                    ("merge::%s::mlp_residual" % name,
                     "residual_merge", False),
                    ("merge::%s::cnn_transformer_concat" % name,
                     "concat_merge", False),
                ])
        return rows
