"""Semantic adapters for supported SPN depth-completion models."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from spn_quant.adapters.base import ModelSemanticAdapter
from spn_quant.adapters.cspn import CSPNSemanticAdapter
from spn_quant.adapters.dyspn import DySPNSemanticAdapter
from spn_quant.adapters.nlspn import NLSPNSemanticAdapter
from spn_quant.adapters.completionformer import CompletionFormerSemanticAdapter
from spn_quant.adapters.completionformer_joint import (
    CompletionFormerJointAdapter,
)
from spn_quant.runtime import EdgeQDQRuntime


def detect_model_name(model: nn.Module) -> str:
    if hasattr(model, "post_process_layer") and hasattr(model, "gud_up_proj_layer6"):
        return "cspn"
    if hasattr(model, "base") and any(
            name.startswith("dyspn_") for name, _ in model.named_modules()):
        return "dyspn"
    if hasattr(model, "backbone") and hasattr(model, "prop_layer"):
        return "completionformer"
    if hasattr(model, "prop_layer") and hasattr(model, "id_dec0"):
        return "nlspn"
    raise ValueError("unable to identify supported SPN model")


def install_model_semantic_adapter(
        model: nn.Module, model_name: Optional[str] = None,
        runtime: Optional[EdgeQDQRuntime] = None,
        merge_policy: str = "shared", group_size: Optional[int] = None,
        strict: bool = True) -> ModelSemanticAdapter:
    name = detect_model_name(model) if model_name is None else str(model_name)
    adapters = {
        "cspn": CSPNSemanticAdapter,
        "dyspn": DySPNSemanticAdapter,
        "nlspn": NLSPNSemanticAdapter,
        "completionformer": CompletionFormerSemanticAdapter,
    }
    try:
        adapter_class = adapters[name]
    except KeyError:
        raise ValueError("unknown semantic adapter: %s" % name)
    return adapter_class(
        model, runtime=runtime, merge_policy=merge_policy,
        group_size=group_size, strict=strict)


__all__ = [
    "ModelSemanticAdapter",
    "CSPNSemanticAdapter",
    "DySPNSemanticAdapter",
    "NLSPNSemanticAdapter",
    "CompletionFormerSemanticAdapter",
    "CompletionFormerJointAdapter",
    "detect_model_name",
    "install_model_semantic_adapter",
]
