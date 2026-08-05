"""Compatibility exports for model-level semantic adapters."""

from spn_quant.adapters import (
    CompletionFormerSemanticAdapter,
    CSPNSemanticAdapter,
    DySPNSemanticAdapter,
    NLSPNSemanticAdapter,
    detect_model_name,
    install_model_semantic_adapter,
)

__all__ = [
    "CompletionFormerSemanticAdapter",
    "CSPNSemanticAdapter",
    "DySPNSemanticAdapter",
    "NLSPNSemanticAdapter",
    "detect_model_name",
    "install_model_semantic_adapter",
]
