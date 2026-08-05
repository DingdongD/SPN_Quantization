"""Semantic quantization contracts for SPN depth-completion models."""

from spn_quant.integration import (
    EdgeAwareInstrumentorAdapter,
    EdgeAwareQuantizerProxy,
    EdgeQDQRuntime,
)
from spn_quant.merge import MergeSiteController
from spn_quant.specs import DEFAULT_W4A4_ACTIVATION_SPEC, QuantSpec
from spn_quant.sites import (
    QuantSite,
    QuantSiteRegistry,
    TracedTensorEdge,
    build_module_site_registry,
    trace_module_tensor_edges,
)

__all__ = [
    "DEFAULT_W4A4_ACTIVATION_SPEC",
    "EdgeAwareInstrumentorAdapter",
    "EdgeAwareQuantizerProxy",
    "EdgeQDQRuntime",
    "MergeSiteController",
    "QuantSpec",
    "QuantSite",
    "QuantSiteRegistry",
    "TracedTensorEdge",
    "build_module_site_registry",
    "trace_module_tensor_edges",
]
