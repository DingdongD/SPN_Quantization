"""Semantic quantization contracts for SPN depth-completion models."""

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
    AdaptiveRoundingParametrization,
    LinearTemperatureDecay,
    select_weight_modules,
)
from spn_quant.integration import (
    EdgeAwareInstrumentorAdapter,
    EdgeAwareQuantizerProxy,
    EdgeQDQRuntime,
)
from spn_quant.merge import MergeSiteController
from spn_quant.reconstruction import (
    ActivationReconstructionController,
    CalibrationRecord,
    LearnedActivationQuantizer,
    ModuleIOCache,
    ReconstructionConfig,
    ReconstructionResult,
    SemanticBlockReconstructor,
    reconstruction_loss,
)
from spn_quant.specs import DEFAULT_W4A4_ACTIVATION_SPEC, QuantSpec
from spn_quant.sites import (
    QuantSite,
    QuantSiteRegistry,
    TracedTensorEdge,
    build_module_site_registry,
    trace_module_tensor_edges,
)

__all__ = [
    "ActivationReconstructionController",
    "AdaptiveRoundingConfig",
    "AdaptiveRoundingController",
    "AdaptiveRoundingParametrization",
    "CalibrationRecord",
    "DEFAULT_W4A4_ACTIVATION_SPEC",
    "EdgeAwareInstrumentorAdapter",
    "EdgeAwareQuantizerProxy",
    "EdgeQDQRuntime",
    "LearnedActivationQuantizer",
    "LinearTemperatureDecay",
    "MergeSiteController",
    "ModuleIOCache",
    "QuantSpec",
    "QuantSite",
    "QuantSiteRegistry",
    "ReconstructionConfig",
    "ReconstructionResult",
    "SemanticBlockReconstructor",
    "TracedTensorEdge",
    "build_module_site_registry",
    "reconstruction_loss",
    "select_weight_modules",
    "trace_module_tensor_edges",
]
