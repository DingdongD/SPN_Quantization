"""Semantic quantization contracts for SPN depth-completion models."""

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
    AdaptiveRoundingParametrization,
    CosineTemperatureDecay,
    LinearTemperatureDecay,
    is_supported_weight_module,
    select_weight_modules,
)
from spn_quant.deployment_contract import (
    StrictContractInstrumentor,
    build_deployment_contract,
    dequantize_weight_contract,
    export_rounding_contracts,
    file_sha256,
    load_deployment_contract,
    save_deployment_contract,
    tensor_sha256,
    validate_graph_preparation,
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
from spn_quant.strict_reconstruction import (
    StrictBlockReconstructor,
    StrictCalibrationRecord,
    StrictReconstructionConfig,
    StrictReconstructionResult,
    strict_reconstruction_loss,
)

__all__ = [
    "ActivationReconstructionController",
    "AdaptiveRoundingConfig",
    "AdaptiveRoundingController",
    "AdaptiveRoundingParametrization",
    "CosineTemperatureDecay",
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
    "StrictBlockReconstructor",
    "StrictCalibrationRecord",
    "StrictContractInstrumentor",
    "StrictReconstructionConfig",
    "StrictReconstructionResult",
    "TracedTensorEdge",
    "build_deployment_contract",
    "build_module_site_registry",
    "dequantize_weight_contract",
    "export_rounding_contracts",
    "file_sha256",
    "is_supported_weight_module",
    "load_deployment_contract",
    "reconstruction_loss",
    "save_deployment_contract",
    "select_weight_modules",
    "strict_reconstruction_loss",
    "tensor_sha256",
    "trace_module_tensor_edges",
    "validate_graph_preparation",
]
