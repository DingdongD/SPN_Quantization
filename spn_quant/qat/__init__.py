"""Quantization-aware training for strict SPN deployment graphs."""

from spn_quant.qat.ste import hard_forward_proxy, round_ste
from spn_quant.qat.quantizers import (
    ActivationSTEQuantizer,
    PerOutputChannelWeightFakeQuantizer,
)
from spn_quant.qat.cspn import (
    CSPNActivationQATController,
    CSPNQATConfig,
    CSPNQATController,
    CSPNQATPropagationController,
    CSPNWeightQATController,
    cspn_hard_activation_bits,
)
from spn_quant.qat.cspn_task_loss import (
    CSPNTaskLossWeights,
    cspn_task_aware_loss,
    depth_boundary_mask,
)
from spn_quant.qat.method_config import (
    CSPNMethodExperimentConfig,
    load_method_config,
)
from spn_quant.qat.lsqplus import (
    LSQPlusActivationQuantizer,
    LSQPlusWeightParametrization,
)
from spn_quant.qat.hawq import HAWQActivationQuantizer
from spn_quant.qat.cspn_methods import (
    CSPNMethodQATConfig,
    CSPNMethodQATController,
)
from spn_quant.qat.model_methods import (
    ModelHardDeploymentController,
    ModelMethodQATConfig,
    ModelMethodQATController,
)
from spn_quant.qat.task_loss import (
    ModelTaskCapture,
    ModelTaskLoss,
    ModelTaskLossWeights,
    model_task_aware_loss,
)


__all__ = (
    "hard_forward_proxy",
    "round_ste",
    "ActivationSTEQuantizer",
    "PerOutputChannelWeightFakeQuantizer",
    "CSPNActivationQATController",
    "CSPNQATConfig",
    "CSPNQATController",
    "CSPNQATPropagationController",
    "CSPNWeightQATController",
    "cspn_hard_activation_bits",
    "CSPNTaskLossWeights",
    "cspn_task_aware_loss",
    "depth_boundary_mask",
    "CSPNMethodExperimentConfig",
    "load_method_config",
    "LSQPlusActivationQuantizer",
    "LSQPlusWeightParametrization",
    "HAWQActivationQuantizer",
    "CSPNMethodQATConfig",
    "CSPNMethodQATController",
    "ModelHardDeploymentController",
    "ModelMethodQATConfig",
    "ModelMethodQATController",
    "ModelTaskCapture",
    "ModelTaskLoss",
    "ModelTaskLossWeights",
    "model_task_aware_loss",
)
