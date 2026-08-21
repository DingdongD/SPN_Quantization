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
)
