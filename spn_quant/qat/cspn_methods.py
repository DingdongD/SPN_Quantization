"""CSPN compatibility wrapper for contract-neutral method QAT state."""

from __future__ import annotations

from spn_quant.qat.cspn import (
    CSPNQATPropagationController,
    _hard_activation_quantizers,
)
from spn_quant.qat.model_methods import (
    ModelMethodQATConfig,
    _MethodQATControllerBase,
    _activation_unsigned,
)


CSPNMethodQATConfig = ModelMethodQATConfig


class CSPNMethodQATController(_MethodQATControllerBase):
    """Retain CSPN graph bindings around shared method QAT machinery."""

    def __init__(self, model, instrumentor, boundary_controller,
                 hard_propagation, config: CSPNMethodQATConfig) -> None:
        if not isinstance(config, CSPNMethodQATConfig):
            raise TypeError("config must be CSPNMethodQATConfig")
        if any("gud_up_proj_layer6" in name
               for name, bits in config.weight_bits):
            raise ValueError("guidance weights must remain FP32")
        if any("gud_up_proj_layer6" in owner[0]
               for owner, bits in config.activation_bits):
            raise ValueError("guidance activations must remain FP32")
        hard_quantizers = _hard_activation_quantizers(
            instrumentor, boundary_controller)
        declared = dict(config.activation_bits)
        if set(declared) != set(hard_quantizers):
            raise ValueError(
                "CSPN method activation owner coverage does not match hard path")
        unsigned = dict(
            (owner, _activation_unsigned(hard_quantizers[owner]))
            for owner in declared)
        super().__init__(model, config, unsigned)
        self.instrumentor = instrumentor
        self.boundary_controller = boundary_controller
        self.hard_propagation = hard_propagation
        self.propagation = CSPNQATPropagationController(hard_propagation)
        self.original_quantizers = dict(instrumentor.quantizers)
        self.original_relu_quantizers = dict(instrumentor.relu_quantizers)
        self.original_structural = dict(
            boundary_controller.active_quantizers)

    def _install_activations(self) -> None:
        self.instrumentor.quantizers = dict(
            (key, self.activation_by_owner[(str(key[0]), str(key[1]))])
            for key in self.original_quantizers)
        self.instrumentor.relu_quantizers = dict(
            (key, self.activation_by_owner[(str(key), "relu_output")])
            for key in self.original_relu_quantizers)
        self.boundary_controller.active_quantizers = dict(
            (key, self.activation_by_owner[
                ("boundary_controller.%s" % key, "boundary")])
            for key in self.original_structural)

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("CSPN method QAT is already installed")
        if not self.activations_initialized:
            raise RuntimeError("CSPN method activations are not initialized")
        if self.hard_propagation.controller.config != self.config.propagation:
            raise ValueError("CSPN hard propagation config does not match QAT")
        self._install_weights()
        self._install_activations()
        self.propagation.install()
        self.installed = True

    def manifest(self):
        propagation = self.config.propagation
        return {
            "method": self.config.method,
            "weight_bits": self.config.weight_bits,
            "activation_bits": self.config.activation_bits,
            "guidance": "fp32",
            "bias": "fp32",
            "propagation": {
                "affinity_bits": propagation.affinity_bits,
                "confidence_bits": propagation.confidence_bits,
                "offset_bits": propagation.offset_bits,
                "state_bits": propagation.state_bits,
                "coefficient_fraction_bits":
                    propagation.coefficient_fraction_bits,
                "proxy_states": int(self.propagation.module.prop_time),
            },
        }

    def set_runtime_statistics(self, enabled: bool) -> None:
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        self.instrumentor.set_runtime_statistics(enabled)
        self.hard_propagation.set_runtime_statistics(enabled)

    def remove(self) -> None:
        if not self.installed:
            raise RuntimeError("CSPN method QAT is not installed")
        self.propagation.remove()
        self.instrumentor.quantizers = self.original_quantizers
        self.instrumentor.relu_quantizers = self.original_relu_quantizers
        self.boundary_controller.active_quantizers = self.original_structural
        self._remove_weights()
        self.installed = False


__all__ = (
    "CSPNMethodQATConfig",
    "CSPNMethodQATController",
)
