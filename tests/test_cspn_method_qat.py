from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from scripts.hardware_aligned_quantization import (
    SymmetricActivationQuantizer,
    UnsignedActivationQuantizer,
)
from spn_quant.propagation.adapters import CSPNPropagationAdapter
from spn_quant.propagation.controller import PropagationQuantConfig
from spn_quant.qat.cspn_methods import (
    CSPNMethodQATConfig,
    CSPNMethodQATController,
)


class ToyPropagation(nn.Module):
    def __init__(self, steps=24):
        super().__init__()
        self.prop_time = int(steps)
        self.norm_type = "8sum"

    def forward(self, guidance, initial, sparse=None):
        del guidance
        return initial if sparse is None else torch.where(
            sparse != 0, initial, initial)


def _hard_propagation(steps=24):
    module = ToyPropagation(steps)
    adapter = CSPNPropagationAdapter(module)
    guidance = torch.randn(1, 8, 3, 4)
    initial = torch.rand(1, 1, 3, 4)
    sparse = torch.zeros_like(initial)
    adapter.observe()
    module(guidance, initial, sparse)
    adapter.freeze()
    adapter.configure(PropagationQuantConfig(
        affinity_bits=8,
        confidence_bits=8,
        offset_bits=8,
        state_bits=8,
        coefficient_fraction_bits=13,
    ))
    return adapter


def _components():
    model = nn.Sequential(nn.Conv2d(2, 3, 1))
    instrumentor = SimpleNamespace(
        quantizers={
            ("0", "input"): SymmetricActivationQuantizer(4, 2.0),
        },
        relu_quantizers={
            "relu#0": UnsignedActivationQuantizer(4, 3.0),
        },
    )
    boundary = SimpleNamespace(
        active_quantizers={
            "decoder_entry": SymmetricActivationQuantizer(4, 4.0),
        })
    return model, instrumentor, boundary, _hard_propagation()


def _config(method, bits, momentum=0.95):
    return CSPNMethodQATConfig(
        method=method,
        weight_bits=(("0", bits),),
        activation_bits=(
            (("0", "input"), bits),
            (("boundary_controller.decoder_entry", "boundary"), bits),
            (("relu#0", "relu_output"), bits),
        ),
        propagation=PropagationQuantConfig(
            affinity_bits=8,
            confidence_bits=8,
            offset_bits=8,
            state_bits=8,
            coefficient_fraction_bits=13,
        ),
        hawq_range_momentum=momentum,
    )


def _initialization_rows():
    return (
        (("0", "input"), torch.tensor([-1.0, 2.0])),
        (("boundary_controller.decoder_entry", "boundary"),
         torch.tensor([-3.0, 4.0])),
        (("relu#0", "relu_output"), torch.tensor([0.0, 5.0])),
    )


def test_lsqplus_controller_exports_separate_method_and_master_state():
    model, instrumentor, boundary, propagation = _components()
    reference = model[0].weight.detach().clone()
    controller = CSPNMethodQATController(
        model, instrumentor, boundary, propagation,
        _config("lsqplus", 4))

    controller.initialize_activations(_initialization_rows())
    controller.install()

    state = controller.method_state_dict()
    assert "weight.0.step" in state
    assert "activation.('0', 'input').step" in state
    assert "activation.('0', 'input').offset" in state
    master = controller.canonical_model_state_dict()
    assert torch.equal(master["0.weight"], reference)
    assert not any("parametrizations" in key for key in master)
    assert parametrize.is_parametrized(model[0], "weight")

    controller.remove()
    propagation.close()


def test_lsqplus_method_state_round_trip_preserves_output():
    model, instrumentor, boundary, propagation = _components()
    controller = CSPNMethodQATController(
        model, instrumentor, boundary, propagation,
        _config("lsqplus", 6))
    controller.initialize_activations(_initialization_rows())
    controller.install()
    tensor = torch.tensor([[[[-0.7]], [[1.2]]]])
    first = instrumentor.quantizers[("0", "input")](tensor)
    state = controller.method_state_dict()
    controller.load_method_state_dict(state)
    second = instrumentor.quantizers[("0", "input")](tensor)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    controller.remove()
    propagation.close()


def test_canonical_model_state_round_trip_restores_master_weight():
    model, instrumentor, boundary, propagation = _components()
    controller = CSPNMethodQATController(
        model, instrumentor, boundary, propagation,
        _config("lsqplus", 4))
    controller.initialize_activations(_initialization_rows())
    controller.install()
    state = controller.canonical_model_state_dict()
    with torch.no_grad():
        model[0].parametrizations.weight.original.add_(3.0)

    controller.load_canonical_model_state_dict(state)

    assert torch.equal(
        model[0].parametrizations.weight.original, state["0.weight"])
    controller.remove()
    propagation.close()


def test_hawq_controller_uses_assignment_and_freezes_ranges():
    model, instrumentor, boundary, propagation = _components()
    controller = CSPNMethodQATController(
        model, instrumentor, boundary, propagation,
        _config("hawq", 6, momentum=0.9))
    controller.initialize_activations(_initialization_rows())
    controller.install()

    controller.freeze_activation_ranges()
    manifest = controller.manifest()

    assert manifest["weight_bits"] == (("0", 6),)
    assert manifest["activation_bits"] == _config(
        "hawq", 6, momentum=0.9).activation_bits
    assert manifest["guidance"] == "fp32"
    assert manifest["propagation"] == {
        "affinity_bits": 8,
        "confidence_bits": 8,
        "offset_bits": 8,
        "state_bits": 8,
        "coefficient_fraction_bits": 13,
        "proxy_states": 24,
    }
    assert all(not quantizer.running_range
               for quantizer in controller.activation_quantizers())

    controller.remove()
    propagation.close()


def test_method_controller_restores_original_activation_quantizers():
    model, instrumentor, boundary, propagation = _components()
    originals = (
        dict(instrumentor.quantizers),
        dict(instrumentor.relu_quantizers),
        dict(boundary.active_quantizers),
    )
    controller = CSPNMethodQATController(
        model, instrumentor, boundary, propagation,
        _config("hawq", 4))
    controller.initialize_activations(_initialization_rows())
    controller.install()
    controller.remove()

    assert instrumentor.quantizers == originals[0]
    assert instrumentor.relu_quantizers == originals[1]
    assert boundary.active_quantizers == originals[2]
    propagation.close()


def test_method_controller_rejects_guidance_assignment():
    model, instrumentor, boundary, propagation = _components()
    config = CSPNMethodQATConfig(
        method="hawq",
        weight_bits=(("gud_up_proj_layer6.conv1", 4),),
        activation_bits=_config("hawq", 4).activation_bits,
        propagation=_config("hawq", 4).propagation,
        hawq_range_momentum=0.95,
    )

    with pytest.raises(ValueError, match="guidance"):
        CSPNMethodQATController(
            model, instrumentor, boundary, propagation, config)
    propagation.close()


def test_method_config_rejects_unsupported_bits_and_method():
    propagation = _config("hawq", 4).propagation
    with pytest.raises(ValueError, match="method"):
        CSPNMethodQATConfig(
            method="rtn", weight_bits=(("0", 4),),
            activation_bits=((('0', 'input'), 4),),
            propagation=propagation, hawq_range_momentum=0.95)
    with pytest.raises(ValueError, match="LSQ"):
        CSPNMethodQATConfig(
            method="lsqplus", weight_bits=(("0", 8),),
            activation_bits=((('0', 'input'), 8),),
            propagation=propagation, hawq_range_momentum=0.95)
