from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from scripts.hardware_aligned_quantization import (
    SymmetricActivationQuantizer,
    UnsignedActivationQuantizer,
)
from spn_quant.qat.cspn import (
    CSPNActivationQATController,
    CSPNWeightQATController,
)
from spn_quant.qat.quantizers import ActivationSTEQuantizer


def test_weight_controller_exports_master_weight_under_standard_key():
    model = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1))
    reference = model[0].weight.detach().clone()
    controller = CSPNWeightQATController(model, ("0",))

    controller.install()
    model(torch.randn(2, 4, 8, 8)).sum().backward()

    assert parametrize.is_parametrized(model[0], "weight")
    master = model[0].parametrizations.weight.original
    assert torch.isfinite(master.grad).all()
    state = controller.canonical_state_dict()
    assert torch.equal(state["0.weight"], reference)
    assert "0.parametrizations.weight.original" not in state

    fresh = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1))
    fresh.load_state_dict(state)
    assert torch.equal(fresh[0].weight, reference)

    controller.remove()
    assert not parametrize.is_parametrized(model[0], "weight")
    assert torch.equal(model[0].weight, reference)


def test_weight_controller_rejects_unknown_and_duplicate_install():
    model = nn.Sequential(nn.Conv2d(4, 8, 1))
    with pytest.raises(KeyError):
        CSPNWeightQATController(model, ("missing",)).install()

    controller = CSPNWeightQATController(model, ("0",))
    controller.install()
    with pytest.raises(RuntimeError, match="already installed"):
        controller.install()
    controller.remove()


def _activation_fixture():
    input_quantizer = SymmetricActivationQuantizer(bits=4, maximum=2.0)
    relu_quantizer = UnsignedActivationQuantizer(bits=4, maximum=3.0)
    boundary_quantizer = SymmetricActivationQuantizer(bits=4, maximum=4.0)
    instrumentor = SimpleNamespace(
        quantizers={("encoder.conv", "input"): input_quantizer},
        relu_quantizers={"encoder.relu#0": relu_quantizer},
    )
    rotation = SimpleNamespace(
        active_quantizers={"decoder_entry": boundary_quantizer})
    return instrumentor, rotation


def test_activation_controller_wraps_and_restores_exact_owners():
    instrumentor, rotation = _activation_fixture()
    original_quantizers = dict(instrumentor.quantizers)
    original_relu_quantizers = dict(instrumentor.relu_quantizers)
    original_structural = dict(rotation.active_quantizers)
    controller = CSPNActivationQATController(instrumentor, rotation)

    controller.install()

    assert controller.ordinary_owners == {
        ("encoder.conv", "input"), "encoder.relu#0"}
    assert controller.structural_owners == {"decoder_entry"}
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in instrumentor.quantizers.values())
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in instrumentor.relu_quantizers.values())
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in rotation.active_quantizers.values())

    controller.remove()

    assert instrumentor.quantizers == original_quantizers
    assert instrumentor.relu_quantizers == original_relu_quantizers
    assert rotation.active_quantizers == original_structural
    with pytest.raises(RuntimeError, match="not installed"):
        controller.remove()


def test_activation_controller_rejects_guidance_owner():
    instrumentor, rotation = _activation_fixture()
    instrumentor.quantizers[("gud_up_proj_layer6", "output")] = \
        SymmetricActivationQuantizer(bits=4, maximum=1.0)
    controller = CSPNActivationQATController(instrumentor, rotation)

    with pytest.raises(RuntimeError, match="guidance"):
        controller.install()
