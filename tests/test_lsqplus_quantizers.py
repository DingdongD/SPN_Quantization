import pytest
import torch

from spn_quant.qat import lsqplus
from spn_quant.qat.lsqplus import (
    LSQPlusActivationQuantizer,
    LSQPlusWeightParametrization,
)


def test_lsqplus_activation_matches_affine_hard_reference():
    quantizer = LSQPlusActivationQuantizer(bits=4, unsigned=False)
    quantizer.initialize(torch.tensor([-2.0, -0.3, 0.7, 3.0]))
    current = torch.tensor([-2.5, -0.2, 0.8, 3.5], requires_grad=True)
    scale = quantizer.step.detach()
    offset = quantizer.offset.detach()
    expected = torch.round(
        (current.detach() - offset) / scale).clamp(-8, 7)
    expected = expected * scale + offset

    torch.testing.assert_close(quantizer(current).detach(), expected)


def test_lsqplus_unsigned_uses_all_sixteen_a4_codes():
    quantizer = LSQPlusActivationQuantizer(bits=4, unsigned=True)
    quantizer.initialize(torch.tensor([0.0, 15.0]))

    _, codes = quantizer.quantize_with_codes(torch.arange(16.0))

    assert torch.equal(codes, torch.arange(16, dtype=torch.int64))


def test_lsqplus_weight_step_is_per_logical_output_channel():
    weight = torch.tensor([
        [[[1.0, 2.0]]],
        [[[4.0, 8.0]]],
    ])
    quantizer = LSQPlusWeightParametrization(
        bits=4, channel_dim=0, initial_weight=weight)

    assert quantizer.step.shape == (2, 1, 1, 1)
    assert quantizer(weight).shape == weight.shape


def test_lsqplus_conv_transpose_uses_logical_output_axis():
    weight = torch.arange(24.0).reshape(2, 3, 2, 2)
    quantizer = LSQPlusWeightParametrization(
        bits=6, channel_dim=1, initial_weight=weight)

    assert quantizer.step.shape == (1, 3, 1, 1)
    assert quantizer(weight).shape == weight.shape


def test_lsqplus_requires_explicit_initialization():
    quantizer = LSQPlusActivationQuantizer(bits=4, unsigned=False)

    with pytest.raises(RuntimeError, match="not initialized"):
        quantizer(torch.ones(2))


def test_lsqplus_step_offset_and_master_weight_receive_gradients():
    activation = LSQPlusActivationQuantizer(bits=4, unsigned=False)
    activation.initialize(torch.tensor([-1.0, 2.0]))
    weight = torch.tensor([[[[0.5, 1.0]]]], requires_grad=True)
    weight_quantizer = LSQPlusWeightParametrization(
        4, 0, weight.detach())
    current = torch.tensor([0.25], requires_grad=True)

    loss = activation(current).sum() + weight_quantizer(weight).sum()
    loss.backward()

    assert activation.step.grad is not None
    assert activation.offset.grad is not None
    assert weight_quantizer.step.grad is not None
    assert weight.grad is not None
    assert current.grad is not None


def test_lsqplus_hot_forward_does_not_repeat_full_tensor_validation(
        monkeypatch):
    activation = LSQPlusActivationQuantizer(bits=4, unsigned=False)
    activation.initialize(torch.tensor([-1.0, 2.0]))
    weight = torch.tensor([[[[0.5, 1.0]]]])
    weight_quantizer = LSQPlusWeightParametrization(
        bits=4, channel_dim=0, initial_weight=weight)
    monkeypatch.setattr(
        lsqplus, "_require_finite",
        lambda name, tensor: (_ for _ in ()).throw(
            AssertionError("hot-path full tensor validation")))
    monkeypatch.setattr(
        torch.Tensor, "item",
        lambda self: (_ for _ in ()).throw(
            AssertionError("hot-path device synchronization")))

    activation(torch.tensor([0.25]))
    weight_quantizer(weight)


def test_lsqplus_state_reload_preserves_hard_output():
    source = LSQPlusActivationQuantizer(6, False)
    source.initialize(torch.tensor([-3.0, 5.0]))
    state = source.state_dict()
    target = LSQPlusActivationQuantizer(6, False)
    target.load_state_dict(state)
    current = torch.linspace(-4.0, 6.0, 101)

    torch.testing.assert_close(source(current), target(current))


def test_lsqplus_negative_raw_steps_preserve_positive_quantization_grid():
    activation = LSQPlusActivationQuantizer(6, False)
    activation.initialize(torch.tensor([-2.0, 3.0]))
    weight = torch.tensor([[[[1.0, 0.25]]]])
    weight_quantizer = LSQPlusWeightParametrization(6, 0, weight)
    current = torch.tensor([-1.25, 0.75])
    activation_expected = activation(current).detach()
    weight_expected = weight_quantizer(weight).detach()
    activation.step.data.neg_()
    weight_quantizer.step.data.neg_()

    torch.testing.assert_close(
        activation(current).detach(), activation_expected,
        rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        weight_quantizer(weight).detach(), weight_expected,
        rtol=0.0, atol=0.0)
