import pytest
import torch

from scripts.hardware_aligned_quantization import (
    DynamicGroupedActivationQuantizer,
    GroupedActivationQuantizer,
    symmetric_weight_qdq,
)
from spn_quant.qat.quantizers import (
    ActivationSTEQuantizer,
    PerOutputChannelWeightFakeQuantizer,
)
from spn_quant.activation_boundaries import SignedActivationQuantizer


def test_static_activation_ste_matches_hard_grouped_ptq():
    hard = GroupedActivationQuantizer(
        bits=4,
        minimum=torch.zeros(2),
        maximum=torch.tensor([3.0, 6.0]),
        channel_dim=1,
        group_size=8,
        channels=16,
        unsigned=True,
    )
    quantizer = ActivationSTEQuantizer(hard)
    value = torch.linspace(
        0.0, 7.0, 32).reshape(1, 16, 1, 2).requires_grad_()

    expected, expected_codes = hard.quantize_with_codes(value.detach())
    actual, actual_codes = quantizer.quantize_with_codes(value)

    assert torch.equal(actual.detach(), expected)
    assert torch.equal(actual_codes, expected_codes)
    actual.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_dynamic_activation_ste_keeps_per_sample_hard_values():
    hard = DynamicGroupedActivationQuantizer(
        bits=4, channel_dim=1, group_size=8,
        channels=8, unsigned=False)
    quantizer = ActivationSTEQuantizer(hard)
    value = torch.cat((
        torch.ones(1, 8, 2, 2),
        torch.full((1, 8, 2, 2), 100.0),
    )).requires_grad_()

    expected, expected_codes = hard.quantize_with_codes(value.detach())
    actual, actual_codes = quantizer.quantize_with_codes(value)

    assert torch.equal(actual.detach(), expected)
    assert torch.equal(actual_codes, expected_codes)
    assert not torch.equal(
        hard.scale_for(value.detach())[:1],
        hard.scale_for(value.detach())[1:])
    actual.square().mean().backward()
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("bits", (4, 8))
def test_weight_fake_quant_matches_existing_qdq(bits):
    weight = torch.tensor(
        [[[[3.0, -1.0]]], [[[0.25, 2.0]]]], requires_grad=True)
    quantizer = PerOutputChannelWeightFakeQuantizer(
        bits=bits, channel_dim=0)

    actual = quantizer(weight)
    expected, scale = symmetric_weight_qdq(
        weight.detach(), bits=bits, channel_dim=0)

    assert torch.equal(actual.detach(), expected)
    assert torch.equal(quantizer.scale.detach(), scale)
    actual.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))


def test_structural_activation_ste_declares_group_metadata():
    hard = SignedActivationQuantizer(
        bits=4, channel_maximum=torch.ones(16), group_size=8)

    qat = ActivationSTEQuantizer(hard)

    assert qat.granularity == "group"
    assert not qat.unsigned
    assert qat.scale_count == 2


def test_weight_fake_quant_rejects_unsupported_configuration():
    with pytest.raises(ValueError, match="must be 4, 6, or 8"):
        PerOutputChannelWeightFakeQuantizer(bits=5, channel_dim=0)


def test_activation_ste_rejects_nonfinite_input():
    hard = DynamicGroupedActivationQuantizer(
        bits=4, channel_dim=1, group_size=8,
        channels=8, unsigned=False)
    quantizer = ActivationSTEQuantizer(hard)
    value = torch.zeros(1, 8, 1, 1)
    value[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        quantizer(value)
