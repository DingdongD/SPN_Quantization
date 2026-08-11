import copy

import pytest
import torch

from spn_quant.qdrop_activation import (
    ExactActivationQuantizer,
    QDropActivationQuantizer,
)


def make_signed(seed=7):
    return QDropActivationQuantizer(
        site="activation::conv#0",
        bits=4,
        signed=True,
        symmetric=True,
        scale_minimum=1.0e-8,
        seed=seed)


def make_unsigned(seed=11):
    return QDropActivationQuantizer(
        site="activation::relu#0",
        bits=4,
        signed=False,
        symmetric=False,
        scale_minimum=1.0e-8,
        seed=seed)


def test_signed_symmetric_a4_has_fixed_zero_point():
    quantizer = make_signed()
    values = torch.tensor([-2.0, 0.0, 2.0])
    quantizer.initialize(values)
    quantizer.start_reconstruction(quant_probability=1.0)

    quantized, codes = quantizer.quantize_with_codes(values)

    assert codes.tolist() == [-7, 0, 7]
    assert quantizer.qmin == -7
    assert quantizer.qmax == 7
    assert quantizer.zero_point_parameter is None
    torch.testing.assert_close(quantized, values)


def test_unsigned_relu_a4_preserves_zero():
    quantizer = make_unsigned()
    values = torch.tensor([0.0, 1.0, 3.0])
    quantizer.initialize(values)
    quantizer.start_reconstruction(quant_probability=1.0)

    quantized, codes = quantizer.quantize_with_codes(values)

    assert codes.tolist() == [0, 5, 15]
    assert codes.dtype == torch.uint8
    assert quantized[0].item() == 0.0


def test_asymmetric_zero_point_receives_gradient():
    quantizer = QDropActivationQuantizer(
        site="activation::affine#0",
        bits=4,
        signed=True,
        symmetric=False,
        scale_minimum=1.0e-8,
        seed=13)
    values = torch.tensor([-1.1, -0.2, 0.7, 2.3])
    quantizer.initialize(values)
    quantizer.start_reconstruction(quant_probability=1.0)

    quantized, _ = quantizer.quantize_with_codes(values)
    quantized.square().sum().backward()

    assert quantizer.scale_parameter.grad is not None
    assert quantizer.zero_point_parameter.grad is not None
    assert bool(torch.isfinite(quantizer.scale_parameter.grad).all().item())
    assert bool(torch.isfinite(
        quantizer.zero_point_parameter.grad).all().item())


@pytest.mark.parametrize("probability", (-0.1, 1.1))
def test_rejects_invalid_quantization_probability(probability):
    quantizer = make_signed()
    quantizer.initialize(torch.tensor([-1.0, 1.0]))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quantizer.start_reconstruction(probability)


def test_probability_zero_returns_original_values():
    quantizer = make_signed()
    values = torch.tensor([-0.8, -0.1, 0.2, 0.9])
    quantizer.initialize(values)
    quantizer.start_reconstruction(quant_probability=0.0)

    output, _ = quantizer.quantize_with_codes(values)

    torch.testing.assert_close(output, values)


def test_probability_one_returns_quantized_values():
    quantizer = make_signed()
    calibration = torch.tensor([-1.0, 1.0])
    values = torch.tensor([-0.8, -0.1, 0.2, 0.9])
    quantizer.initialize(calibration)
    quantizer.start_reconstruction(quant_probability=1.0)

    output, codes = quantizer.quantize_with_codes(values)
    scale = quantizer.scale_parameter.detach().abs().clamp_min(1.0e-8)
    expected = codes.to(values.dtype) * scale

    torch.testing.assert_close(output, expected)


def test_half_probability_matches_official_elementwise_mask():
    quantizer = make_signed(seed=17)
    calibration = torch.tensor([-1.0, 1.0])
    values = torch.tensor([-0.91, -0.47, -0.12, 0.18, 0.44, 0.83])
    quantizer.initialize(calibration)
    quantizer.start_reconstruction(quant_probability=0.5)

    output, codes = quantizer.quantize_with_codes(values)
    scale = quantizer.scale_parameter.detach().abs().clamp_min(1.0e-8)
    deterministic = codes.to(values.dtype) * scale
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    mask = torch.rand(
        values.shape, dtype=torch.float32,
        generator=generator) < 0.5

    torch.testing.assert_close(
        output, torch.where(mask, deterministic, values))


def test_same_seed_reproduces_masks_and_different_seed_changes_them():
    values = torch.linspace(-0.97, 0.97, steps=128)
    outputs = []
    for seed in (23, 23, 29):
        quantizer = make_signed(seed=seed)
        quantizer.initialize(torch.tensor([-1.0, 1.0]))
        quantizer.start_reconstruction(quant_probability=0.5)
        outputs.append(quantizer.quantize_with_codes(values)[0])

    torch.testing.assert_close(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])


def test_freeze_disables_randomness_and_contract_round_trips():
    quantizer = make_unsigned(seed=31)
    values = torch.tensor([0.0, 0.4, 1.2, 2.0])
    quantizer.initialize(values)
    quantizer.start_reconstruction(quant_probability=0.5)
    quantizer.quantize_with_codes(values)
    quantizer.freeze()

    first, first_codes = quantizer.quantize_with_codes(values)
    second, second_codes = quantizer.quantize_with_codes(values)
    replay = ExactActivationQuantizer.from_contract(quantizer.contract())
    replayed, replayed_codes = replay.quantize_with_codes(values)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, replayed)
    torch.testing.assert_close(first_codes, second_codes)
    torch.testing.assert_close(first_codes, replayed_codes)
    assert not quantizer.scale_parameter.requires_grad
    assert not quantizer.zero_point_parameter.requires_grad


def test_contract_tampering_is_rejected():
    quantizer = make_signed(seed=37)
    quantizer.initialize(torch.tensor([-1.0, 1.0]))
    quantizer.start_reconstruction(quant_probability=1.0)
    quantizer.freeze()
    contract = copy.deepcopy(quantizer.contract())
    contract["scale"] = contract["scale"] * 2.0

    with pytest.raises(RuntimeError, match="fingerprint"):
        ExactActivationQuantizer.from_contract(contract)


def test_nonfinite_calibration_is_rejected():
    quantizer = make_signed()

    with pytest.raises(ValueError, match="finite"):
        quantizer.initialize(torch.tensor([0.0, float("inf")]))
