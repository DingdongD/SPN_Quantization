import pytest
import torch

from spn_quant.fp_formats import (
    FP4E2M1Quantizer,
    FP6E3M2Quantizer,
    FP8E4M3FNQuantizer,
    FORMAT_SPECS,
    make_quantizer,
)


def test_fp4_e2m1_uses_the_declared_codebook():
    quantizer = FP4E2M1Quantizer(torch.tensor(6.0))
    values, codes = quantizer.quantize_with_codes(
        torch.tensor([-7.0, -3.2, -1.2, -0.2, 0.2, 1.2, 3.2, 7.0]))
    assert torch.equal(codes, torch.tensor([-7, -5, -2, 0, 0, 2, 5, 7], dtype=torch.int8))
    assert torch.equal(values, torch.tensor([-6.0, -3.0, -1.0, 0.0, 0.0, 1.0, 3.0, 6.0]))


def test_fp8_e4m3fn_round_trip_uses_float8_e4m3fn():
    quantizer = FP8E4M3FNQuantizer(torch.tensor(448.0))
    values, codes = quantizer.quantize_with_codes(
        torch.tensor([-448.0, -1.0, -0.5, 0.5, 1.0, 448.0]))
    expected = (torch.tensor([-448.0, -1.0, -0.5, 0.5, 1.0, 448.0])
                .to(torch.float8_e4m3fn).to(torch.float32))
    assert torch.equal(values, expected)
    assert codes.dtype == torch.uint8


def test_fp6_e3m2_uses_the_declared_finite_codebook():
    quantizer = FP6E3M2Quantizer(torch.tensor(28.0))
    values, codes = quantizer.quantize_with_codes(
        torch.tensor([-28.0, -10.0, -3.2, -0.2, 0.2, 3.2, 10.0, 28.0]))
    assert torch.equal(
        codes,
        torch.tensor([-31, -25, -18, -3, 3, 18, 25, 31], dtype=torch.int8),
    )
    assert torch.equal(
        values,
        torch.tensor([-28.0, -10.0, -3.0, -0.1875, 0.1875, 3.0, 10.0, 28.0]),
    )


def test_format_calibration_rejects_nonfinite_and_zero_scale():
    with pytest.raises(ValueError, match="finite"):
        FP4E2M1Quantizer(torch.tensor(float("inf")))
    with pytest.raises(ValueError, match="positive"):
        FP8E4M3FNQuantizer(torch.tensor(0.0))


def test_format_specs_are_explicit_and_immutable():
    assert FORMAT_SPECS["fp4_e2m1"].bits == 4
    assert FORMAT_SPECS["fp6_e3m2"].bits == 6
    assert FORMAT_SPECS["fp8_e4m3fn"].bits == 8
    assert FORMAT_SPECS["fp4_e2m1"].maximum == 6.0
    assert FORMAT_SPECS["fp6_e3m2"].maximum == 28.0
    assert FORMAT_SPECS["fp8_e4m3fn"].maximum == 448.0


def test_fp_format_supports_head_broadcast_scales():
    quantizer = FP4E2M1Quantizer(
        torch.tensor([1.0, 2.0]), broadcast_shape=(1, 2, 1, 1))
    tensor = torch.tensor([[[[1.0]], [[2.0]]]])
    reconstructed, _ = quantizer.quantize_with_codes(tensor)
    assert reconstructed.shape == tensor.shape
    assert torch.isfinite(reconstructed).all()


def test_fp16_ieee_uses_explicit_finite_cast_qdq():
    quantizer = make_quantizer("fp16_ieee", torch.tensor(10.0))
    tensor = torch.tensor([0.0, 1.0001, -2.0001, 70000.0])
    reconstructed, codes = quantizer.quantize_with_codes(tensor)
    expected = tensor.clamp(-65504.0, 65504.0).half().float()
    torch.testing.assert_close(reconstructed, expected, rtol=0.0, atol=0.0)
    assert quantizer.bits == 16
    assert quantizer.scale.item() == 1.0
    assert quantizer.numel == 4
    assert quantizer.zero_codes == 1
    assert quantizer.saturated == 1
    assert codes.shape == tensor.shape
    assert FORMAT_SPECS["fp16_ieee"].maximum == 65504.0


def test_bf16_uses_explicit_cast_qdq_without_calibrated_scaling():
    quantizer = make_quantizer("bf16", torch.tensor(10.0))
    tensor = torch.tensor([0.0, 1.003, -2.007, 1.0e20])

    reconstructed, codes = quantizer.quantize_with_codes(tensor)

    expected = tensor.to(torch.bfloat16).float()
    torch.testing.assert_close(reconstructed, expected, rtol=0.0, atol=0.0)
    assert quantizer.bits == 16
    assert quantizer.scale.item() == 1.0
    assert quantizer.numel == 4
    assert quantizer.zero_codes == 1
    assert quantizer.saturated == 0
    assert codes.shape == tensor.shape
    assert FORMAT_SPECS["bf16"].bits == 16
