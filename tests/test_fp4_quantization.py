import unittest

import torch

from scripts.fp4_quantization import E2M1ActivationQuantizer


class E2M1ActivationQuantizerTest(unittest.TestCase):
    def test_scale_one_reconstructs_complete_signed_codebook(self):
        quantizer = E2M1ActivationQuantizer(torch.tensor(6.0))
        values = torch.tensor([
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        ])

        quantized, codes = quantizer.quantize_with_codes(values)

        torch.testing.assert_close(quantized, values)
        self.assertEqual(codes.dtype, torch.int32)
        self.assertEqual(len(torch.unique(codes)), 15)

    def test_midpoints_use_round_to_nearest_even(self):
        quantizer = E2M1ActivationQuantizer(torch.tensor(6.0))
        values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

        quantized, _ = quantizer.quantize_with_codes(values)

        torch.testing.assert_close(
            quantized,
            torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]))

    def test_saturation_zero_and_negative_symmetry_are_reported(self):
        quantizer = E2M1ActivationQuantizer(torch.tensor(12.0))
        values = torch.tensor([-20.0, -1.0, 0.0, 1.0, 20.0])

        quantized, codes = quantizer.quantize_with_codes(values)

        torch.testing.assert_close(
            quantized, torch.tensor([-12.0, -1.0, 0.0, 1.0, 12.0]))
        self.assertEqual(quantizer.saturated_count(values), 2)
        self.assertEqual(quantizer.zero_code_count(codes), 1)

    def test_per_channel_scale_uses_requested_axis(self):
        quantizer = E2M1ActivationQuantizer(
            torch.tensor([6.0, 60.0]), channel_dim=1)
        values = torch.tensor([[[[1.0]], [[10.0]]]])

        quantized, _ = quantizer.quantize_with_codes(values)

        torch.testing.assert_close(quantized, values)
        torch.testing.assert_close(
            quantizer.scale, torch.tensor([1.0, 10.0]))

    def test_zero_range_uses_unit_scale_and_nonfinite_range_is_rejected(self):
        quantizer = E2M1ActivationQuantizer(torch.tensor(0.0))

        self.assertEqual(float(quantizer.scale), 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            E2M1ActivationQuantizer(torch.tensor(float("nan")))


if __name__ == "__main__":
    unittest.main()
