import unittest

import torch

from spn_quant.integer_ops import (
    add_requantized_int32,
    batched_int8_mm_int32,
    batched_uint8_int8_mm_int32,
    int8_mm_int32,
    integer_im2col,
    quantize_signed,
    quantize_unsigned,
    requantize_int32,
    uint8_int8_mm_int32,
)


class IntegerCodeTest(unittest.TestCase):
    def test_signed_codes_use_symmetric_a4_domain(self):
        codes, scale = quantize_signed(
            torch.tensor([-2.0, 0.0, 2.0]), bits=4, maximum=2.0)

        self.assertEqual(codes.dtype, torch.int8)
        self.assertEqual(codes.tolist(), [-7, 0, 7])
        self.assertAlmostEqual(scale, 2.0 / 7.0)

    def test_unsigned_codes_use_full_a8_domain(self):
        codes, scale = quantize_unsigned(
            torch.tensor([0.0, 0.5, 1.0]), bits=8, maximum=1.0)

        self.assertEqual(codes.dtype, torch.uint8)
        self.assertEqual(codes.tolist(), [0, 128, 255])
        self.assertAlmostEqual(scale, 1.0 / 255.0)

    def test_quantizers_reject_nonfinite_or_negative_maximum(self):
        with self.assertRaises(ValueError):
            quantize_signed(torch.ones(1), bits=4, maximum=float("inf"))
        with self.assertRaises(ValueError):
            quantize_unsigned(torch.ones(1), bits=8, maximum=-1.0)


class RequantizationTest(unittest.TestCase):
    def test_independent_branch_codes_requantize_before_int32_add(self):
        left_codes = torch.tensor([1, 2], dtype=torch.int32)
        right_codes = torch.tensor([10, 20], dtype=torch.int32)

        output = add_requantized_int32(
            ((left_codes, 0.1), (right_codes, 1.0)),
            output_scale=0.25, qmin=-127, qmax=127)

        self.assertEqual(output.dtype, torch.int32)
        self.assertEqual(output.tolist(), [40, 81])

    def test_branch_add_saturates_only_after_wide_accumulation(self):
        first = torch.tensor([100], dtype=torch.int32)
        second = torch.tensor([-100], dtype=torch.int32)

        output = add_requantized_int32(
            ((first, 1.0), (second, 1.0)),
            output_scale=1.0, qmin=-7, qmax=7)

        self.assertEqual(output.tolist(), [0])

    def test_q31_requantization_is_signed_and_saturating(self):
        values = torch.tensor([-30, -3, 3, 30], dtype=torch.int32)

        output = requantize_int32(
            values, torch.tensor(0.5), torch.tensor(1.0), -7, 7)

        self.assertEqual(output.dtype, torch.int32)
        self.assertEqual(output.tolist(), [-7, -2, 2, 7])

    def test_q31_requantization_broadcasts_per_channel_scale(self):
        values = torch.tensor([[2, 2], [4, 4]], dtype=torch.int32)

        output = requantize_int32(
            values,
            torch.tensor([[0.5], [0.25]]),
            torch.tensor([[1.0], [1.0]]),
            -127, 127)

        self.assertEqual(output.tolist(), [[1, 1], [1, 1]])

    def test_q31_requantization_rejects_zero_scale(self):
        with self.assertRaises(ValueError):
            requantize_int32(
                torch.ones(1, dtype=torch.int32),
                torch.tensor(0.0), torch.tensor(1.0), -7, 7)


class IntegerMatrixMultiplicationTest(unittest.TestCase):
    def test_int8_mm_returns_exact_int32_dot_products(self):
        left = torch.tensor([[1, -2, 3]], dtype=torch.int8)
        right = torch.tensor([[4], [5], [-6]], dtype=torch.int8)

        output = int8_mm_int32(left, right)

        self.assertEqual(output.dtype, torch.int32)
        self.assertEqual(int(output.item()), -24)

    def test_shifted_u8_int8_mm_matches_unsigned_reference(self):
        probability = torch.tensor([[0, 128, 255]], dtype=torch.uint8)
        value = torch.tensor([[2], [-3], [4]], dtype=torch.int8)
        expected = probability.to(torch.int32) @ value.to(torch.int32)

        actual = uint8_int8_mm_int32(probability, value)

        torch.testing.assert_close(actual, expected)

    def test_batched_int8_mm_preserves_leading_dimensions(self):
        left = torch.tensor(
            [[[[1, 2], [3, 4]]], [[[2, 1], [1, 2]]]], dtype=torch.int8)
        right = torch.tensor(
            [[[[1, 0], [0, 1]]], [[[1, 2], [3, 4]]]], dtype=torch.int8)
        expected = torch.stack([
            left[index, 0].to(torch.int32) @
            right[index, 0].to(torch.int32)
            for index in range(2)
        ]).unsqueeze(1)

        actual = batched_int8_mm_int32(left, right)

        torch.testing.assert_close(actual, expected)

    def test_batched_unsigned_mm_matches_integer_reference(self):
        left = torch.tensor(
            [[[[0, 255], [128, 1]]]], dtype=torch.uint8)
        right = torch.tensor(
            [[[[2, -1], [3, 4]]]], dtype=torch.int8)
        expected = (
            left[0, 0].to(torch.int32) @ right[0, 0].to(torch.int32)
        ).reshape(1, 1, 2, 2)

        actual = batched_uint8_int8_mm_int32(left, right)

        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_int_mm_zero_pads_non_aligned_official_shapes(self):
        left = torch.arange(15 * 7, device="cuda", dtype=torch.int8).reshape(
            15, 7) % 11 - 5
        right = torch.arange(7 * 13, device="cuda", dtype=torch.int8).reshape(
            7, 13) % 7 - 3
        expected = left.cpu().to(torch.int32) @ right.cpu().to(torch.int32)

        actual = int8_mm_int32(left, right)

        torch.testing.assert_close(actual.cpu(), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_unsigned_av_pads_non_aligned_token_dimension(self):
        probability = torch.arange(
            19 * 21, device="cuda", dtype=torch.int64).reshape(19, 21)
        probability = (probability % 256).to(torch.uint8)
        value = torch.arange(
            21 * 7, device="cuda", dtype=torch.int64).reshape(21, 7)
        value = (value % 15 - 7).to(torch.int8)
        expected = probability.cpu().to(torch.int32) @ \
            value.cpu().to(torch.int32)

        actual = uint8_int8_mm_int32(probability, value)

        torch.testing.assert_close(actual.cpu(), expected)


class IntegerIm2ColTest(unittest.TestCase):
    def test_integer_im2col_preserves_patch_order(self):
        tensor = torch.tensor(
            [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.int8)

        patches, output_shape = integer_im2col(
            tensor, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0),
            dilation=(1, 1))

        self.assertEqual(output_shape, (2, 2))
        self.assertEqual(patches.dtype, torch.int8)
        self.assertEqual(patches.shape, (4, 4))
        self.assertEqual(patches[0].tolist(), [1, 2, 4, 5])
        self.assertEqual(patches[-1].tolist(), [5, 6, 8, 9])

    def test_integer_im2col_rejects_dilation(self):
        with self.assertRaises(ValueError):
            integer_im2col(
                torch.ones(1, 1, 3, 3, dtype=torch.int8),
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                dilation=(2, 2))


if __name__ == "__main__":
    unittest.main()
