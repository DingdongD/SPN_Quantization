import unittest

import torch
import torch.nn as nn

from spn_quant.im2col_diagnostics import (
    ConvIm2ColLayout,
    local_output_error,
)


class ConvIm2ColLayoutTest(unittest.TestCase):
    def setUp(self):
        self.module = nn.Conv2d(
            2, 3, kernel_size=(2, 3), stride=(2, 1),
            padding=(1, 0), bias=True)
        with torch.no_grad():
            self.module.weight.copy_(torch.arange(
                self.module.weight.numel(), dtype=torch.float32
            ).reshape_as(self.module.weight) / 20.0)
            self.module.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        self.inputs = torch.arange(
            2 * 2 * 5 * 6, dtype=torch.float32).reshape(2, 2, 5, 6) / 10.0

    def test_matrix_product_matches_asymmetric_conv2d(self):
        layout = ConvIm2ColLayout.from_module("conv", self.module)

        patches = layout.unfold(self.inputs)
        weights = layout.flatten_weight(self.module.weight)
        output_height, output_width = layout.output_shape(self.inputs)
        matrix_output = torch.einsum("bkm,ok->bom", patches, weights)
        matrix_output = matrix_output.reshape(
            self.inputs.shape[0], self.module.out_channels,
            output_height, output_width)
        matrix_output += self.module.bias.reshape(1, -1, 1, 1)

        torch.testing.assert_close(matrix_output, self.module(self.inputs))

    def test_k_order_is_channel_then_kernel_row_then_column(self):
        layout = ConvIm2ColLayout.from_module("conv", self.module)

        self.assertEqual(layout.decode_k(0), (0, 0, 0))
        self.assertEqual(layout.decode_k(5), (0, 1, 2))
        self.assertEqual(layout.decode_k(6), (1, 0, 0))
        self.assertEqual(layout.decode_k(11), (1, 1, 2))

    def test_rejects_grouped_and_transposed_convolution(self):
        grouped = nn.Conv2d(4, 4, kernel_size=3, groups=2)
        transposed = nn.ConvTranspose2d(2, 3, kernel_size=3)

        with self.assertRaises(ValueError):
            ConvIm2ColLayout.from_module("grouped", grouped)
        with self.assertRaises(TypeError):
            ConvIm2ColLayout.from_module("transposed", transposed)


class LocalOutputErrorTest(unittest.TestCase):
    def test_reports_exact_per_token_joint_weight_activation_error(self):
        fp_patches = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        quantized_patches = torch.tensor([[[1.0, 1.5], [2.5, 4.0]]])
        fp_weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        quantized_weight = torch.tensor([[1.0, 1.5], [-0.5, 0.5]])
        fp_output = torch.einsum("bkm,ok->bom", fp_patches, fp_weight)
        quantized_output = torch.einsum(
            "bkm,ok->bom", quantized_patches, quantized_weight)
        expected = (quantized_output - fp_output).square().sum(dim=1)

        actual = local_output_error(
            fp_patches, quantized_patches, fp_weight, quantized_weight)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
