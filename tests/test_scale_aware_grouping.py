import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from spn_quant.scale_aware_grouping import (
    build_scale_aware_grouping,
    grouped_channel_maximum,
    inverse_permutation,
    permute_activation,
    permute_input_weight,
)


class ScaleAwareGroupingTest(unittest.TestCase):
    def test_stable_rms_order_forms_exact_groups(self):
        rms = torch.tensor([
            1.0, 100.0, 2.0, 90.0, 3.0, 80.0, 4.0, 70.0,
            5.0, 60.0, 6.0, 50.0, 7.0, 40.0, 8.0, 30.0,
        ])

        grouping = build_scale_aware_grouping(
            rms, group_size=8, epsilon=1e-12)

        self.assertEqual(grouping.permutation.tolist(), [
            0, 2, 4, 6, 8, 10, 12, 14,
            15, 13, 11, 9, 7, 5, 3, 1,
        ])
        self.assertEqual(tuple(grouping.group_rms.shape), (2, 8))
        self.assertLess(
            float(grouping.scale_aware_dispersion.sum()),
            float(grouping.contiguous_dispersion.sum()))

    def test_zero_and_tied_rms_preserve_channel_index_order(self):
        rms = torch.tensor([
            0.0, 0.0, 2.0, 1.0, 1.0, 3.0, 2.0, 3.0,
        ])

        grouping = build_scale_aware_grouping(
            rms, group_size=8, epsilon=1e-12)

        self.assertEqual(
            grouping.permutation.tolist(), [0, 1, 3, 4, 2, 6, 5, 7])
        self.assertTrue(bool(torch.isfinite(
            grouping.scale_aware_dispersion).all()))

    def test_inverse_permutation_restores_original_channel_order(self):
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = inverse_permutation(permutation)
        values = torch.arange(4).reshape(1, 4, 1, 1)

        permuted = permute_activation(values, permutation, channel_dim=1)
        restored = permute_activation(permuted, inverse, channel_dim=1)

        torch.testing.assert_close(restored, values)

    def test_grouped_maximum_uses_rms_permutation(self):
        maximum = torch.tensor([1.0, 20.0, 3.0, 40.0])
        permutation = torch.tensor([0, 2, 1, 3])

        grouped = grouped_channel_maximum(
            maximum, permutation, group_size=2)

        torch.testing.assert_close(grouped, torch.tensor([3.0, 40.0]))

    def test_conv2d_weight_permutation_preserves_fp_output(self):
        torch.manual_seed(41)
        module = nn.Conv2d(4, 3, 3, padding=1, bias=True).eval()
        activation = torch.randn(2, 4, 5, 5)
        permutation = torch.tensor([2, 0, 3, 1])
        reference = module(activation)

        permuted_activation = permute_activation(
            activation, permutation, channel_dim=1)
        permuted_weight = permute_input_weight(
            module, module.weight, permutation)
        candidate = F.conv2d(
            permuted_activation, permuted_weight, module.bias,
            module.stride, module.padding, module.dilation, module.groups)

        torch.testing.assert_close(candidate, reference)

    def test_conv_transpose2d_weight_permutation_preserves_fp_output(self):
        torch.manual_seed(42)
        module = nn.ConvTranspose2d(
            4, 3, 3, padding=1, bias=True).eval()
        activation = torch.randn(2, 4, 5, 5)
        permutation = torch.tensor([2, 0, 3, 1])
        reference = module(activation)

        permuted_activation = permute_activation(
            activation, permutation, channel_dim=1)
        permuted_weight = permute_input_weight(
            module, module.weight, permutation)
        candidate = F.conv_transpose2d(
            permuted_activation, permuted_weight, module.bias,
            module.stride, module.padding, module.output_padding,
            module.groups, module.dilation)

        torch.testing.assert_close(candidate, reference)

    def test_invalid_grouping_inputs_fail_directly(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            build_scale_aware_grouping(
                torch.tensor([0.0] * 7 + [float("nan")]), 8, 1e-12)
        with self.assertRaisesRegex(ValueError, "divide"):
            build_scale_aware_grouping(torch.ones(7), 8, 1e-12)
        with self.assertRaisesRegex(ValueError, "bijection"):
            inverse_permutation(torch.tensor([0, 0, 1, 2]))
        with self.assertRaisesRegex(TypeError, "Conv"):
            permute_input_weight(
                nn.Linear(4, 2), torch.ones(2, 4), torch.arange(4))


if __name__ == "__main__":
    unittest.main()
