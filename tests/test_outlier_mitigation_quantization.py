import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts import outlier_mitigation_quantization as mitigation


class OutlierMitigationPrimitiveTest(unittest.TestCase):
    def test_percentile_overrides_merge_call_sites_conservatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "activation_percentiles.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "module", "call_index", "kind", "p99_9"])
                writer.writeheader()
                writer.writerows([
                    {"module": "linear", "call_index": 0,
                     "kind": "input", "p99_9": 2.0},
                    {"module": "linear", "call_index": 1,
                     "kind": "input", "p99_9": 5.0},
                ])

            overrides = mitigation.load_percentile_overrides(tmp, "p99_9")

            self.assertEqual(overrides[("linear", "input")], 5.0)

    def test_smoothquant_linear_transform_preserves_fp32_output(self):
        torch.manual_seed(3)
        activation = torch.randn(2, 5, 4)
        weight = torch.randn(3, 4)
        bias = torch.randn(3)
        activation_absmax = activation.abs().amax(dim=(0, 1))

        scale = mitigation.smoothquant_scale(
            weight, activation_absmax, alpha=0.5, input_channel_dim=1)
        transformed_weight = mitigation.apply_input_scale_to_weight(
            weight, scale, input_channel_dim=1)
        transformed_activation = activation / scale

        torch.testing.assert_close(
            F.linear(transformed_activation, transformed_weight, bias),
            F.linear(activation, weight, bias), atol=1e-5, rtol=1e-5)
        self.assertEqual(tuple(scale.shape), (4,))

    def test_smoothquant_conv_transform_preserves_fp32_output(self):
        torch.manual_seed(5)
        activation = torch.randn(1, 3, 8, 8)
        weight = torch.randn(4, 3, 3, 3)
        activation_absmax = activation.abs().amax(dim=(0, 2, 3))

        scale = mitigation.smoothquant_scale(
            weight, activation_absmax, alpha=0.25, input_channel_dim=1)
        transformed_weight = mitigation.apply_input_scale_to_weight(
            weight, scale, input_channel_dim=1)

        torch.testing.assert_close(
            F.conv2d(activation / scale.reshape(1, -1, 1, 1),
                     transformed_weight, padding=1),
            F.conv2d(activation, weight, padding=1), atol=1e-4, rtol=1e-5)

    def test_smoothquant_conv_transpose_uses_input_channel_dimension_zero(self):
        torch.manual_seed(9)
        activation = torch.randn(1, 3, 6, 6)
        weight = torch.randn(3, 2, 3, 3)
        activation_absmax = activation.abs().amax(dim=(0, 2, 3))

        scale = mitigation.smoothquant_scale(
            weight, activation_absmax, alpha=0.25, input_channel_dim=0)
        transformed_weight = mitigation.apply_input_scale_to_weight(
            weight, scale, input_channel_dim=0)

        torch.testing.assert_close(
            F.conv_transpose2d(
                activation / scale.reshape(1, -1, 1, 1),
                transformed_weight, padding=1),
            F.conv_transpose2d(activation, weight, padding=1),
            atol=1e-4, rtol=1e-5)

    def test_awq_clipping_changes_range_before_w4_rounding(self):
        weight = torch.tensor([[[[0.1, 1.0, 10.0]]]])

        quantized, scale = mitigation.clipped_symmetric_weight_qdq(
            weight, bits=4, clip_ratio=0.8)

        self.assertAlmostEqual(float(scale.reshape(-1)[0]), 8.0 / 7.0, places=6)
        self.assertEqual(float(quantized.max()), 8.0)


class ChannelMaximaLoaderTest(unittest.TestCase):
    def test_loader_merges_repeated_input_calls_by_channel_maximum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez_compressed(
                str(root / "channel_absmax.npz"),
                site_00000=np.array([1.0, 5.0], dtype=np.float32),
                site_00001=np.array([3.0, 2.0], dtype=np.float32),
            )
            with (root / "channel_absmax_index.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "key", "module", "kind"])
                writer.writeheader()
                writer.writerows([
                    {"key": "site_00000", "module": "linear", "kind": "input"},
                    {"key": "site_00001", "module": "linear", "kind": "input"},
                ])

            maxima = mitigation.load_input_channel_maxima(root)

            torch.testing.assert_close(
                maxima["linear"], torch.tensor([3.0, 5.0]))


if __name__ == "__main__":
    unittest.main()
