import unittest

import torch
import torch.nn as nn

from scripts.hardware_aligned_quantization import SymmetricActivationQuantizer
from spn_quant.im2col_diagnostics import (
    CSPNW8A8Im2ColRecorder,
    ConvIm2ColAccumulator,
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


class ConvIm2ColAccumulatorTest(unittest.TestCase):
    def setUp(self):
        self.module = nn.Conv2d(
            2, 2, kernel_size=(1, 2), padding=(0, 1), bias=False)
        with torch.no_grad():
            self.module.weight.copy_(torch.tensor([
                [[[0.5, -0.25]], [[1.0, 0.5]]],
                [[[-0.5, 0.75]], [[0.25, -1.0]]],
            ]))
        self.layout = ConvIm2ColLayout.from_module("conv", self.module)
        self.reference = torch.tensor([[[
            [0.001, 0.25, -0.75],
            [0.5, -0.125, 1.0],
        ], [
            [0.375, -0.001, 0.625],
            [-1.0, 0.125, -0.5],
        ]]])
        self.quantizer = SymmetricActivationQuantizer(8, 1.0)
        self.quantized, self.codes = self.quantizer.quantize_with_codes(
            self.reference)
        self.quantized_weight = torch.round(
            self.module.weight.detach() * 64.0) / 64.0

    def _accumulator(self, chunk):
        accumulator = ConvIm2ColAccumulator(
            self.layout, percentile_capacity=32, token_topk=3)
        accumulator.update(
            self.reference, self.quantized, self.codes, self.quantizer,
            self.module.weight.detach(), self.quantized_weight,
            sample_index=17, token_chunk=chunk)
        return accumulator

    def test_chunk_size_preserves_exact_statistics(self):
        left = self._accumulator(1)
        right = self._accumulator(7)

        self.assertEqual(left.exact_state(), right.exact_state())
        self.assertEqual(left.top_token_rows(), right.top_token_rows())

    def test_channel_offset_rows_preserve_k_order_and_layer_energy(self):
        accumulator = self._accumulator(2)

        rows = accumulator.channel_offset_rows()
        layer = accumulator.layer_row()

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [(row["channel"], row["kernel_row"], row["kernel_col"])
             for row in rows],
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)])
        self.assertAlmostEqual(
            sum(row["activation_signal_energy"] for row in rows),
            layer["activation_signal_energy"])
        self.assertAlmostEqual(
            sum(row["activation_error_energy"] for row in rows),
            layer["activation_error_energy"])
        self.assertGreater(layer["activation_new_zero_count"], 0)
        self.assertAlmostEqual(
            rows[0]["activation_rms"],
            (rows[0]["activation_signal_energy"] /
             rows[0]["elements"]) ** 0.5)
        self.assertGreater(rows[0]["activation_mean_abs"], 0.0)
        self.assertAlmostEqual(
            rows[0]["weight_rms"],
            (rows[0]["weight_signal_energy"] /
             rows[0]["weight_elements"]) ** 0.5)

    def test_spatial_arrays_match_direct_local_output_error(self):
        accumulator = self._accumulator(3)
        arrays = accumulator.spatial_arrays(17)
        fp_patches = self.layout.unfold(self.reference)
        quantized_patches = self.layout.unfold(self.quantized)
        expected = local_output_error(
            fp_patches, quantized_patches,
            self.layout.flatten_weight(self.module.weight.detach()),
            self.layout.flatten_weight(self.quantized_weight))
        height, width = self.layout.output_shape(self.reference)

        torch.testing.assert_close(
            torch.from_numpy(arrays["local_output_error"]),
            expected[0].reshape(height, width))
        self.assertAlmostEqual(
            accumulator.layer_row()["local_output_error_energy"],
            float(expected.to(torch.float64).sum().item()))
        self.assertGreater(
            accumulator.layer_row()["local_output_signal_energy"], 0.0)
        self.assertEqual(arrays["patch_rms"].shape, (height, width))
        self.assertEqual(arrays["activation_new_zero_count"].shape,
                         (height, width))

    def test_channel_and_offset_reductions_cover_every_cell(self):
        accumulator = self._accumulator(3)

        channels = accumulator.channel_rows()
        offsets = accumulator.offset_rows()

        self.assertEqual([row["channel"] for row in channels], [0, 1])
        self.assertEqual(
            [(row["kernel_row"], row["kernel_col"]) for row in offsets],
            [(0, 0), (0, 1)])
        self.assertAlmostEqual(
            sum(row["activation_error_energy"] for row in channels),
            sum(row["activation_error_energy"] for row in offsets))


class CSPNW8A8Im2ColRecorderTest(unittest.TestCase):
    def setUp(self):
        self.first = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.second = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.linear = nn.Linear(2, 2)
        self.modules = {
            "first": self.first,
            "second": self.second,
            "linear": self.linear,
        }
        self.original = dict(
            (name, module.weight.detach().cpu().clone())
            for name, module in self.modules.items())
        self.quantizer = SymmetricActivationQuantizer(8, 1.0)

    def _record(self, recorder, name, tensor):
        quantized, codes = self.quantizer.quantize_with_codes(tensor)
        recorder.record(
            name, "input", 0, "encoder", tensor, quantized, codes,
            self.quantizer, 1)

    def test_validates_sample_coverage_and_excludes_linear(self):
        recorder = CSPNW8A8Im2ColRecorder(
            self.modules, self.original, percentile_capacity=16,
            token_topk=2, token_chunk=4)
        recorder.begin_sample(5)

        self._record(recorder, "first", torch.ones(1, 1, 2, 2))
        self._record(recorder, "second", torch.ones(1, 2, 2, 2))
        arrays = recorder.end_sample()

        self.assertEqual(set(arrays), {"first", "second"})
        self.assertEqual(
            [row["module"] for row in recorder.module_manifest_rows()],
            ["first", "linear", "second"])
        excluded = dict(
            (row["module"], row["status"])
            for row in recorder.module_manifest_rows())
        self.assertEqual(excluded["linear"], "excluded_linear")

    def test_rejects_missing_or_repeated_conv_input(self):
        recorder = CSPNW8A8Im2ColRecorder(
            self.modules, self.original, percentile_capacity=16,
            token_topk=2, token_chunk=4)
        recorder.begin_sample(5)
        self._record(recorder, "first", torch.ones(1, 1, 2, 2))

        with self.assertRaises(RuntimeError):
            recorder.end_sample()

        quantized, codes = self.quantizer.quantize_with_codes(
            torch.ones(1, 2, 2, 2))
        with self.assertRaises(ValueError):
            recorder.record(
                "second", "input", 1, "encoder", quantized, quantized,
                codes, self.quantizer, 1)


if __name__ == "__main__":
    unittest.main()
