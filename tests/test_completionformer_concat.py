import unittest

import torch
import torch.nn.functional as F

from spn_quant.completionformer_concat import (
    SplitConcatConvController,
    split_concat_branches,
)


def make_module(bias=False):
    module = torch.nn.Conv2d(4, 2, kernel_size=1, bias=bias).eval()
    with torch.no_grad():
        module.weight.copy_(torch.tensor([
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            [[[-1.0]], [[1.0]], [[2.0]], [[-2.0]]],
        ]))
        if bias:
            module.bias.copy_(torch.tensor([0.5, -0.25]))
    return module


def make_controller(module, clip_factors=(1.0,)):
    return SplitConcatConvController(
        name="backbone.former.block1.0.concat_conv",
        module=module,
        branch_channels=2,
        weight_bits=4,
        activation_bits=4,
        output_bits=4,
        clip_factors=clip_factors,
        search_rounds=1,
        cache_sample_limit=2,
        cache_byte_limit=1 << 20)


def calibration_input():
    transformer = torch.tensor([[[[0.25, 0.5]], [[1.0, -0.5]]]])
    cnn = torch.tensor([[[[4.0, 8.0]], [[-2.0, 6.0]]]])
    return torch.cat((transformer, cnn), dim=1)


class SplitConcatCalibrationTest(unittest.TestCase):
    def test_split_preserves_official_channel_order(self):
        merged = calibration_input()

        transformer, cnn = split_concat_branches(merged, channels=2)

        self.assertEqual(transformer.tolist(), merged[:, :2].tolist())
        self.assertEqual(cnn.tolist(), merged[:, 2:].tolist())

    def test_freeze_keeps_branch_scales_independent(self):
        module = make_module()
        controller = make_controller(module)
        merged = calibration_input()
        target = module(merged)

        controller.observe(merged, target)
        controller.freeze()
        manifest = controller.manifest()

        self.assertNotEqual(
            manifest["transformer_scale"], manifest["cnn_scale"])
        self.assertEqual(len(manifest["weight_scales"]), 2)
        self.assertEqual(len(manifest["target_accumulator_scales"]), 2)

    def test_freeze_requires_observation(self):
        with self.assertRaises(RuntimeError):
            make_controller(make_module()).freeze()


class SplitConcatExecutionTest(unittest.TestCase):
    def setUp(self):
        self.module = make_module(bias=True)
        self.controller = make_controller(self.module)
        self.merged = calibration_input()
        self.target = self.module(self.merged)
        self.controller.observe(self.merged, self.target)
        self.controller.freeze()
        self.controller.enable()

    def test_partial_accumulators_match_direct_integer_products(self):
        result = self.controller.execute(self.merged)
        transformer = result.transformer_codes.permute(
            0, 2, 3, 1).reshape(-1, 2).to(torch.int32)
        cnn = result.cnn_codes.permute(
            0, 2, 3, 1).reshape(-1, 2).to(torch.int32)
        transformer_weight = result.weight_codes[:, :2, 0, 0].to(torch.int32)
        cnn_weight = result.weight_codes[:, 2:, 0, 0].to(torch.int32)
        expected_transformer = (
            transformer @ transformer_weight.t()).reshape(1, 1, 2, 2).permute(
                0, 3, 1, 2)
        expected_cnn = (
            cnn @ cnn_weight.t()).reshape(1, 1, 2, 2).permute(0, 3, 1, 2)

        torch.testing.assert_close(
            result.transformer_accumulator, expected_transformer)
        torch.testing.assert_close(result.cnn_accumulator, expected_cnn)

    def test_bias_uses_target_accumulator_scale(self):
        result = self.controller.execute(self.merged)
        manifest = self.controller.manifest()

        self.assertEqual(
            manifest["bias_scales"], manifest["target_accumulator_scales"])
        self.assertEqual(result.bias_codes.dtype, torch.int32)
        self.assertEqual(result.output_codes.dtype, torch.int32)
        self.assertGreaterEqual(int(result.output_codes.min()), -7)
        self.assertLessEqual(int(result.output_codes.max()), 7)

    def test_quantize_records_finite_metrics_and_update_count(self):
        output = self.controller.quantize(self.merged)
        rows = self.controller.statistics()

        self.assertEqual(output.shape, self.target.shape)
        self.assertEqual(rows[0]["updates"], 1)
        self.assertEqual(rows[0]["transformer_saturation_ratio"], 0.0)
        self.assertEqual(rows[0]["cnn_saturation_ratio"], 0.0)
        for key in ("transformer_sqnr_db", "cnn_sqnr_db",
                    "partial_requantization_mse", "output_sqnr_db",
                    "block_mse"):
            self.assertTrue(torch.isfinite(torch.tensor(rows[0][key])))

    def test_disabled_controller_rejects_execution(self):
        self.controller.disable()

        with self.assertRaises(RuntimeError):
            self.controller.execute(self.merged)

    def test_statistics_can_be_reset_between_ablation_configs(self):
        self.controller.quantize(self.merged)

        self.controller.reset_statistics()

        with self.assertRaisesRegex(RuntimeError, "no quantized observations"):
            self.controller.statistics()


class SplitConcatSearchTest(unittest.TestCase):
    def test_search_rows_cover_all_joint_parameters(self):
        module = make_module()
        controller = make_controller(module, clip_factors=(1.0, 0.75))
        merged = calibration_input()
        controller.observe(merged, F.conv2d(
            merged, module.weight, module.bias, module.stride,
            module.padding, module.dilation, module.groups))

        controller.freeze()
        rows = controller.search_rows()

        self.assertEqual(len(rows), 8)
        self.assertEqual(
            {row["parameter"] for row in rows},
            {"transformer", "cnn", "accumulator", "output"})

    def test_reconfigure_searches_a8_from_the_same_calibration_cache(self):
        module = make_module()
        controller = make_controller(module, clip_factors=(1.0, 0.75))
        merged = calibration_input()
        controller.observe(merged, module(merged))
        controller.freeze()

        controller.reconfigure(activation_bits=8, output_bits=8)

        manifest = controller.manifest()
        self.assertEqual(manifest["activation_bits"], 8)
        self.assertEqual(manifest["output_bits"], 8)
        self.assertTrue(all(
            row["activation_bits"] == 8 and row["output_bits"] == 8
            for row in controller.search_rows()))


if __name__ == "__main__":
    unittest.main()
