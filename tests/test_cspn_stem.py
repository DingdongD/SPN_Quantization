import unittest

import torch
import torch.nn.functional as F

from spn_quant.cspn_stem import (
    CSPNStemController,
    STEM_CONFIGS,
    split_rgb_depth,
)
from scripts.hardware_aligned_quantization import symmetric_weight_qdq


def make_stem(kernel_size=1):
    module = torch.nn.Conv2d(
        4, 2, kernel_size=kernel_size,
        padding=kernel_size // 2, bias=False).eval()
    with torch.no_grad():
        if kernel_size == 1:
            module.weight.copy_(torch.tensor([
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
                [[[-1.0]], [[1.0]], [[2.0]], [[-2.0]]],
            ]))
        else:
            module.weight.copy_(torch.linspace(
                -0.8, 0.9, steps=module.weight.numel()).reshape_as(
                    module.weight))
    return module


def calibration_input():
    rgb = torch.tensor([[
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.2, 0.4], [0.6, 0.8]],
        [[0.3, 0.6], [0.9, 1.0]],
    ]])
    depth = torch.tensor([[[[0.0, 4.0], [0.0, 8.0]]]])
    return torch.cat((rgb, depth), dim=1)


def calibrated_controller(module=None):
    module = make_stem() if module is None else module
    controller = CSPNStemController(module)
    controller.observe()
    module(calibration_input())
    controller.freeze()
    return module, controller


class CSPNStemLayoutTest(unittest.TestCase):
    def test_configuration_order_is_fixed(self):
        self.assertEqual(STEM_CONFIGS, (
            "STRICT_W4A4", "STEM_W4A8", "STEM_W8A8",
            "STEM_FP16", "STEM_BRANCH_A4",
        ))

    def test_rgb_depth_split_preserves_official_channel_order(self):
        merged = calibration_input()

        rgb, depth = split_rgb_depth(merged)

        torch.testing.assert_close(rgb, merged[:, :3])
        torch.testing.assert_close(depth, merged[:, 3:4])

    def test_controller_requires_official_bias_free_four_channel_conv(self):
        with self.assertRaisesRegex(ValueError, "four input channels"):
            CSPNStemController(torch.nn.Conv2d(3, 2, 1, bias=False))
        with self.assertRaisesRegex(ValueError, "bias-free"):
            CSPNStemController(torch.nn.Conv2d(4, 2, 1, bias=True))


class CSPNStemCalibrationTest(unittest.TestCase):
    def test_freeze_requires_observation(self):
        controller = CSPNStemController(make_stem())

        with self.assertRaisesRegex(RuntimeError, "observation"):
            controller.freeze()

    def test_branch_scales_are_independent(self):
        _, controller = calibrated_controller()
        controller.configure("STEM_BRANCH_A4")

        contract = controller.contract()

        self.assertAlmostEqual(contract["rgb_maximum"], 1.0)
        self.assertAlmostEqual(contract["depth_maximum"], 8.0)
        self.assertAlmostEqual(contract["rgb_scale"], 1.0 / 15.0)
        self.assertAlmostEqual(contract["depth_scale"], 8.0 / 15.0)
        self.assertEqual(contract["activation_scales"], 2)

    def test_merged_configs_use_one_rgbd_scale(self):
        _, controller = calibrated_controller()

        controller.configure("STRICT_W4A4")
        strict = controller.contract()
        controller.configure("STEM_W4A8")
        activation_promoted = controller.contract()
        controller.configure("STEM_W8A8")
        promoted = controller.contract()

        self.assertEqual(strict["activation_scales"], 1)
        self.assertEqual(activation_promoted["activation_scales"], 1)
        self.assertEqual(promoted["activation_scales"], 1)
        self.assertAlmostEqual(strict["input_scale"], 8.0 / 15.0)
        self.assertAlmostEqual(
            activation_promoted["input_scale"], 8.0 / 255.0)
        self.assertAlmostEqual(promoted["input_scale"], 8.0 / 255.0)
        self.assertEqual(strict["weight_bits"], 4)
        self.assertEqual(activation_promoted["weight_bits"], 4)
        self.assertEqual(activation_promoted["activation_bits"], 8)
        self.assertEqual(promoted["weight_bits"], 8)

    def test_weight_rounding_matches_existing_hardware_quantizer(self):
        module = torch.nn.Conv2d(4, 1, 1, bias=False).eval()
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[[[1.0]],
                                                [[0.2142857164144516]],
                                                [[0.0]], [[0.0]]]]))
        _, controller = calibrated_controller(module)
        controller.configure("STRICT_W4A4")
        quantized, scale = symmetric_weight_qdq(module.weight, 4)
        expected_codes = torch.round(module.weight / scale).to(torch.int8)

        torch.testing.assert_close(
            controller.weight_codes, expected_codes.cpu())
        torch.testing.assert_close(
            controller.weight_codes.float() * controller.weight_scales,
            quantized.cpu())


class CSPNStemExecutionTest(unittest.TestCase):
    def test_explicit_integer_bits_match_manual_qdq(self):
        module, controller = calibrated_controller()
        merged = calibration_input()

        for weight_bits in (2, 4, 6, 8):
            for activation_bits in (2, 4, 6, 8):
                controller.configure_integer(weight_bits, activation_bits)

                observed = module(merged)
                quantized, _, _ = controller._unsigned_qdq(
                    merged, controller.merged_maximum, activation_bits)
                expected = controller._float_convolution(
                    quantized, controller._quantized_weight(merged))
                contract = controller.contract()

                torch.testing.assert_close(observed, expected)
                self.assertEqual(contract["weight_bits"], weight_bits)
                self.assertEqual(
                    contract["activation_bits"], activation_bits)
                self.assertEqual(
                    contract["config"],
                    "STEM_W%dA%d" % (weight_bits, activation_bits))

    def test_explicit_integer_bits_reject_unsupported_values(self):
        _, controller = calibrated_controller()

        with self.assertRaisesRegex(ValueError, "stem weight bits"):
            controller.configure_integer(3, 4)
        with self.assertRaisesRegex(ValueError, "stem activation bits"):
            controller.configure_integer(4, 3)

    def test_w4a8_matches_manual_qdq_convolution(self):
        module, controller = calibrated_controller()
        controller.configure("STEM_W4A8")
        merged = calibration_input()
        contract = controller.contract()

        output = module(merged)
        input_codes = torch.round(
            merged / contract["input_scale"]).clamp(0, 255)
        expected = F.conv2d(
            input_codes * contract["input_scale"],
            controller.weight_codes.float() * controller.weight_scales,
            stride=module.stride, padding=module.padding,
            dilation=module.dilation, groups=module.groups)

        self.assertEqual(controller.weight_codes.dtype, torch.int8)
        self.assertEqual(contract["weight_bits"], 4)
        self.assertEqual(contract["activation_bits"], 8)
        torch.testing.assert_close(output, expected)

    def test_strict_w4a4_matches_manual_qdq_convolution(self):
        module, controller = calibrated_controller()
        controller.configure("STRICT_W4A4")
        merged = calibration_input()
        contract = controller.contract()

        output = module(merged)
        input_codes = torch.round(
            merged / contract["input_scale"]).clamp(0, 15)
        weight_codes = controller.weight_codes.to(torch.float32)
        expected = F.conv2d(
            input_codes * contract["input_scale"],
            weight_codes * controller.weight_scales,
            stride=module.stride, padding=module.padding,
            dilation=module.dilation, groups=module.groups)

        torch.testing.assert_close(output, expected)

    def test_branch_a4_uses_two_integer_partial_convolutions(self):
        module, controller = calibrated_controller()
        controller.configure("STEM_BRANCH_A4")

        output = module(calibration_input())
        result = controller.last_integer_result()

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape, (1, 2, 2, 2))
        self.assertEqual(result.rgb_codes.dtype, torch.uint8)
        self.assertEqual(result.depth_codes.dtype, torch.uint8)
        self.assertEqual(result.weight_codes.dtype, torch.int8)
        self.assertEqual(result.rgb_accumulator.dtype, torch.int32)
        self.assertEqual(result.depth_accumulator.dtype, torch.int32)
        self.assertEqual(result.accumulator.dtype, torch.int32)
        self.assertEqual(
            result.weight_codes[:, :3].tolist(),
            controller.weight_codes[:, :3].tolist())
        self.assertEqual(
            result.weight_codes[:, 3:].tolist(),
            controller.weight_codes[:, 3:].tolist())

    def test_fp16_path_uses_fp16_operands_and_returns_fp32(self):
        module, controller = calibrated_controller()
        controller.configure("STEM_FP16")
        merged = calibration_input()

        output = module(merged)
        expected = F.conv2d(
            merged.half(), module.weight.detach().half(),
            stride=module.stride, padding=module.padding,
            dilation=module.dilation, groups=module.groups).float()

        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, expected)

    def test_statistics_record_branch_zero_collapse_and_output_error(self):
        module, controller = calibrated_controller()
        controller.configure("STEM_BRANCH_A4")
        module(calibration_input())

        rows = controller.statistics()

        self.assertEqual(
            {row["signal"] for row in rows},
            {"rgb_input", "depth_input", "rgb_partial",
             "depth_partial", "stem_output"})
        self.assertTrue(all(row["updates"] == 1 for row in rows))
        self.assertTrue(all(torch.isfinite(torch.tensor(
            row["mse"])).item() for row in rows))

    def test_reconfiguration_resets_runtime_statistics(self):
        module, controller = calibrated_controller()
        controller.configure("STRICT_W4A4")
        module(calibration_input())
        controller.configure("STEM_W8A8")

        with self.assertRaisesRegex(RuntimeError, "no observations"):
            controller.statistics()

    def test_close_restores_original_forward(self):
        module = make_stem()
        original = module(calibration_input())
        controller = CSPNStemController(module)
        controller.observe()
        module(calibration_input())
        controller.freeze()
        controller.configure("STRICT_W4A4")
        controller.close()

        restored = module(calibration_input())

        torch.testing.assert_close(restored, original)


if __name__ == "__main__":
    unittest.main()
