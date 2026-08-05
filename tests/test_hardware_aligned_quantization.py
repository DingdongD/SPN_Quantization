import unittest

import torch
import torch.nn as nn

from scripts import hardware_aligned_quantization as haq
from scripts.lognp_quantization import LogNPActivationQuantizer


class HardwareQuantizationPrimitiveTest(unittest.TestCase):
    def test_signed_w4_activation_uses_symmetric_fifteen_level_range(self):
        quantizer = haq.SymmetricActivationQuantizer(bits=4, maximum=7.0)
        values = torch.tensor([-9.0, -7.0, -1.0, 0.0, 1.0, 7.0, 9.0])

        quantized, codes = quantizer.quantize_with_codes(values)

        self.assertEqual((int(codes.min()), int(codes.max())), (-7, 7))
        torch.testing.assert_close(quantized, torch.tensor(
            [-7.0, -7.0, -1.0, 0.0, 1.0, 7.0, 7.0]))
        self.assertEqual(quantizer.zero_point, 0)

    def test_relu_w4_activation_uses_unsigned_sixteen_level_range(self):
        quantizer = haq.UnsignedActivationQuantizer(bits=4, maximum=15.0)
        values = torch.tensor([-1.0, 0.0, 1.0, 15.0, 18.0])

        quantized, codes = quantizer.quantize_with_codes(values)

        self.assertEqual((int(codes.min()), int(codes.max())), (0, 15))
        torch.testing.assert_close(
            quantized, torch.tensor([0.0, 0.0, 1.0, 15.0, 15.0]))
        self.assertEqual(quantizer.zero_point, 0)

    def test_weight_and_bias_scales_follow_integer_conv_contract(self):
        weight = torch.tensor([
            [[[7.0, -7.0]]],
            [[[3.5, -3.5]]],
        ])
        bias = torch.tensor([0.7, -0.7])

        quantized_weight, weight_scale = haq.symmetric_weight_qdq(weight, bits=4)
        quantized_bias, bias_codes, bias_scale = haq.int32_bias_qdq(
            bias, input_scale=0.1, weight_scale=weight_scale)

        self.assertEqual(tuple(weight_scale.shape), (2, 1, 1, 1))
        torch.testing.assert_close(quantized_weight, weight)
        torch.testing.assert_close(bias_scale, torch.tensor([0.1, 0.05]))
        torch.testing.assert_close(
            bias_codes, torch.tensor([7, -14], dtype=torch.int32))
        torch.testing.assert_close(quantized_bias, bias)
        self.assertEqual(bias_codes.dtype, torch.int32)


class ConvBatchNormFoldingTest(unittest.TestCase):
    def test_executed_conv_bn_pair_is_folded_before_observation(self):
        torch.manual_seed(4)
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        ).eval()
        model[1].running_mean.copy_(torch.tensor([0.2, -0.1, 0.3, 0.0]))
        model[1].running_var.copy_(torch.tensor([0.5, 1.5, 2.0, 0.8]))
        sample = torch.randn(2, 3, 8, 8)
        reference = model(sample)
        observer = haq.HardwareMinMaxObserver()

        pairs = haq.discover_conv_bn_pairs(model, (sample,))
        manifest = haq.fold_conv_bn_pairs(model, pairs)

        self.assertEqual(pairs, [("0", "1")])
        self.assertEqual(manifest[0]["conv"], "0")
        self.assertIsInstance(model[1], nn.Identity)
        self.assertIsNotNone(model[0].bias)
        self.assertFalse(observer.observed)
        torch.testing.assert_close(model(sample), reference, atol=1e-5, rtol=1e-5)

        observer.update(model(sample))
        self.assertTrue(observer.observed)

    def test_bn_pair_with_pre_bn_fanout_is_explicitly_left_unfolded(self):
        class FanoutNet(nn.Module):
            def __init__(self):
                super(FanoutNet, self).__init__()
                self.conv = nn.Conv2d(1, 1, 1, bias=False)
                self.bn = nn.BatchNorm2d(1)

            def forward(self, value):
                raw = self.conv(value)
                return self.bn(raw) + raw

        model = FanoutNet().eval()
        sample = torch.randn(1, 1, 3, 3)

        preparation = haq.prepare_hardware_model(
            model, (sample,), excluded_pairs=[("conv", "bn")])

        self.assertEqual(preparation["folded_pairs"], [])
        self.assertEqual(preparation["unfolded_fanout_pairs"], [
            {"conv": "conv", "bn": "bn"},
        ])
        self.assertIsInstance(model.bn, nn.BatchNorm2d)
        self.assertEqual(preparation["max_abs_error"], 0.0)

    def test_instrumentor_folds_then_calibrates_relu_and_int32_bias(self):
        class TinyNet(nn.Module):
            def __init__(self):
                super(TinyNet, self).__init__()
                self.conv1 = nn.Conv2d(2, 3, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(3)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(3, 1, 1, bias=True)

            def forward(self, value):
                return self.conv2(self.relu(self.bn1(self.conv1(value))))

        torch.manual_seed(8)
        model = TinyNet().eval()
        sample = torch.randn(1, 2, 6, 6)

        preparation = haq.prepare_hardware_model(model, (sample,))
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            preparation["fused_relu_producers"])

        self.assertLess(preparation["max_abs_error"], 1e-5)
        self.assertEqual(preparation["folded_pairs"], [
            {"conv": "conv1", "bn": "bn1"},
        ])
        with self.assertRaises(RuntimeError):
            instrumentor.configure(4, 4, {"encoder"})

        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"encoder"})
        model(sample)

        manifest = instrumentor.manifest()
        relu_rows = [row for row in manifest if row["kind"] == "relu_output"]
        bias_rows = [row for row in instrumentor.statistics()
                     if row["kind"] == "bias"]
        self.assertEqual(relu_rows[0]["unsigned"], True)
        self.assertEqual(relu_rows[0]["qmin"], 0)
        self.assertEqual(relu_rows[0]["qmax"], 15)
        self.assertEqual({row["module"] for row in bias_rows}, {"conv1", "conv2"})
        self.assertTrue(all(row["scale_min"] > 0 for row in bias_rows))
        instrumentor.close()

    def test_instrumentor_supports_lognp_with_unsigned_relu_and_float_bias(self):
        class TinyNet(nn.Module):
            def __init__(self):
                super(TinyNet, self).__init__()
                self.conv1 = nn.Conv2d(2, 3, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(3)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(3, 1, 1, bias=True)

            def forward(self, value):
                return self.conv2(self.relu(self.bn1(self.conv1(value))))

        torch.manual_seed(19)
        model = TinyNet().eval()
        sample = torch.randn(1, 2, 6, 6)
        preparation = haq.prepare_hardware_model(model, (sample,))
        original_bias = model.conv2.bias.detach().clone()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            preparation["fused_relu_producers"])

        instrumentor.observe(activation_mode="lognp")
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(
            8, 4, {"encoder"}, activation_mode="lognp", alpha_factor=1.0)
        output = model(sample)

        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertIsInstance(
            instrumentor.lognp_quantizers[("conv2", "input")],
            LogNPActivationQuantizer)
        relu_rows = [row for row in instrumentor.manifest()
                     if row["kind"] == "relu_output"]
        self.assertEqual(relu_rows[0]["unsigned"], True)
        self.assertEqual(relu_rows[0]["qmin"], 0)
        self.assertEqual(relu_rows[0]["qmax"], 15)
        manifest = {(row["module"], row["kind"]): row
                    for row in instrumentor.manifest()}
        self.assertEqual(len(manifest[("conv1", "input")]["alpha"]), 2)
        self.assertEqual(len(manifest[("conv2", "input")]["alpha"]), 3)
        self.assertNotIn(("conv2", "bias"), instrumentor.stats)
        self.assertEqual(
            instrumentor.metadata()["bias_contract"],
            "reference_float_reconstruction")
        instrumentor.disable()
        torch.testing.assert_close(model.conv2.bias, original_bias)
        instrumentor.close()


class OutlierMitigationInstrumentorTest(unittest.TestCase):
    def _calibrated_linear(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=True)).eval()
        sample = torch.tensor([[1.0, 100.0]])
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        return model, instrumentor, sample

    def test_percentile_override_replaces_unsigned_input_minmax(self):
        model, instrumentor, sample = self._calibrated_linear()

        instrumentor.configure(
            4, 4, {"encoder"},
            activation_overrides={("0", "input"): 2.0})

        quantizer = instrumentor.quantizers[("0", "input")]
        self.assertEqual(quantizer.qmax, 15)
        self.assertAlmostEqual(quantizer.scale, 2.0 / 15.0)
        model(sample)
        instrumentor.close()

    def test_lognp_weight_compensation_uses_bounded_calibration_rows(self):
        torch.manual_seed(23)
        model = nn.Sequential(nn.Linear(3, 2, bias=True)).eval()
        samples = torch.randn(32, 3)
        reference = model(samples).detach()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.enable_compensation_capture(
            modules=["0"], sample_limit=64)
        instrumentor.observe(activation_mode="lognp")
        model(samples)
        instrumentor.freeze()
        instrumentor.configure(
            8, 4, {"encoder"}, activation_mode="lognp")
        uncorrected = model(samples).detach()
        uncorrected_error = torch.mean((uncorrected - reference) ** 2)
        rows = instrumentor.apply_compensation(method="weight")
        corrected = model(samples).detach()
        corrected_error = torch.mean((corrected - reference) ** 2)

        self.assertLess(float(corrected_error), float(uncorrected_error))
        self.assertEqual(rows[0]["method"], "weight")
        self.assertEqual(rows[0]["module"], "0")
        self.assertEqual(rows[0]["rows"], 32)
        instrumentor.close()

    def test_lognp_conv_compensation_preserves_unfolded_row_contract(self):
        torch.manual_seed(24)
        model = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, bias=True)).eval()
        samples = torch.randn(4, 2, 4, 4)
        reference = model(samples).detach()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.enable_compensation_capture(
            modules=["0"], sample_limit=64)
        instrumentor.observe(activation_mode="lognp")
        model(samples)
        instrumentor.freeze()
        instrumentor.configure(
            8, 4, {"encoder"}, activation_mode="lognp")
        rows = instrumentor.apply_compensation(method="weight")
        after = torch.mean((model(samples).detach() - reference) ** 2)

        self.assertTrue(bool(torch.isfinite(after)))
        self.assertLessEqual(rows[0]["after_mse"], rows[0]["before_mse"] * 1.01)
        self.assertEqual(rows[0]["rows"], 64)
        instrumentor.close()

    def test_smoothquant_scale_is_applied_and_parameters_restore(self):
        model, instrumentor, sample = self._calibrated_linear()
        original = model[0].weight.detach().clone()

        instrumentor.configure(
            4, 8, {"encoder"},
            smooth_channel_maxima={"0": torch.tensor([1.0, 100.0])},
            smooth_alpha=0.5)

        self.assertEqual(tuple(instrumentor.smooth_scales["0"].shape), (2,))
        self.assertFalse(torch.equal(model[0].weight, original))
        model(sample)
        instrumentor.disable()
        torch.testing.assert_close(model[0].weight, original)
        instrumentor.close()

    def test_awq_weight_clip_ratio_reduces_quantized_weight_range(self):
        model, instrumentor, sample = self._calibrated_linear()
        original_max = model[0].weight.detach().abs().amax(dim=1)

        instrumentor.configure(
            4, 8, {"encoder"}, weight_clip_ratio=0.8)

        quantized_max = model[0].weight.detach().abs().amax(dim=1)
        torch.testing.assert_close(
            quantized_max, original_max * 0.8, atol=1e-6, rtol=1e-5)
        model(sample)
        instrumentor.close()


class MixedActivationBitInstrumentorTest(unittest.TestCase):
    def _calibrated_model(self):
        model = nn.Sequential(
            nn.Conv2d(1, 2, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(2, 1, 1, bias=True),
        ).eval()
        sample = torch.rand(1, 1, 4, 4)
        preparation = haq.prepare_hardware_model(model, (sample,))
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            preparation["fused_relu_producers"])
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        return model, instrumentor, sample

    def test_selected_module_uses_a8_while_other_module_remains_a4(self):
        model, instrumentor, sample = self._calibrated_model()

        instrumentor.configure(
            4, 4, {"encoder"}, activation_bit_overrides={"0": 8})

        self.assertEqual(instrumentor.quantizers[("0", "input")].bits, 8)
        self.assertEqual(instrumentor.quantizers[("2", "input")].bits, 4)
        manifest = dict(((row["module"], row["kind"]), row)
                        for row in instrumentor.manifest())
        self.assertEqual(manifest[("0", "input")]["bits"], 8)
        self.assertEqual(manifest[("2", "input")]["bits"], 4)
        model(sample)
        instrumentor.close()

    def test_fused_relu_follows_direct_producer_activation_bits(self):
        model, instrumentor, sample = self._calibrated_model()

        instrumentor.configure(
            4, 4, {"encoder"}, activation_bit_overrides={"0": 8})

        relu = [row for row in instrumentor.manifest()
                if row["kind"] == "relu_output"]
        self.assertEqual(len(relu), 1)
        self.assertEqual(relu[0]["bits"], 8)
        self.assertEqual(relu[0]["qmax"], 255)
        model(sample)
        instrumentor.close()


if __name__ == "__main__":
    unittest.main()
