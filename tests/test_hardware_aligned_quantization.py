import unittest

import torch
import torch.nn as nn

from scripts import hardware_aligned_quantization as haq
from scripts.lognp_quantization import LogNPActivationQuantizer
from spn_quant.runtime import EdgeAwareQuantizerProxy, EdgeQDQRuntime
from spn_quant.specs import QuantSpec


class HardwareQuantizationPrimitiveTest(unittest.TestCase):
    def test_channel_observer_accumulates_exact_rms_across_updates(self):
        observer = haq.ChannelMinMaxObserver(channel_dim=1)
        observer.update(torch.tensor([[[[3.0, 4.0]], [[0.0, 0.0]]]]))
        observer.update(torch.tensor([[[[0.0, 0.0]], [[6.0, 8.0]]]]))

        torch.testing.assert_close(
            observer.channel_rms(), torch.tensor([2.5, 5.0]))

    def test_channel_observer_rejects_rms_before_observation(self):
        observer = haq.ChannelMinMaxObserver(channel_dim=1)

        with self.assertRaisesRegex(RuntimeError, "observations"):
            observer.channel_rms()

    def test_group_a4_uses_one_scale_per_contiguous_channel_group(self):
        observer = haq.ChannelMinMaxObserver(channel_dim=1)
        observer.update(torch.tensor(
            [[[[1.0]], [[2.0]], [[10.0]], [[20.0]]]]))
        spec = QuantSpec(
            bits=4, scheme="affine", granularity="group",
            axis=1, group_size=2, signed=False, preserve_zero=True)

        quantizer = observer.quantizer_for(spec)

        self.assertEqual(quantizer.scale_count, 2)
        torch.testing.assert_close(
            quantizer.scale, torch.tensor([2.0 / 15.0, 20.0 / 15.0]))
        values = torch.tensor(
            [[[[0.1]], [[2.0]], [[1.0]], [[20.0]]]])
        quantized, codes = quantizer.quantize_with_codes(values)
        self.assertGreater(float(quantized[0, 0, 0, 0]), 0.0)
        self.assertEqual(codes.dtype, torch.int32)

    def test_group_a4_rejects_nondivisible_channel_count(self):
        observer = haq.ChannelMinMaxObserver(channel_dim=1)
        observer.update(torch.ones(1, 3, 2, 2))
        spec = QuantSpec(
            bits=4, granularity="group", axis=1, group_size=2)

        with self.assertRaisesRegex(ValueError, "divide"):
            observer.quantizer_for(spec)

    def test_activation_spec_signedness_must_match_observed_site(self):
        observer = haq.ChannelMinMaxObserver(channel_dim=1)
        observer.update(torch.ones(1, 2, 2, 2))

        with self.assertRaisesRegex(ValueError, "unsigned"):
            observer.quantizer_for(QuantSpec.signed_tensor(4), unsigned=True)

    def test_instrumentor_quantizes_explicit_weight_source(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=False)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        sample = torch.randn(1, 2, 3, 3)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        source = torch.tensor([
            [[[0.9]], [[-0.2]]],
            [[[0.1]], [[-0.7]]],
        ])
        expected, _ = haq.symmetric_weight_qdq(source, bits=4)

        instrumentor.configure(
            4, 4, {"encoder"}, quantize_bias=False,
            weight_source_overrides={"0": source})

        torch.testing.assert_close(model[0].weight, expected)
        instrumentor.close()

    def test_instrumentor_refreshes_parameter_sources_after_checkpoint_load(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=True)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        with torch.no_grad():
            model[0].weight.fill_(0.75)
            model[0].bias.fill_(-0.25)

        instrumentor.refresh_parameter_sources()

        torch.testing.assert_close(
            instrumentor.original_weights["0"], model[0].weight.cpu())
        torch.testing.assert_close(
            instrumentor.original_biases["0"], model[0].bias.cpu())
        instrumentor.close()

    def test_edge_proxy_preserves_per_channel_scale_for_statistics(self):
        values = torch.tensor([
            [[[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]],
             [[-20.0, -10.0, 0.0], [0.0, 10.0, 20.0]]],
        ])
        base = haq.ChannelActivationQuantizer(
            4, torch.tensor([-2.0, -20.0]),
            torch.tensor([2.0, 20.0]), channel_dim=1)
        runtime = EdgeQDQRuntime()
        quantizer = EdgeAwareQuantizerProxy(base, runtime, "decoder:input")
        stats = haq.QuantizationStats()
        runtime.begin_forward()

        quantized, codes = quantizer.quantize_with_codes(values)
        haq.update_activation_stats(
            stats, quantizer, values, quantized, codes, values)

        self.assertEqual(stats.numel, values.numel())
        self.assertEqual(stats.saturated, 0)

    def test_reused_edge_without_codes_does_not_update_activation_stats(self):
        stats = haq.QuantizationStats()
        quantizer = haq.SymmetricActivationQuantizer(bits=8, maximum=1.0)
        values = torch.tensor([-0.5, 0.0, 0.5])

        haq.update_activation_stats(
            stats, quantizer, values, values, None, values)

        self.assertEqual(stats.numel, 0)

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

    def test_dynamic_signed_group_a4_uses_independent_sample_ranges(self):
        quantizer = haq.DynamicGroupedActivationQuantizer(
            bits=4, channel_dim=1, group_size=2, channels=4,
            unsigned=False)
        values = torch.tensor([
            [-7.0, -1.0, -14.0, -2.0],
            [70.0, 10.0, 140.0, 20.0],
        ]).reshape(2, 4, 1, 1)

        quantized, codes = quantizer.quantize_with_codes(values)

        self.assertEqual((int(codes.min()), int(codes.max())), (-7, 7))
        torch.testing.assert_close(quantized, values)
        self.assertEqual(
            tuple(quantizer.scale_for(values).shape), (2, 4, 1, 1))

    def test_dynamic_unsigned_group_a4_preserves_zero_range(self):
        quantizer = haq.DynamicGroupedActivationQuantizer(
            bits=4, channel_dim=1, group_size=2, channels=4,
            unsigned=True)
        values = torch.tensor([[[[0.0]], [[0.0]], [[1.0]], [[15.0]]]])

        quantized, codes = quantizer.quantize_with_codes(values)

        torch.testing.assert_close(
            quantized[:, :2], torch.zeros(1, 2, 1, 1))
        self.assertEqual((int(codes.min()), int(codes.max())), (0, 15))

    def test_dynamic_tensor_a4_uses_one_scale_per_sample(self):
        quantizer = haq.DynamicTensorActivationQuantizer(
            bits=4, unsigned=False)
        values = torch.tensor([[-7.0, 1.0], [-70.0, 10.0]])

        scale = quantizer.scale_for(values)

        torch.testing.assert_close(
            scale.reshape(-1), torch.tensor([1.0, 10.0]))

    def test_dynamic_activation_rejects_nonfinite_input(self):
        quantizer = haq.DynamicTensorActivationQuantizer(
            bits=4, unsigned=False)

        with self.assertRaisesRegex(ValueError, "finite"):
            quantizer(torch.tensor([[float("inf")]]))

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


class ActivationRecorderTest(unittest.TestCase):
    class RecordingSink(object):
        def __init__(self):
            self.rows = []

        def record(self, module, kind, call_index, group, reference,
                   quantized, codes, quantizer, channel_dim):
            self.rows.append({
                "module": module,
                "kind": kind,
                "call_index": call_index,
                "group": group,
                "reference": reference.detach().clone(),
                "quantized": quantized.detach().clone(),
                "codes": codes.detach().clone(),
                "quantizer": quantizer,
                "channel_dim": channel_dim,
            })

    def test_recorder_observes_real_qdq_and_shared_calls(self):
        class SharedConv(nn.Module):
            def __init__(self):
                super(SharedConv, self).__init__()
                self.conv = nn.Conv2d(1, 1, 1, bias=False)

            def forward(self, value):
                first = self.conv(value)
                return self.conv(first + 1.0)

        model = SharedConv().eval()
        sink = self.RecordingSink()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.set_activation_recorder(sink)
        sample = torch.ones(1, 1, 2, 2)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"encoder"})

        model(sample)

        inputs = [row for row in sink.rows if row["kind"] == "input"]
        outputs = [row for row in sink.rows if row["kind"] == "output"]
        self.assertEqual([row["call_index"] for row in inputs], [0, 1])
        self.assertEqual([row["call_index"] for row in outputs], [0, 1])
        self.assertTrue(all(row["module"] == "conv" for row in sink.rows))
        self.assertTrue(all(row["group"] == "encoder" for row in sink.rows))
        self.assertTrue(all(row["channel_dim"] == 1 for row in sink.rows))
        self.assertTrue(all(row["codes"].dtype == torch.int32
                            for row in sink.rows))
        for row in sink.rows:
            expected, expected_codes = \
                row["quantizer"].quantize_with_codes(row["reference"])
            torch.testing.assert_close(row["quantized"], expected)
            torch.testing.assert_close(row["codes"], expected_codes)

        instrumentor.close()

    def test_recorder_preserves_layernorm_channel_dimension(self):
        class ConvNorm(nn.Module):
            def __init__(self):
                super(ConvNorm, self).__init__()
                self.proj = nn.Conv2d(2, 2, 1)
                self.norm = nn.LayerNorm(2)

            def forward(self, value):
                value = self.proj(value)
                value = value.permute(0, 2, 3, 1)
                return self.norm(value)

        model = ConvNorm().eval()
        sink = self.RecordingSink()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "attention")
        instrumentor.set_activation_recorder(sink)
        sample = torch.randn(1, 2, 3, 3)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"attention"})

        model(sample)

        norm = [row for row in sink.rows if row["module"] == "norm"]
        self.assertEqual(len(norm), 1)
        self.assertEqual(norm[0]["channel_dim"], -1)
        instrumentor.close()

    def test_recorder_includes_relu_owned_output(self):
        model = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False),
            nn.ReLU(),
        ).eval()
        sink = self.RecordingSink()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.set_activation_recorder(sink)
        sample = torch.tensor([[[[-1.0, 2.0]]]])
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"encoder"})

        model(sample)

        rows = [row for row in sink.rows
                if row["kind"] == "relu_output"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["module"], "1#0")
        self.assertEqual(rows[0]["call_index"], 0)
        self.assertTrue(rows[0]["quantizer"].unsigned)
        instrumentor.close()

    def test_recorder_preserves_per_channel_quantizer(self):
        model = nn.Sequential(nn.Conv2d(2, 1, 1, bias=False)).eval()
        sink = self.RecordingSink()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            per_channel_activation_inputs={"0"})
        instrumentor.set_activation_recorder(sink)
        sample = torch.tensor([[[[1.0, 2.0]], [[10.0, 20.0]]]])
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"encoder"})

        model(sample)

        row = next(row for row in sink.rows if row["kind"] == "input")
        self.assertEqual(row["quantizer"].channel_dim, 1)
        self.assertEqual(row["quantizer"].scale.numel(), 2)
        torch.testing.assert_close(
            row["quantizer"].scale, torch.tensor([2.0 / 15.0, 20.0 / 15.0]))
        instrumentor.close()

    def test_instrumentor_applies_group_spec_to_declared_input_site(self):
        model = nn.Sequential(nn.Conv2d(4, 1, 1, bias=False)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        sample = torch.tensor(
            [[[[1.0]], [[2.0]], [[10.0]], [[20.0]]]])
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        spec = QuantSpec(
            bits=4, scheme="affine", granularity="group",
            axis=1, group_size=2, signed=False, preserve_zero=True)

        instrumentor.configure(
            4, 4, {"encoder"},
            activation_specs={("0", "input"): spec},
            quantize_bias=False)

        quantizer = instrumentor.quantizers[("0", "input")]
        self.assertEqual(quantizer.granularity, "group")
        self.assertEqual(quantizer.scale_count, 2)
        self.assertEqual(
            instrumentor.quantizers[("0", "output")].granularity,
            "tensor")
        instrumentor.close()

    def test_instrumentor_builds_dynamic_group_quantizer(self):
        model = nn.Sequential(nn.Conv2d(4, 1, 1, bias=False)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        sample = torch.tensor(
            [[[[1.0]], [[2.0]], [[10.0]], [[20.0]]]])
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})
        specs[("0", "input")] = QuantSpec.unsigned_group(
            4, axis=1, group_size=2).with_dynamic()

        instrumentor.configure_components_with_ranges(
            4, 4, set(), {"encoder"}, specs, False,
            activation_maxima={})

        self.assertIsInstance(
            instrumentor.quantizers[("0", "input")],
            haq.DynamicGroupedActivationQuantizer)
        self.assertIsInstance(
            instrumentor.quantizers[("0", "output")],
            haq.SymmetricActivationQuantizer)
        instrumentor.close()

    def test_dynamic_spec_rejects_static_maximum(self):
        observer = haq.ChannelMinMaxObserver(channel_dim=1)
        observer.update(torch.ones(1, 4, 1, 1))
        spec = QuantSpec.unsigned_group(
            4, axis=1, group_size=2).with_dynamic()

        with self.assertRaisesRegex(ValueError, "dynamic.*maximum"):
            observer.quantizer_for(spec, maximum=torch.ones(2))

    def test_dynamic_overhead_rows_count_runtime_work(self):
        model = nn.Sequential(nn.Conv2d(4, 1, 1, bias=False)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        sample = torch.ones(2, 4, 3, 3)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})
        specs[("0", "input")] = QuantSpec.unsigned_group(
            4, axis=1, group_size=2).with_dynamic()
        instrumentor.configure_components_with_ranges(
            4, 4, set(), {"encoder"}, specs, False,
            activation_maxima={})

        model(sample)
        rows = instrumentor.dynamic_activation_rows()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["runtime_scale_count"], 4)
        self.assertEqual(rows[0]["reduction_elements"], sample.numel())
        instrumentor.close()

    def test_instrumentor_rejects_unknown_activation_spec_site(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(torch.ones(1, 1, 1, 1))
        instrumentor.freeze()

        with self.assertRaisesRegex(ValueError, "unknown activation specs"):
            instrumentor.configure(
                4, 4, {"encoder"},
                activation_specs={
                    ("missing", "input"): QuantSpec.signed_tensor(4),
                })
        instrumentor.close()


class ComponentQuantizationTest(unittest.TestCase):
    @staticmethod
    def _scale_aware_model(module):
        model = nn.Sequential(module).eval()
        sample = torch.arange(
            1, 9, dtype=torch.float32).reshape(1, 8, 1, 1)
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, current: "encoder")
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})
        specs[("0", "input")] = QuantSpec.unsigned_group(
            4, axis=1, group_size=8)
        return model, instrumentor, sample, specs

    def test_scale_aware_input_permutes_weight_before_w4(self):
        torch.manual_seed(43)
        model, instrumentor, sample, specs = self._scale_aware_model(
            nn.Conv2d(8, 3, 1, bias=False))
        del sample
        original = instrumentor.original_weights["0"].clone()
        permutation = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        expected_source = original[:, permutation]
        expected_weight, expected_scale = haq.symmetric_weight_qdq(
            expected_source, bits=4)

        instrumentor.configure_components_with_ranges(
            4, 4, {"encoder"}, {"encoder"}, specs, False,
            activation_maxima={},
            activation_permutations={("0", "input"): permutation})

        torch.testing.assert_close(model[0].weight, expected_weight)
        torch.testing.assert_close(
            instrumentor.weight_scales["0"], expected_scale)
        baseline_weight, baseline_scale = haq.symmetric_weight_qdq(
            original, bits=4)
        torch.testing.assert_close(expected_scale, baseline_scale)
        torch.testing.assert_close(
            torch.linalg.vector_norm(expected_weight - expected_source),
            torch.linalg.vector_norm(baseline_weight - original))
        instrumentor.disable()
        torch.testing.assert_close(model[0].weight, original)
        instrumentor.close()

    def test_scale_aware_input_reports_original_channel_order(self):
        class Recorder(object):
            def __init__(self):
                self.rows = []

            def record(self, module, kind, call_index, group, reference,
                       quantized, codes, quantizer, channel_dim):
                self.rows.append((
                    module, kind, reference.clone(), quantized.clone(),
                    codes.clone(), torch.as_tensor(
                        quantizer.scale_for(reference)).clone(),
                    channel_dim))

        torch.manual_seed(44)
        model, instrumentor, sample, specs = self._scale_aware_model(
            nn.Conv2d(8, 3, 1, bias=False))
        permutation = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        instrumentor.configure_components_with_ranges(
            4, 4, {"encoder"}, {"encoder"}, specs, False,
            activation_maxima={},
            activation_permutations={("0", "input"): permutation})
        recorder = Recorder()
        instrumentor.set_activation_recorder(recorder)

        model(sample)

        row = [row for row in recorder.rows
               if row[0] == "0" and row[1] == "input"][0]
        torch.testing.assert_close(row[2], sample)
        self.assertEqual(tuple(row[3].shape), tuple(sample.shape))
        self.assertEqual(tuple(row[4].shape), tuple(sample.shape))
        self.assertEqual(tuple(row[5].shape), tuple(sample.shape))
        instrumentor.close()

    def test_scale_aware_input_requires_paired_weight_quantization(self):
        model, instrumentor, sample, specs = self._scale_aware_model(
            nn.Conv2d(8, 3, 1, bias=False))
        del model, sample

        with self.assertRaisesRegex(ValueError, "weight"):
            instrumentor.configure_components_with_ranges(
                4, 4, set(), {"encoder"}, specs, False,
                activation_maxima={},
                activation_permutations={
                    ("0", "input"): torch.arange(8)})
        instrumentor.close()

    def test_scale_aware_input_rejects_output_and_nonbijective_mapping(self):
        model, instrumentor, sample, specs = self._scale_aware_model(
            nn.Conv2d(8, 3, 1, bias=False))
        del model, sample

        with self.assertRaisesRegex(ValueError, "input"):
            instrumentor.configure_components_with_ranges(
                4, 4, {"encoder"}, {"encoder"}, specs, False,
                activation_maxima={},
                activation_permutations={
                    ("0", "output"): torch.arange(3)})
        with self.assertRaisesRegex(ValueError, "bijection"):
            instrumentor.configure_components_with_ranges(
                4, 4, {"encoder"}, {"encoder"}, specs, False,
                activation_maxima={},
                activation_permutations={
                    ("0", "input"): torch.zeros(8, dtype=torch.long)})
        instrumentor.close()

    def test_outlier_isolation_changes_only_declared_input_activation_scales(self):
        torch.manual_seed(47)
        model, instrumentor, sample, specs = self._scale_aware_model(
            nn.Conv2d(8, 3, 1, bias=False))
        original = instrumentor.original_weights["0"].clone()
        expected_weight, expected_weight_scale = haq.symmetric_weight_qdq(
            original, bits=4)

        instrumentor.configure_components_with_ranges(
            4, 4, {"encoder"}, {"encoder"}, specs, False,
            activation_maxima={},
            activation_isolations={("0", "input"): (7,)})

        quantizer = instrumentor.quantizers[("0", "input")]
        expected_scales = torch.tensor([
            7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0,
            7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0, 8.0 / 15.0,
        ]).reshape(1, 8, 1, 1)
        torch.testing.assert_close(quantizer.scale_for(sample), expected_scales)
        torch.testing.assert_close(model[0].weight, expected_weight)
        torch.testing.assert_close(
            instrumentor.weight_scales["0"], expected_weight_scale)

        output = model(sample)
        self.assertEqual(tuple(output.shape), (1, 3, 1, 1))
        instrumentor.close()

    def test_outlier_isolation_quantizes_declared_relu_owner(self):
        model = nn.Sequential(
            nn.Conv2d(8, 8, 1, bias=False), nn.ReLU()).eval()
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(8).reshape(8, 8, 1, 1))
        sample = torch.arange(
            1, 9, dtype=torch.float32).reshape(1, 8, 1, 1)
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, current: "encoder",
            fused_relu_producers={"1#0": "0"},
            externally_owned_inputs=("0",))
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})
        specs["1#0"] = QuantSpec.unsigned_group(
            4, axis=1, group_size=8)

        instrumentor.configure_components_with_ranges(
            4, 4, {"encoder"}, {"encoder"}, specs, False,
            activation_maxima={},
            activation_isolations={"1#0": (7,)})

        quantizer = instrumentor.relu_quantizers["1#0"]
        expected_scales = torch.tensor([
            7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0,
            7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0, 8.0 / 15.0,
        ]).reshape(1, 8, 1, 1)
        torch.testing.assert_close(
            quantizer.scale_for(sample), expected_scales)
        probe = sample.clone()
        probe[:, 0] = 0.25
        output = model(probe)
        self.assertGreater(float(output[0, 0, 0, 0]), 0.0)
        instrumentor.close()

    def test_outlier_isolation_rejects_signed_output_and_nonmaximum_channel(self):
        model, instrumentor, sample, specs = self._scale_aware_model(
            nn.Conv2d(8, 8, 1, bias=False))
        del model, sample
        specs[("0", "output")] = QuantSpec.signed_group(
            4, axis=1, group_size=8)

        with self.assertRaisesRegex(ValueError, "unsigned affine A4"):
            instrumentor.configure_components_with_ranges(
                4, 4, {"encoder"}, {"encoder"}, specs, False,
                activation_maxima={},
                activation_isolations={("0", "output"): (2,)})
        with self.assertRaisesRegex(ValueError, "maximum channel"):
            instrumentor.configure_components_with_ranges(
                4, 4, {"encoder"}, {"encoder"}, specs, False,
                activation_maxima={},
                activation_isolations={("0", "input"): (6,)})
        instrumentor.close()
    def test_observe_forwards_declared_fp_references_to_calibration_recorder(self):
        class Recorder(object):
            def __init__(self):
                self.rows = []

            def record_reference(self, module, kind, group, tensor,
                                 channel_dim):
                self.rows.append((
                    module, kind, group, tensor.detach().clone(), channel_dim))

        model = nn.Sequential(
            nn.Conv2d(2, 2, 1, bias=False), nn.ReLU()).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        sample = torch.tensor([[[[-1.0]], [[2.0]]]])
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        owners = set(
            haq_key if isinstance(haq_key, tuple) else
            (haq_key, "relu_output")
            for haq_key in instrumentor.activation_site_keys({"encoder"}))
        recorder = Recorder()

        instrumentor.set_calibration_recorder(recorder, owners)
        instrumentor.observe()
        model(sample)

        self.assertEqual(
            {(row[0], row[1]) for row in recorder.rows}, owners)
        for row in recorder.rows:
            self.assertFalse(row[3].requires_grad)
        instrumentor.close()

    @staticmethod
    def _calibrated_model():
        torch.manual_seed(31)
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=False)).eval()
        sample = torch.tensor(
            [[[[0.2, 0.9]], [[2.0, 9.0]]]], dtype=torch.float32)
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        return model, instrumentor, sample

    def test_weight_only_keeps_w4_weights_and_bypasses_activation_qdq(self):
        model, instrumentor, sample = self._calibrated_model()
        reference_weight = model[0].weight.detach().clone()

        instrumentor.configure_components(
            4, 4, {"encoder"}, set(), {}, False)
        model(sample)

        self.assertFalse(torch.equal(model[0].weight, reference_weight))
        self.assertEqual(instrumentor.quantizers, {})
        self.assertEqual(instrumentor.relu_quantizers, {})
        self.assertTrue(all(key[1] == "weight" for key in instrumentor.stats))
        instrumentor.close()

    def test_activation_only_restores_fp_weights_and_applies_a4_qdq(self):
        model, instrumentor, sample = self._calibrated_model()
        reference_weight = model[0].weight.detach().clone()
        specs = instrumentor.tensor_activation_specs(
            4, {"encoder"})

        instrumentor.configure_components(
            4, 4, set(), {"encoder"}, specs, False)
        quantized = model(sample).detach()
        instrumentor.disable()
        reference = model(sample).detach()

        torch.testing.assert_close(model[0].weight, reference_weight)
        self.assertFalse(torch.equal(quantized, reference))
        self.assertNotEqual(instrumentor.quantizers, {})
        instrumentor.close()

    def test_component_configuration_rejects_unknown_groups(self):
        model, instrumentor, sample = self._calibrated_model()
        del model, sample

        with self.assertRaisesRegex(ValueError, "unknown activation groups"):
            instrumentor.configure_components(
                4, 4, set(), {"missing"}, {}, False)
        instrumentor.close()

    def test_component_configuration_requires_every_activation_spec(self):
        model, instrumentor, sample = self._calibrated_model()
        del model, sample

        with self.assertRaisesRegex(ValueError, "activation spec coverage"):
            instrumentor.configure_components(
                4, 4, set(), {"encoder"}, {}, False)
        instrumentor.close()

    def test_component_configuration_uses_per_site_spec_bits(self):
        model, instrumentor, sample = self._calibrated_model()
        del model, sample
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})
        specs[("0", "output")] = specs[("0", "output")].with_bits(8)

        instrumentor.configure_components(
            4, 4, set(), {"encoder"}, specs, False)

        self.assertEqual(instrumentor.quantizers[("0", "input")].bits, 4)
        self.assertEqual(instrumentor.quantizers[("0", "output")].bits, 8)
        instrumentor.close()

    def test_component_range_configuration_uses_explicit_maximum(self):
        model, instrumentor, sample = self._calibrated_model()
        del model, sample
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})

        instrumentor.configure_components_with_ranges(
            4, 4, set(), {"encoder"}, specs, False,
            activation_maxima={("0", "input"): 2.0})

        quantizer = instrumentor.quantizers[("0", "input")]
        expected = 2.0 / float(quantizer.qmax)
        self.assertAlmostEqual(float(quantizer.scale), expected)
        instrumentor.close()

    def test_component_range_configuration_uses_relu_maximum(self):
        model = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False), nn.ReLU()).eval()
        sample = torch.tensor([[[[0.5, 4.0]]]])
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})

        instrumentor.configure_components_with_ranges(
            4, 4, set(), {"encoder"}, specs, False,
            activation_maxima={"1#0": 1.5})

        quantizer = instrumentor.relu_quantizers["1#0"]
        self.assertAlmostEqual(float(quantizer.scale), 1.5 / 15.0)
        instrumentor.close()

    def test_component_smoothquant_aggregates_transformed_group_ranges(self):
        model = nn.Sequential(nn.Conv2d(4, 2, 1, bias=False)).eval()
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([
                [[[1.0]], [[2.0]], [[1.0]], [[4.0]]],
                [[[0.5]], [[1.0]], [[2.0]], [[2.0]]],
            ]))
        sample = torch.tensor([[[[1.0]], [[2.0]], [[10.0]], [[20.0]]]])
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})
        specs[("0", "input")] = QuantSpec(
            bits=4, scheme="affine", granularity="group", axis=1,
            group_size=2, signed=False, preserve_zero=True)
        maxima = torch.tensor([1.0, 2.0, 10.0, 20.0])
        expected_smooth = haq.smoothquant_scale(
            instrumentor.original_weights["0"], maxima, 0.5,
            input_channel_dim=1)
        expected_maxima = (maxima / expected_smooth).reshape(2, 2).amax(1)

        instrumentor.configure_components_with_ranges(
            4, 4, {"encoder"}, {"encoder"}, specs, False,
            activation_maxima={}, smooth_channel_maxima={"0": maxima},
            smooth_alpha=0.5)

        quantizer = instrumentor.quantizers[("0", "input")]
        self.assertEqual(quantizer.scale_count, 2)
        torch.testing.assert_close(
            quantizer.scale, expected_maxima / float(quantizer.qmax))
        instrumentor.close()

    def test_component_smoothquant_without_qdq_preserves_fp32_output(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=False)).eval()
        sample = torch.tensor([[[[1.0, 2.0]], [[10.0, 20.0]]]])
        reference = model(sample).detach()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()

        instrumentor.configure_components_with_ranges(
            4, 4, set(), set(), {}, False, activation_maxima={},
            smooth_channel_maxima={"0": torch.tensor([2.0, 20.0])},
            smooth_alpha=0.5)
        transformed = model(sample).detach()

        torch.testing.assert_close(transformed, reference, rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.equal(
            model[0].weight, instrumentor.original_weights["0"]))
        instrumentor.close()


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

    def test_fold_validation_uses_primary_prediction_and_keeps_auxiliary_error(self):
        reference = {
            "pred": torch.tensor([1.0, 2.0]),
            "offset": torch.tensor([0.0, 0.0]),
        }
        candidate = {
            "pred": torch.tensor([1.001, 1.999]),
            "offset": torch.tensor([0.2, -0.2]),
        }

        self.assertLess(
            haq._primary_output_error(reference, candidate), 0.01)
        self.assertAlmostEqual(
            haq._maximum_output_error(reference, candidate), 0.2)

    def test_conv_layernorm_fusion_quantizes_input_and_normalized_output(self):
        class PatchStem(nn.Module):
            def __init__(self):
                super(PatchStem, self).__init__()
                self.proj = nn.Conv2d(1, 2, 1)
                self.norm = nn.LayerNorm(2)

            def forward(self, value):
                output = self.proj(value)
                output = output.flatten(2).transpose(1, 2)
                return self.norm(output)

        model = PatchStem().eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        self.assertEqual(
            instrumentor.layernorm_fusions(), [
                {"conv": "proj", "layernorm": "norm"},
            ])
        self.assertNotIn(("proj", "output"), instrumentor.observers)
        self.assertIn(("proj", "input"), instrumentor.observers)
        self.assertIn(("norm", "output"), instrumentor.observers)
        instrumentor.observe()
        sample = torch.randn(1, 1, 4, 4)
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"encoder"})
        output = model(sample)
        quantizer = instrumentor.quantizers[("norm", "output")]
        codes = output / quantizer.scale
        torch.testing.assert_close(codes, codes.round())
        self.assertIn(("proj", "bias"), instrumentor.stats)
        self.assertEqual((quantizer.qmin, quantizer.qmax), (-7, 7))
        metadata = instrumentor.metadata()
        self.assertFalse(metadata["conv_layernorm_kernel_fused"])
        self.assertEqual(
            metadata["conv_layernorm_contract"],
            "Aq input -> Wq Conv -> high-precision accumulator LayerNorm -> "
            "Aq output")
        instrumentor.close()

    def test_conv_layernorm_fusion_can_be_disabled_for_ablation(self):
        class PatchStem(nn.Module):
            def __init__(self):
                super(PatchStem, self).__init__()
                self.proj = nn.Conv2d(1, 2, 1)
                self.norm = nn.LayerNorm(2)

            def forward(self, value):
                output = self.proj(value)
                return self.norm(output.flatten(2).transpose(1, 2))

        model = PatchStem().eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            fuse_layernorm=False)
        self.assertEqual(instrumentor.layernorm_fusions(), [])
        self.assertIn(("proj", "input"), instrumentor.observers)
        self.assertIn(("proj", "output"), instrumentor.observers)
        self.assertNotIn(("norm", "output"), instrumentor.observers)
        instrumentor.close()

    def test_explicit_concat_input_uses_per_channel_activation_quantization(self):
        class Fusion(nn.Module):
            def __init__(self):
                super(Fusion, self).__init__()
                self.concat_conv = nn.Conv2d(2, 2, 1, bias=False)

            def forward(self, value):
                return self.concat_conv(value)

        model = Fusion().eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            per_channel_activation_inputs={"concat_conv"})
        self.assertEqual(
            instrumentor.per_channel_activation_modules(), ["concat_conv"])
        self.assertIn(("concat_conv", "output"), instrumentor.observers)
        instrumentor.observe()
        model(torch.randn(1, 2, 4, 4))
        instrumentor.freeze()
        instrumentor.configure(4, 8, {"encoder"})
        self.assertEqual(
            instrumentor.quantizers[("concat_conv", "input")].scale.numel(), 2)
        self.assertEqual(
            torch.as_tensor(
                instrumentor.quantizers[("concat_conv", "output")].scale
            ).numel(), 1)
        instrumentor.close()

    def test_unknown_per_channel_activation_input_fails(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()

        with self.assertRaisesRegex(
                ValueError, "unknown per-channel activation inputs"):
            haq.HardwareAlignedInstrumentor(
                model, lambda name, module: "encoder",
                per_channel_activation_inputs={"missing"})

    def test_conv_transpose_bn_is_folded_and_quantized_per_output_channel(self):
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.ConvTranspose2d(3, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        ).eval()
        model[1].running_mean.copy_(torch.tensor([0.2, -0.3]))
        model[1].running_var.copy_(torch.tensor([0.7, 1.4]))
        sample = torch.randn(1, 3, 5, 5)
        reference = model(sample)

        preparation = haq.prepare_hardware_model(model, (sample,))

        self.assertEqual(
            preparation["folded_pairs"], [{"conv": "0", "bn": "1"}])
        self.assertIsInstance(model[0], nn.ConvTranspose2d)
        self.assertIsInstance(model[1], nn.Identity)
        torch.testing.assert_close(model(sample), reference, atol=1e-5, rtol=1e-5)

        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "decoder")
        self.assertIn("0", instrumentor.modules)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(
            4, 4, {"decoder"}, quantize_bias=False)

        self.assertEqual(tuple(instrumentor.weight_scales["0"].shape),
                         (1, 2, 1, 1))
        instrumentor.close()

    def test_externally_owned_output_keeps_weight_and_input_qdq_only(self):
        model = nn.Sequential(nn.Conv2d(1, 3, 1)).eval()

        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "propagation_head",
            externally_owned_outputs={"0"})

        self.assertIn(("0", "input"), instrumentor.observers)
        self.assertIn(("0", "output"), instrumentor.observers)
        self.assertEqual(instrumentor.externally_owned_outputs(), ["0"])
        instrumentor.observe()
        sample = torch.randn(1, 1, 3, 3)
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"propagation_head"})
        model(sample)
        self.assertIn(("0", "weight"), instrumentor.stats)
        self.assertIn(("0", "bias"), instrumentor.stats)
        self.assertIn(("0", "input"), instrumentor.stats)
        self.assertNotIn(("0", "output"), instrumentor.stats)
        self.assertEqual(
            instrumentor.metadata()["externally_owned_outputs"], ["0"])

        instrumentor.configure(
            4, 4, {"propagation_head"},
            external_output_ownership=False)
        model(sample)
        self.assertIn(("0", "output"), instrumentor.stats)
        self.assertGreater(instrumentor.stats[("0", "output")].numel, 0)
        instrumentor.close()

    def test_externally_owned_input_keeps_weight_and_output_qdq_only(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=False)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            externally_owned_inputs={"0"})

        self.assertIn(("0", "input"), instrumentor.observers)
        self.assertIn(("0", "output"), instrumentor.observers)
        self.assertEqual(instrumentor.externally_owned_inputs(), ["0"])
        instrumentor.observe()
        sample = torch.randn(1, 2, 3, 3)
        model(sample)
        instrumentor.freeze()
        instrumentor.configure(4, 4, {"encoder"})
        model(sample)

        self.assertIn(("0", "weight"), instrumentor.stats)
        self.assertNotIn(("0", "input"), instrumentor.stats)
        self.assertIn(("0", "output"), instrumentor.stats)
        self.assertEqual(
            instrumentor.metadata()["externally_owned_inputs"], ["0"])
        self.assertEqual(
            instrumentor.metadata()["active_externally_owned_inputs"], ["0"])
        instrumentor.close()

    def test_external_ownership_can_be_disabled_for_reconstruction(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=True)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            externally_owned_inputs={"0"},
            externally_owned_outputs={"0"})
        instrumentor.observe()
        sample = torch.randn(1, 2, 3, 3)
        model(sample)
        instrumentor.freeze()

        instrumentor.set_external_ownership(inputs=set(), outputs=set())
        instrumentor.configure(4, 4, {"encoder"})
        model(sample)

        self.assertIn(("0", "input"), instrumentor.stats)
        self.assertIn(("0", "output"), instrumentor.stats)
        self.assertIn(("0", "bias"), instrumentor.stats)
        self.assertEqual(
            instrumentor.metadata()["active_externally_owned_inputs"], [])
        self.assertEqual(
            instrumentor.metadata()["active_externally_owned_outputs"], [])
        instrumentor.close()

    def test_fully_owned_operation_skips_generic_weight_and_bias(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=True)).eval()
        original_weight = model[0].weight.detach().clone()
        original_bias = model[0].bias.detach().clone()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            externally_owned_inputs={"0"},
            externally_owned_outputs={"0"})
        instrumentor.observe()
        model(torch.randn(1, 2, 3, 3))
        instrumentor.freeze()

        instrumentor.configure(4, 4, {"encoder"})

        self.assertNotIn(("0", "weight"), instrumentor.stats)
        self.assertNotIn(("0", "bias"), instrumentor.stats)
        self.assertNotIn(("0", "input"), instrumentor.stats)
        self.assertNotIn(("0", "output"), instrumentor.stats)
        torch.testing.assert_close(model[0].weight, original_weight)
        torch.testing.assert_close(model[0].bias, original_bias)
        instrumentor.close()

    def test_externally_owned_input_with_generic_bias_fails(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=True)).eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            externally_owned_inputs={"0"})
        instrumentor.observe()
        model(torch.randn(1, 2, 3, 3))
        instrumentor.freeze()

        with self.assertRaisesRegex(RuntimeError, "externally owned input bias"):
            instrumentor.configure(4, 4, {"encoder"})

        instrumentor.close()

    def test_unknown_externally_owned_input_fails(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()

        with self.assertRaisesRegex(
                ValueError, "unknown externally owned inputs"):
            haq.HardwareAlignedInstrumentor(
                model, lambda name, module: "encoder",
                externally_owned_inputs={"missing"})

    def test_group_only_configuration_quantizes_only_owned_relu_sites(self):
        class TwoGroups(nn.Module):
            def __init__(self):
                super(TwoGroups, self).__init__()
                self.encoder_conv = nn.Conv2d(1, 1, 1)
                self.encoder_relu = nn.ReLU()
                self.head_conv = nn.Conv2d(1, 1, 1)
                self.head_relu = nn.ReLU()

            def forward(self, value):
                value = self.encoder_relu(self.encoder_conv(value))
                return self.head_relu(self.head_conv(value))

        def group_fn(name, module):
            del module
            return "encoder" if name.startswith("encoder") else "head"

        model = TwoGroups().eval()
        instrumentor = haq.HardwareAlignedInstrumentor(model, group_fn)
        sample = torch.randn(1, 1, 3, 3)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()

        instrumentor.configure(4, 4, {"encoder"})

        self.assertIn("encoder_relu#0", instrumentor.relu_quantizers)
        self.assertNotIn("head_relu#0", instrumentor.relu_quantizers)
        model(sample)
        relu_rows = [row for row in instrumentor.statistics()
                     if row["kind"] == "relu_output"]
        self.assertEqual(
            [(row["module"], row["group"]) for row in relu_rows],
            [("encoder_relu#0", "encoder")])
        instrumentor.close()

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

    def test_selected_module_uses_w8_while_other_module_remains_w4(self):
        model, instrumentor, sample = self._calibrated_model()

        instrumentor.configure(
            4, 4, {"encoder"}, weight_bit_overrides={"0": 8})

        self.assertEqual(instrumentor.weight_bits_by_module(), {
            "0": 8,
            "2": 4,
        })
        expected_w8, expected_w8_scale = haq.symmetric_weight_qdq(
            instrumentor.original_weights["0"], 8, channel_dim=0)
        expected_w4, expected_w4_scale = haq.symmetric_weight_qdq(
            instrumentor.original_weights["2"], 4, channel_dim=0)
        torch.testing.assert_close(model[0].weight.cpu(), expected_w8)
        torch.testing.assert_close(model[2].weight.cpu(), expected_w4)
        torch.testing.assert_close(
            instrumentor.weight_scales["0"], expected_w8_scale)
        torch.testing.assert_close(
            instrumentor.weight_scales["2"], expected_w4_scale)
        model(sample)
        instrumentor.close()

    def test_components_with_weight_bit_overrides_preserve_mixed_weights(self):
        model, instrumentor, sample = self._calibrated_model()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})

        instrumentor.configure_components_with_ranges(
            4, 4, {"encoder"}, {"encoder"}, specs, False, {},
            weight_bit_overrides={"0": 8})

        self.assertEqual(instrumentor.weight_bits_by_module(), {
            "0": 8,
            "2": 4,
        })
        model(sample)
        instrumentor.close()

    def test_weight_bit_overrides_accept_two_and_six_bits(self):
        model, instrumentor, sample = self._calibrated_model()
        specs = instrumentor.tensor_activation_specs(4, {"encoder"})

        instrumentor.configure_components_with_ranges(
            4, 4, {"encoder"}, {"encoder"}, specs, False, {},
            weight_bit_overrides={"0": 2, "2": 6})

        self.assertEqual(instrumentor.weight_bits_by_module(), {
            "0": 2,
            "2": 6,
        })
        expected_w2, expected_w2_scale = haq.symmetric_weight_qdq(
            instrumentor.original_weights["0"], 2, channel_dim=0)
        expected_w6, expected_w6_scale = haq.symmetric_weight_qdq(
            instrumentor.original_weights["2"], 6, channel_dim=0)
        torch.testing.assert_close(model[0].weight.cpu(), expected_w2)
        torch.testing.assert_close(model[2].weight.cpu(), expected_w6)
        torch.testing.assert_close(
            instrumentor.weight_scales["0"], expected_w2_scale)
        torch.testing.assert_close(
            instrumentor.weight_scales["2"], expected_w6_scale)
        model(sample)
        instrumentor.close()

    def test_unknown_weight_bit_override_fails(self):
        _, instrumentor, _ = self._calibrated_model()

        with self.assertRaisesRegex(ValueError, "unknown weight bit overrides"):
            instrumentor.configure(
                4, 4, {"encoder"},
                weight_bit_overrides={"missing": 8})

        instrumentor.close()

    def test_invalid_weight_bit_override_fails(self):
        _, instrumentor, _ = self._calibrated_model()

        with self.assertRaisesRegex(
                ValueError, "weight bits must be one of.*2.*4.*6.*8"):
            instrumentor.configure(
                4, 4, {"encoder"}, weight_bit_overrides={"0": 3})

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

    def test_e2m1_quantizes_conv_relu_and_keeps_bias_fp32(self):
        model, instrumentor, sample = self._calibrated_model()
        original_biases = [
            model[0].bias.detach().clone(),
            model[2].bias.detach().clone(),
        ]

        instrumentor.configure(
            8, 4, {"encoder"}, activation_mode="e2m1",
            quantize_bias=False)
        model(sample)

        self.assertEqual(
            instrumentor.quantizers[("0", "input")].format, "e2m1")
        self.assertEqual(
            instrumentor.relu_quantizers["1#0"].format, "e2m1")
        self.assertNotIn(("0", "bias"), instrumentor.stats)
        torch.testing.assert_close(model[0].bias, original_biases[0])
        torch.testing.assert_close(model[2].bias, original_biases[1])
        self.assertEqual(
            instrumentor.metadata()["bias_contract"], "fp32_isolation")
        activation_rows = [
            row for row in instrumentor.statistics()
            if row["kind"] not in ("weight", "bias")
        ]
        self.assertTrue(
            all("zero_code_rate" in row for row in activation_rows))
        self.assertTrue(
            all("nonfinite_rate" in row for row in activation_rows))
        instrumentor.close()

    def test_e2m1_override_keeps_semantic_site_uniform_a8(self):
        model, instrumentor, sample = self._calibrated_model()

        instrumentor.configure(
            8, 4, {"encoder"}, activation_mode="e2m1",
            activation_bit_overrides={("0", "input"): 8},
            activation_format_overrides={("0", "input"): "uniform"},
            quantize_bias=False)

        self.assertEqual(
            instrumentor.quantizers[("0", "input")].format, "uniform")
        self.assertEqual(
            instrumentor.quantizers[("0", "input")].bits, 8)
        self.assertEqual(
            instrumentor.quantizers[("2", "input")].format, "e2m1")
        model(sample)
        instrumentor.close()

    def test_e2m1_layernorm_output_retains_per_channel_scale(self):
        class PatchStem(nn.Module):
            def __init__(self):
                super(PatchStem, self).__init__()
                self.proj = nn.Conv2d(1, 2, 1)
                self.norm = nn.LayerNorm(2)

            def forward(self, value):
                output = self.proj(value)
                output = output.flatten(2).transpose(1, 2)
                return self.norm(output)

        model = PatchStem().eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        sample = torch.randn(1, 1, 4, 4)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()

        instrumentor.configure(
            8, 4, {"encoder"}, activation_mode="e2m1",
            quantize_bias=False)

        quantizer = instrumentor.quantizers[("norm", "output")]
        self.assertEqual(quantizer.format, "e2m1")
        self.assertEqual(quantizer.scale.numel(), 2)
        model(sample)
        instrumentor.close()

    def test_e2m1_concat_input_retains_per_channel_scale(self):
        class Fusion(nn.Module):
            def __init__(self):
                super(Fusion, self).__init__()
                self.concat_conv = nn.Conv2d(2, 2, 1, bias=False)

            def forward(self, value):
                return self.concat_conv(value)

        model = Fusion().eval()
        instrumentor = haq.HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder",
            per_channel_activation_inputs={"concat_conv"})
        sample = torch.randn(1, 2, 4, 4)
        instrumentor.observe()
        model(sample)
        instrumentor.freeze()

        instrumentor.configure(
            8, 4, {"encoder"}, activation_mode="e2m1",
            quantize_bias=False)

        quantizer = instrumentor.quantizers[("concat_conv", "input")]
        self.assertEqual(quantizer.format, "e2m1")
        self.assertEqual(quantizer.scale.numel(), 2)
        model(sample)
        instrumentor.close()


if __name__ == "__main__":
    unittest.main()
