import unittest

import torch

from scripts import activation_histograms as histograms
from scripts import hardware_aligned_quantization as haq


class RangeCollectorTest(unittest.TestCase):
    def test_quantizer_descriptor_preserves_scalar_and_vector_scales(self):
        scalar = haq.SymmetricActivationQuantizer(4, 7.0)
        scalar_row = histograms.quantizer_descriptor(scalar, channel_dim=1)

        self.assertEqual(scalar_row["bits"], 4)
        self.assertFalse(scalar_row["unsigned"])
        self.assertEqual(scalar_row["granularity"], "per_tensor")
        self.assertEqual(scalar_row["scale"].shape, (1,))

        channel = haq.ChannelActivationQuantizer(
            4, torch.zeros(2), torch.tensor([1.0, 10.0]),
            channel_dim=1, unsigned=True)
        channel_row = histograms.quantizer_descriptor(channel, channel_dim=1)

        self.assertTrue(channel_row["unsigned"])
        self.assertEqual(channel_row["qmin"], 0)
        self.assertEqual(channel_row["qmax"], 15)
        self.assertEqual(channel_row["granularity"], "per_channel")
        torch.testing.assert_close(
            torch.from_numpy(channel_row["scale"]),
            channel.scale.double())

    def test_preserves_exact_counts_and_energies(self):
        quantizer = haq.UnsignedActivationQuantizer(4, 3.0)
        reference = torch.tensor([[[[0.0, 0.1, 1.0, 3.5]]]])
        quantized, codes = quantizer.quantize_with_codes(reference)
        collector = histograms.RangeCollector(capacity=32, per_update=32)

        collector.update(
            reference, quantized, codes, quantizer, channel_dim=1)
        row = collector.summary()

        self.assertEqual(row["elements"], 4)
        self.assertEqual(row["updates"], 1)
        self.assertEqual(row["reference_zeros"], 1)
        self.assertEqual(row["zero_codes"], 2)
        self.assertEqual(row["saturated_values"], 1)
        self.assertEqual(row["endpoint_codes"], 3)
        self.assertAlmostEqual(
            row["signal_energy"], float((reference.double() ** 2).sum()))
        self.assertAlmostEqual(
            row["error_energy"],
            float(((reference.double() - quantized.double()) ** 2).sum()))
        self.assertEqual(tuple(collector.channel_absmax.shape), (1,))

    def test_per_channel_scale_is_used_for_saturation(self):
        quantizer = haq.ChannelActivationQuantizer(
            4, torch.zeros(2), torch.tensor([1.0, 10.0]),
            channel_dim=1, unsigned=True)
        reference = torch.tensor([[[[1.2, 0.5]], [[12.0, 5.0]]]])
        quantized, codes = quantizer.quantize_with_codes(reference)
        collector = histograms.RangeCollector(capacity=32, per_update=32)

        collector.update(
            reference, quantized, codes, quantizer, channel_dim=1)
        row = collector.summary()

        self.assertEqual(row["saturated_values"], 2)
        self.assertEqual(row["endpoint_codes"], 2)
        torch.testing.assert_close(
            collector.channel_absmax, torch.tensor([1.2, 12.0]))

    def test_rejects_nonfinite_reference(self):
        quantizer = haq.SymmetricActivationQuantizer(4, 1.0)
        reference = torch.tensor([0.0, float("nan")])
        quantized, codes = quantizer.quantize_with_codes(reference)
        collector = histograms.RangeCollector(capacity=32, per_update=32)

        with self.assertRaisesRegex(ValueError, "reference contains nonfinite"):
            collector.update(
                reference, quantized, codes, quantizer, channel_dim=0)

    def test_sparse_magnitude_range_uses_observed_positive_value(self):
        quantizer = haq.UnsignedActivationQuantizer(4, 2.0)
        reference = torch.zeros(200)
        reference[-1] = 0.25
        quantized, codes = quantizer.quantize_with_codes(reference)
        collector = histograms.RangeCollector(capacity=256, per_update=256)

        collector.update(
            reference, quantized, codes, quantizer, channel_dim=0)
        row = collector.summary()

        self.assertEqual(row["p99"], 0.0)
        self.assertEqual(row["magnitude_normalizer"], 0.25)
        self.assertEqual(
            row["magnitude_normalizer_kind"], "minimum_positive_sample")

    def test_all_zero_magnitude_range_is_explicit(self):
        quantizer = haq.UnsignedActivationQuantizer(4, 0.0)
        reference = torch.zeros(32)
        quantized, codes = quantizer.quantize_with_codes(reference)
        collector = histograms.RangeCollector(capacity=32, per_update=32)

        collector.update(
            reference, quantized, codes, quantizer, channel_dim=0)
        row = collector.summary()

        self.assertEqual(row["magnitude_normalizer"], quantizer.scale)
        self.assertEqual(
            row["magnitude_normalizer_kind"], "quantizer_scale_all_zero")


class HistogramAccumulatorTest(unittest.TestCase):
    def test_conserves_reference_error_and_code_counts(self):
        quantizer = haq.SymmetricActivationQuantizer(4, 4.0)
        reference = torch.linspace(-5.0, 5.0, 101).reshape(1, 1, 1, -1)
        reference[..., 50] = 0.0
        quantized, codes = quantizer.quantize_with_codes(reference)
        ranges = histograms.RangeCollector(capacity=256, per_update=256)
        ranges.update(reference, quantized, codes, quantizer, channel_dim=1)
        accumulator = histograms.HistogramAccumulator.from_range(
            ranges, bin_count=64)

        accumulator.update(reference, quantized, codes)

        self.assertEqual(accumulator.reference_total, reference.numel())
        self.assertEqual(accumulator.magnitude_total, reference.numel())
        self.assertEqual(accumulator.error_total, reference.numel())
        self.assertEqual(accumulator.code_total, reference.numel())
        self.assertGreater(accumulator.reference_underflow, 0)
        self.assertGreater(accumulator.reference_overflow, 0)
        self.assertEqual(accumulator.reference_zeros, 1)

    def test_multiple_updates_are_additive(self):
        quantizer = haq.UnsignedActivationQuantizer(4, 2.0)
        reference = torch.tensor([0.0, 0.5, 1.0, 2.0])
        quantized, codes = quantizer.quantize_with_codes(reference)
        ranges = histograms.RangeCollector(capacity=32, per_update=32)
        ranges.update(reference, quantized, codes, quantizer, channel_dim=0)
        accumulator = histograms.HistogramAccumulator.from_range(
            ranges, bin_count=16)

        accumulator.update(reference, quantized, codes)
        accumulator.update(reference, quantized, codes)

        self.assertEqual(accumulator.reference_total, 8)
        self.assertEqual(accumulator.magnitude_total, 8)
        self.assertEqual(accumulator.error_total, 8)
        self.assertEqual(accumulator.code_total, 8)
        self.assertEqual(accumulator.reference_zeros, 2)


if __name__ == "__main__":
    unittest.main()
