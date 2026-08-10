import unittest
import csv
from pathlib import Path
import tempfile

import numpy as np
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


class ActivationHistogramRecorderTest(unittest.TestCase):
    def _record_input(self, model_name, module, channels):
        recorder = histograms.ActivationHistogramRecorder(
            model_name=model_name, phase="range",
            capacity=128, per_update=128)
        values = torch.arange(
            1, channels + 1, dtype=torch.float32).reshape(1, channels, 1, 1)
        quantizer = haq.UnsignedActivationQuantizer(8, float(channels))
        quantized, codes = quantizer.quantize_with_codes(values)
        recorder.record(
            module, "input", 0, "encoder",
            values, quantized, codes, quantizer, channel_dim=1)
        return recorder

    def test_cspn_combined_input_is_split_into_rgb_and_depth(self):
        recorder = self._record_input("cspn", "conv1_1", 4)

        self.assertEqual(set(recorder.site_names()), {
            "conv1_1#0:input",
            "input_rgb#0:input",
            "input_depth#0:input",
        })
        self.assertEqual(
            recorder.site_metadata["input_rgb#0:input"]["channels"], 3)
        self.assertEqual(
            recorder.site_metadata["input_depth#0:input"]["channels"], 1)

    def test_separate_model_stems_are_labeled_as_rgb_and_depth(self):
        stems = {
            "dyspn": ("base.conv1_rgb.0", "base.conv1_dep.0"),
            "nlspn": ("conv1_rgb.0", "conv1_dep.0"),
            "completionformer": (
                "backbone.conv1_rgb.0", "backbone.conv1_dep.0"),
        }
        for model_name in stems:
            with self.subTest(model=model_name):
                recorder = self._record_input(
                    model_name, stems[model_name][0], 3)
                depth = self._record_input(
                    model_name, stems[model_name][1], 1)
                self.assertIn("input_rgb#0:input", recorder.site_names())
                self.assertIn("input_depth#0:input", depth.site_names())

    def test_histogram_pass_rejects_quantizer_changes(self):
        recorder = self._record_input("nlspn", "conv1_rgb.0", 3)
        recorder.freeze_ranges(bin_count=16)
        recorder.begin_histogram_pass()
        values = torch.ones(1, 3, 1, 1)
        changed = haq.UnsignedActivationQuantizer(4, 3.0)
        quantized, codes = changed.quantize_with_codes(values)

        with self.assertRaisesRegex(ValueError, "quantizer changed"):
            recorder.record(
                "conv1_rgb.0", "input", 0, "encoder",
                values, quantized, codes, changed, channel_dim=1)

    def test_coverage_requires_every_real_site_on_every_sample(self):
        recorder = self._record_input("nlspn", "conv1_rgb.0", 3)
        recorder.freeze_ranges(bin_count=16)
        recorder.begin_histogram_pass()
        values = torch.arange(1, 4, dtype=torch.float32).reshape(1, 3, 1, 1)
        quantizer = haq.UnsignedActivationQuantizer(8, 3.0)
        quantized, codes = quantizer.quantize_with_codes(values)
        recorder.record(
            "conv1_rgb.0", "input", 0, "encoder",
            values, quantized, codes, quantizer, channel_dim=1)

        with self.assertRaisesRegex(ValueError, "missing manifest sites"):
            recorder.validate(
                {"conv1_rgb.0#0:input", "missing#0:output"},
                expected_updates=1)
        recorder.validate({"conv1_rgb.0#0:input"}, expected_updates=1)

    def test_npz_csv_round_trip_indexes_every_array(self):
        recorder = self._record_input("cspn", "conv1_1", 4)
        recorder.freeze_ranges(bin_count=16)
        recorder.begin_histogram_pass()
        values = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1, 1)
        quantizer = haq.UnsignedActivationQuantizer(8, 4.0)
        quantized, codes = quantizer.quantize_with_codes(values)
        recorder.record(
            "conv1_1", "input", 0, "encoder",
            values, quantized, codes, quantizer, channel_dim=1)
        recorder.validate({"conv1_1#0:input"}, expected_updates=1)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            recorder.write(output)
            with (output / "histogram_index.csv").open(
                    newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            with np.load(output / "histogram_data.npz") as arrays:
                indexed = {
                    row[field]
                    for row in rows
                    for field in row
                    if field.endswith("_key")
                }
                self.assertEqual(indexed, set(arrays.files))
                for row in rows:
                    self.assertEqual(
                        int(arrays[row["code_counts_key"]].sum()),
                        int(row["elements"]))

            with (output / "outlier_summary.csv").open(
                    newline="", encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 3)
            real = [row for row in summary
                    if int(row["synthetic_slice"]) == 0]
            synthetic = [row for row in summary
                         if int(row["synthetic_slice"]) == 1]
            self.assertAlmostEqual(sum(
                float(row["local_error_energy_share"]) for row in real), 1.0)
            self.assertTrue(all(
                int(row["excluded_from_aggregate"]) == 1
                for row in synthetic))


if __name__ == "__main__":
    unittest.main()
