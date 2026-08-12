import math
import unittest

import torch

from scripts.hardware_aligned_quantization import (
    SymmetricActivationQuantizer,
    UnsignedActivationQuantizer,
)
from spn_quant.activation_resolution import (
    ActivationResolutionAccumulator,
    ActivationResolutionRecorder,
    BoundedChannelSampler,
)


class BoundedChannelSamplerTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_bounded(self):
        values = torch.arange(40, dtype=torch.float32).reshape(2, 20)
        first = BoundedChannelSampler(channels=2, capacity=8)
        second = BoundedChannelSampler(channels=2, capacity=8)

        first.update(values)
        second.update(values)

        self.assertEqual(first.sample_count, 8)
        torch.testing.assert_close(
            first.percentiles((0.75, 0.99)),
            second.percentiles((0.75, 0.99)))


class ActivationResolutionAccumulatorTest(unittest.TestCase):
    def test_error_partition_conserves_total_energy(self):
        quantizer = SymmetricActivationQuantizer(4, 7.0)
        reference = torch.tensor([-9.0, -0.2, 0.0, 0.4, 9.0])
        quantized, codes = quantizer.quantize_with_codes(reference)
        accumulator = ActivationResolutionAccumulator(
            channel_dim=0, capacity=32)

        accumulator.update(reference, quantized, codes, quantizer)

        row = accumulator.tensor_summary()
        partition = (
            row["zero_collapse_error_energy"] +
            row["rounding_error_energy"] +
            row["clipping_error_energy"])
        self.assertAlmostEqual(partition, row["total_error_energy"], places=10)
        self.assertGreater(row["zero_collapse_error_energy"], 0.0)
        self.assertGreater(row["clipping_error_energy"], 0.0)

    def test_new_zero_rate_excludes_reference_zeros(self):
        quantizer = SymmetricActivationQuantizer(4, 7.0)
        reference = torch.tensor([0.0, 0.2, 1.0])
        quantized, codes = quantizer.quantize_with_codes(reference)
        accumulator = ActivationResolutionAccumulator(
            channel_dim=0, capacity=16)

        accumulator.update(reference, quantized, codes, quantizer)

        row = accumulator.tensor_summary()
        self.assertAlmostEqual(row["reference_zero_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(row["new_zero_rate"], 0.5)
        self.assertEqual(row["nonzero_elements"], 2)
        self.assertEqual(row["new_zero_elements"], 1)

    def test_channel_error_shares_sum_to_one(self):
        quantizer = SymmetricActivationQuantizer(4, 7.0)
        reference = torch.tensor([[[[0.2]], [[1.4]]]])
        quantized, codes = quantizer.quantize_with_codes(reference)
        accumulator = ActivationResolutionAccumulator(
            channel_dim=1, capacity=16)

        accumulator.update(reference, quantized, codes, quantizer)

        rows = accumulator.channel_summaries()
        self.assertEqual([row["channel"] for row in rows], [0, 1])
        self.assertAlmostEqual(
            sum(row["error_energy_share"] for row in rows), 1.0)

    def test_effective_code_count_uses_code_entropy(self):
        quantizer = SymmetricActivationQuantizer(4, 1.0)
        reference = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        quantized, codes = quantizer.quantize_with_codes(reference)
        accumulator = ActivationResolutionAccumulator(
            channel_dim=0, capacity=16)

        accumulator.update(reference, quantized, codes, quantizer)

        self.assertAlmostEqual(
            accumulator.tensor_summary()["effective_code_count"], 2.0)

    def test_per_channel_percentiles_use_absolute_magnitude(self):
        quantizer = UnsignedActivationQuantizer(4, 14.0)
        reference = torch.tensor([[[[1.0, 2.0]], [[10.0, 15.0]]]])
        quantized, codes = quantizer.quantize_with_codes(reference)
        accumulator = ActivationResolutionAccumulator(
            channel_dim=1, capacity=16)

        accumulator.update(reference, quantized, codes, quantizer)

        rows = accumulator.channel_summaries()
        self.assertLess(rows[0]["p99"], rows[1]["p99"])
        self.assertEqual(rows[1]["maximum_abs"], 15.0)
        self.assertTrue(math.isfinite(rows[0]["sqnr_db"]))


class ActivationResolutionRecorderTest(unittest.TestCase):
    def test_recorder_adds_site_metadata_and_split(self):
        recorder = ActivationResolutionRecorder(
            split="evaluation", capacity=16)
        quantizer = SymmetricActivationQuantizer(4, 1.0)
        reference = torch.ones(1, 1, 1, 1)
        quantized, codes = quantizer.quantize_with_codes(reference)

        recorder.record(
            "conv", "input", 0, "encoder", reference,
            quantized, codes, quantizer, channel_dim=1)

        tensor_row = recorder.tensor_rows()[0]
        channel_row = recorder.channel_rows()[0]
        self.assertEqual(tensor_row["site"], "conv#0:input")
        self.assertEqual(tensor_row["split"], "evaluation")
        self.assertEqual(tensor_row["group"], "encoder")
        self.assertEqual(channel_row["site"], "conv#0:input")
        self.assertEqual(channel_row["channel"], 0)

    def test_recorder_rejects_unknown_split(self):
        with self.assertRaisesRegex(ValueError, "split"):
            ActivationResolutionRecorder(split="test", capacity=16)


if __name__ == "__main__":
    unittest.main()
