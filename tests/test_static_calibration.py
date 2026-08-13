import unittest

import torch

from spn_quant.static_calibration import (
    GroupedHistogramObserver,
    HistogramSite,
    StaticCalibrationRecorder,
)


class GroupedHistogramObserverTest(unittest.TestCase):
    def test_grouped_update_counts_each_entity_independently(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([4.0, 8.0]), channels=4,
            channel_dim=1, group_size=2, bins=4, signed=True)
        values = torch.tensor([[[[-4.0]], [[2.0]], [[0.0]], [[8.0]]]])

        observer.update(values)

        torch.testing.assert_close(observer.counts, torch.tensor([
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ], dtype=torch.int64))

    def test_updates_accumulate_without_changing_range(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([2.0]), channels=2,
            channel_dim=1, group_size=2, bins=2, signed=False)

        observer.update(torch.tensor([[[[0.0]], [[2.0]]]]))
        observer.update(torch.tensor([[[[0.5]], [[1.5]]]]))

        torch.testing.assert_close(
            observer.counts, torch.tensor([[2, 2]], dtype=torch.int64))
        self.assertEqual(observer.updates, 2)

    def test_zero_range_accepts_only_zero_values(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([0.0]), channels=1,
            channel_dim=1, group_size=1, bins=8, signed=False)

        observer.update(torch.zeros(1, 1, 2, 2))

        self.assertEqual(int(observer.counts[0, 0]), 4)
        with self.assertRaisesRegex(ValueError, "zero-range"):
            observer.update(torch.ones(1, 1, 1, 1))

    def test_nonfinite_input_fails_directly(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([1.0]), channels=1,
            channel_dim=1, group_size=1, bins=8, signed=True)

        with self.assertRaisesRegex(ValueError, "finite"):
            observer.update(torch.tensor([[[[float("nan")]]]]))


class HistogramThresholdTest(unittest.TestCase):
    def test_percentile_selects_upper_bin_boundary(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([8.0]), channels=1,
            channel_dim=1, group_size=1, bins=8, signed=False)
        observer.counts[0] = torch.tensor(
            [90, 5, 3, 1, 1, 0, 0, 0], dtype=torch.int64)
        observer.updates = 1

        threshold = observer.thresholds("percentile", percentile=0.99)

        torch.testing.assert_close(threshold, torch.tensor([4.0]))

    def test_signed_histogram_mse_uses_seven_positive_codes(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([1.0]), channels=1,
            channel_dim=1, group_size=1, bins=16, signed=True)
        observer.counts[0] = torch.tensor(
            [100, 100, 100, 100, 20, 20, 10, 10,
             2, 2, 1, 1, 1, 1, 1, 1], dtype=torch.int64)
        observer.updates = 1

        threshold = observer.thresholds("hist_mse", bits=4)

        self.assertGreater(float(threshold[0]), 0.0)
        self.assertLessEqual(float(threshold[0]), 1.0)
        self.assertEqual(observer.qmax(4), 7)

    def test_unsigned_histogram_mse_uses_fifteen_codes(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([1.0]), channels=1,
            channel_dim=1, group_size=1, bins=16, signed=False)

        self.assertEqual(observer.qmax(4), 15)

    def test_histogram_mse_tie_selects_larger_threshold(self):
        observer = GroupedHistogramObserver(
            maximum=torch.tensor([1.0]), channels=1,
            channel_dim=1, group_size=1, bins=4, signed=False)
        observer.counts[0] = torch.tensor([1, 0, 0, 0], dtype=torch.int64)
        observer.updates = 1

        threshold = observer.thresholds("hist_mse", bits=4)

        torch.testing.assert_close(threshold, torch.tensor([1.0]))


class StaticCalibrationRecorderTest(unittest.TestCase):
    def test_recorder_requires_and_summarizes_every_declared_owner(self):
        sites = (
            HistogramSite(
                "conv", "input", "encoder", 2, 1, 2,
                torch.tensor([2.0]), True),
            HistogramSite(
                "relu#0", "relu_output", "encoder", 2, 1, 2,
                torch.tensor([1.0]), False),
        )
        recorder = StaticCalibrationRecorder(sites, bins=8)

        recorder.record_reference(
            "conv", "input", "encoder",
            torch.tensor([[[[-2.0]], [[1.0]]]]), 1)
        recorder.record_reference(
            "relu#0", "relu_output", "encoder",
            torch.tensor([[[[0.0]], [[1.0]]]]), 1)
        thresholds = recorder.thresholds("percentile_p999", bits=4)
        rows = recorder.rows("percentile_p999", thresholds)

        self.assertEqual(set(thresholds), {
            ("conv", "input"), ("relu#0", "relu_output")})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["scale_count"] == 1 for row in rows))

    def test_recorder_rejects_undeclared_owner(self):
        recorder = StaticCalibrationRecorder((
            HistogramSite(
                "conv", "input", "encoder", 1, 1, 1,
                torch.tensor([1.0]), True),
        ), bins=8)

        with self.assertRaisesRegex(ValueError, "undeclared"):
            recorder.record_reference(
                "other", "input", "encoder", torch.ones(1, 1, 1, 1), 1)


if __name__ == "__main__":
    unittest.main()
