import unittest

import torch
import torch.nn as nn

from scripts import activation_outlier_analysis as analysis


class BoundedActivationSamplerTest(unittest.TestCase):
    def test_default_update_budget_can_fill_capacity(self):
        self.assertEqual(analysis.per_update_budget(1000000, 128), 7813)

    def test_percentiles_are_monotonic_and_sampling_is_bounded(self):
        sampler = analysis.BoundedActivationSampler(capacity=64, per_update=16)
        values = torch.arange(400, dtype=torch.float32).reshape(1, 4, 10, 10)

        for _ in range(8):
            sampler.update(values, channel_axis=1)
        row = sampler.statistics()

        self.assertEqual(row["retained_values"], 64)
        self.assertEqual(row["total_values"], 3200)
        percentiles = [row[name] for name in (
            "p75", "p90", "p99", "p99_9", "p99_99", "maximum")]
        self.assertEqual(percentiles, sorted(percentiles))
        self.assertGreaterEqual(row["max_over_p99_99"], 1.0)

    def test_sampling_is_deterministic_and_channel_maxima_are_exact(self):
        values = torch.tensor([
            [[[1.0, -2.0]], [[3.0, -9.0]], [[4.0, -5.0]]],
        ])
        first = analysis.BoundedActivationSampler(capacity=4, per_update=2)
        second = analysis.BoundedActivationSampler(capacity=4, per_update=2)

        first.update(values, channel_axis=1)
        second.update(values, channel_axis=1)

        self.assertEqual(first.statistics(), second.statistics())
        torch.testing.assert_close(
            first.channel_absmax, torch.tensor([2.0, 9.0, 5.0]))
        row = first.statistics()
        self.assertAlmostEqual(row["channel_max_over_median"], 9.0 / 5.0)


class ActivationOutlierProfilerTest(unittest.TestCase):
    def test_shared_module_calls_are_profiled_separately(self):
        class SharedLinear(nn.Module):
            def __init__(self):
                super(SharedLinear, self).__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, value):
                first = self.linear(value)
                return self.linear(first * 10.0)

        model = SharedLinear().eval()
        profiler = analysis.ActivationOutlierProfiler(
            model, lambda name, module: "encoder", capacity=128, per_update=64)

        model(torch.ones(2, 4))
        rows = profiler.rows()

        input_sites = [row["site"] for row in rows if row["kind"] == "input"]
        self.assertEqual(input_sites, ["linear#0", "linear#1"])
        self.assertGreater(
            next(row["maximum"] for row in rows if row["site"] == "linear#1"
                 and row["kind"] == "input"), 1.0)
        profiler.close()

    def test_occupancy_shares_sum_to_one_per_measure(self):
        rows = [
            {"model": "dyspn", "config": "HW_W4A4_full", "group": "encoder",
             "kind": "weight", "numel": "75"},
            {"model": "dyspn", "config": "HW_W4A4_full", "group": "decoder",
             "kind": "weight", "numel": "25"},
            {"model": "dyspn", "config": "HW_W4A4_full", "group": "encoder",
             "kind": "input", "numel": "60"},
            {"model": "dyspn", "config": "HW_W4A4_full", "group": "decoder",
             "kind": "output", "numel": "40"},
        ]

        result = analysis.occupancy_rows(rows, config="HW_W4A4_full")

        for measure in ("parameter_elements", "activation_elements", "boundaries"):
            self.assertAlmostEqual(
                sum(row["share"] for row in result if row["measure"] == measure),
                1.0)


if __name__ == "__main__":
    unittest.main()
