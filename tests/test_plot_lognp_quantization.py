import unittest

from scripts.plot_lognp_quantization import (
    aggregate_lognp_rows,
    compare_lognp_configs,
)


class LogNPReportTest(unittest.TestCase):
    def test_aggregation_is_weighted_and_retains_tail_and_nonfinite_counts(self):
        rows = [
            {"model": "cspn", "config": "LOGNP_W8A4_tensor",
             "module": "a", "kind": "input", "numel": "2",
             "mse": "1.0", "p50": "1.0", "p75": "2.0",
             "p99": "3.0", "p99_9": "4.0", "nonfinite_rate": "0.1"},
            {"model": "cspn", "config": "LOGNP_W8A4_tensor",
             "module": "b", "kind": "input", "numel": "6",
             "mse": "3.0", "p50": "3.0", "p75": "4.0",
             "p99": "5.0", "p99_9": "6.0", "nonfinite_rate": "0.2"},
        ]

        result = aggregate_lognp_rows(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["numel"], 8)
        self.assertAlmostEqual(result[0]["mse"], 2.5)
        self.assertAlmostEqual(result[0]["p99_9"], 5.5)
        self.assertAlmostEqual(result[0]["nonfinite_rate"], 0.175)

    def test_comparison_adds_deltas_against_tensor_baseline(self):
        rows = [
            {"model": "dyspn", "config": "LOGNP_W8A4_tensor",
             "numel": 10, "mse": 4.0, "p99_9": 8.0,
             "nonfinite_rate": 0.0},
            {"model": "dyspn", "config": "LOGNP_W8A4_channel",
             "numel": 10, "mse": 2.0, "p99_9": 5.0,
             "nonfinite_rate": 0.0},
        ]

        result = compare_lognp_configs(rows)
        channel = [row for row in result
                   if row["config"] == "LOGNP_W8A4_channel"][0]

        self.assertAlmostEqual(channel["mse_delta"], -2.0)
        self.assertAlmostEqual(channel["p99_9_delta"], -3.0)


if __name__ == "__main__":
    unittest.main()
