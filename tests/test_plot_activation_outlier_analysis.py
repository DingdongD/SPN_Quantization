import unittest

from scripts import plot_activation_outlier_analysis as plotting


class ActivationOutlierPlotDataTest(unittest.TestCase):
    def test_nonfinite_rate_includes_finite_and_nonfinite_pixels(self):
        rows = [
            {"model": "cspn", "config": "HW_W4A4_MinMax",
             "nonfinite_pixels": "75", "num_pixels": "25"},
            {"model": "cspn", "config": "HW_W4A4_MinMax",
             "nonfinite_pixels": "5", "num_pixels": "95"},
        ]

        rates = plotting.nonfinite_rates(rows)

        self.assertAlmostEqual(rates[("cspn", "HW_W4A4_MinMax")], 0.4)


    def test_encoder_occupancy_keeps_measure_semantics(self):
        rows = [
            {"model": "nlspn", "config": "HW_W4A4_full",
             "measure": "parameter_elements", "group": "encoder",
             "value": "90", "share": "0.9"},
            {"model": "nlspn", "config": "HW_W4A4_full",
             "measure": "parameter_elements", "group": "decoder",
             "value": "10", "share": "0.1"},
            {"model": "nlspn", "config": "HW_W4A4_full",
             "measure": "activation_elements", "group": "encoder",
             "value": "55", "share": "0.55"},
        ]

        result = plotting.encoder_occupancy_rows(rows)

        self.assertEqual([row["measure"] for row in result],
                         ["parameter_elements", "activation_elements"])
        self.assertEqual([row["encoder_share"] for row in result], [0.9, 0.55])

    def test_percentile_tail_rows_select_worst_spatial_tail(self):
        rows = [
            {"model": "cspn", "metric": "spatial_tail", "rank": 2,
             "site": "less#0", "group": "decoder", "ratio": 2.0,
             "p75": 1.0, "p99": 2.0, "p99_9": 3.0,
             "p99_99": 4.0, "maximum": 8.0},
            {"model": "cspn", "metric": "spatial_tail", "rank": 1,
             "site": "worst#0", "group": "encoder", "ratio": 5.0,
             "p75": 0.5, "p99": 2.0, "p99_9": 4.0,
             "p99_99": 6.0, "maximum": 10.0},
        ]

        result = plotting.percentile_tail_rows(rows)

        self.assertEqual([row["percentile"] for row in result],
                         ["p75", "p99", "p99.9", "p99.99", "max"])
        self.assertEqual(set(row["site"] for row in result), {"worst#0"})
        self.assertEqual([row["over_p99"] for row in result],
                         [0.25, 1.0, 2.0, 3.0, 5.0])


if __name__ == "__main__":
    unittest.main()
