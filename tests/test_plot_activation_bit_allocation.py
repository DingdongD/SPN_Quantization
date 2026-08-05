import math
import unittest

from scripts import plot_activation_bit_allocation as plotting


class ActivationAllocationPlottingTest(unittest.TestCase):
    def test_short_label_preserves_distinct_allocation_meaning(self):
        self.assertEqual(plotting.short_config_label("MP_W4A4_base"), "A4 base")
        self.assertEqual(plotting.short_config_label("MP_W4A8_full"), "A8 full")
        self.assertEqual(plotting.short_config_label("MP_site01_A8"), "Site 1 A8")
        self.assertEqual(plotting.short_config_label("MP_top4_A8"), "Top 4 A8")
        self.assertEqual(plotting.short_config_label(
            "MP_propagation_head_A8"), "Propagation head A8")
        self.assertEqual(plotting.short_config_label(
            "MP_W4A4_stateA16"), "A4 + state A16")

    def test_select_best_sparse_excludes_dense_and_state_configs(self):
        rows = [
            {"config": "MP_W4A8_full", "selection": "full_a8",
             "RMSE": 0.2, "nonfinite_rate": 0.0},
            {"config": "MP_site01_A8", "selection": "single_site",
             "RMSE": 1.1, "nonfinite_rate": 0.1},
            {"config": "MP_heads_A8", "selection": "heads",
             "RMSE": 1.3, "nonfinite_rate": 0.0},
            {"config": "MP_W4A4_stateA16", "selection": "state",
             "RMSE": 0.8, "nonfinite_rate": 0.0},
        ]

        result = plotting.select_best_sparse(rows)

        self.assertEqual(result["config"], "MP_heads_A8")

    def test_pareto_front_uses_cost_and_quality(self):
        rows = [
            {"config": "base", "extra_activation_ratio": 0.0,
             "RMSE": 3.0, "nonfinite_rate": 0.0},
            {"config": "dominated", "extra_activation_ratio": 0.5,
             "RMSE": 2.5, "nonfinite_rate": 0.0},
            {"config": "front", "extra_activation_ratio": 0.4,
             "RMSE": 2.0, "nonfinite_rate": 0.0},
            {"config": "invalid", "extra_activation_ratio": math.nan,
             "RMSE": 0.1, "nonfinite_rate": 0.0},
        ]

        result = plotting.pareto_front(rows)

        self.assertEqual([row["config"] for row in result], ["base", "front"])

    def test_scatter_rows_exclude_fp32_and_zero_traffic_state_controls(self):
        rows = [
            {"config": "FP32", "selection": "fp32"},
            {"config": "MP_W4A4_base", "selection": "baseline"},
            {"config": "MP_site01_A8", "selection": "single_site"},
            {"config": "MP_W4A4_stateA16", "selection": "state"},
        ]

        result = plotting.scatter_rows(rows)

        self.assertEqual([row["config"] for row in result],
                         ["MP_W4A4_base", "MP_site01_A8"])


if __name__ == "__main__":
    unittest.main()
