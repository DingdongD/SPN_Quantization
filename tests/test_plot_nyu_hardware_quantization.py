import unittest

from scripts import plot_nyu_hardware_quantization as plotter


class HardwareQuantizationPlotTest(unittest.TestCase):
    def test_comparison_summary_keeps_backend_order_and_nonfinite_rate(self):
        rtn_regional = [
            {"model": "cspn", "config": "FP32", "region": "all", "RMSE": "0.2"},
            {"model": "cspn", "config": "W4A8_full", "region": "all", "RMSE": "0.3"},
            {"model": "cspn", "config": "W4A4_full", "region": "all", "RMSE": "1.0"},
        ]
        hardware_regional = [
            {"model": "cspn", "config": "FP32", "region": "all", "RMSE": "0.2"},
            {"model": "cspn", "config": "HW_W4A8_full", "region": "all", "RMSE": "0.25"},
            {"model": "cspn", "config": "HW_W4A4_full", "region": "all", "RMSE": "2.0"},
        ]
        hardware_samples = [
            {"model": "cspn", "config": "HW_W4A4_full",
             "nonfinite_pixels": "20", "num_pixels": "80"},
        ]

        rows = plotter.comparison_summary_rows(
            rtn_regional, [], hardware_regional, hardware_samples)

        self.assertEqual([row["label"] for row in rows], [
            "FP32", "RTN W4A8", "HW W4A8", "RTN W4A4", "HW W4A4",
        ])
        self.assertAlmostEqual(rows[2]["rmse_ratio"], 1.25)
        self.assertAlmostEqual(rows[-1]["nonfinite_rate"], 0.2)

    def test_signal_summary_uses_median_finite_sqnr(self):
        rows = [
            {"model": "nlspn", "config": "HW_W4A4_full", "signal": "affinity",
             "iteration": "0", "sqnr_db": "-4", "zeroed_rate": "0.2"},
            {"model": "nlspn", "config": "HW_W4A4_full", "signal": "affinity",
             "iteration": "0", "sqnr_db": "2", "zeroed_rate": "0.4"},
            {"model": "nlspn", "config": "HW_W4A4_full",
             "signal": "propagation_states", "iteration": "1",
             "sqnr_db": "99", "zeroed_rate": "0"},
        ]

        summary = plotter.signal_damage_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["signal"], "affinity")
        self.assertAlmostEqual(summary[0]["median_sqnr_db"], -1.0)
        self.assertAlmostEqual(summary[0]["median_zeroed_rate"], 0.3)

    def test_layer_summary_groups_module_statistics(self):
        rows = [
            {"model": "dyspn", "config": "HW_W4A8_full", "group": "decoder",
             "kind": "input", "sqnr_db": "10", "saturation_rate": "0.01"},
            {"model": "dyspn", "config": "HW_W4A8_full", "group": "decoder",
             "kind": "input", "sqnr_db": "14", "saturation_rate": "0.03"},
        ]

        summary = plotter.layer_damage_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["median_sqnr_db"], 12.0)
        self.assertAlmostEqual(summary[0]["max_saturation_rate"], 0.03)


if __name__ == "__main__":
    unittest.main()
