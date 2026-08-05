import csv
import math
import tempfile
import unittest
from pathlib import Path

from scripts import plot_nyu_rtn_quantization as plotter


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RTNPlotAggregationTest(unittest.TestCase):
    def test_full_summary_reports_relative_delta_and_nonfinite_rate(self):
        regional = [
            {"model": "cspn", "config": "FP32", "region": "all", "RMSE": "0.2"},
            {"model": "cspn", "config": "W8A8_full", "region": "all", "RMSE": "0.21"},
            {"model": "cspn", "config": "W4A8_full", "region": "all", "RMSE": "0.3"},
            {"model": "cspn", "config": "W4A4_full", "region": "all", "RMSE": "2.0"},
        ]
        samples = [
            {"model": "cspn", "config": "W4A4_full", "nonfinite_pixels": "10",
             "num_pixels": "90"},
        ]

        rows = plotter.full_quantization_summary(regional, samples)
        by_config = dict((row["config"], row) for row in rows)

        self.assertAlmostEqual(by_config["W8A8_full"]["delta_percent"], 5.0)
        self.assertAlmostEqual(by_config["W4A8_full"]["delta_percent"], 50.0)
        self.assertAlmostEqual(by_config["W4A4_full"]["nonfinite_rate"], 0.1)
        self.assertEqual([row["config"] for row in rows],
                         ["FP32", "W8A8_full", "W4A8_full", "W4A4_full"])

    def test_module_sensitivity_extracts_group_from_config(self):
        regional = [
            {"model": "nlspn", "config": "FP32", "region": "all", "RMSE": "0.1"},
            {"model": "nlspn", "config": "W4A4_encoder_only", "region": "all", "RMSE": "1.0"},
            {"model": "nlspn", "config": "W4A4_propagation_head_only", "region": "all", "RMSE": "0.5"},
        ]

        samples = [
            {"model": "nlspn", "config": "W4A4_encoder_only",
             "nonfinite_pixels": "0", "num_pixels": "100"},
            {"model": "nlspn", "config": "W4A4_propagation_head_only",
             "nonfinite_pixels": "25", "num_pixels": "75"},
        ]

        rows = plotter.module_sensitivity_rows(regional, samples, bits=4)

        self.assertEqual([row["group"] for row in rows], ["encoder", "propagation_head"])
        self.assertEqual([row["rmse_ratio"] for row in rows], [10.0, 5.0])
        self.assertEqual([row["nonfinite_rate"] for row in rows], [0.0, 0.25])

    def test_signal_summary_preserves_structural_damage_metrics(self):
        rows = [
            {"model": "nlspn", "config": "W4A4_full", "signal": "affinity",
             "iteration": "0", "sqnr_db": "-7", "cosine": "0.2",
             "sign_flip_rate": "0.1", "dominant_neighbor_change_rate": "0.8",
             "endpoint_error": ""},
            {"model": "nlspn", "config": "W4A4_full", "signal": "offset",
             "iteration": "0", "sqnr_db": "1", "cosine": "0.4",
             "sign_flip_rate": "0.0", "dominant_neighbor_change_rate": "",
             "endpoint_error": "2.5"},
        ]

        summary = plotter.signal_summary_rows(rows)
        by_signal = dict((row["signal"], row) for row in summary)

        self.assertAlmostEqual(
            by_signal["affinity"]["median_dominant_neighbor_change_rate"], 0.8)
        self.assertTrue(math.isnan(by_signal["affinity"]["median_endpoint_error"]))
        self.assertAlmostEqual(by_signal["offset"]["median_endpoint_error"], 2.5)

    def test_regional_summary_reports_absolute_rmse_delta(self):
        rows = [
            {"model": "dyspn", "config": "FP32", "region": "sparse_anchor",
             "RMSE": "0.000001"},
            {"model": "dyspn", "config": "W4A4_full", "region": "sparse_anchor",
             "RMSE": "0.006001"},
        ]

        summary = plotter.regional_ratio_rows(rows)

        self.assertAlmostEqual(summary[0]["rmse_delta"], 0.006)

    def test_collect_tables_merges_model_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in ("cspn", "dyspn"):
                model_dir = Path(tmp) / model
                model_dir.mkdir()
                write_rows(model_dir / "regional_metrics.csv", [
                    {"model": model, "config": "FP32", "region": "all", "RMSE": "0.1"},
                ])
                write_rows(model_dir / "sample_metrics.csv", [
                    {"model": model, "config": "FP32", "nonfinite_pixels": "0",
                     "num_pixels": "10"},
                ])

            tables = plotter.collect_tables(tmp)

            self.assertEqual(len(tables["regional"]), 2)
            self.assertEqual(len(tables["samples"]), 2)

    def test_heatmap_annotation_contrasts_dark_and_light_cells(self):
        self.assertEqual(plotter.heatmap_annotation_color(1.0, 1.0, 20.0), "white")
        self.assertEqual(plotter.heatmap_annotation_color(20.0, 1.0, 20.0), "black")

    def test_heatmap_value_precision_tracks_panel_scale(self):
        self.assertEqual(plotter.format_heatmap_value(0.034, panel_max=0.06), "0.034")
        self.assertEqual(plotter.format_heatmap_value(2.34, panel_max=3.0), "2.3")

    def test_heatmap_width_scales_for_three_precision_panels(self):
        width, height = plotter.heatmap_figure_size(3)

        self.assertGreaterEqual(width, 18.0)
        self.assertEqual(height, 4.8)


if __name__ == "__main__":
    unittest.main()
