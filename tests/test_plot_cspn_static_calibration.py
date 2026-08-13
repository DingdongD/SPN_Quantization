from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import plot_cspn_static_calibration as plotter


class StaticCalibrationPlotTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=".")
        self.root = Path(self.directory.name)
        self.configs = ("W4A4_G8_MINMAX", "W4A4_G8_PERCENTILE_P9999")

    def tearDown(self):
        self.directory.cleanup()

    def _write_payload(self, config, index, error):
        directory = self.root / "predictions" / config
        directory.mkdir(parents=True, exist_ok=True)
        gt = np.ones((4, 6), dtype=np.float32) * 2.0
        pred = gt + np.float32(error)
        np.savez_compressed(
            directory / ("sample_%05d.npz" % index),
            gt=gt, fp32=gt, pred=pred, abs_err=np.abs(pred - gt),
            valid_gt=np.ones_like(gt, dtype=bool),
            nonfinite=np.zeros_like(gt, dtype=bool),
            sample_index=np.int64(index), model=np.array("cspn"),
            config=np.array(config), sparse=np.zeros_like(gt),
            rgb=np.ones((4, 6, 3), dtype=np.float32) * 0.5)

    def test_load_predictions_requires_exact_declared_directories(self):
        for config_index, config in enumerate(self.configs):
            for index in range(4):
                self._write_payload(config, index, 0.1 * config_index)

        predictions = plotter.load_predictions(
            self.root, self.configs, expected_samples=4)

        self.assertEqual(sorted(predictions), [0, 1, 2, 3])
        self.assertEqual(set(predictions[0]), set(self.configs))

    def test_metric_plotters_create_nonempty_files(self):
        (self.root / "aggregate_metrics.csv").write_text(
            "config,RMSE\n"
            "W4A4_G8_MINMAX,0.31\n"
            "W4A4_G8_PERCENTILE_P999,1.35\n"
            "W4A4_G8_PERCENTILE_P9999,0.49\n"
            "W4A4_G8_HIST_MSE,0.99\n", encoding="utf-8")
        (self.root / "activation_resolution_metrics.csv").write_text(
            "config,signal_energy,total_error_energy,clipping_error_energy,"
            "rounding_error_energy,zero_collapse_error_energy\n"
            "W4A4_G8_MINMAX,10,1,0.1,0.4,0.5\n"
            "W4A4_G8_PERCENTILE_P999,10,1,0.4,0.4,0.2\n"
            "W4A4_G8_PERCENTILE_P9999,10,1,0.2,0.5,0.3\n"
            "W4A4_G8_HIST_MSE,10,1,0.3,0.5,0.2\n",
            encoding="utf-8")

        rmse = plotter.render_rmse(
            self.root, self.root / "rmse.png", dpi=40)
        errors = plotter.render_error_composition(
            self.root, self.root / "errors.png", dpi=40)

        self.assertGreater(rmse.stat().st_size, 0)
        self.assertGreater(errors.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
