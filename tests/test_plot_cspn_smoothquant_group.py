from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import plot_cspn_smoothquant_group as plotter


class SmoothQuantPredictionPlotTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=".")
        self.root = Path(self.directory.name)
        self.configs = (
            "FP32", "W4A4_GROUP16", "SQ_W4A4_GROUP16_A050",
            "W4A4_GROUP8", "SQ_W4A4_GROUP8_A050",
        )

    def tearDown(self):
        self.directory.cleanup()

    def write_payload(self, config, sample_index, error):
        directory = self.root / "predictions" / config
        directory.mkdir(parents=True, exist_ok=True)
        gt = np.ones((4, 6), dtype=np.float32) * 2.0
        pred = gt + np.float32(error)
        np.savez_compressed(
            directory / ("sample_%05d.npz" % sample_index),
            gt=gt, fp32=gt, pred=pred, abs_err=np.abs(pred - gt),
            valid_gt=np.ones_like(gt, dtype=bool),
            nonfinite=np.zeros_like(gt, dtype=bool),
            sample_index=np.int64(sample_index), model=np.array("cspn"),
            config=np.array(config), sparse=np.zeros_like(gt),
            rgb=np.ones((4, 6, 3), dtype=np.float32) * 0.5)

    def populate(self):
        for config_index, config in enumerate(self.configs):
            for sample_index in range(4):
                self.write_payload(
                    config, sample_index,
                    0.05 * config_index * (sample_index + 1))

    def test_load_predictions_requires_declared_configurations(self):
        self.populate()

        predictions = plotter.load_predictions(
            self.root, self.configs, expected_samples=4)

        self.assertEqual(set(predictions[0]), set(self.configs))
        self.assertEqual(sorted(predictions), [0, 1, 2, 3])

    def test_render_prediction_detail_creates_file(self):
        self.populate()
        predictions = plotter.load_predictions(
            self.root, self.configs, expected_samples=4)

        output = plotter.render_prediction_detail(
            predictions, "SQ_W4A4_GROUP8_A050",
            self.root / "detail.png", dpi=40)

        self.assertGreater(output.stat().st_size, 0)

    def test_render_scale_pareto_creates_file(self):
        names = (
            "W4A4_GROUP16", "SQ_W4A4_GROUP16_A025",
            "SQ_W4A4_GROUP16_A050", "SQ_W4A4_GROUP16_A075",
            "W4A4_GROUP8", "SQ_W4A4_GROUP8_A025",
            "SQ_W4A4_GROUP8_A050", "SQ_W4A4_GROUP8_A075",
        )
        metrics = "config,RMSE\n" + "\n".join(
            "%s,%.3f" % (name, 0.3 + index * 0.01)
            for index, name in enumerate(names)) + "\n"
        manifest = "config,activation_scales\n" + "\n".join(
            "%s,%d" % (name, 100 if "GROUP16" in name else 200)
            for name in names) + "\n"
        (self.root / "aggregate_metrics.csv").write_text(
            metrics, encoding="utf-8")
        (self.root / "config_manifest.csv").write_text(
            manifest, encoding="utf-8")

        output = plotter.render_scale_pareto(
            self.root, self.root / "pareto.png", dpi=40)

        self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
