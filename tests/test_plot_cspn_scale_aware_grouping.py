from pathlib import Path
import shutil
import unittest

import numpy as np

from scripts import plot_cspn_scale_aware_grouping as plotter


class ScaleAwarePlotTest(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / "test_outputs" / \
            "test_plot_cspn_scale_aware_grouping"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.prediction_root = self.root / "cspn" / "predictions"
        shape = (3, 4)
        for config, offset in (
                (plotter.BASELINE, 0.1), (plotter.SCALE_AWARE, 0.2)):
            directory = self.prediction_root / config
            directory.mkdir(parents=True)
            for index in (1, 2):
                gt = np.full(shape, 2.0, dtype=np.float32)
                pred = gt + offset * index
                np.savez_compressed(
                    directory / ("sample_%05d.npz" % index),
                    gt=gt,
                    fp32=gt,
                    pred=pred,
                    abs_err=np.abs(pred - gt),
                    valid_gt=np.ones(shape, dtype=bool),
                    nonfinite=np.zeros(shape, dtype=bool),
                    sample_index=np.asarray(index, dtype=np.int64),
                    model=np.asarray("cspn"),
                    config=np.asarray(config),
                    sparse=np.zeros(shape, dtype=np.float32),
                    rgb=np.zeros(shape + (3,), dtype=np.float32))

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_load_predictions_requires_paired_sample_identity(self):
        predictions = plotter.load_predictions(
            self.root / "cspn", expected_samples=2)

        self.assertEqual(set(predictions), {1, 2})
        self.assertEqual(
            set(predictions[1]), {plotter.BASELINE, plotter.SCALE_AWARE})

    def test_sample_deltas_use_scale_aware_minus_baseline(self):
        predictions = plotter.load_predictions(
            self.root / "cspn", expected_samples=2)

        rows = plotter.sample_delta_rows(predictions)

        self.assertEqual([row["sample_index"] for row in rows], [1, 2])
        self.assertGreater(rows[0]["rmse_delta"], 0.0)
        self.assertGreater(rows[1]["rmse_delta"], rows[0]["rmse_delta"])

    def test_render_prediction_detail_is_nonempty(self):
        predictions = plotter.load_predictions(
            self.root / "cspn", expected_samples=2)
        output = self.root / "figures" / "detail.png"

        plotter.render_prediction_detail(predictions, output, dpi=80)

        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
