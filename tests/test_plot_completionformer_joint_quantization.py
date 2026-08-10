import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import plot_completionformer_joint_quantization as plotter


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root, samples=2):
    root = Path(root)
    metric_rows = []
    for config_index, config in enumerate(plotter.CONFIG_ORDER):
        prediction_dir = root / "predictions" / config
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(samples):
            gt = np.full((4, 5), 2.0 + sample_index, dtype=np.float32)
            pred = gt + 0.01 * config_index
            np.savez_compressed(
                prediction_dir / ("sample_%05d.npz" % sample_index),
                gt=gt,
                fp32=gt,
                pred=pred,
                sample_index=np.array(sample_index),
                config=np.array(config))
            metric_rows.append({
                "model": "completionformer",
                "config": config,
                "sample_index": sample_index,
                "RMSE": 0.1 + 0.01 * config_index,
                "MAE": 0.05,
                "ABS_REL": 0.01,
            })
    write_csv(root / "sample_metrics.csv", metric_rows)
    write_csv(root / "attention_metrics.csv", [{
        "config": "JIQ_Joint_W4A4",
        "family": "attention",
        "module": "backbone.former.block1.0.attn",
        "updates": samples,
        "score_sqnr_db": 12.0,
        "context_mse": 0.02,
        "probability_kl": 0.001,
    }])
    write_csv(root / "concat_metrics.csv", [{
        "config": "JIQ_Joint_W4A4",
        "family": "concat",
        "module": "backbone.former.block1.0.concat_conv",
        "updates": samples,
        "output_sqnr_db": 10.0,
        "block_mse": 0.03,
        "partial_requantization_mse": 0.002,
    }])


class CompletionFormerJointPlotTest(unittest.TestCase):
    def test_config_columns_match_joint_ablation_contract(self):
        self.assertEqual(plotter.PANEL_ORDER, (
            "GT",
            "FP32",
            "JIQ_RTN_W4A4",
            "JIQ_Attention_W4A4",
            "JIQ_Concat_W4A4",
            "JIQ_Joint_W4A4",
            "JIQ_W4A8",
        ))

    def test_generate_figures_requires_and_renders_every_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "completionformer"
            output = Path(directory) / "analysis"
            root.mkdir(parents=True)
            build_fixture(root, samples=2)

            paths = plotter.generate_figures(
                root=root,
                out_dir=output,
                expected_samples=2,
                dpi=40)

            self.assertEqual(set(paths), {
                "aggregate", "local", "predictions"})
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_missing_prediction_config_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "completionformer"
            root.mkdir(parents=True)
            build_fixture(root, samples=2)
            missing = root / "predictions" / "JIQ_Concat_W4A4" / \
                "sample_00001.npz"
            missing.unlink()

            with self.assertRaisesRegex(ValueError, "JIQ_Concat_W4A4"):
                plotter.load_prediction_grid(root, expected_samples=2)

    def test_exact_zero_log_metrics_break_the_line(self):
        values = plotter.positive_log_values([1.0, 0.0, 2.0])

        self.assertEqual(values[0], 1.0)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(values[2], 2.0)


if __name__ == "__main__":
    unittest.main()
