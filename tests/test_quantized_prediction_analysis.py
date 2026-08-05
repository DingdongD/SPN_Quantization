import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import quantized_prediction_analysis as analysis


class PredictionMetricsTest(unittest.TestCase):
    def test_metrics_separate_valid_gt_from_finite_predictions(self):
        gt = np.array([[1.0, 0.0], [2.0, 3.0]], np.float32)
        pred = np.array([[2.0, 4.0], [np.nan, 5.0]], np.float32)

        result = analysis.prediction_metrics(gt, pred)

        self.assertAlmostEqual(result["RMSE"], np.sqrt(2.5))
        self.assertAlmostEqual(result["MAE"], 1.5)
        self.assertEqual(result["nonfinite_pixels"], 1)
        self.assertEqual(result["num_pixels"], 2)
        self.assertEqual(result["valid_gt_pixels"], 3)
        self.assertAlmostEqual(result["nonfinite_rate"], 1.0 / 3.0)

    def test_cross_validation_reports_metric_mismatch(self):
        computed = [{"model": "dyspn", "config": "q", "sample_index": 1,
                     "RMSE": 1.0, "MAE": 0.5}]
        formal = [{"model": "dyspn", "config": "q", "sample_index": "1",
                   "RMSE": "1.2", "MAE": "0.5"}]

        with self.assertRaisesRegex(ValueError, "RMSE"):
            analysis.cross_validate_metrics(computed, formal)


class RepresentativeSelectionTest(unittest.TestCase):
    @staticmethod
    def rows(model, base_values, sparse_values, nonfinite=None):
        output = []
        nonfinite = nonfinite or {}
        for sample_index, value in base_values.items():
            output.append({
                "model": model, "role": "w4a4", "sample_index": sample_index,
                "RMSE": value, "nonfinite_rate": nonfinite.get(sample_index, 0.0),
            })
        for sample_index, value in sparse_values.items():
            output.append({
                "model": model, "role": "sparse_a8", "sample_index": sample_index,
                "RMSE": value, "nonfinite_rate": 0.0,
            })
        return output

    def test_selects_unique_quantiles_maximum_and_sparse_recovery(self):
        base = dict((index, float(index)) for index in range(1, 7))
        sparse = dict((index, float(index) - (4.0 if index == 2 else 0.1))
                      for index in range(1, 7))

        result = analysis.select_representative_samples(
            self.rows("nlspn", base, sparse), "nlspn")

        self.assertEqual([row["reason"] for row in result], [
            "median_w4a4", "p90_w4a4", "maximum_w4a4",
            "maximum_sparse_recovery",
        ])
        self.assertEqual(len(set(row["sample_index"] for row in result)), 4)
        self.assertEqual(result[-1]["sample_index"], 2)

    def test_cspn_fourth_sample_prioritizes_nonfinite_rate(self):
        base = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0,
                9: float("nan")}
        sparse = dict((index, 0.5) for index in base)

        result = analysis.select_representative_samples(
            self.rows("cspn", base, sparse, {9: 0.95}), "cspn")

        self.assertEqual(result[-1], {
            "model": "cspn", "sample_index": 9,
            "reason": "maximum_nonfinite",
        })


class SampleSetValidationTest(unittest.TestCase):
    def test_requires_each_config_to_have_exact_expected_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for config in ("FP32", "quant"):
                path = root / config
                path.mkdir()
                for index in (3, 7):
                    np.savez_compressed(
                        str(path / ("sample_%05d.npz" % index)),
                        sample_index=np.array(index))

            analysis.validate_sample_sets(root, ["FP32", "quant"], [3, 7])

            (root / "quant" / "sample_00007.npz").unlink()
            with self.assertRaisesRegex(ValueError, "quant"):
                analysis.validate_sample_sets(
                    root, ["FP32", "quant"], [3, 7])


if __name__ == "__main__":
    unittest.main()
