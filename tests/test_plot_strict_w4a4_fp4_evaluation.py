import unittest

import numpy as np

from scripts import plot_strict_w4a4_fp4_evaluation as plotting


def make_rows(values):
    series = (
        ("rtn", "FP4V_W4A4"),
        ("rtn", "FP4V_W4E2M1"),
        ("adaround", "FP4V_W4A4"),
        ("adaround", "FP4V_W4E2M1"),
        ("brecq", "FP4V_W4A4"),
        ("brecq", "FP4V_W4E2M1"),
    )
    rows = []
    for sample_index, rmses in values.items():
        for (method, config), rmse in zip(series, rmses):
            rows.append({
                "method": method,
                "config": config,
                "sample_index": str(sample_index),
                "RMSE": str(rmse),
            })
    return rows


class StrictW4A4FP4PlotTest(unittest.TestCase):
    def test_prediction_panels_include_all_methods_and_formats(self):
        self.assertEqual(
            plotting.prediction_panels(),
            (
                ("GT", "GT"),
                ("FP32", "FP32"),
                ("rtn:FP4V_W4A4", "RTN A4"),
                ("rtn:FP4V_W4E2M1", "RTN E2M1"),
                ("adaround:FP4V_W4A4", "AdaRound A4"),
                ("adaround:FP4V_W4E2M1", "AdaRound E2M1"),
                ("brecq:FP4V_W4A4", "BRECQ A4"),
                ("brecq:FP4V_W4E2M1", "BRECQ E2M1"),
            ))

    def test_visual_sample_uses_largest_finite_cross_method_spread(self):
        rows = make_rows({
            11: [0.2, 0.3, 0.21, 0.31, 0.22, 0.32],
            17: [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            23: [0.5, 0.6, 0.51, 0.61, 0.52, 0.62],
        })

        self.assertEqual(plotting.select_visual_index(rows), 17)

    def test_visual_sample_rejects_nonfinite_or_incomplete_sets(self):
        rows = make_rows({11: [0.2, 0.3, 0.21, 0.31, 0.22, np.nan]})
        with self.assertRaisesRegex(ValueError, "no fully finite visual sample"):
            plotting.select_visual_index(rows)

        rows = make_rows({11: [0.2, 0.3, 0.21, 0.31, 0.22, 0.32]})[:-1]
        with self.assertRaisesRegex(ValueError, "visual sample series"):
            plotting.select_visual_index(rows)

    def test_rgba_distinguishes_invalid_gt_and_nonfinite_prediction(self):
        values = np.asarray([[1.0, np.nan], [2.0, 3.0]])
        valid_gt = np.asarray([[True, True], [False, True]])
        nonfinite = ~np.isfinite(values)

        rgba = plotting.depth_rgba(values, valid_gt, nonfinite)

        np.testing.assert_allclose(rgba[0, 1], plotting.NONFINITE_RGBA)
        np.testing.assert_allclose(rgba[1, 0], plotting.INVALID_GT_RGBA)
        self.assertFalse(np.allclose(rgba[0, 0], plotting.NONFINITE_RGBA))


if __name__ == "__main__":
    unittest.main()
