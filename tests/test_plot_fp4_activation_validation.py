import unittest

import numpy as np

from scripts import plot_fp4_activation_validation as plotting


class FP4PlotContractTest(unittest.TestCase):
    def test_panel_specification_keeps_matched_activation_formats(self):
        self.assertEqual(
            [panel["config"] for panel in plotting.panel_specifications("W4")],
            ["GT", "FP32", "FP4V_W4A4", "FP4V_W4E2M1",
             "FP4V_W4A8"])

    def test_depth_rgba_distinguishes_invalid_gt_and_nonfinite_prediction(self):
        depth = np.array([[1.0, 2.0, np.nan]], dtype=np.float32)
        valid = np.array([[True, False, True]])
        nonfinite = np.array([[False, False, True]])

        rgba = plotting.depth_rgba(depth, valid, nonfinite)

        np.testing.assert_allclose(rgba[0, 1], plotting.INVALID_GT_RGBA)
        np.testing.assert_allclose(rgba[0, 2], plotting.NONFINITE_RGBA)

    def test_visual_sample_selection_is_unique_and_deterministic(self):
        rows = []
        for sample_index, int4, e2m1 in (
                (3, 0.4, 0.3), (7, 0.8, 0.5),
                (9, 1.5, 1.0), (11, 2.0, 0.8)):
            rows.extend([
                {"model": "cspn", "config": "FP4V_W8A4",
                 "sample_index": str(sample_index), "RMSE": str(int4)},
                {"model": "cspn", "config": "FP4V_W8E2M1",
                 "sample_index": str(sample_index), "RMSE": str(e2m1)},
            ])

        first = plotting.select_visual_indices(rows, "W8", 3)
        second = plotting.select_visual_indices(rows, "W8", 3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(first[-1], 11)


if __name__ == "__main__":
    unittest.main()
