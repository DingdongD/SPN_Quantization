import unittest

import numpy as np

from scripts import plot_quantized_prediction_comparison as plotting


class SemanticColorTest(unittest.TestCase):
    def test_depth_rgba_distinguishes_invalid_gt_and_nonfinite_prediction(self):
        depth = np.array([[1.0, 4.0], [np.nan, 3.0]], np.float32)
        valid_gt = np.array([[True, False], [True, True]])
        nonfinite = np.array([[False, False], [True, False]])

        rgba = plotting.depth_rgba(depth, valid_gt, nonfinite)

        self.assertTrue(np.allclose(
            rgba[0, 1], plotting.INVALID_GT_RGBA))
        self.assertTrue(np.allclose(
            rgba[1, 0], plotting.NONFINITE_RGBA))
        self.assertFalse(np.allclose(rgba[0, 0], plotting.INVALID_GT_RGBA))

    def test_error_rgba_uses_same_semantic_override_order(self):
        error = np.array([[0.5, np.nan], [np.nan, 2.0]], np.float32)
        valid_gt = np.array([[True, False], [True, True]])
        nonfinite = np.array([[False, False], [True, False]])

        rgba = plotting.error_rgba(error, valid_gt, nonfinite)

        self.assertTrue(np.allclose(
            rgba[0, 1], plotting.INVALID_GT_RGBA))
        self.assertTrue(np.allclose(
            rgba[1, 0], plotting.NONFINITE_RGBA))


class PanelSpecificationTest(unittest.TestCase):
    def test_each_model_has_six_unrotated_columns(self):
        cspn = plotting.panel_specifications("cspn")
        nlspn = plotting.panel_specifications("nlspn")

        self.assertEqual([row["config"] for row in cspn], [
            "GT", "FP32", "MP_W8A8_full", "MP_W4A4_base",
            "MP_heads_A8", "MP_W4A8_full",
        ])
        self.assertEqual(nlspn[4]["config"], "MP_site01_A8")
        self.assertEqual(len(cspn), 6)
        self.assertTrue(all(row["rotation"] == 0 for row in cspn))


if __name__ == "__main__":
    unittest.main()
