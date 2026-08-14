import unittest

import numpy as np

from scripts import export_cspn_sequence_predictions as sequence


class DepthInputTest(unittest.TestCase):
    def test_sanitize_depth_rejects_nonfinite_nonpositive_and_over_range(self):
        depth = np.array(
            [[1.0, np.inf, 0.0, -1.0, 10.1]], dtype=np.float32)

        clean, valid = sequence.sanitize_depth(depth, max_depth=10.0)

        np.testing.assert_array_equal(
            valid, [[True, False, False, False, False]])
        np.testing.assert_array_equal(
            clean, [[1.0, 0.0, 0.0, 0.0, 0.0]])

    def test_shared_sparse_depth_uses_same_500_coordinates(self):
        depths = np.stack([
            np.full((24, 32), 1.0, dtype=np.float32),
            np.full((24, 32), 2.0, dtype=np.float32),
        ])

        sparse, mask = sequence.build_shared_sparse_depths(
            depths, depths > 0.0, count=500, seed=2026)

        self.assertEqual(int(mask.sum()), 500)
        self.assertEqual(int(np.count_nonzero(sparse[0])), 500)
        self.assertEqual(int(np.count_nonzero(sparse[1])), 500)
        np.testing.assert_array_equal(sparse[0] > 0.0, sparse[1] > 0.0)

    def test_shared_sparse_depth_rejects_too_few_common_pixels(self):
        depths = np.ones((2, 4, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "common valid pixels"):
            sequence.build_shared_sparse_depths(
                depths, depths > 0.0, count=500, seed=2026)


class MetricTest(unittest.TestCase):
    def test_frame_metrics_use_only_valid_ground_truth(self):
        gt = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
        pred = np.array([[2.0, 2.0, 9.0]], dtype=np.float32)

        result = sequence.frame_metrics(gt, pred, gt > 0.0)

        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5))
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["abs_rel"], 0.5)

    def test_temporal_metrics_measure_change_residual(self):
        gt = np.array([[[1.0, 2.0]], [[2.0, 4.0]]], dtype=np.float32)
        pred = np.array([[[1.0, 2.0]], [[3.0, 5.0]]], dtype=np.float32)
        valid = np.ones_like(gt, dtype=bool)

        rows, maps = sequence.temporal_metrics(gt, pred, valid, [1, 2])

        self.assertEqual(rows[0]["pair"], "0001->0002")
        self.assertAlmostEqual(rows[0]["rmse"], 1.0)
        np.testing.assert_array_equal(maps[0]["residual"], [[1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
