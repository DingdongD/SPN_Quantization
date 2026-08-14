from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

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


class PreprocessingTest(unittest.TestCase):
    def test_load_rgb_returns_uint8_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jpg"
            Image.fromarray(
                np.full((6, 8, 3), 127, dtype=np.uint8), mode="RGB").save(path)

            rgb = sequence.load_rgb(path)

        self.assertEqual(rgb.shape, (6, 8, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_read_exr_depth_selects_equal_channel(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.exr"
            depth = np.arange(20, dtype=np.float32).reshape(4, 5)
            self.assertTrue(cv2.imwrite(
                str(path), np.repeat(depth[..., None], 3, axis=2)))

            loaded = sequence.read_exr_depth(path)

        np.testing.assert_array_equal(loaded, depth)

    def test_read_exr_depth_rejects_different_channels(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.exr"
            depth = np.ones((4, 5, 3), dtype=np.float32)
            depth[..., 2] = 2.0
            self.assertTrue(cv2.imwrite(str(path), depth))

            with self.assertRaisesRegex(ValueError, "channels differ"):
                sequence.read_exr_depth(path)

    def test_preprocess_pair_has_cspn_geometry(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 2.0, dtype=np.float32)

        rgb_out, depth_out, valid = sequence.preprocess_pair(rgb, depth)

        self.assertEqual(rgb_out.shape, (3, 228, 304))
        self.assertEqual(depth_out.shape, (228, 304))
        self.assertEqual(valid.shape, (228, 304))
        self.assertEqual(rgb_out.dtype, np.float32)
        self.assertTrue(valid.all())


class CheckpointTest(unittest.TestCase):
    def test_normalize_state_accepts_only_known_legacy_difference(self):
        model = torch.nn.Linear(2, 1)
        state = {
            "module.weight": model.weight.detach().clone(),
            "module.bias": model.bias.detach().clone(),
            "module.post_process_layer.sum_conv.weight": torch.ones(
                1, 8, 1, 1, 1),
        }

        report = sequence.load_compatible_state(
            model, state, allowed_missing=())

        self.assertEqual(
            report["ignored"], ["post_process_layer.sum_conv.weight"])
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["unexpected"], [])

    def test_normalize_state_rejects_unknown_key(self):
        model = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        state["unknown"] = torch.ones(1)

        with self.assertRaisesRegex(RuntimeError, "unexpected checkpoint keys"):
            sequence.load_compatible_state(model, state, allowed_missing=())

    def test_normalize_state_rejects_unknown_missing_key(self):
        model = torch.nn.Linear(2, 1)
        state = {"weight": model.weight.detach().clone()}

        with self.assertRaisesRegex(RuntimeError, "missing checkpoint keys.*bias"):
            sequence.load_compatible_state(model, state, allowed_missing=())

    def test_normalize_state_rejects_invalid_legacy_sum_kernel(self):
        model = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        state["post_process_layer.sum_conv.weight"] = torch.zeros(
            1, 8, 1, 1, 1)

        with self.assertRaisesRegex(RuntimeError, "invalid CSPN fixed sum kernel"):
            sequence.load_compatible_state(model, state, allowed_missing=())


if __name__ == "__main__":
    unittest.main()
