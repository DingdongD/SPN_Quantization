import unittest

import numpy as np
import torch

from scripts import nyu_quantization_analysis as analysis


class ModuleGroupingTest(unittest.TestCase):
    def test_cspn_heads_are_separated(self):
        self.assertEqual(analysis.classify_module("cspn", "layer2.0.conv1"), "encoder")
        self.assertEqual(analysis.classify_module("cspn", "gud_up_proj_layer3.conv1"), "decoder")
        self.assertEqual(analysis.classify_module("cspn", "gud_up_proj_layer5.conv1"), "depth_head")
        self.assertEqual(analysis.classify_module("cspn", "gud_up_proj_layer6.conv1"), "propagation_head")

    def test_dyspn_joint_output_is_a_propagation_head(self):
        self.assertEqual(analysis.classify_module("dyspn", "base.conv4.0.conv1"), "encoder")
        self.assertEqual(analysis.classify_module("dyspn", "base.gd_dec1_.0"), "decoder")
        self.assertEqual(
            analysis.classify_module("dyspn", "base.gd_dec0_dyspn_9_5.0"),
            "propagation_head",
        )
        self.assertEqual(
            analysis.classify_module("dyspn", "dyspn_9_5.conv_offset_aff"),
            "propagation_head",
        )

    def test_nlspn_and_completionformer_groups(self):
        self.assertEqual(analysis.classify_module("nlspn", "id_dec0.0"), "depth_head")
        self.assertEqual(analysis.classify_module("nlspn", "cf_dec0.0"), "propagation_head")
        self.assertEqual(
            analysis.classify_module("completionformer", "backbone.former.block1.0.attn.q"),
            "attention",
        )
        self.assertEqual(
            analysis.classify_module("completionformer", "backbone.former.block1.0.mlp.fc1"),
            "attention",
        )
        self.assertEqual(
            analysis.classify_module("completionformer", "backbone.dep_dec0.0"),
            "depth_head",
        )


class InformationMetricTest(unittest.TestCase):
    def test_tensor_metrics_report_sign_and_cosine_damage(self):
        reference = torch.tensor([1.0, -1.0, 2.0, -2.0])
        candidate = torch.tensor([1.0, 1.0, 1.0, -2.0])

        metrics = analysis.tensor_metrics(reference, candidate)

        self.assertAlmostEqual(metrics["sign_flip_rate"], 0.25)
        self.assertLess(metrics["cosine"], 1.0)
        self.assertGreater(metrics["mse"], 0.0)

    def test_affinity_metrics_count_dominant_neighbor_changes(self):
        reference = torch.tensor([[[[0.9]], [[0.1]]]])
        candidate = torch.tensor([[[[0.2]], [[0.8]]]])

        metrics = analysis.affinity_metrics(reference, candidate, channel_dim=1)

        self.assertEqual(metrics["dominant_neighbor_change_rate"], 1.0)

    def test_offset_metrics_report_endpoint_error(self):
        reference = torch.zeros(1, 4, 1, 1)
        candidate = torch.tensor([[[[3.0]], [[4.0]], [[0.0]], [[0.0]]]])

        metrics = analysis.offset_metrics(reference, candidate, channel_dim=1)

        self.assertAlmostEqual(metrics["endpoint_error"], 2.5)

    def test_output_signal_comparison_keeps_offset_endpoint_error(self):
        reference = {"offset": torch.zeros(1, 2, 1, 1)}
        candidate = {"offset": torch.tensor([[[[3.0]], [[4.0]]]])}

        rows = analysis.compare_output_signals(reference, candidate)

        self.assertAlmostEqual(rows[0]["endpoint_error"], 5.0)

    def test_regional_metrics_separate_edges_sparse_points_and_depth_bins(self):
        gt = np.array([[1.0, 1.0, 4.0], [1.0, 1.0, 7.0]], dtype=np.float32)
        pred = gt + 1.0
        sparse = np.zeros_like(gt)
        sparse[0, 0] = gt[0, 0]

        rows = analysis.regional_depth_metrics(gt, pred, sparse, edge_threshold=0.5)
        by_region = dict((row["region"], row) for row in rows)

        self.assertEqual(by_region["sparse_anchor"]["num_pixels"], 1)
        self.assertEqual(by_region["holes"]["num_pixels"], 5)
        self.assertEqual(by_region["near_0_2m"]["num_pixels"], 4)
        self.assertEqual(by_region["mid_2_5m"]["num_pixels"], 1)
        self.assertEqual(by_region["far_5_10m"]["num_pixels"], 1)
        self.assertGreater(by_region["boundary"]["num_pixels"], 0)

    def test_nonfinite_prediction_marks_region_metrics_as_failed(self):
        gt = np.array([[1.0, 2.0]], dtype=np.float32)
        pred = np.array([[1.0, np.nan]], dtype=np.float32)
        sparse = np.zeros_like(gt)

        rows = analysis.regional_depth_metrics(gt, pred, sparse)
        all_pixels = dict((row["region"], row) for row in rows)["all"]

        self.assertEqual(all_pixels["num_pixels"], 2)
        self.assertTrue(np.isinf(all_pixels["RMSE"]))


if __name__ == "__main__":
    unittest.main()
