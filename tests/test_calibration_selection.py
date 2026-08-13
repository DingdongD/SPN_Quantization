import unittest

import numpy as np
import torch

from spn_quant.calibration_selection import (
    FeatureSchema,
    activation_range_coverage,
    build_disjoint_splits,
    descriptor_coverage,
    deterministic_kmedoids,
    deterministic_weighted_kmedoids,
    fit_robust_normalizer,
    greedy_kcenter,
    greedy_kcenter_features,
    grouped_pairwise_distance,
    nearest_distance_summary,
    raw_descriptor,
    representative_weights,
    select_tail_cover,
)


class RawDescriptorTest(unittest.TestCase):
    def test_raw_descriptor_records_depth_rgb_and_sparse_statistics(self):
        height, width = 20, 25
        luminance = torch.linspace(0.0, 1.0, height * width).reshape(
            height, width)
        rgb = torch.stack((luminance, luminance, luminance))
        depth = torch.arange(
            1, height * width + 1, dtype=torch.float32).reshape(
                1, height, width)
        sparse = depth.clone()

        row = raw_descriptor(rgb, depth, sparse)

        self.assertAlmostEqual(row["depth_mean"], 250.5)
        self.assertAlmostEqual(row["depth_p50"], 250.5)
        self.assertAlmostEqual(row["depth_p95"], 475.05, places=4)
        self.assertEqual(row["depth_max"], 500.0)
        self.assertEqual(row["depth_valid_ratio"], 1.0)
        self.assertAlmostEqual(row["rgb_luminance_mean"], 0.5, places=6)
        self.assertAlmostEqual(
            row["rgb_contrast"], row["rgb_luminance_std"], places=6)
        self.assertEqual(row["sparse_valid_count"], 500.0)
        self.assertAlmostEqual(
            sum(row["sparse_quadrant_%d" % index] for index in range(4)),
            1.0)
        self.assertEqual(row["sparse_grid_occupancy"], 1.0)
        self.assertGreater(row["sparse_centroid_spread"], 0.0)
        self.assertGreaterEqual(row["rgb_edge_density"], 0.0)
        self.assertLessEqual(row["rgb_edge_density"], 1.0)

    def test_raw_descriptor_records_sparse_count_below_requested_budget(self):
        rgb = torch.zeros(3, 20, 25)
        depth = torch.ones(1, 20, 25)
        sparse = depth.clone()
        sparse[0, 0, 0] = 0.0

        row = raw_descriptor(rgb, depth, sparse)

        self.assertEqual(row["sparse_valid_count"], 499.0)

    def test_raw_descriptor_uses_sixteen_by_sixteen_sparse_occupancy(self):
        rgb = torch.zeros(3, 32, 32)
        depth = torch.ones(1, 32, 32)
        sparse = torch.zeros_like(depth)
        sparse[0, 0, 0] = 1.0
        sparse[0, 0, 31] = 1.0
        sparse[0, 31, 0] = 1.0
        sparse[0, 31, 31] = 1.0

        row = raw_descriptor(rgb, depth, sparse)

        self.assertEqual(row["sparse_grid_occupancy"], 4.0 / 256.0)

    def test_raw_descriptor_rejects_nonfinite_and_excess_sparse_count(self):
        rgb = torch.zeros(3, 20, 26)
        depth = torch.ones(1, 20, 26)
        sparse = depth.clone()
        with self.assertRaisesRegex(ValueError, "at most 500"):
            raw_descriptor(rgb, depth, sparse)
        rgb[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            raw_descriptor(rgb, depth, torch.zeros_like(depth))


class FeatureSpaceTest(unittest.TestCase):
    def test_robust_normalizer_uses_median_iqr_and_excludes_diagnostic(self):
        schema = FeatureSchema(
            names=("depth", "rgb", "count"),
            groups=("depth", "rgb", "diagnostic"),
            diagnostic=(False, False, True))
        values = np.asarray([
            [0.0, 10.0, 500.0],
            [1.0, 12.0, 500.0],
            [2.0, 14.0, 500.0],
            [3.0, 16.0, 500.0],
            [4.0, 18.0, 500.0],
        ], dtype=np.float64)

        normalizer = fit_robust_normalizer(values, schema)
        transformed = normalizer.transform(values)

        np.testing.assert_allclose(normalizer.median, [2.0, 14.0])
        np.testing.assert_allclose(normalizer.iqr, [2.0, 4.0])
        np.testing.assert_allclose(transformed[2], [0.0, 0.0])
        self.assertEqual(normalizer.names, ("depth", "rgb"))

    def test_robust_normalizer_rejects_zero_iqr(self):
        schema = FeatureSchema(
            names=("constant",), groups=("depth",), diagnostic=(False,))
        with self.assertRaisesRegex(ValueError, "IQR"):
            fit_robust_normalizer(np.ones((4, 1)), schema)

    def test_grouped_distance_weights_feature_groups_equally(self):
        values = np.asarray([
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 4.0],
        ])
        distances = grouped_pairwise_distance(
            values, ("activation", "activation", "depth"))

        self.assertAlmostEqual(float(distances[0, 1]), 10.0)
        np.testing.assert_allclose(np.diag(distances), 0.0)


class SelectionAlgorithmTest(unittest.TestCase):
    def test_representative_weights_count_nearest_population_samples(self):
        reference = np.asarray([[0.0], [1.0], [2.0], [9.0], [10.0]])
        centers = np.asarray([[0.0], [10.0]])

        weights = representative_weights(reference, centers, ("depth",))

        np.testing.assert_array_equal(weights, [3.0, 2.0])

    def test_weighted_kmedoids_keeps_fixed_tail_and_represents_dense_body(self):
        indices = np.asarray([10, 11, 12, 13])
        values = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        distances = grouped_pairwise_distance(values, ("depth",))

        result = deterministic_weighted_kmedoids(
            indices, distances, np.asarray([1.0, 10.0, 1.0, 1.0]),
            count=1, fixed_indices=(10,))

        self.assertEqual(result.medoid_indices, (11,))
        self.assertEqual(len(result.assignments), 4)
        self.assertEqual(result.assignments[0], 0)
        self.assertEqual(result.assignments[1], 1)

    def test_tail_cover_covers_low_high_conditions_and_fills_by_tail_score(self):
        indices = np.asarray([10, 11, 12, 13, 14, 15])
        values = np.asarray([
            [-10.0, 0.0],
            [0.0, -10.0],
            [10.0, 10.0],
            [0.0, 0.0],
            [8.0, 0.0],
            [0.0, 8.0],
        ])

        result = select_tail_cover(indices, values, ("a", "b"), budget=3)

        self.assertEqual(result.selected_indices, (12, 10, 11))
        self.assertEqual(len(result.covered_conditions), 4)
        self.assertEqual(len(result.selection_reasons), 3)

    def test_tail_cover_fails_when_budget_cannot_cover_conditions(self):
        indices = np.arange(4)
        values = np.asarray([
            [-10.0, 0.0],
            [10.0, 0.0],
            [0.0, -10.0],
            [0.0, 10.0],
        ])
        with self.assertRaisesRegex(ValueError, "tail conditions"):
            select_tail_cover(indices, values, ("a", "b"), budget=1)

    def test_greedy_kcenter_uses_existing_centers_and_index_ties(self):
        indices = np.asarray([10, 11, 12, 13])
        positions = np.asarray([0.0, 2.0, 5.0, 9.0])
        distances = (positions[:, None] - positions[None, :]) ** 2

        selected = greedy_kcenter(
            indices, distances, count=3, initial_indices=(10,))

        self.assertEqual(selected, (10, 13, 12))

    def test_kmedoids_returns_real_deterministic_cluster_centers(self):
        indices = np.asarray([10, 11, 12, 20, 21, 22])
        positions = np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        distances = (positions[:, None] - positions[None, :]) ** 2

        result = deterministic_kmedoids(indices, distances, count=2)

        self.assertEqual(set(result.medoid_indices), {11, 21})
        self.assertEqual(len(result.assignments), 6)

    def test_incremental_feature_kcenter_matches_distance_matrix_selection(self):
        indices = np.asarray([10, 11, 12, 13])
        values = np.asarray([[0.0], [2.0], [5.0], [9.0]])

        selected = greedy_kcenter_features(
            indices, values, ("raw",), count=3,
            initial_indices=(10,))

        self.assertEqual(selected, (10, 13, 12))

    def test_disjoint_splits_reserve_audit_outside_current_baseline(self):
        result = build_disjoint_splits(
            length=20, baseline_count=4, baseline_seed=3,
            audit_count=5, audit_seed=7, random_seeds=(11, 13))

        self.assertEqual(len(result.baseline_indices), 4)
        self.assertEqual(len(result.audit_indices), 5)
        self.assertEqual(len(result.eligible_indices), 15)
        self.assertFalse(
            set(result.baseline_indices) & set(result.audit_indices))
        for baseline in result.random_baselines:
            self.assertEqual(len(baseline), 4)
            self.assertFalse(set(baseline) & set(result.audit_indices))


class CoverageMetricTest(unittest.TestCase):
    def test_descriptor_coverage_reports_quantiles_range_and_wasserstein(self):
        calibration = np.asarray([[0.0], [5.0], [10.0]])
        audit = np.asarray([[1.0], [3.0], [7.0], [9.0]])

        rows = descriptor_coverage(
            "stratified", calibration, audit, ("depth",), ("depth",))

        row = rows[0]
        self.assertEqual(row["configuration"], "stratified")
        self.assertEqual(row["range_coverage"], 1.0)
        self.assertAlmostEqual(row["audit_p50"], 5.0)
        self.assertGreaterEqual(row["wasserstein"], 0.0)

    def test_nearest_distance_summary_uses_equal_group_distance(self):
        calibration = np.asarray([[0.0, 0.0], [10.0, 10.0]])
        audit = np.asarray([[1.0, 1.0], [8.0, 8.0]])

        row = nearest_distance_summary(
            "stratified", calibration, audit, ("raw", "raw"))

        self.assertAlmostEqual(row["nearest_p50"], 2.5)
        self.assertAlmostEqual(row["nearest_p95"], 3.85)

    def test_activation_range_coverage_counts_audit_max_exceedance(self):
        names = ("stem_max", "decoder_max")
        calibration = np.asarray([[1.0, 5.0], [2.0, 6.0]])
        audit = np.asarray([[1.5, 7.0], [1.0, 4.0]])

        rows = activation_range_coverage(
            "stratified", calibration, audit, names)

        self.assertEqual(rows[0]["audit_exceeds_calibration"], 0)
        self.assertEqual(rows[1]["audit_exceeds_calibration"], 1)
        self.assertAlmostEqual(rows[1]["maximum_ratio"], 7.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
