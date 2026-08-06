import unittest

import numpy as np

from scripts.analyze_fp4_activation_validation import (
    aggregate_sample_rows,
    build_paired_comparisons,
    propagation_diagnostics_clear,
    validate_sample_rows,
)
from scripts.fp4_activation_validation import (
    FP4_CONFIG_NAMES,
    a4_to_a8_recovery,
    build_fp4_validation_configurations,
    paired_bootstrap_rmse_difference,
    resolve_semantic_a8_overrides,
)


class FP4ValidationConfigurationTest(unittest.TestCase):
    def test_matrix_contains_two_matched_triplets(self):
        configs = build_fp4_validation_configurations(
            {"encoder", "decoder"})

        self.assertEqual(
            [config["name"] for config in configs],
            [
                "FP32",
                "FP4V_W8A4",
                "FP4V_W8E2M1",
                "FP4V_W8A8",
                "FP4V_W4A4",
                "FP4V_W4E2M1",
                "FP4V_W4A8",
            ])
        for config in configs[1:]:
            self.assertFalse(config["quantize_bias"])
            self.assertEqual(
                config["propagation"],
                {
                    "affinity_bits": 8,
                    "confidence_bits": 8,
                    "offset_bits": 8,
                    "state_bits": 8,
                    "coefficient_fraction_bits": 13,
                })

    def test_triplets_only_change_activation_format_and_precision(self):
        configs = build_fp4_validation_configurations({"encoder"})
        by_name = {config["name"]: config for config in configs}

        self.assertEqual(by_name["FP4V_W8A4"]["w_bits"], 8)
        self.assertEqual(by_name["FP4V_W8E2M1"]["w_bits"], 8)
        self.assertEqual(by_name["FP4V_W8A8"]["w_bits"], 8)
        self.assertEqual(by_name["FP4V_W4A4"]["w_bits"], 4)
        self.assertEqual(by_name["FP4V_W4E2M1"]["w_bits"], 4)
        self.assertEqual(by_name["FP4V_W4A8"]["w_bits"], 4)
        self.assertEqual(
            by_name["FP4V_W8E2M1"]["activation_mode"], "e2m1")
        self.assertEqual(
            by_name["FP4V_W4E2M1"]["activation_mode"], "e2m1")
        self.assertEqual(by_name["FP4V_W8A4"]["a_bits"], 4)
        self.assertEqual(by_name["FP4V_W8A8"]["a_bits"], 8)


class FP4SemanticBoundaryTest(unittest.TestCase):
    def test_completionformer_boundaries_keep_semantic_tensors_a8(self):
        modules = {
            "backbone.conv1_dep.0",
            "backbone.dep_dec0.0",
            "backbone.gd_dec0.0",
            "backbone.cf_dec0.0",
        }

        bits, formats, manifest = resolve_semantic_a8_overrides(
            "completionformer", modules)

        expected = {
            ("backbone.conv1_dep.0", "input"),
            ("backbone.dep_dec0.0", "output"),
            ("backbone.gd_dec0.0", "output"),
            ("backbone.cf_dec0.0", "output"),
        }
        self.assertEqual(set(bits), expected)
        self.assertEqual(set(formats), expected)
        self.assertEqual(set(bits.values()), {8})
        self.assertEqual(set(formats.values()), {"uniform"})
        self.assertEqual(
            {row["role"] for row in manifest},
            {"sparse_depth_input", "initial_depth", "guidance", "confidence"})

    def test_cspn_combined_rgbd_input_is_kept_a8(self):
        modules = {
            "conv1_1",
            "gud_up_proj_layer5.conv1",
            "gud_up_proj_layer6.conv1",
        }

        bits, formats, manifest = resolve_semantic_a8_overrides(
            "cspn", modules)

        self.assertEqual(bits[("conv1_1", "input")], 8)
        self.assertEqual(formats[("conv1_1", "input")], "uniform")
        input_row = [
            row for row in manifest if row["role"] == "sparse_depth_input"
        ][0]
        self.assertEqual(input_row["module"], "conv1_1")

    def test_missing_required_boundary_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "initial_depth"):
            resolve_semantic_a8_overrides("nlspn", {"conv1_dep.0"})

    def test_multiple_boundary_matches_fail_closed(self):
        modules = {
            "base.conv1_dep.0",
            "base.gd_dec0_dyspn_6_5.0",
            "base.gd_dec0_dyspn_7_5.0",
        }
        with self.assertRaisesRegex(RuntimeError, "combined_prop_inputs"):
            resolve_semantic_a8_overrides("dyspn", modules)


class FP4PairedStatisticsTest(unittest.TestCase):
    def test_bootstrap_reports_e2m1_minus_int4_difference(self):
        int4 = np.array([1.0, 1.2, 1.4, 1.6])
        e2m1 = np.array([0.6, 0.8, 1.0, 1.2])

        result = paired_bootstrap_rmse_difference(
            int4, e2m1, resamples=1000, seed=20260806)

        self.assertAlmostEqual(result["mean_difference"], -0.4)
        self.assertLess(result["ci_lower"], 0.0)
        self.assertLess(result["ci_upper"], 0.0)
        self.assertEqual(result["samples"], 4)

    def test_bootstrap_is_deterministic_and_rejects_invalid_pairs(self):
        int4 = np.array([1.0, 2.0, 3.0])
        e2m1 = np.array([0.8, 2.2, 2.4])

        first = paired_bootstrap_rmse_difference(
            int4, e2m1, resamples=200, seed=31)
        second = paired_bootstrap_rmse_difference(
            int4, e2m1, resamples=200, seed=31)

        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "identical shape"):
            paired_bootstrap_rmse_difference(
                int4, e2m1[:2], resamples=200, seed=31)
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_bootstrap_rmse_difference(
                int4, np.array([0.8, np.nan, 2.4]),
                resamples=200, seed=31)

    def test_recovery_uses_matched_mean_rmse_and_marks_invalid_denominator(self):
        self.assertAlmostEqual(a4_to_a8_recovery(1.2, 0.8, 0.4), 0.5)
        self.assertIsNone(a4_to_a8_recovery(0.8, 0.7, 0.8))
        self.assertIsNone(a4_to_a8_recovery(0.8, 0.7, 0.9))


class FP4ResultAnalysisTest(unittest.TestCase):
    @staticmethod
    def sample_rows():
        values = {
            "FP32": (0.1, 0.2),
            "FP4V_W8A4": (1.0, 1.2),
            "FP4V_W8E2M1": (0.6, 0.8),
            "FP4V_W8A8": (0.4, 0.5),
            "FP4V_W4A4": (1.4, 1.6),
            "FP4V_W4E2M1": (1.0, 1.2),
            "FP4V_W4A8": (0.8, 0.9),
        }
        rows = []
        for config, rmses in values.items():
            for sample_index, rmse in zip((3, 7), rmses):
                rows.append({
                    "model": "cspn",
                    "config": config,
                    "sample_index": str(sample_index),
                    "RMSE": str(rmse),
                    "MAE": str(rmse / 2.0),
                    "ABS_REL": str(rmse / 10.0),
                    "nonfinite_pixels": "0",
                    "num_pixels": "10",
                })
        return rows

    def test_sample_rows_require_exact_configuration_and_indices(self):
        rows = self.sample_rows()
        validate_sample_rows(rows, "cspn", FP4_CONFIG_NAMES, (3, 7))

        with self.assertRaisesRegex(ValueError, "sample rows"):
            validate_sample_rows(rows[:-1], "cspn", FP4_CONFIG_NAMES, (3, 7))

    def test_aggregation_and_paired_comparison_use_matched_samples(self):
        rows = self.sample_rows()

        aggregates = aggregate_sample_rows(rows, "cspn", FP4_CONFIG_NAMES)
        comparisons = build_paired_comparisons(
            rows, "cspn", resamples=500, seed=20260806,
            constraints_clear={"FP4V_W8E2M1": True,
                               "FP4V_W4E2M1": True},
            diagnostics_clear={"FP4V_W8E2M1": True,
                               "FP4V_W4E2M1": True})

        w8a4 = [row for row in aggregates
                if row["config"] == "FP4V_W8A4"][0]
        self.assertAlmostEqual(w8a4["mean_sample_RMSE"], 1.1)
        self.assertEqual(w8a4["nonfinite_samples"], 0)
        self.assertEqual(len(comparisons), 2)
        self.assertAlmostEqual(comparisons[0]["mean_difference"], -0.4)
        self.assertGreater(comparisons[0]["recovery"], 0.0)
        self.assertTrue(comparisons[0]["effective"])

    def test_propagation_diagnostic_rejects_step_amplification(self):
        rows = [
            {"config": "q", "signal": "pred_init", "iteration": "0",
             "rmse": "1.0"},
            {"config": "q", "signal": "pred", "iteration": "0",
             "rmse": "0.8"},
            {"config": "q", "signal": "propagation_states",
             "iteration": "1", "rmse": "0.7"},
            {"config": "q", "signal": "propagation_states",
             "iteration": "2", "rmse": "0.9"},
        ]

        self.assertFalse(propagation_diagnostics_clear(rows, "q"))
        rows[-1]["rmse"] = "0.6"
        self.assertTrue(propagation_diagnostics_clear(rows, "q"))


if __name__ == "__main__":
    unittest.main()
