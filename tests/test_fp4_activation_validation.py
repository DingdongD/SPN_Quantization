import unittest

from scripts.fp4_activation_validation import (
    build_fp4_validation_configurations,
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


if __name__ == "__main__":
    unittest.main()
