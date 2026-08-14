import unittest

from spn_quant import cspn_sensitivity as sensitivity


def registry():
    modules = tuple(
        name
        for block in sensitivity.BLOCK_ORDER
        for name in sensitivity.WEIGHT_MODULES_BY_BLOCK[block]
    )
    owners = tuple(
        owner
        for block in sensitivity.BLOCK_ORDER
        for owner in sensitivity.ACTIVATION_OWNERS_BY_BLOCK[block]
    )
    return sensitivity.build_candidate_registry(modules, owners)


class CandidateRegistryTest(unittest.TestCase):
    def test_registry_matches_all_official_decoder_candidates(self):
        result = registry()

        self.assertEqual(tuple(result.weights_by_block), sensitivity.BLOCK_ORDER)
        self.assertEqual(
            sum(len(values) for values in result.weights_by_block.values()),
            16)
        self.assertEqual(
            sum(len(values) for values in result.activations_by_block.values()),
            30)
        self.assertEqual(
            result.input_dependencies["gud_up_proj_layer4.conv1_1"], (
                ("gud_up_proj_layer4.relu#0", "relu_output"),
                ("rotation.layer4_signed_skip", "boundary"),
            ))

    def test_registry_rejects_missing_weight_module(self):
        modules = tuple(
            name
            for block in sensitivity.BLOCK_ORDER
            for name in sensitivity.WEIGHT_MODULES_BY_BLOCK[block]
        )[:-1]
        owners = tuple(
            owner
            for block in sensitivity.BLOCK_ORDER
            for owner in sensitivity.ACTIVATION_OWNERS_BY_BLOCK[block]
        )

        with self.assertRaisesRegex(ValueError, "weight registry"):
            sensitivity.build_candidate_registry(modules, owners)

    def test_registry_rejects_unknown_activation_owner(self):
        modules = tuple(
            name
            for block in sensitivity.BLOCK_ORDER
            for name in sensitivity.WEIGHT_MODULES_BY_BLOCK[block]
        )
        owners = tuple(
            owner
            for block in sensitivity.BLOCK_ORDER
            for owner in sensitivity.ACTIVATION_OWNERS_BY_BLOCK[block]
        ) + (("unknown", "input"),)

        with self.assertRaisesRegex(ValueError, "activation registry"):
            sensitivity.build_candidate_registry(modules, owners)

    def test_stage1_contains_strict_and_three_modes_per_block(self):
        candidates = sensitivity.build_stage1_candidates(registry())

        self.assertEqual(len(candidates), 16)
        self.assertEqual(candidates[0].name, "STRICT_W4A4")
        self.assertEqual(
            [candidate.mode for candidate in candidates[1:4]],
            ["W4A8", "W8A4", "W8A8"])
        self.assertEqual(
            candidates[3].weight_modules,
            sensitivity.WEIGHT_MODULES_BY_BLOCK["decoder_layer1"])
        self.assertEqual(
            candidates[3].activation_owners,
            sensitivity.ACTIVATION_OWNERS_BY_BLOCK["decoder_layer1"])


class StageSelectionTest(unittest.TestCase):
    def test_selects_two_blocks_by_best_mode_rmse(self):
        candidates = sensitivity.build_stage1_candidates(registry())
        rows = [{"config": "STRICT_W4A4", "RMSE": 1.0}]
        best = {
            "decoder_layer1": 0.95,
            "decoder_layer2": 1.02,
            "decoder_layer3": 0.90,
            "decoder_layer4": 0.98,
            "initial_depth": 1.01,
        }
        for candidate in candidates[1:]:
            offset = {"W4A8": 0.02, "W8A4": 0.01, "W8A8": 0.0}[candidate.mode]
            rows.append({
                "config": candidate.name,
                "RMSE": best[candidate.block] + offset,
            })

        selected = sensitivity.select_sensitive_blocks(rows, candidates)

        self.assertEqual(selected, ("decoder_layer3", "decoder_layer1"))

    def test_no_improvement_selects_least_regressed_blocks_stably(self):
        candidates = sensitivity.build_stage1_candidates(registry())
        rows = [{"config": "STRICT_W4A4", "RMSE": 1.0}]
        best = {
            "decoder_layer1": 1.03,
            "decoder_layer2": 1.01,
            "decoder_layer3": 1.02,
            "decoder_layer4": 1.01,
            "initial_depth": 1.04,
        }
        for candidate in candidates[1:]:
            rows.append({
                "config": candidate.name,
                "RMSE": best[candidate.block],
            })

        selected = sensitivity.select_sensitive_blocks(rows, candidates)

        self.assertEqual(selected, ("decoder_layer2", "decoder_layer4"))

    def test_stage2_builds_activation_weight_and_paired_candidates(self):
        candidates = sensitivity.build_stage2_candidates(
            registry(), ("initial_depth",))

        self.assertEqual(len(candidates), 3)
        by_mode = {candidate.mode: candidate for candidate in candidates}
        self.assertEqual(
            by_mode["W4A8"].activation_owners,
            (("gud_up_proj_layer5.conv1", "input"),))
        self.assertEqual(
            by_mode["W8A4"].weight_modules,
            ("gud_up_proj_layer5.conv1",))
        self.assertEqual(
            by_mode["W8A8"].activation_owners,
            (("gud_up_proj_layer5.conv1", "input"),))


class CumulativeAndParetoTest(unittest.TestCase):
    def test_cumulative_candidates_union_promotions_without_duplicates(self):
        sites = sensitivity.build_stage2_candidates(
            registry(), ("initial_depth",))
        rows = []
        rmse = {"W4A8": 0.95, "W8A4": 0.97, "W8A8": 0.90}
        cost = {"W4A8": 0.01, "W8A4": 0.02, "W8A8": 0.03}
        for candidate in sites:
            rows.append({
                "config": candidate.name,
                "RMSE": rmse[candidate.mode],
                "normalized_added_bit_cost": cost[candidate.mode],
            })

        cumulative = sensitivity.build_cumulative_candidates(
            rows, sites, strict_rmse=1.0)

        self.assertEqual(len(cumulative), 1)
        self.assertEqual(cumulative[0].mode, "MIXED")
        self.assertEqual(
            cumulative[0].weight_modules,
            ("gud_up_proj_layer5.conv1",))
        self.assertEqual(
            cumulative[0].activation_owners,
            (("gud_up_proj_layer5.conv1", "input"),))

    def test_normalized_cost_counts_weight_and_activation_elements(self):
        candidate = sensitivity.SensitivityCandidate(
            "candidate", "site", "initial_depth", "W8A8",
            ("head",), (("head", "input"),))
        weights = (
            {"module": "head", "weight_elements": 20, "macs": 100},
            {"module": "other", "weight_elements": 80, "macs": 900},
        )
        activations = (
            {"module": "head", "kind": "input", "elements": 50},
            {"module": "other", "kind": "input", "elements": 50},
        )

        cost = sensitivity.precision_cost(weights, activations, candidate)

        self.assertAlmostEqual(cost["normalized_added_bit_cost"], 70 / 200)
        self.assertAlmostEqual(cost["w8_weight_mac_fraction"], 0.1)
        self.assertAlmostEqual(cost["w8_weight_element_fraction"], 0.2)
        self.assertAlmostEqual(cost["a8_activation_element_fraction"], 0.5)

    def test_pareto_rows_remove_dominated_configs(self):
        rows = (
            {"config": "strict", "RMSE": 1.0,
             "normalized_added_bit_cost": 0.0},
            {"config": "good", "RMSE": 0.8,
             "normalized_added_bit_cost": 0.1},
            {"config": "dominated", "RMSE": 0.9,
             "normalized_added_bit_cost": 0.2},
            {"config": "accurate", "RMSE": 0.7,
             "normalized_added_bit_cost": 0.3},
        )

        pareto = sensitivity.pareto_rows(rows)

        self.assertEqual(
            [row["config"] for row in pareto],
            ["strict", "good", "accurate"])


if __name__ == "__main__":
    unittest.main()
