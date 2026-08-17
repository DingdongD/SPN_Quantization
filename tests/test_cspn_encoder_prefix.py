import unittest

from spn_quant import cspn_encoder_prefix as prefix


def stable_union(sequences):
    values = []
    for sequence in sequences:
        for value in sequence:
            if value not in values:
                values.append(value)
    return tuple(values)


def registry():
    modules = stable_union(
        prefix.WEIGHT_MODULES_BY_UNIT[unit]
        for unit in prefix.ALL_UNIT_ORDER)
    owners = stable_union(
        prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
        for unit in prefix.ALL_UNIT_ORDER)
    return prefix.build_unit_registry(modules, owners)


class UnitRegistryTest(unittest.TestCase):
    def test_registry_matches_official_prefix_and_tail_units(self):
        result = registry()

        self.assertEqual(tuple(result.weights_by_unit), prefix.ALL_UNIT_ORDER)
        self.assertEqual(len(stable_union(result.weights_by_unit.values())), 25)
        self.assertEqual(
            len(stable_union(result.activations_by_unit.values())), 49)
        self.assertEqual(
            result.weights_by_unit["stem"], ("conv1_1",))
        self.assertEqual(len(result.weights_by_unit["encoder_layer1"]), 4)
        self.assertEqual(len(result.weights_by_unit["encoder_layer2"]), 5)
        self.assertIn(
            ("rotation.layer4_signed_skip", "boundary"),
            result.activations_by_unit["stem"])
        self.assertIn(
            ("rotation.layer4_signed_skip", "boundary"),
            result.activations_by_unit["decoder_layer4"])

    def test_registry_rejects_missing_weight(self):
        modules = stable_union(
            prefix.WEIGHT_MODULES_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER)[:-1]
        owners = stable_union(
            prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER)

        with self.assertRaisesRegex(ValueError, "weight registry"):
            prefix.build_unit_registry(modules, owners)

    def test_registry_rejects_extra_owner(self):
        modules = stable_union(
            prefix.WEIGHT_MODULES_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER)
        owners = stable_union(
            prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER) + (("unknown", "input"),)

        with self.assertRaisesRegex(ValueError, "activation registry"):
            prefix.build_unit_registry(modules, owners)

    def test_registry_rejects_duplicate_executed_values(self):
        modules = stable_union(
            prefix.WEIGHT_MODULES_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER)
        owners = stable_union(
            prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER)

        with self.assertRaisesRegex(ValueError, "duplicates"):
            prefix.build_unit_registry(modules + (modules[0],), owners)


class CandidateMatrixTest(unittest.TestCase):
    def test_builds_complete_prefix_major_matrix(self):
        candidates = prefix.build_candidates(registry())

        self.assertEqual(len(candidates), 24)
        self.assertEqual(len({candidate.name for candidate in candidates}), 24)
        self.assertEqual(candidates[0].name, "PREFIX_P0__TAIL_T0")
        self.assertEqual(candidates[1].name, "PREFIX_P0__TAIL_T1")
        self.assertEqual(candidates[4].name, "PREFIX_P1__TAIL_T0")
        self.assertEqual(candidates[-1].name, "PREFIX_P5__TAIL_T3")

    def test_strict_and_full_candidates_have_exact_promotions(self):
        candidates = prefix.build_candidates(registry())
        strict = candidates[0]
        full = candidates[-1]

        self.assertEqual(strict.weight_modules, ())
        self.assertEqual(strict.activation_owners, ())
        self.assertFalse(strict.stem_w8a8)
        self.assertEqual(full.encoder_units, prefix.ENCODER_UNIT_ORDER)
        self.assertEqual(full.tail_units, prefix.TAIL_STATES[-1])
        self.assertEqual(len(full.weight_modules), 25)
        self.assertEqual(len(full.activation_owners), 49)
        self.assertTrue(full.stem_w8a8)

    def test_shared_skip_boundary_is_present_once(self):
        candidate = next(
            value for value in prefix.build_candidates(registry())
            if value.prefix_index == 1 and value.tail_index == 1)

        self.assertEqual(
            candidate.activation_owners.count(
                ("rotation.layer4_signed_skip", "boundary")), 1)

    def test_prefixes_are_cumulative(self):
        candidates = prefix.build_candidates(registry())
        rows = [candidate for candidate in candidates
                if candidate.tail_index == 0]

        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(
                current.encoder_units[:-1], previous.encoder_units)
            self.assertTrue(
                set(previous.weight_modules) < set(current.weight_modules))


class CostInteractionAndParetoTest(unittest.TestCase):
    def test_cost_counts_shared_owner_once(self):
        candidate = next(
            value for value in prefix.build_candidates(registry())
            if value.prefix_index == 1 and value.tail_index == 1)
        weights = tuple({
            "module": name,
            "weight_elements": 1,
            "macs": 10,
        } for name in stable_union(
            prefix.WEIGHT_MODULES_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER))
        owners = stable_union(
            prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER)
        activations = tuple({
            "module": owner[0],
            "kind": owner[1],
            "elements": 1,
        } for owner in owners)

        result = prefix.precision_cost(weights, activations, candidate)

        self.assertEqual(len(candidate.weight_modules), 5)
        self.assertEqual(len(candidate.activation_owners), 10)
        self.assertAlmostEqual(
            result["normalized_added_bit_cost"], 15 / 74)
        self.assertAlmostEqual(result["w8_weight_mac_fraction"], 5 / 25)
        self.assertAlmostEqual(
            result["a8_activation_element_fraction"], 10 / 49)

    def test_interaction_uses_complete_factorial_baselines(self):
        rows = []
        for prefix_index in range(6):
            for tail_index in range(4):
                rmse = 1.0 - 0.1 * prefix_index - 0.05 * tail_index
                if prefix_index == 2 and tail_index == 3:
                    rmse -= 0.02
                rows.append({
                    "config": "P%dT%d" % (prefix_index, tail_index),
                    "prefix_index": prefix_index,
                    "tail_index": tail_index,
                    "RMSE": rmse,
                })

        result = prefix.interaction_rows(rows)
        by_cell = {
            (row["prefix_index"], row["tail_index"]): row
            for row in result
        }

        self.assertEqual(len(result), 24)
        self.assertAlmostEqual(by_cell[(0, 3)]["interaction_rmse"], 0.0)
        self.assertAlmostEqual(by_cell[(2, 0)]["interaction_rmse"], 0.0)
        self.assertAlmostEqual(by_cell[(2, 3)]["interaction_rmse"], -0.02)

    def test_pareto_supports_both_cost_fields(self):
        rows = (
            {"config": "strict", "RMSE": 1.0,
             "normalized_added_bit_cost": 0.0,
             "w8_weight_mac_fraction": 0.0},
            {"config": "good", "RMSE": 0.8,
             "normalized_added_bit_cost": 0.1,
             "w8_weight_mac_fraction": 0.2},
            {"config": "dominated", "RMSE": 0.9,
             "normalized_added_bit_cost": 0.2,
             "w8_weight_mac_fraction": 0.3},
            {"config": "mac_tradeoff", "RMSE": 0.75,
             "normalized_added_bit_cost": 0.4,
             "w8_weight_mac_fraction": 0.25},
        )

        normalized = prefix.pareto_rows(
            rows, "normalized_added_bit_cost")
        mac = prefix.pareto_rows(rows, "w8_weight_mac_fraction")

        self.assertEqual(
            [row["config"] for row in normalized],
            ["strict", "good", "mac_tradeoff"])
        self.assertEqual(
            [row["config"] for row in mac],
            ["strict", "good", "mac_tradeoff"])

    def test_prediction_selection_is_deterministic_and_unique(self):
        rows = []
        candidates = prefix.build_candidates(registry())
        for candidate in candidates:
            rows.append({
                "config": candidate.name,
                "RMSE": 1.0 if candidate.prefix_index == 0 and
                candidate.tail_index == 0 else 0.9,
                "normalized_added_bit_cost":
                0.01 * (candidate.prefix_index + candidate.tail_index),
            })
        pareto = (
            rows[0], rows[5], rows[-1],
        )

        names = prefix.prediction_candidate_names(rows, pareto)

        self.assertEqual(names[0], "PREFIX_P0__TAIL_T0")
        self.assertIn("PREFIX_P5__TAIL_T3", names)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
