import unittest

from spn_quant.cspn_encoder_prefix import (
    ACTIVATION_OWNERS_BY_UNIT,
    ALL_UNIT_ORDER,
    UnitRegistry,
    WEIGHT_MODULES_BY_UNIT,
)
from spn_quant.cspn_selective_w4a8 import (
    ACTIVATION_UNIT_ORDER,
    INITIAL_DEPTH_WEIGHT,
    build_cumulative_path,
    build_single_demotions,
    build_stage1_candidates,
    pareto_rows,
    rank_demotions,
    select_stage1_anchor,
    select_winner,
)


def registry():
    return UnitRegistry(
        weights_by_unit=dict(
            (unit, WEIGHT_MODULES_BY_UNIT[unit]) for unit in ALL_UNIT_ORDER),
        activations_by_unit=dict(
            (unit, ACTIVATION_OWNERS_BY_UNIT[unit])
            for unit in ALL_UNIT_ORDER),
    )


def feasible_row(config, cost, a8_fraction, rmse):
    return {
        "config": config,
        "samples": 64,
        "RMSE": rmse,
        "normalized_added_bit_cost": cost,
        "a8_activation_element_fraction": a8_fraction,
        "nonfinite_ratio": 0.0,
        "nonpositive_ratio": 0.0,
        "coefficient_sum_max_error": 0.0,
        "contraction_violation_rate": 0.0,
        "anchor_max_error": 0.0,
    }


class Stage1CandidateTest(unittest.TestCase):
    def test_complete_four_unit_mask_search(self):
        candidates = build_stage1_candidates(registry())

        self.assertEqual(ACTIVATION_UNIT_ORDER, (
            "stem", "encoder_layer1", "encoder_layer2", "decoder_layer4"))
        self.assertEqual(len(candidates), 16)
        self.assertEqual([candidate.mask for candidate in candidates],
                         list(range(16)))
        self.assertEqual(candidates[0].selected_units, ())
        self.assertEqual(candidates[0].activation_owners, ())
        self.assertEqual(candidates[-1].selected_units, ACTIVATION_UNIT_ORDER)
        self.assertEqual(candidates[-1].weight_modules,
                         (INITIAL_DEPTH_WEIGHT,))
        self.assertEqual(
            candidates[-1].activation_owners.count(
                ("rotation.layer4_signed_skip", "boundary")), 1)

    def test_stage1_anchor_uses_cost_then_a8_then_rmse(self):
        candidates = build_stage1_candidates(registry())
        rows = [feasible_row(
            candidate.name, 0.3 + candidate.mask / 100.0,
            0.4 + candidate.mask / 100.0, 0.19)
                for candidate in candidates]
        rows[5] = feasible_row(candidates[5].name, 0.2, 0.3, 0.176)
        rows[6] = feasible_row(candidates[6].name, 0.2, 0.25, 0.177)

        anchor = select_stage1_anchor(rows, candidates, 0.1773)

        self.assertEqual(anchor["config"], candidates[6].name)

    def test_stage1_anchor_rejects_nonpositive_predictions(self):
        candidates = build_stage1_candidates(registry())
        rows = [feasible_row(
            candidate.name, 0.2, 0.3, 0.19) for candidate in candidates]
        rows[0] = feasible_row(candidates[0].name, 0.1, 0.2, 0.17)
        rows[0]["nonpositive_ratio"] = 0.001

        with self.assertRaisesRegex(RuntimeError, "target"):
            select_stage1_anchor(rows, candidates, 0.1773)


class BoundaryDemotionTest(unittest.TestCase):
    def test_single_demotions_remove_exactly_one_owner(self):
        anchor = build_stage1_candidates(registry())[-1]

        demotions = build_single_demotions(anchor)

        self.assertEqual(len(demotions), len(anchor.activation_owners))
        for demotion in demotions:
            self.assertNotIn(demotion.owner, demotion.activation_owners)
            self.assertEqual(
                len(demotion.activation_owners),
                len(anchor.activation_owners) - 1)
            self.assertEqual(demotion.weight_modules,
                             (INITIAL_DEPTH_WEIGHT,))

    def test_ranking_uses_propagation_error_per_saved_cost(self):
        anchor = build_stage1_candidates(registry())[3]
        demotions = build_single_demotions(anchor)[:3]
        anchor_row = {"propagation_mse": 1.0}
        rows = [
            {
                "config": demotions[0].name,
                "module": demotions[0].owner[0],
                "kind": demotions[0].owner[1],
                "propagation_mse": 1.02,
                "downstream_mse": 0.04,
                "anchor_downstream_mse": 0.01,
                "saved_cost": 0.02,
            },
            {
                "config": demotions[1].name,
                "module": demotions[1].owner[0],
                "kind": demotions[1].owner[1],
                "propagation_mse": 0.99,
                "downstream_mse": 0.02,
                "anchor_downstream_mse": 0.01,
                "saved_cost": 0.01,
            },
            {
                "config": demotions[2].name,
                "module": demotions[2].owner[0],
                "kind": demotions[2].owner[1],
                "propagation_mse": 1.01,
                "downstream_mse": 0.03,
                "anchor_downstream_mse": 0.01,
                "saved_cost": 0.02,
            },
        ]

        ranking = rank_demotions(anchor_row, rows, demotions)

        self.assertEqual(
            [(row["module"], row["kind"]) for row in ranking],
            [demotions[1].owner, demotions[2].owner, demotions[0].owner])
        self.assertEqual([row["rank"] for row in ranking], [1, 2, 3])
        self.assertAlmostEqual(ranking[0]["score"], 0.0)
        self.assertAlmostEqual(ranking[1]["score"], 0.5)

    def test_cumulative_path_follows_frozen_ranking(self):
        anchor = build_stage1_candidates(registry())[3]
        demotions = build_single_demotions(anchor)
        ordered = (demotions[1], demotions[0]) + demotions[2:]
        ranking = [
            {"rank": rank, "module": demotion.owner[0],
             "kind": demotion.owner[1]}
            for rank, demotion in enumerate(ordered, start=1)]

        path = build_cumulative_path(anchor, ranking)

        self.assertEqual(
            [candidate.step for candidate in path],
            list(range(len(anchor.activation_owners) + 1)))
        self.assertEqual(path[1].demoted_owners, (demotions[1].owner,))
        self.assertEqual(
            path[2].demoted_owners,
            (demotions[1].owner, demotions[0].owner))
        self.assertGreater(
            len(path[0].activation_owners),
            len(path[1].activation_owners))
        self.assertGreater(
            len(path[1].activation_owners),
            len(path[2].activation_owners))
        self.assertEqual(path[-1].activation_owners, ())


class WinnerTest(unittest.TestCase):
    def test_winner_requires_rerun_and_uses_lexicographic_cost(self):
        expensive = feasible_row("expensive", 0.2, 0.3, 0.17)
        cheap = feasible_row("cheap", 0.1, 0.2, 0.176)
        for row in (expensive, cheap):
            row["rerun_RMSE"] = row["RMSE"]

        winner = select_winner((expensive, cheap), 0.1773)

        self.assertEqual(winner["config"], "cheap")

    def test_pareto_supports_cost_and_a8_fraction(self):
        rows = [
            feasible_row("low", 0.1, 0.2, 0.176),
            feasible_row("middle", 0.2, 0.3, 0.17),
            feasible_row("dominated", 0.3, 0.4, 0.18),
        ]

        cost = pareto_rows(rows, "normalized_added_bit_cost")
        activation = pareto_rows(rows, "a8_activation_element_fraction")

        self.assertEqual([row["config"] for row in cost], ["low", "middle"])
        self.assertEqual(
            [row["config"] for row in activation], ["low", "middle"])


if __name__ == "__main__":
    unittest.main()
