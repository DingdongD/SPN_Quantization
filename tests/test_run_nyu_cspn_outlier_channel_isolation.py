import unittest

from scripts import run_nyu_cspn_outlier_channel_isolation as runner


class OutlierIsolationRunnerTest(unittest.TestCase):
    def test_candidate_ranking_uses_calibration_energy_then_count_and_identity(self):
        rows = [
            {"module": "b", "kind": "input", "group_index": 0,
             "outlier_channel": 3,
             "rescued_energy": 4.0, "rescued_elements": 10},
            {"module": "a", "kind": "relu_output", "group_index": 1,
             "outlier_channel": 9,
             "rescued_energy": 5.0, "rescued_elements": 8},
            {"module": "a", "kind": "input", "group_index": 0,
             "outlier_channel": 2,
             "rescued_energy": 4.0, "rescued_elements": 11},
            {"module": "c", "kind": "input", "group_index": 0,
             "outlier_channel": 1,
             "rescued_energy": 0.0, "rescued_elements": 0},
        ]

        ranked = runner.rank_candidates(rows)

        self.assertEqual(
            [(row["module"], row["outlier_channel"]) for row in ranked],
            [("a", 9), ("a", 2), ("b", 3)])

    def test_budgets_are_cumulative_and_end_with_all_positive_candidates(self):
        rows = [
            {"module": "m%d" % index, "kind": "input", "group_index": 0,
             "outlier_channel": index, "rescued_energy": 10.0 - index,
             "rescued_elements": 10 - index}
            for index in range(6)
        ]

        budgets = runner.build_budgets(rows, prefix_limits=(1, 2, 4))

        self.assertEqual([len(budget["selected"]) for budget in budgets],
                         [0, 1, 2, 4, 6])
        self.assertEqual(budgets[0]["name"], "W4A4_G8_CONTIGUOUS")
        self.assertEqual(budgets[-1]["name"], "W4A4_G8_OCI_ALL")

    def test_isolation_mapping_preserves_local_qdq_owner_identity(self):
        selected = (
            {"module": "layer1.0.conv1", "kind": "input",
             "outlier_channel": 3},
            {"module": "layer1.0.conv1", "kind": "input",
             "outlier_channel": 11},
            {"module": "relu", "kind": "relu_output",
             "outlier_channel": 7},
        )

        mapping = runner.isolation_mapping(selected)

        self.assertEqual(mapping, (
            (("layer1.0.conv1", "input"), (3, 11)),
            ("relu", (7,)),
        ))


if __name__ == "__main__":
    unittest.main()
