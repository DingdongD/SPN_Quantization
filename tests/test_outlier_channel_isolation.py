import unittest

import torch

from spn_quant.outlier_channel_isolation import (
    OutlierHarmAccumulator,
    build_outlier_candidates,
    isolated_channel_scales,
)


class OutlierChannelIsolationTest(unittest.TestCase):
    def test_candidates_use_contiguous_group_maximum_and_second_maximum(self):
        maximum = torch.tensor([
            1.0, 2.0, 3.0, 12.0, 4.0, 5.0, 6.0, 7.0,
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
        ])

        candidates = build_outlier_candidates(maximum, group_size=8)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].group_index, 0)
        self.assertEqual(candidates[0].outlier_channel, 3)
        self.assertEqual(candidates[0].victim_channels, (0, 1, 2, 4, 5, 6, 7))
        self.assertEqual(candidates[0].outlier_maximum, 12.0)
        self.assertEqual(candidates[0].remaining_maximum, 7.0)
        self.assertAlmostEqual(candidates[0].original_zero_threshold, 12.0 / 30.0)
        self.assertAlmostEqual(candidates[0].isolated_zero_threshold, 7.0 / 30.0)
        self.assertEqual(candidates[1].outlier_channel, 8)

    def test_isolated_scales_keep_outlier_and_reduce_only_its_group_victims(self):
        maximum = torch.tensor([
            1.0, 2.0, 3.0, 12.0, 4.0, 5.0, 6.0, 7.0,
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
        ])

        scales = isolated_channel_scales(
            maximum, bits=4, group_size=8, isolated_channels=(3,))

        torch.testing.assert_close(
            scales[:8], torch.tensor([
                7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0, 12.0 / 15.0,
                7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0, 7.0 / 15.0,
            ]))
        torch.testing.assert_close(
            scales[8:], torch.full((8,), 9.0 / 15.0))

    def test_isolation_rejects_non_outlier_and_duplicate_group_declarations(self):
        maximum = torch.tensor([
            1.0, 2.0, 3.0, 12.0, 4.0, 5.0, 6.0, 7.0,
        ])

        with self.assertRaisesRegex(ValueError, "maximum channel"):
            isolated_channel_scales(
                maximum, bits=4, group_size=8, isolated_channels=(2,))
        with self.assertRaisesRegex(ValueError, "one isolated channel"):
            isolated_channel_scales(
                maximum, bits=4, group_size=8,
                isolated_channels=(3, 3))

    def test_candidates_reject_invalid_maxima_and_group_size(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            build_outlier_candidates(
                torch.tensor([1.0, float("nan")]), group_size=2)
        with self.assertRaisesRegex(ValueError, "divide"):
            build_outlier_candidates(torch.ones(7), group_size=8)
        with self.assertRaisesRegex(ValueError, "positive"):
            build_outlier_candidates(torch.ones(8), group_size=0)

    def test_harm_accumulator_counts_only_values_rescued_by_isolation(self):
        candidate = build_outlier_candidates(torch.tensor([
            1.0, 2.0, 3.0, 12.0, 4.0, 5.0, 6.0, 7.0,
        ]), group_size=8)[0]
        accumulator = OutlierHarmAccumulator((candidate,), channel_dim=1)
        values = torch.tensor([
            0.0, 0.20, 0.30, 10.0, 0.25, 0.50, 0.39, 0.10,
        ]).reshape(1, 8, 1, 1)

        accumulator.update(values)
        candidate_rows, victim_rows = accumulator.rows(
            "conv", "relu_output", "encoder")

        self.assertEqual(candidate_rows[0]["kind"], "relu_output")
        self.assertEqual(candidate_rows[0]["rescued_elements"], 3)
        self.assertAlmostEqual(
            candidate_rows[0]["rescued_energy"],
            0.30 ** 2 + 0.25 ** 2 + 0.39 ** 2, places=6)
        self.assertEqual(candidate_rows[0]["affected_victim_channels"], 3)
        self.assertAlmostEqual(candidate_rows[0]["harm_score"], 3.0)
        self.assertAlmostEqual(
            candidate_rows[0]["harm_probability_mean"], 3.0 / 7.0)
        rescued = {
            row["channel"]: row["rescued_elements"]
            for row in victim_rows
        }
        self.assertEqual(rescued, {2: 1, 4: 1, 6: 1})


if __name__ == "__main__":
    unittest.main()
