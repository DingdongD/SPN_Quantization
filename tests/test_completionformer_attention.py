import unittest

import torch

from spn_quant.completionformer_attention import IntegerAttentionController


def reference_context(q, k, v, head_scale):
    score = torch.matmul(q, k.transpose(-2, -1)) * head_scale
    return torch.matmul(torch.softmax(score, dim=-1), v)


def make_controller(clip_factors=(1.0,)):
    return IntegerAttentionController(
        name="backbone.former.block1.0.attn",
        num_heads=2,
        head_dim=4,
        head_scale=0.5,
        qkv_bits=4,
        probability_bits=8,
        clip_factors=clip_factors,
        search_rounds=1,
        cache_sample_limit=2,
        cache_byte_limit=1 << 20)


class IntegerAttentionCalibrationTest(unittest.TestCase):
    def test_freeze_keeps_q_k_v_scales_independent_by_head(self):
        controller = make_controller()
        q = torch.ones(1, 2, 3, 4)
        q[:, 1] *= 3.0
        k = torch.ones(1, 2, 5, 4) * 2.0
        k[:, 1] *= 2.0
        v = torch.ones(1, 2, 5, 4) * 16.0
        v[:, 1] *= 2.0
        target = reference_context(q, k, v, 0.5)

        controller.observe(q, k, v, target)
        controller.freeze()

        self.assertEqual(controller.scales["q"].shape, (2,))
        self.assertEqual(controller.scales["k"].shape, (2,))
        self.assertEqual(controller.scales["v"].shape, (2,))
        self.assertFalse(torch.equal(
            controller.scales["k"], controller.scales["v"]))
        self.assertFalse(torch.equal(
            controller.scales["q"], controller.scales["k"]))

    def test_observe_rejects_wrong_head_shape(self):
        controller = make_controller()
        q = torch.ones(1, 1, 3, 4)
        k = torch.ones(1, 1, 5, 4)
        v = torch.ones(1, 1, 5, 4)
        target = torch.ones(1, 1, 3, 4)

        with self.assertRaises(ValueError):
            controller.observe(q, k, v, target)

    def test_freeze_requires_observations(self):
        with self.assertRaises(RuntimeError):
            make_controller().freeze()


class IntegerAttentionExecutionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.q = torch.randn(1, 2, 3, 4)
        self.k = torch.randn(1, 2, 5, 4)
        self.v = torch.randn(1, 2, 5, 4)
        self.target = reference_context(self.q, self.k, self.v, 0.5)
        self.controller = make_controller()
        self.controller.observe(self.q, self.k, self.v, self.target)
        self.controller.freeze()
        self.controller.enable()

    def test_qk_and_av_accumulators_match_explicit_integer_sums(self):
        result = self.controller.execute(self.q, self.k, self.v)
        expected_score = torch.stack([
            q_codes.to(torch.int32) @
            k_codes.to(torch.int32).transpose(-2, -1)
            for q_codes, k_codes in zip(
                result.q_codes[0], result.k_codes[0])
        ]).unsqueeze(0)
        expected_context = torch.stack([
            probability.to(torch.int32) @ value.to(torch.int32)
            for probability, value in zip(
                result.probability_codes[0], result.v_codes[0])
        ]).unsqueeze(0)

        torch.testing.assert_close(result.score_accumulator, expected_score)
        torch.testing.assert_close(result.context_accumulator, expected_context)

    def test_probability_uses_full_unsigned_a8_domain(self):
        result = self.controller.execute(self.q, self.k, self.v)

        self.assertEqual(result.probability_codes.dtype, torch.uint8)
        self.assertGreaterEqual(int(result.probability_codes.min()), 0)
        self.assertLessEqual(int(result.probability_codes.max()), 255)
        self.assertAlmostEqual(result.probability_scale, 1.0 / 255.0)

    def test_context_shape_and_metrics_are_finite(self):
        output = self.controller.quantize(self.q, self.k, self.v)
        rows = self.controller.statistics()

        self.assertEqual(output.shape, self.target.shape)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["module"], self.controller.name)
        self.assertEqual(rows[0]["updates"], 1)
        for key in ("q_sqnr_db", "k_sqnr_db", "v_sqnr_db",
                    "score_sqnr_db", "context_mse", "probability_kl"):
            self.assertTrue(torch.isfinite(torch.tensor(rows[0][key])))

    def test_manifest_has_one_row_per_head(self):
        rows = self.controller.manifest()

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["head"] for row in rows], [0, 1])
        self.assertTrue(all(row["qkv_bits"] == 4 for row in rows))
        self.assertTrue(all(row["probability_bits"] == 8 for row in rows))

    def test_disabled_controller_rejects_quantization(self):
        self.controller.disable()

        with self.assertRaises(RuntimeError):
            self.controller.quantize(self.q, self.k, self.v)


class IntegerAttentionSearchTest(unittest.TestCase):
    def test_search_rows_cover_every_role_and_factor(self):
        controller = make_controller(clip_factors=(1.0, 0.75))
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        controller.observe(q, k, v, reference_context(q, k, v, 0.5))

        controller.freeze()
        rows = controller.search_rows()

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["parameter"] for row in rows}, {"q", "k", "v"})
        self.assertTrue(all(
            row["module"] == controller.name for row in rows))


if __name__ == "__main__":
    unittest.main()
