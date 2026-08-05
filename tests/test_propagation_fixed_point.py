import unittest

import torch


class PropagationFixedPointTest(unittest.TestCase):
    def test_signed_code_normalization_derives_exact_center_residual(self):
        from spn_quant.propagation.fixed_point import (
            Q13_ONE,
            normalize_signed_codes_q13,
        )

        codes = torch.tensor([[[[-7]], [[0]], [[0]]]], dtype=torch.int8)
        neighbor, center, normalized_codes = normalize_signed_codes_q13(
            codes, scale=1.0 / 7.0, denominator_floor=False, eps=0.0)

        self.assertEqual(normalized_codes.dtype, torch.int16)
        self.assertEqual(int(center.item()), 2 * Q13_ONE)
        self.assertEqual(
            int(center.item() + normalized_codes.sum().item()), Q13_ONE)
        torch.testing.assert_close(
            neighbor, normalized_codes.float() / float(Q13_ONE))

    def test_denominator_floor_is_applied_in_integer_code_domain(self):
        from spn_quant.propagation.fixed_point import (
            Q13_ONE,
            normalize_signed_codes_q13,
        )

        codes = torch.tensor([[[[1]], [[-1]]]], dtype=torch.int8)
        _, center, normalized_codes = normalize_signed_codes_q13(
            codes, scale=0.1, denominator_floor=True, eps=0.0)

        self.assertEqual(normalized_codes.flatten().tolist(), [819, -819])
        self.assertEqual(int(center.item()), Q13_ONE)
        self.assertEqual(
            int(center.item() + normalized_codes.sum().item()), Q13_ONE)

    def test_softmax_q13_is_nonnegative_and_has_exact_sum(self):
        from spn_quant.propagation.fixed_point import Q13_ONE, softmax_codes_q13

        codes = torch.tensor([[[[7]], [[2]], [[-7]]]], dtype=torch.int8)
        values, normalized_codes = softmax_codes_q13(
            codes, scale=3.0 / 7.0, dim=1)

        self.assertTrue(bool(torch.all(normalized_codes >= 0)))
        self.assertEqual(int(normalized_codes.sum(dim=1).item()), Q13_ONE)
        torch.testing.assert_close(
            values.sum(dim=1), torch.ones(1, 1, 1), atol=0.0, rtol=0.0)

    def test_unsigned_unit_a8_preserves_endpoints(self):
        from spn_quant.propagation.fixed_point import unsigned_unit_qdq

        values, codes = unsigned_unit_qdq(
            torch.tensor([0.0, 0.5, 1.0]), bits=8)

        self.assertEqual(codes.tolist(), [0, 128, 255])
        self.assertEqual(float(values[0]), 0.0)
        self.assertEqual(float(values[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
