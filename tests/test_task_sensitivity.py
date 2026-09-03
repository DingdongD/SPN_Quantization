import math
import unittest

import torch

from spn_quant.task_sensitivity import (
    gradient_weighted_error,
    marginal_score_per_saved_bit,
    normalized_gradient_weighted_error,
    propagation_metric_row,
)


class TaskSensitivityFormulaTest(unittest.TestCase):
    def test_gradient_weighted_error_uses_elementwise_absolute_product(self):
        gradient = torch.tensor([[-2.0, 3.0]])
        reference = torch.tensor([[4.0, -5.0]])
        quantized = torch.tensor([[3.0, -4.0]])

        self.assertAlmostEqual(
            gradient_weighted_error(gradient, reference, quantized), 5.0)

    def test_normalized_score_uses_reference_l1(self):
        gradient = torch.tensor([2.0, -4.0])
        reference = torch.tensor([2.0, 4.0])
        quantized = torch.tensor([1.0, 5.0])

        self.assertAlmostEqual(
            normalized_gradient_weighted_error(
                gradient, reference, quantized), 6.0 / 6.0)

    def test_marginal_score_rejects_nonpositive_memory_saving(self):
        with self.assertRaisesRegex(ValueError, "memory saving"):
            marginal_score_per_saved_bit(1.0, 2.0, 0.0)

    def test_marginal_score_is_lower_for_smaller_degradation(self):
        self.assertAlmostEqual(
            marginal_score_per_saved_bit(1.0, 1.5, 2.0), 0.25)

    def test_propagation_metric_row_contains_required_numeric_metrics(self):
        row = propagation_metric_row(
            "cspn", "sample", "affinity", 3,
            torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))

        self.assertEqual(row["model"], "cspn")
        self.assertEqual(row["signal"], "affinity")
        self.assertEqual(row["iteration"], 3)
        self.assertAlmostEqual(row["mse"], 0.5)
        self.assertTrue(math.isfinite(row["sqnr_db"]))


if __name__ == "__main__":
    unittest.main()
