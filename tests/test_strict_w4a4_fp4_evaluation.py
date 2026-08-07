import unittest

from scripts.strict_w4a4_fp4_evaluation import (
    METHOD_ORDER,
    PRIMARY_CONFIGS,
    STRESS_CONFIGS,
    performance_decision,
)


class StrictW4A4FP4ContractTest(unittest.TestCase):
    def test_matrix_is_fixed_and_keeps_stress_separate(self):
        self.assertEqual(METHOD_ORDER, ("rtn", "adaround", "brecq"))
        self.assertEqual(
            PRIMARY_CONFIGS,
            ("FP32", "FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8"))
        self.assertEqual(STRESS_CONFIGS, ("FP32", "HW_W4A4_full"))

    def test_performance_decision_requires_all_three_conditions(self):
        accepted = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.08, rtn_rmse=1.09,
            nonfinite_samples=0, nonfinite_pixels=0)
        self.assertEqual(accepted["status"], "preserved")

        degraded = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.11, rtn_rmse=1.12,
            nonfinite_samples=0, nonfinite_pixels=0)
        self.assertEqual(
            degraded["status"], "rejected_fp32_degradation")

        regression = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.08, rtn_rmse=1.07,
            nonfinite_samples=0, nonfinite_pixels=0)
        self.assertEqual(regression["status"], "rejected_rtn_regression")

        invalid = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.02, rtn_rmse=1.03,
            nonfinite_samples=1, nonfinite_pixels=10)
        self.assertEqual(invalid["status"], "rejected_nonfinite")

    def test_performance_decision_rejects_invalid_rmse_values(self):
        with self.assertRaisesRegex(ValueError, "finite positive RMSE"):
            performance_decision(
                fp32_rmse=0.0, quant_rmse=1.0, rtn_rmse=1.0,
                nonfinite_samples=0, nonfinite_pixels=0)


if __name__ == "__main__":
    unittest.main()
