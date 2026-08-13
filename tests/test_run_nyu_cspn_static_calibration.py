import unittest

import torch

from scripts import run_nyu_cspn_static_calibration as runner


class StaticCalibrationConfigurationTest(unittest.TestCase):
    def test_method_matrix_is_exact_and_static_group8(self):
        methods = runner.build_methods()

        self.assertEqual(
            tuple(method["config"] for method in methods),
            runner.EXPECTED_CONFIGURATIONS)
        self.assertEqual(
            tuple(method["calibration"] for method in methods),
            ("minmax", "percentile_p999", "percentile_p9999", "hist_mse"))
        self.assertTrue(all(method["group_size"] == 8 for method in methods))
        self.assertTrue(all(method["dynamic"] is False for method in methods))

    def test_runtime_contract_requires_71_sites_and_1673_scales(self):
        runner.validate_activation_contract(71, 1673)

        with self.assertRaisesRegex(ValueError, "site"):
            runner.validate_activation_contract(70, 1673)
        with self.assertRaisesRegex(ValueError, "scale"):
            runner.validate_activation_contract(71, 1672)

    def test_thresholds_split_into_exact_ordinary_and_rotation_keys(self):
        thresholds = {
            ("conv", "input"): torch.tensor([1.0]),
            ("relu#0", "relu_output"): torch.tensor([2.0]),
            ("rotation.decoder_entry", "boundary"): torch.tensor([3.0]),
            ("rotation.layer4_signed_skip", "boundary"): torch.tensor([4.0]),
        }
        ordinary_keys = {
            ("conv", "input"): ("conv", "input"),
            ("relu#0", "relu_output"): "relu#0",
        }

        ordinary, rotation = runner.split_thresholds(
            thresholds, ordinary_keys,
            ("decoder_entry", "layer4_signed_skip"))

        self.assertEqual(set(ordinary), {("conv", "input"), "relu#0"})
        self.assertEqual(set(rotation), {
            "decoder_entry", "layer4_signed_skip"})


if __name__ == "__main__":
    unittest.main()
