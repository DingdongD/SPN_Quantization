from pathlib import Path
import tempfile
import unittest

from scripts import run_nyu_cspn_dynamic_group_a4 as runner


class DynamicGroupConfigurationTest(unittest.TestCase):
    def test_evaluation_matrix_is_fixed(self):
        configs = runner.build_configurations()

        self.assertEqual(
            tuple(config["name"] for config in configs),
            runner.EXPECTED_CONFIGURATIONS)
        self.assertEqual(len(configs), 6)

    def test_static_dynamic_pairs_differ_only_in_range_source(self):
        configs = dict(
            (config["name"], config)
            for config in runner.build_configurations())
        pairs = (
            ("A4_ONLY_G8_STATIC", "A4_ONLY_G8_DYNAMIC"),
            ("W4A4_G8_STATIC", "W4A4_G8_DYNAMIC"),
        )

        for static_name, dynamic_name in pairs:
            static = dict(configs[static_name])
            dynamic = dict(configs[dynamic_name])
            self.assertFalse(static.pop("dynamic"))
            self.assertTrue(dynamic.pop("dynamic"))
            static["name"] = "paired"
            dynamic["name"] = "paired"
            self.assertEqual(static, dynamic)

    def test_quantized_configs_keep_group8_and_propagation_contract(self):
        for config in runner.build_configurations():
            self.assertNotIn("propagation_head", config["weight_groups"])
            self.assertNotIn("propagation_head", config["activation_groups"])
            if config["activation_groups"]:
                self.assertEqual(config["group_size"], 8)
                self.assertEqual(
                    config["propagation"], runner.PROPAGATION_A8_Q13)

    def test_runtime_contract_requires_exact_sample_counts(self):
        runner.validate_runtime_contract(128, list(range(64)))

        with self.assertRaisesRegex(ValueError, "128"):
            runner.validate_runtime_contract(64, list(range(64)))
        with self.assertRaisesRegex(ValueError, "64"):
            runner.validate_runtime_contract(128, list(range(63)))


class PredictionCoverageTest(unittest.TestCase):
    def test_prediction_coverage_requires_every_exported_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "predictions"
            indices = (3, 7)
            for config in runner.PREDICTION_CONFIGURATIONS:
                output = root / config
                output.mkdir(parents=True)
                for index in indices:
                    (output / ("sample_%05d.npz" % index)).touch()

            runner.validate_prediction_coverage(
                Path(directory), indices)

            missing = root / runner.PREDICTION_CONFIGURATIONS[0] / \
                "sample_00003.npz"
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "coverage"):
                runner.validate_prediction_coverage(
                    Path(directory), indices)


if __name__ == "__main__":
    unittest.main()
