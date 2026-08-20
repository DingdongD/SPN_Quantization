import unittest
import inspect

from scripts import run_w4a4_activation_histograms as runner


class W4A4ActivationHistogramRunnerTest(unittest.TestCase):
    def test_select_w4a4_config_requires_one_exact_configuration(self):
        config = {
            "name": "PA_W4A4_PROP_A8",
            "w_bits": 4,
            "a_bits": 4,
            "groups": {"encoder"},
            "propagation": {"state_bits": 8},
        }

        self.assertIs(runner.select_w4a4_config([config]), config)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            runner.select_w4a4_config([])
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            runner.select_w4a4_config([config, dict(config)])

    def test_select_w4a4_config_rejects_policy_mismatch(self):
        config = {
            "name": "PA_W4A4_PROP_A8",
            "w_bits": 4,
            "a_bits": 8,
            "groups": {"encoder"},
            "propagation": {"state_bits": 8},
        }

        with self.assertRaisesRegex(ValueError, "activation bits"):
            runner.select_w4a4_config([config])

    def test_runner_has_no_fp4_or_strict_artifact_dependency(self):
        source = inspect.getsource(runner)
        self.assertNotIn("fp4_activation_validation", source)
        self.assertNotIn("fp4_quantization", source)
        self.assertNotIn("--strict-root", source)

    def test_expected_sites_expand_manifest_using_observed_call_indices(self):
        manifest = [
            {"module": "shared", "kind": "input"},
            {"module": "shared", "kind": "output"},
        ]
        site_metadata = {
            "shared#0:input": {
                "module": "shared", "kind": "input",
                "synthetic_slice": 0,
            },
            "shared#1:input": {
                "module": "shared", "kind": "input",
                "synthetic_slice": 0,
            },
            "shared#0:output": {
                "module": "shared", "kind": "output",
                "synthetic_slice": 0,
            },
            "input_rgb#0:input": {
                "module": "input_rgb", "kind": "input",
                "synthetic_slice": 1,
            },
        }

        self.assertEqual(runner.expected_site_names(
            manifest, site_metadata), {
                "shared#0:input", "shared#1:input", "shared#0:output",
            })
        with self.assertRaisesRegex(ValueError, "manifest site mismatch"):
            runner.expected_site_names(manifest[:1], site_metadata)

    def test_profile_indices_are_an_explicit_prefix_of_calibration_set(self):
        self.assertEqual(runner.profile_indices([4, 9, 2], 2), [4, 9])
        with self.assertRaisesRegex(ValueError, "profile sample count"):
            runner.profile_indices([4, 9, 2], 0)
        with self.assertRaisesRegex(ValueError, "profile sample count"):
            runner.profile_indices([4, 9, 2], 4)


if __name__ == "__main__":
    unittest.main()
