import unittest

from scripts import run_w4a4_activation_histograms as runner


class W4A4ActivationHistogramRunnerTest(unittest.TestCase):
    def test_select_w4a4_config_requires_one_exact_configuration(self):
        config = {
            "name": "FP4V_W4A4",
            "w_bits": 4,
            "a_bits": 4,
            "activation_mode": "uniform",
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
            "name": "FP4V_W4A4",
            "w_bits": 4,
            "a_bits": 8,
            "activation_mode": "uniform",
            "groups": {"encoder"},
            "propagation": {"state_bits": 8},
        }

        with self.assertRaisesRegex(ValueError, "activation bits"):
            runner.select_w4a4_config([config])

    def test_strict_identity_uses_content_hashes_and_sample_indices(self):
        metadata = {
            "model": "nlspn",
            "seed": 7,
            "calibration_samples": 2,
            "calibration_indices": [4, 9],
            "quant_backend": "fp4",
            "configs": ["FP32", "FP4V_W4A4"],
            "model_provenance": {
                "model_class": "NLSPNModel",
                "model_module": "model.nlspnmodel",
                "source_sha256": "source-hash",
                "checkpoint_sha256": "checkpoint-hash",
            },
        }
        provenance = {
            "model_class": "NLSPNModel",
            "model_module": "model.nlspnmodel",
            "source_sha256": "source-hash",
            "checkpoint_sha256": "checkpoint-hash",
        }

        runner.validate_strict_identity(
            metadata, "nlspn", 7, 2, [4, 9], provenance)
        changed = dict(provenance, source_sha256="different")
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            runner.validate_strict_identity(
                metadata, "nlspn", 7, 2, [4, 9], changed)

    def test_semantic_rows_are_compared_with_typed_fields(self):
        expected = [{
            "model": "cspn", "role": "sparse_depth_input",
            "module": "conv1_1", "kind": "input",
            "bits": "8", "format": "uniform",
        }]
        actual = [{
            "model": "cspn", "role": "sparse_depth_input",
            "module": "conv1_1", "kind": "input",
            "bits": 8, "format": "uniform",
        }]

        runner.validate_semantic_rows(expected, actual)
        actual[0]["bits"] = 4
        with self.assertRaisesRegex(ValueError, "semantic A8"):
            runner.validate_semantic_rows(expected, actual)

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


if __name__ == "__main__":
    unittest.main()
