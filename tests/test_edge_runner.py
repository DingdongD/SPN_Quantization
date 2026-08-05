import unittest

from scripts.run_nyu_edge_quantization import (
    normalize_merge_manifest,
    parse_edge_args,
)


class EdgeRunnerTest(unittest.TestCase):
    def test_default_policy_preserves_shared_scale_baseline(self):
        options, remaining = parse_edge_args(["--run-dir", "run"])
        self.assertEqual(options.merge_policy, "shared")
        self.assertEqual(remaining, ["--run-dir", "run"])

    def test_grouped_policy_requires_positive_group_size(self):
        with self.assertRaises(SystemExit):
            parse_edge_args(["--merge-policy", "grouped"])
        options, remaining = parse_edge_args([
            "--merge-policy", "grouped", "--merge-group-size", "16",
            "--quant-backend", "hardware",
        ])
        self.assertEqual(options.merge_group_size, 16)
        self.assertEqual(remaining, ["--quant-backend", "hardware"])

    def test_group_size_is_rejected_for_non_grouped_policy(self):
        with self.assertRaises(SystemExit):
            parse_edge_args([
                "--merge-policy", "independent", "--merge-group-size", "16"])

    def test_manifest_normalization_preserves_legacy_columns(self):
        row = normalize_merge_manifest([{
            "merge": "cat#0", "policy": "independent", "scales": "0.1;1.0"
        }])[0]
        self.assertEqual(row["scale"], "")
        self.assertEqual(row["scales"], "0.1;1.0")


if __name__ == "__main__":
    unittest.main()
