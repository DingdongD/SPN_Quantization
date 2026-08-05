import unittest

from scripts.run_nyu_edge_quantization import parse_edge_args


class SemanticEdgeRunnerTest(unittest.TestCase):
    def test_parser_accepts_non_strict_adapter_bringup(self):
        options, remaining = parse_edge_args([
            "--merge-policy", "independent",
            "--no-strict-semantic-sites", "--run-dir", "example",
        ])
        self.assertEqual(options.merge_policy, "independent")
        self.assertTrue(options.no_strict_semantic_sites)
        self.assertEqual(remaining, ["--run-dir", "example"])

    def test_grouped_policy_requires_positive_group_size(self):
        with self.assertRaises(SystemExit):
            parse_edge_args(["--merge-policy", "grouped"])
        with self.assertRaises(SystemExit):
            parse_edge_args([
                "--merge-policy", "grouped", "--merge-group-size", "0",
            ])


if __name__ == "__main__":
    unittest.main()
