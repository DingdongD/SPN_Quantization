import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts.run_nyu_edge_quantization import (
    load_reconstruction_manifest,
    parse_edge_args,
)


class SemanticEdgeRunnerTest(unittest.TestCase):
    def test_direct_cli_bootstraps_repository_root(self):
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        if "PYTHONPATH" in environment:
            del environment["PYTHONPATH"]
        result = subprocess.run(
            [sys.executable, "scripts/run_nyu_edge_quantization.py", "--help"],
            cwd=str(repository), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stderr)

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

    def test_reconstruction_manifest_rejects_removed_semantic_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({
                "method": "brecq",
                "weight_bits": 4,
                "activation_bits": 4,
                "targets": ["backbone.dep_dec0"],
                "activation_manifest": [{
                    "site": "backbone.dep_dec0.0",
                    "maximum": 3.25,
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "strict reconstruction manifest required"):
                load_reconstruction_manifest(str(path))

    def test_parser_rejects_two_contract_sources(self):
        with self.assertRaises(SystemExit):
            parse_edge_args([
                "--reconstruction-manifest", "manifest.json",
                "--deployment-contract", "contract.pt",
            ])

    def test_strict_manifest_loads_deployment_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.pt"
            torch.save({
                "format_version": 1,
                "strict": 1,
                "method": "adaround_strict",
                "targets": ["conv"],
                "weight_contracts": {
                    "conv": {"bits": 4},
                },
            }, contract_path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "strict": 1,
                "method": "adaround_strict",
                "weight_bits": 4,
                "activation_bits": 0,
                "targets": ["conv"],
                "activation_manifest": [],
                "deployment_contract": str(contract_path),
            }), encoding="utf-8")

            payload = load_reconstruction_manifest(
                str(manifest_path))

        self.assertEqual(payload["method"], "adaround_strict")
        self.assertIsNotNone(payload["strict_contract"])
        self.assertEqual(payload["strict"], 1)
        self.assertEqual(
            payload["activation_policy"], "evaluation_backend_owned")


if __name__ == "__main__":
    unittest.main()
