import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import export_nyu_predictions as predictions


class ModelLoadingTest(unittest.TestCase):
    def test_exact_checkpoint_state_is_accepted(self):
        model = torch.nn.Linear(2, 1)

        report = predictions.load_model_state(
            model, dict(model.state_dict()), "dyspn")

        self.assertEqual(report, {
            "ignored_checkpoint_keys": [],
            "missing_keys": [],
            "unexpected_keys": [],
        })

    def test_unexpected_checkpoint_key_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        state["unexpected"] = torch.ones(1)

        with self.assertRaisesRegex(
                RuntimeError, "unexpected checkpoint keys.*unexpected"):
            predictions.load_model_state(model, state, "dyspn")

    def test_missing_checkpoint_key_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        del state["bias"]

        with self.assertRaisesRegex(RuntimeError, "missing checkpoint keys.*bias"):
            predictions.load_model_state(model, state, "nlspn")

    def test_cspn_dynamic_fixed_sum_kernel_is_the_only_ignored_key(self):
        model = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        state["post_process_layer.sum_conv.weight"] = torch.ones(
            1, 8, 1, 1, 1)

        report = predictions.load_model_state(model, state, "cspn")

        self.assertEqual(report["ignored_checkpoint_keys"], [
            "post_process_layer.sum_conv.weight",
        ])

    def test_cspn_rejects_invalid_dynamic_sum_kernel(self):
        model = torch.nn.Linear(2, 1)
        state = dict(model.state_dict())
        state["post_process_layer.sum_conv.weight"] = torch.zeros(
            1, 8, 1, 1, 1)

        with self.assertRaisesRegex(RuntimeError, "invalid CSPN fixed sum kernel"):
            predictions.load_model_state(model, state, "cspn")


class ModelProvenanceTest(unittest.TestCase):
    def test_file_sha256_matches_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"official-checkpoint")

            digest = predictions.file_sha256(path)

        self.assertEqual(
            digest, hashlib.sha256(b"official-checkpoint").hexdigest())

    def test_source_path_must_be_inside_expected_official_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "external" / "DySPN"
            source = root / "DySPN" / "base.py"
            source.parent.mkdir(parents=True)
            source.write_text("class Model: pass\n", encoding="utf-8")
            outside = Path(tmp) / "replacement.py"
            outside.write_text("class Model: pass\n", encoding="utf-8")

            accepted = predictions.validate_model_source(
                "dyspn", source, source_roots={"dyspn": root})
            with self.assertRaisesRegex(RuntimeError, "outside official source root"):
                predictions.validate_model_source(
                    "dyspn", outside, source_roots={"dyspn": root})

        self.assertEqual(accepted, source.resolve())


if __name__ == "__main__":
    unittest.main()
