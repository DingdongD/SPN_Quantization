import unittest

import torch
import torch.nn as nn

from scripts import run_nyu_cspn_stratified_calibration as runner


class RunnerContractTest(unittest.TestCase):
    def test_cli_requires_exact_declared_selection_contract(self):
        args = runner.parse_args([
            "--run-dir", "run", "--checkpoint", "best.pt",
            "--data-root", "data", "--out-dir", "output",
            "--device", "cuda:0", "--baseline-seed", "20260812",
            "--audit-seed", "20260813", "--calibration-samples", "128",
            "--audit-samples", "512", "--candidate-samples", "1024",
            "--candidate-tail-samples", "256",
            "--final-tail-samples", "32",
            "--random-baseline-seeds",
            "31", "37", "41", "43", "47", "53", "59", "61",
            "67", "71", "73", "79", "83", "89", "97", "101",
            "--parity-samples", "8", "--workers", "4",
        ])

        self.assertEqual(args.calibration_samples, 128)
        self.assertEqual(args.audit_samples, 512)
        self.assertEqual(args.candidate_samples, 1024)
        self.assertEqual(args.candidate_tail_samples, 256)
        self.assertEqual(args.final_tail_samples, 32)
        self.assertEqual(len(args.random_baseline_seeds), 16)
        self.assertFalse(hasattr(args, "evaluation_list"))

    def test_artifact_contract_contains_no_visual_outputs(self):
        names = runner.artifact_filenames()
        self.assertEqual(len(names), 11)
        self.assertIn("calibration_indices.json", names)
        self.assertIn("coverage_report.md", names)
        self.assertFalse(any(
            name.endswith((".png", ".pdf")) for name in names))

    def test_activation_owner_contract_is_fixed_and_unique(self):
        identities = tuple(owner.identity for owner in runner.ACTIVATION_OWNERS)
        self.assertEqual(len(identities), 6)
        self.assertEqual(len(set(identities)), 6)
        self.assertIn("encoder_stem_relu", identities)
        self.assertIn("decoder_layer4_fusion", identities)
        self.assertIn("decoder_layer4_relu", identities)


class ActivationDescriptorCollectorTest(unittest.TestCase):
    class ReusedReluModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(2, 2, 1, bias=False)
            self.relu = nn.ReLU()
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))

        def forward(self, value):
            first = self.relu(self.proj(value))
            return self.relu(first + 1.0)

    def test_collector_captures_declared_relu_call_once_per_forward(self):
        model = self.ReusedReluModel().eval()
        owners = (
            runner.ActivationOwner(
                "projection", "proj", 0, "output"),
            runner.ActivationOwner(
                "second_relu", "relu", 1, "relu_output"),
        )
        collector = runner.ActivationDescriptorCollector(model, owners)
        sample = torch.tensor([[[[-2.0, 2.0]], [[4.0, 8.0]]]])

        row = collector.capture(7, (sample,))

        self.assertEqual(row["sample_index"], 7)
        self.assertAlmostEqual(row["projection_max"], 8.0)
        self.assertAlmostEqual(row["second_relu_max"], 9.0)
        self.assertGreater(row["projection_channel_imbalance"], 1.0)
        self.assertGreaterEqual(row["second_relu_p99_over_max"], 0.0)
        self.assertLessEqual(row["second_relu_p99_over_max"], 1.0)
        collector.close()

    def test_collector_resets_call_counts_between_forwards(self):
        model = self.ReusedReluModel().eval()
        owners = (
            runner.ActivationOwner(
                "first_relu", "relu", 0, "relu_output"),
            runner.ActivationOwner(
                "second_relu", "relu", 1, "relu_output"),
        )
        collector = runner.ActivationDescriptorCollector(model, owners)
        sample = torch.ones(1, 2, 1, 1)

        first = collector.capture(1, (sample,))
        second = collector.capture(2, (sample * 2.0,))

        self.assertEqual(first["first_relu_max"], 1.0)
        self.assertEqual(second["first_relu_max"], 2.0)
        collector.close()

    def test_collector_rejects_missing_declared_call(self):
        model = self.ReusedReluModel().eval()
        owners = (
            runner.ActivationOwner(
                "third_relu", "relu", 2, "relu_output"),
        )
        collector = runner.ActivationDescriptorCollector(model, owners)

        with self.assertRaisesRegex(RuntimeError, "missing"):
            collector.capture(1, (torch.ones(1, 2, 1, 1),))
        collector.close()


if __name__ == "__main__":
    unittest.main()
