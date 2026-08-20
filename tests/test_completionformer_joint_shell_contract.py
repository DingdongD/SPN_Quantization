import unittest
from pathlib import Path


class CompletionFormerJointShellContractTest(unittest.TestCase):
    def test_reference_metrics_are_explicitly_required(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" /
                  "run_completionformer_joint_quantization.sh").read_text(
                      encoding="utf-8")

        self.assertIn(
            'REFERENCE_METRICS="${COMPLETIONFORMER_REFERENCE_METRICS:?COMPLETIONFORMER_REFERENCE_METRICS is required}"',
            script)


if __name__ == "__main__":
    unittest.main()
