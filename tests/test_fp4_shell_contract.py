import unittest
from pathlib import Path


class FP4ShellContractTest(unittest.TestCase):
    def test_formal_runner_does_not_skip_conv_bn_fold(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "run_fp4_activation_validation.sh").read_text(
            encoding="utf-8")

        self.assertNotIn("--skip-conv-bn-fold", script)

    def test_formal_runner_passes_explicit_data_root(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "run_fp4_activation_validation.sh").read_text(
            encoding="utf-8")

        self.assertIn('--data-root "$SPN_DATA_ROOT"', script)


if __name__ == "__main__":
    unittest.main()
