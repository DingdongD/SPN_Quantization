import unittest
from pathlib import Path


class StrictW4A4FP4ShellContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.script = (
            root / "scripts" / "run_strict_w4a4_fp4_evaluation.sh"
        ).read_text(encoding="utf-8")

    def test_uses_edge_backend_for_primary_and_stress(self):
        self.assertIn("scripts/run_nyu_edge_quantization.py", self.script)
        self.assertIn("--merge-policy independent", self.script)
        self.assertIn("--quant-backend fp4", self.script)
        self.assertIn("--quant-backend hardware", self.script)

    def test_full_protocol_uses_64_calibration_and_evaluation_samples(self):
        self.assertIn("CALIBRATION_SAMPLES=64", self.script)
        self.assertIn("EVALUATION_SAMPLES=64", self.script)
        self.assertIn("--bootstrap-resamples 10000", self.script)

    def test_requires_contracts_and_original_checkpoints(self):
        self.assertIn(
            '${model}/${method}_strict/strict_reconstruction_manifest.json',
            self.script)
        self.assertIn('[[ -f "$manifest" ]]', self.script)
        self.assertIn('--checkpoint best.pt', self.script)

    def test_cuda_extension_processes_use_visible_device_zero(self):
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPUS[$model]}"', self.script)
        self.assertIn('--device cuda:0', self.script)

    def test_rejects_existing_output_root(self):
        self.assertIn('[[ ! -e "$STRICT_W4A4_FP4_OUTPUT_ROOT" ]]', self.script)


if __name__ == "__main__":
    unittest.main()
