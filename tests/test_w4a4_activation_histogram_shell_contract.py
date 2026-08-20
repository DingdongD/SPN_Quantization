import unittest
from pathlib import Path


class W4A4ActivationHistogramShellContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.script = (
            root / "scripts" / "run_w4a4_activation_histograms.sh"
        ).read_text(encoding="utf-8")

    def test_full_protocol_profiles_fixed_64_samples(self):
        self.assertIn("CALIBRATION_SAMPLES=64", self.script)
        self.assertIn("PROFILE_SAMPLES=64", self.script)

    def test_smoke_protocol_keeps_strict_calibration_and_profiles_one(self):
        self.assertIn("PROFILE_SAMPLES=1", self.script)
        self.assertIn('--profile-samples "$PROFILE_SAMPLES"', self.script)

    def test_requires_all_paths_environments_and_devices(self):
        for variable in (
                "SPN_DATA_ROOT", "SPN_EXTERNAL_ROOT",
                "COMPLETIONFORMER_ROOT",
                "W4A4_HISTOGRAM_OUTPUT_ROOT", "CSPN_PYTHON",
                "DYSPN_PYTHON", "NLSPN_PYTHON",
                "COMPLETIONFORMER_PYTHON", "CSPN_GPU", "DYSPN_GPU",
                "NLSPN_GPU", "COMPLETIONFORMER_GPU"):
            self.assertIn(': "${%s:?}"' % variable, self.script)
        self.assertNotIn("STRICT_W4A4_FP4_ROOT", self.script)
        self.assertNotIn("--strict-root", self.script)

    def test_uses_original_checkpoints_and_visible_device_zero(self):
        self.assertIn('--checkpoint "$run_dir/best.pt"', self.script)
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPUS[$model]}"', self.script)
        self.assertIn('--device cuda:0', self.script)

    def test_rejects_existing_output_and_writes_per_model_logs(self):
        self.assertIn('[[ ! -e "$W4A4_HISTOGRAM_OUTPUT_ROOT" ]]', self.script)
        self.assertIn('"$W4A4_HISTOGRAM_OUTPUT_ROOT/logs/${model}.log"',
                      self.script)

    def test_runs_model_plots_and_root_comparison(self):
        self.assertIn("scripts/plot_w4a4_activation_histograms.py", self.script)
        self.assertIn("--models cspn dyspn nlspn completionformer", self.script)


if __name__ == "__main__":
    unittest.main()
