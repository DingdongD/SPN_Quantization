import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts import plot_nyu_random64_predictions as plotter


class Random64PredictionPlotTest(unittest.TestCase):
    def test_script_entrypoint_can_load_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [sys.executable, "scripts/plot_nyu_random64_predictions.py", "--help"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_validate_complete_samples_requires_same_four_models(self):
        by_sample = {
            index: dict((model, {"sample_index": index, "model": model})
                        for model in plotter.MODEL_ORDER)
            for index in range(64)
        }

        sample_ids = plotter.validate_complete_samples(by_sample, expected_samples=64)

        self.assertEqual(sample_ids, list(range(64)))

    def test_validate_complete_samples_rejects_missing_model(self):
        by_sample = {
            index: dict((model, {"sample_index": index, "model": model})
                        for model in plotter.MODEL_ORDER)
            for index in range(64)
        }
        del by_sample[17]["nlspn"]

        with self.assertRaisesRegex(ValueError, "sample 17"):
            plotter.validate_complete_samples(by_sample, expected_samples=64)

    def test_sample_block_positions_form_sixteen_by_four_layout(self):
        positions = [plotter.sample_block_position(i, columns=4) for i in range(64)]

        self.assertEqual(positions[0], (0, 0))
        self.assertEqual(positions[3], (0, 3))
        self.assertEqual(positions[4], (1, 0))
        self.assertEqual(positions[-1], (15, 3))

    def test_panel_order_starts_with_ground_truth(self):
        self.assertEqual(
            plotter.PANEL_ORDER,
            ["gt", "cspn", "dyspn", "nlspn", "completionformer"],
        )

    def test_colorbar_is_outside_the_panel_region(self):
        panel_right = plotter.PANEL_BOUNDS[1]
        colorbar_left = plotter.COLORBAR_RECT[0]

        self.assertGreater(colorbar_left, panel_right)

    def test_export_pdf_embeds_an_existing_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "sheet.png"
            pdf = Path(tmp) / "sheet.pdf"
            Image.new("RGB", (12, 8), color=(20, 40, 60)).save(str(png))

            result = plotter.export_pdf(png, pdf, dpi=100)

            self.assertEqual(result, pdf)
            self.assertGreater(pdf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
