from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import plot_cspn_activation_resolution_predictions as plotter


class CSPNActivationPredictionPlotTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=".")
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write_payload(self, config, sample_index, error):
        directory = self.root / "predictions" / config
        directory.mkdir(parents=True, exist_ok=True)
        gt = np.ones((4, 6), dtype=np.float32) * 2.0
        pred = gt + np.float32(error)
        np.savez_compressed(
            directory / ("sample_%05d.npz" % sample_index),
            gt=gt,
            fp32=gt,
            pred=pred,
            abs_err=np.abs(pred - gt),
            valid_gt=np.ones_like(gt, dtype=bool),
            nonfinite=np.zeros_like(gt, dtype=bool),
            sample_index=np.int64(sample_index),
            model=np.array("cspn"),
            config=np.array(config),
            sparse=np.zeros_like(gt),
            rgb=np.ones((4, 6, 3), dtype=np.float32) * 0.5,
        )

    def populate(self):
        errors = {
            "FP32": (0.0, 0.0, 0.0, 0.0),
            "W4A4_RTN": (0.8, 0.2, 0.6, 0.4),
            "W4A4_CHANNEL": (0.1, 0.3, 0.5, 0.7),
            "W4A4_CALIBRATED_SCALE": (0.2, 0.4, 0.3, 0.6),
        }
        for config in plotter.CONFIG_ORDER:
            for sample_index, error in enumerate(errors[config]):
                self.write_payload(config, sample_index, error)

    def test_load_predictions_requires_exact_four_configs(self):
        self.populate()

        predictions = plotter.load_predictions(
            self.root, expected_samples=4)

        self.assertEqual(sorted(predictions), [0, 1, 2, 3])
        self.assertEqual(set(predictions[0]), set(plotter.CONFIG_ORDER))

    def test_load_predictions_rejects_missing_config(self):
        self.populate()
        missing = self.root / "predictions" / "W4A4_CHANNEL"
        for path in missing.glob("*.npz"):
            path.unlink()
        missing.rmdir()

        with self.assertRaisesRegex(ValueError, "configuration directories"):
            plotter.load_predictions(self.root, expected_samples=4)

    def test_representative_samples_are_distinct_and_deterministic(self):
        self.populate()
        predictions = plotter.load_predictions(
            self.root, expected_samples=4)

        selected = plotter.select_representative_samples(predictions)

        self.assertEqual([row[1] for row in selected], [0, 2, 3, 1])
        self.assertEqual([row[0] for row in selected], [
            "RTN worst", "Largest channel gain", "RTN median",
            "Channel worst",
        ])

    def test_renderers_create_nonempty_png_and_pdf_files(self):
        self.populate()
        predictions = plotter.load_predictions(
            self.root, expected_samples=4)
        output = self.root / "figures"
        output.mkdir()

        detail = plotter.render_detail(
            predictions, output / "detail.png", dpi=40)
        contact = plotter.render_contact_sheet(
            predictions, output / "contact.png", expected_samples=4,
            sample_columns=2, dpi=40)
        detail_pdf = plotter.export_pdf(
            detail, output / "detail.pdf", dpi=40)
        contact_pdf = plotter.export_pdf(
            contact, output / "contact.pdf", dpi=40)

        for path in (detail, contact, detail_pdf, contact_pdf):
            self.assertGreater(path.stat().st_size, 0)
        with Image.open(detail) as image:
            self.assertGreater(image.width, image.height)
        with Image.open(contact) as image:
            self.assertGreater(image.width, image.height)


if __name__ == "__main__":
    unittest.main()
