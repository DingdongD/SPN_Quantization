import math
from pathlib import Path
import tempfile
import unittest

import fitz
import numpy as np
from PIL import Image
import torch

from scripts import activation_histograms as histograms
from scripts import hardware_aligned_quantization as haq
from scripts import plot_w4a4_activation_histograms as plotting


class W4A4ActivationHistogramPlotTest(unittest.TestCase):
    def _record(self, recorder, module, channels, maximum):
        values = torch.linspace(
            0.0, maximum * 1.1, channels * 32,
            dtype=torch.float32).reshape(1, channels, 4, 8)
        quantizer = haq.UnsignedActivationQuantizer(4, maximum)
        quantized, codes = quantizer.quantize_with_codes(values)
        recorder.record(
            module, "input", 0, "encoder", values,
            quantized, codes, quantizer, channel_dim=1)
        return values, quantizer

    def _write_profile(self, root, model_name):
        recorder = histograms.ActivationHistogramRecorder(
            model_name, phase="range", capacity=1024, per_update=1024)
        records = []
        if model_name == "cspn":
            records.append(("conv1_1",) +
                           self._record(recorder, "conv1_1", 4, 4.0))
        elif model_name == "dyspn":
            records.append(("base.conv1_rgb.0",) + self._record(
                recorder, "base.conv1_rgb.0", 3, 1.0))
            records.append(("base.conv1_dep.0",) + self._record(
                recorder, "base.conv1_dep.0", 1, 8.0))
        else:
            raise ValueError("unsupported test model")
        recorder.freeze_ranges(bin_count=32)
        recorder.begin_histogram_pass()
        for module, values, quantizer in records:
            quantized, codes = quantizer.quantize_with_codes(values)
            recorder.record(
                module, "input", 0, "encoder", values,
                quantized, codes, quantizer, channel_dim=1)
        expected = {
            name for name in recorder.site_names()
            if recorder.site_metadata[name]["synthetic_slice"] == 0
        }
        recorder.validate(expected, expected_updates=1)
        recorder.write(Path(root) / model_name)
        return recorder

    def _assert_nonblank(self, path):
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        self.assertGreater(float(pixels.std()), 1.0)

    def test_model_profile_writes_paginated_pdf_and_nonblank_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self._write_profile(directory, "cspn")
            model_dir = Path(directory) / "cspn"

            plotting.plot_model_profile(
                model_dir, sites_per_page=2, critical_limit=2)

            with fitz.open(model_dir / "all_sites_histograms.pdf") as document:
                self.assertEqual(
                    document.page_count,
                    int(math.ceil(len(recorder.site_names()) / 2.0)))
            for filename in (
                    "critical_layers.png",
                    "rgb_depth_input_histograms.png",
                    "group_outlier_distribution.png"):
                self._assert_nonblank(model_dir / filename)

    def test_root_comparison_writes_csv_and_unrotated_model_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_profile(directory, "cspn")
            self._write_profile(directory, "dyspn")

            figure, rows = plotting.plot_root_comparison(
                Path(directory), ("cspn", "dyspn"))

            self.assertEqual([row["model"] for row in rows], [
                "CSPN", "DySPN"])
            self.assertTrue(all(label.get_rotation() == 0.0
                                for label in figure.axes[0].get_xticklabels()))
            self.assertTrue((Path(directory) /
                             "w4a4_activation_outlier_comparison.csv").is_file())
            self._assert_nonblank(
                Path(directory) / "w4a4_activation_outlier_comparison.png")

    def test_nonfinite_metrics_are_bounded_only_for_plotting(self):
        plotted, labels = plotting.finite_plot_values([
            float("-inf"), 2.0, float("inf")])

        self.assertTrue(np.isfinite(plotted).all())
        self.assertEqual(labels, ["-Inf", "", "Inf"])


if __name__ == "__main__":
    unittest.main()
