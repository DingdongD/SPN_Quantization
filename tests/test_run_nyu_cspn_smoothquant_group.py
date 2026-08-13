import unittest

import torch
import torch.nn as nn

from scripts import run_nyu_cspn_smoothquant_group as runner
from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor


class SmoothQuantConfigurationTest(unittest.TestCase):
    def test_evaluation_matrix_is_fixed(self):
        configs = runner.build_configurations()

        self.assertEqual(
            tuple(config["name"] for config in configs),
            runner.EXPECTED_CONFIGURATIONS)
        self.assertEqual(len(configs), 16)

    def test_smoothquant_is_restricted_to_ordinary_groups(self):
        for config in runner.build_configurations():
            if config["smooth_alpha"] is None:
                self.assertEqual(config["smooth_groups"], set())
                continue
            self.assertEqual(
                config["smooth_groups"], set(runner.ORDINARY_GROUPS))
            self.assertNotIn("propagation_head", config["smooth_groups"])

    def test_alpha_sweep_covers_both_group_sizes(self):
        configs = runner.build_configurations()
        full = [
            config for config in configs
            if config["weight_groups"] and config["activation_groups"]
            and config["smooth_alpha"] is not None
        ]

        self.assertEqual(
            {(config["group_size"], config["smooth_alpha"])
             for config in full},
            {(16, 0.25), (16, 0.5), (16, 0.75),
             (8, 0.25), (8, 0.5), (8, 0.75)})


class SmoothQuantSiteTest(unittest.TestCase):
    def test_channel_maxima_include_only_observed_module_inputs(self):
        model = nn.Sequential(
            nn.Conv2d(4, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 2, 1),
        ).eval()
        instrumentor = HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(torch.tensor([[[[1.0]], [[-2.0]], [[3.0]], [[-4.0]]]]))
        instrumentor.freeze()

        maxima = runner.build_smooth_channel_maxima(
            instrumentor, {"encoder"})

        self.assertEqual(set(maxima), {"0", "2"})
        torch.testing.assert_close(
            maxima["0"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertNotIn("1", maxima)
        instrumentor.close()

    def test_group_range_uses_transformed_channel_maxima(self):
        maxima = torch.tensor([1.0, 4.0, 9.0, 16.0])
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0])

        ranges = runner.transformed_group_maxima(maxima, scales, 2)

        torch.testing.assert_close(ranges, torch.tensor([2.0, 4.0]))


class CalibrationSelectionTest(unittest.TestCase):
    def test_selection_uses_calibration_rows_only(self):
        rows = [
            {"split": "calibration", "config": "SQ_W4A4_GROUP8_A025",
             "block_output_mse": 2.0,
             "block_output_sqnr": 3.0},
            {"split": "calibration", "config": "SQ_W4A4_GROUP8_A050",
             "block_output_mse": 1.0,
             "block_output_sqnr": 2.0},
            {"split": "evaluation", "config": "SQ_W4A4_GROUP8_A075",
             "block_output_mse": 0.0,
             "block_output_sqnr": 100.0},
        ]

        selected = runner.select_smooth_configuration(rows, 8)

        self.assertEqual(selected["config"], "SQ_W4A4_GROUP8_A050")


if __name__ == "__main__":
    unittest.main()
