import csv
import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from scripts import plot_completionformer_front_encoder_pareto as plotter


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root, samples=2):
    aggregate = [
        {"config": "FP32", "selected_units": "", "samples": samples,
         "mean_rmse": 0.20, "whole_model_mac_share": "",
         "whole_model_parameter_share": "",
         "whole_model_operator_share": "", "is_pareto": 0},
        {"config": "JIQ_Joint_W4A4", "selected_units": "",
         "samples": samples, "mean_rmse": 0.90,
         "whole_model_mac_share": 0.0,
         "whole_model_parameter_share": 0.0,
         "whole_model_operator_share": 0.0, "is_pareto": 1},
        {"config": "JIQ_W4A8", "selected_units": "", "samples": samples,
         "mean_rmse": 0.40, "whole_model_mac_share": "",
         "whole_model_parameter_share": "",
         "whole_model_operator_share": "", "is_pareto": 0},
        {"config": "FE_W8A8_Stem", "selected_units": "Stem",
         "samples": samples, "mean_rmse": 0.70,
         "whole_model_mac_share": 0.10,
         "whole_model_parameter_share": 0.08,
         "whole_model_operator_share": 0.05, "is_pareto": 1},
        {"config": "FE_W8A8_E10", "selected_units": "Embed1.0",
         "samples": samples, "mean_rmse": 0.85,
         "whole_model_mac_share": 0.20,
         "whole_model_parameter_share": 0.12,
         "whole_model_operator_share": 0.10, "is_pareto": 0},
        {"config": "FE_W8A8_Stem+E10",
         "selected_units": "Stem;Embed1.0", "samples": samples,
         "mean_rmse": 0.50, "whole_model_mac_share": 0.30,
         "whole_model_parameter_share": 0.20,
         "whole_model_operator_share": 0.15, "is_pareto": 1},
    ]
    pareto = [row for row in aggregate if int(row["is_pareto"]) == 1]
    write_csv(root / "front_encoder_final_aggregate.csv", aggregate)
    write_csv(root / "front_encoder_pareto.csv", tuple(reversed(pareto)))
    (root / "metadata.json").write_text(json.dumps({
        "front_encoder_w8a8_pareto": {
            "knee_config": "FE_W8A8_Stem",
            "best_config": "FE_W8A8_Stem+E10",
        },
    }), encoding="utf-8")

    configs = (
        "FP32", "JIQ_Joint_W4A4", "JIQ_W4A8",
        "FE_W8A8_Stem", "FE_W8A8_Stem+E10",
    )
    for config_index, config in enumerate(configs):
        directory = root / "predictions" / config
        directory.mkdir(parents=True, exist_ok=True)
        for sample_index in range(samples):
            gt = np.full((4, 5), 2.0 + sample_index, dtype=np.float32)
            pred = gt + 0.02 * config_index
            np.savez_compressed(
                directory / ("sample_%05d.npz" % sample_index),
                gt=gt, fp32=gt, pred=pred,
                sample_index=np.array(sample_index),
                config=np.array(config))


class CompletionFormerFrontEncoderParetoPlotTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = Path(__file__).parent / ".artifacts" / \
            "completionformer_front_encoder_pareto"
        shutil.rmtree(self.artifacts, ignore_errors=True)
        self.root = self.artifacts / "completionformer"
        self.output = self.artifacts / "analysis"
        self.root.mkdir(parents=True)
        build_fixture(self.root, samples=2)

    def tearDown(self):
        shutil.rmtree(self.artifacts, ignore_errors=True)

    def test_generate_figures_renders_pareto_and_predictions(self):
        paths = plotter.generate_figures(
            self.root, self.output, expected_samples=2, dpi=40)

        self.assertEqual(set(paths), {
            "mac", "parameter", "operator", "predictions",
        })
        self.assertEqual(
            set(path.name for path in paths.values()), {
                "rmse_vs_w8a8_mac_share.png",
                "rmse_vs_w8a8_parameter_share.png",
                "rmse_vs_w8a8_operator_share.png",
                "prediction_comparison_64.png",
            })
        for path in paths.values():
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_contract_separates_frontier_and_dominated_points(self):
        contract = plotter.load_pareto_contract(self.root)

        self.assertEqual(
            [row["config"] for row in contract["frontier"]], [
                "JIQ_Joint_W4A4", "FE_W8A8_Stem",
                "FE_W8A8_Stem+E10",
            ])
        self.assertEqual(
            [row["config"] for row in contract["dominated"]], [
                "FE_W8A8_E10",
            ])
        self.assertEqual(contract["knee_config"], "FE_W8A8_Stem")
        self.assertEqual(
            contract["best_config"], "FE_W8A8_Stem+E10")

        labels = plotter.pareto_point_labels(contract["frontier"])
        self.assertEqual(labels[0], ("P0", "W4A4"))
        self.assertEqual(labels[1], ("P1", "Stem"))
        self.assertEqual(labels[2], ("P2", "Stem+E10"))
        self.assertTrue(all(len(point) <= 3 for point, units in labels))

    def test_missing_prediction_sample_fails(self):
        missing = self.root / "predictions" / "FE_W8A8_Stem" / \
            "sample_00001.npz"
        missing.unlink()

        contract = plotter.load_pareto_contract(self.root)
        with self.assertRaisesRegex(ValueError, "FE_W8A8_Stem"):
            plotter.load_prediction_grid(
                self.root, contract, expected_samples=2)


if __name__ == "__main__":
    unittest.main()
