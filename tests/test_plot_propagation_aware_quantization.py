import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def _write_csv(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class PropagationAwarePlotTest(unittest.TestCase):
    def test_depth_rgba_marks_nonfinite_prediction_magenta(self):
        from scripts import plot_propagation_aware_quantization as plotting

        values = np.array([[1.0, np.nan]], dtype=np.float32)
        valid = np.array([[True, True]])

        rgba = plotting.depth_rgba(values, valid)

        np.testing.assert_allclose(rgba[0, 1], [1.0, 0.0, 1.0, 1.0])

    def test_error_rgba_marks_nonfinite_error_magenta(self):
        from scripts import plot_propagation_aware_quantization as plotting

        values = np.array([[0.1, np.nan]], dtype=np.float32)
        valid = np.array([[True, True]])

        rgba = plotting.error_rgba(values, valid, maximum=1.0)

        np.testing.assert_allclose(rgba[0, 1], [1.0, 0.0, 1.0, 1.0])

    def test_contact_annotations_do_not_repeat_long_titles_per_sample(self):
        from scripts import plot_propagation_aware_quantization as plotting

        first = plotting.contact_annotations(
            sample_index=19, panel="PA_Generic_W4A4", sample_rank=0,
            sample_columns=4)
        later = plotting.contact_annotations(
            sample_index=25, panel="PA_Generic_W4A4", sample_rank=4,
            sample_columns=4)

        self.assertEqual(first, ("Generic W4A4", "#00019"))
        self.assertEqual(later, ("", "#00025"))

    def test_select_details_contains_deterministic_random_and_worst_samples(self):
        from scripts import plot_propagation_aware_quantization as plotting

        rows = [{
            "config": "PA_Generic_W4A4",
            "sample_index": index,
            "RMSE": float(index),
        } for index in range(8)]

        selected = plotting.select_detail_samples(
            rows, seed=7, random_count=2, worst_count=2)

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {row["reason"] for row in selected},
            {"random", "worst_generic_w4a4"})
        worst = [row["sample_index"] for row in selected
                 if row["reason"] == "worst_generic_w4a4"]
        self.assertEqual(worst, [7, 6])

    def test_worst_selection_prioritizes_nonfinite_predictions(self):
        from scripts import plot_propagation_aware_quantization as plotting

        rows = [{
            "config": "PA_Generic_W4A4",
            "sample_index": index,
            "RMSE": 100.0 - index,
            "nonfinite_pixels": 1 if index == 7 else 0,
        } for index in range(8)]

        selected = plotting.select_detail_samples(
            rows, seed=7, random_count=2, worst_count=2)
        worst = [row["sample_index"] for row in selected
                 if row["reason"] == "worst_generic_w4a4"]

        self.assertEqual(worst[0], 7)

    def test_step_series_reports_nonfinite_sample_rate(self):
        from scripts import plot_propagation_aware_quantization as plotting

        rows = [
            {"config": "PA_Generic_W4A4", "signal": "propagation_states",
             "iteration": "1", "rmse": "nan"},
            {"config": "PA_Generic_W4A4", "signal": "propagation_states",
             "iteration": "1", "rmse": "1.0"},
            {"config": "PA_Generic_W4A4", "signal": "propagation_states",
             "iteration": "2", "rmse": "nan"},
            {"config": "PA_Generic_W4A4", "signal": "propagation_states",
             "iteration": "2", "rmse": "inf"},
        ]

        iterations, rmse, failure_rate = plotting.step_series(
            rows, "PA_Generic_W4A4")

        self.assertEqual(iterations, [1, 2])
        self.assertEqual(rmse[0], 1.0)
        self.assertTrue(np.isnan(rmse[1]))
        np.testing.assert_allclose(failure_rate, [0.5, 1.0])

    def test_render_model_writes_contact_detail_step_and_constraint_figures(self):
        from scripts import plot_propagation_aware_quantization as plotting

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input" / "cspn"
            out = Path(tmp) / "figures"
            configs = (
                "FP32", "PA_Generic_W4A4", "PA_Constraint",
                "PA_OffsetA8", "PA_StateA8", "PA_W8A8",
            )
            sample_rows = []
            for config_rank, config in enumerate(configs):
                folder = root / "predictions" / config
                folder.mkdir(parents=True)
                for index in range(4):
                    gt = np.full((3, 4), 1.0 + index, np.float32)
                    sparse = np.zeros_like(gt)
                    sparse[1, 2] = gt[1, 2]
                    pred = gt + 0.05 * config_rank
                    np.savez_compressed(
                        str(folder / ("sample_%05d.npz" % index)),
                        gt=gt,
                        fp32=gt,
                        pred=pred,
                        sparse=sparse,
                        abs_err=np.abs(pred - gt),
                        valid_gt=np.ones_like(gt, dtype=bool),
                        nonfinite=np.zeros_like(gt, dtype=bool),
                        sample_index=np.array(index),
                        model=np.array("cspn"),
                        config=np.array(config),
                    )
                    sample_rows.append({
                        "model": "cspn", "config": config,
                        "sample_index": index,
                        "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
                    })
            _write_csv(root / "sample_metrics.csv", sample_rows)
            _write_csv(root / "signal_metrics.csv", [{
                "model": "cspn", "config": config,
                "sample_index": index,
                "signal": "propagation_states", "iteration": iteration,
                "rmse": 0.1 * iteration + 0.01 * index,
            } for config in configs[1:] for index in range(4)
              for iteration in (1, 2)])
            _write_csv(root / "propagation_quantization_metrics.csv", [{
                "model": "cspn", "config": config,
                "sample_index": index,
                "signal": "affinity_constraints", "iteration": 0,
                "coefficient_sum_max_error": 0.0,
                "contraction_violation_rate": 0.0,
            } for config in configs[2:] for index in range(4)])

            outputs = plotting.render_model(
                root.parent, out, "cspn", expected_samples=4,
                random_count=1, worst_count=1)

            self.assertEqual({path.name for path in outputs}, {
                "cspn_prediction_contact_sheet.png",
                "cspn_prediction_details.png",
                "cspn_propagation_step_error.png",
                "cspn_constraint_violations.png",
            })
            for path in outputs:
                self.assertTrue(path.exists())
                with Image.open(str(path)) as image:
                    self.assertGreater(image.width, 100)
                    self.assertGreater(image.height, 100)
            self.assertTrue((out / "cspn_selected_visual_samples.csv").exists())


if __name__ == "__main__":
    unittest.main()
