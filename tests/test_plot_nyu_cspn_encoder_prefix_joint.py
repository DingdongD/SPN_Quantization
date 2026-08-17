import unittest
from pathlib import Path
import subprocess
import sys

from scripts import plot_nyu_cspn_encoder_prefix_joint as plotter


def aggregate_rows():
    rows = []
    for prefix_index in range(6):
        for tail_index in range(4):
            rows.append({
                "config": "PREFIX_P%d__TAIL_T%d" %
                (prefix_index, tail_index),
                "prefix_index": str(prefix_index),
                "tail_index": str(tail_index),
                "RMSE": str(1.0 - 0.01 * prefix_index -
                            0.02 * tail_index),
                "normalized_added_bit_cost": str(
                    0.1 * prefix_index + 0.01 * tail_index),
                "w8_weight_mac_fraction": str(
                    0.08 * prefix_index + 0.02 * tail_index),
            })
    return rows


class PlotDataTest(unittest.TestCase):
    def test_script_entrypoint_resolves_repository_imports(self):
        script = Path(plotter.__file__).resolve()

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[2], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_aggregate_validation_requires_complete_matrix(self):
        rows = plotter.validate_aggregate_rows(aggregate_rows())

        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]["prefix_index"], 0)
        self.assertEqual(rows[-1]["tail_index"], 3)

    def test_missing_aggregate_cell_fails(self):
        with self.assertRaisesRegex(ValueError, "matrix coverage"):
            plotter.validate_aggregate_rows(aggregate_rows()[:-1])

    def test_duplicate_aggregate_cell_fails(self):
        rows = aggregate_rows()
        rows[-1] = dict(rows[0])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            plotter.validate_aggregate_rows(rows)

    def test_interaction_validation_requires_finite_complete_values(self):
        rows = [{
            "config": row["config"],
            "prefix_index": row["prefix_index"],
            "tail_index": row["tail_index"],
            "RMSE": row["RMSE"],
            "interaction_rmse": "0.0",
        } for row in aggregate_rows()]

        validated = plotter.validate_interaction_rows(rows)

        self.assertEqual(len(validated), 24)
        self.assertEqual(validated[0]["interaction_rmse"], 0.0)

    def test_pareto_validation_rejects_dominated_input(self):
        rows = (
            {"config": "strict", "RMSE": "1.0",
             "normalized_added_bit_cost": "0.0",
             "prefix_index": "0", "tail_index": "0"},
            {"config": "good", "RMSE": "0.8",
             "normalized_added_bit_cost": "0.1",
             "prefix_index": "1", "tail_index": "0"},
            {"config": "bad", "RMSE": "0.9",
             "normalized_added_bit_cost": "0.2",
             "prefix_index": "2", "tail_index": "0"},
        )

        with self.assertRaisesRegex(ValueError, "non-dominated"):
            plotter.validate_pareto_rows(
                rows, "normalized_added_bit_cost")

    def test_missing_input_file_fails(self):
        with self.assertRaises(FileNotFoundError):
            plotter.load_csv(Path("tests/does_not_exist_prefix.csv"))


if __name__ == "__main__":
    unittest.main()
