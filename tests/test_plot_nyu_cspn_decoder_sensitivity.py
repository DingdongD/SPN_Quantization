import unittest
from pathlib import Path
import subprocess
import sys

from scripts import plot_nyu_cspn_decoder_sensitivity as plotter


class ParetoPlotDataTest(unittest.TestCase):
    def test_script_entrypoint_resolves_repository_imports(self):
        script = Path(plotter.__file__).resolve()

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[2], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_valid_rows_preserve_cost_order(self):
        rows = (
            {"config": "strict", "stage": "baseline", "RMSE": "1.0",
             "normalized_added_bit_cost": "0.0"},
            {"config": "candidate", "stage": "site", "RMSE": "0.8",
             "normalized_added_bit_cost": "0.1"},
        )

        validated = plotter.validate_pareto_rows(rows)

        self.assertEqual(
            [row["config"] for row in validated],
            ["strict", "candidate"])

    def test_missing_column_fails(self):
        rows = ({
            "config": "strict", "stage": "baseline", "RMSE": "1.0",
        },)

        with self.assertRaises(KeyError):
            plotter.validate_pareto_rows(rows)

    def test_duplicate_configuration_fails(self):
        rows = (
            {"config": "same", "stage": "baseline", "RMSE": "1.0",
             "normalized_added_bit_cost": "0.0"},
            {"config": "same", "stage": "site", "RMSE": "0.8",
             "normalized_added_bit_cost": "0.1"},
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            plotter.validate_pareto_rows(rows)

    def test_dominated_row_fails_pareto_contract(self):
        rows = (
            {"config": "strict", "stage": "baseline", "RMSE": "1.0",
             "normalized_added_bit_cost": "0.0"},
            {"config": "good", "stage": "site", "RMSE": "0.8",
             "normalized_added_bit_cost": "0.1"},
            {"config": "bad", "stage": "site", "RMSE": "0.9",
             "normalized_added_bit_cost": "0.2"},
        )

        with self.assertRaisesRegex(ValueError, "non-dominated"):
            plotter.validate_pareto_rows(rows)

    def test_missing_input_file_fails(self):
        path = Path("tests/does_not_exist_pareto_metrics.csv")

        with self.assertRaises(FileNotFoundError):
            plotter.load_pareto_rows(path)


if __name__ == "__main__":
    unittest.main()
