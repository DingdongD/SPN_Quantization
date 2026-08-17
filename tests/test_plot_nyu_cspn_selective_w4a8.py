import unittest
from pathlib import Path
import subprocess
import sys

from scripts import plot_nyu_cspn_selective_w4a8 as plotter


def stage1_rows():
    rows = []
    for mask in range(16):
        rows.append({
            "config": "ACT_MASK_%02d" % mask,
            "role": "primary",
            "mask": str(mask),
            "RMSE": str(0.2 - mask / 1000.0),
            "normalized_added_bit_cost": str(mask / 100.0),
            "a8_activation_element_fraction": str(mask / 50.0),
        })
    rows.append({
        "config": "CONTEXT_STRICT_W4A4",
        "role": "context",
        "mask": "",
        "RMSE": "0.3",
        "normalized_added_bit_cost": "0.0",
        "a8_activation_element_fraction": "0.0",
    })
    return rows


class PlotValidationTest(unittest.TestCase):
    def test_script_entrypoint_resolves_repository_imports(self):
        script = Path(plotter.__file__).resolve()

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[2], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stage1_validation_requires_all_sixteen_masks(self):
        rows = plotter.validate_stage1_rows(stage1_rows())

        self.assertEqual(len(rows), 16)
        self.assertEqual([row["mask"] for row in rows], list(range(16)))

    def test_missing_stage1_mask_fails(self):
        with self.assertRaisesRegex(ValueError, "mask coverage"):
            plotter.validate_stage1_rows(stage1_rows()[:-2] + stage1_rows()[-1:])

    def test_boundary_ranking_requires_contiguous_unique_ranks(self):
        rows = [
            {"rank": "1", "module": "a", "kind": "input",
             "score": "0.1", "saved_cost": "0.02"},
            {"rank": "2", "module": "b", "kind": "output",
             "score": "0.2", "saved_cost": "0.01"},
        ]

        validated = plotter.validate_boundary_rows(rows)

        self.assertEqual([row["rank"] for row in validated], [1, 2])

    def test_path_validation_requires_strictly_decreasing_cost(self):
        rows = [
            {"config": "PATH_000", "step": "0", "RMSE": "0.17",
             "normalized_added_bit_cost": "0.3",
             "a8_activation_element_fraction": "0.5"},
            {"config": "PATH_001", "step": "1", "RMSE": "0.171",
             "normalized_added_bit_cost": "0.2",
             "a8_activation_element_fraction": "0.4"},
        ]

        validated = plotter.validate_path_rows(rows)

        self.assertEqual(len(validated), 2)


if __name__ == "__main__":
    unittest.main()
