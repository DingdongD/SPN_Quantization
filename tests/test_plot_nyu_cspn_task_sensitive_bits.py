import json
from pathlib import Path
import shutil
import unittest

import numpy as np

from scripts import plot_nyu_cspn_task_sensitive_bits as plotter
from scripts import run_nyu_cspn_task_sensitive_bits as runner
from scripts import run_nyu_cspn_stem_precision as stem_runner
from scripts.run_nyu_rtn_quantization import write_csv, write_json
from spn_quant import cspn_task_sensitive_bits as allocation


def allocation_rows():
    return (
        {"tensor": "weight", "block": "stem", "module": "conv1",
         "kind": "weight", "bits": "2", "cost": "1",
         "cost_fraction": "0.25", "weighted_bits": "2"},
        {"tensor": "weight", "block": "encoder", "module": "conv2",
         "kind": "weight", "bits": "4", "cost": "3",
         "cost_fraction": "0.75", "weighted_bits": "12"},
        {"tensor": "activation", "block": "stem", "module": "conv1",
         "kind": "input", "bits": "6", "cost": "1",
         "cost_fraction": "0.25", "weighted_bits": "6"},
        {"tensor": "activation", "block": "encoder", "module": "conv2",
         "kind": "input", "bits": "2", "cost": "3",
         "cost_fraction": "0.75", "weighted_bits": "6"},
    )


class PlotDataTest(unittest.TestCase):
    def test_allocation_rows_recompute_separate_budgets_and_fractions(self):
        rows = plotter.validate_allocation_rows(allocation_rows())
        summary = plotter.summarize_bit_fractions(rows)

        self.assertAlmostEqual(
            sum(row["cost_fraction"] for row in rows
                if row["tensor"] == "weight"), 1.0)
        self.assertAlmostEqual(
            sum(row["cost_fraction"] for row in rows
                if row["tensor"] == "activation"), 1.0)
        self.assertAlmostEqual(summary["weight"][2], 0.25)
        self.assertAlmostEqual(summary["activation"][2], 0.75)

    def test_allocation_rows_reject_incorrect_fraction(self):
        rows = list(allocation_rows())
        rows[0] = dict(rows[0], cost_fraction="0.5")

        with self.assertRaisesRegex(ValueError, "cost fraction"):
            plotter.validate_allocation_rows(rows)

    def test_missing_csv_fails(self):
        with self.assertRaises(FileNotFoundError):
            plotter.read_csv(Path("tests/does_not_exist_task_bits.csv"))


class AuditTest(unittest.TestCase):
    root = Path("tests/.cspn_task_sensitive_audit")

    def setUp(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir()
        registry = runner.expected_registry()
        self.assignment = allocation.uniform_assignment(registry, 4, 4)
        self.basis = allocation.CostBasis(
            weight_macs=tuple(
                (module, 1) for module, bits in self.assignment.weight_bits),
            activation_elements=tuple(
                (owner, 1)
                for owner, bits in self.assignment.activation_bits),
        )
        write_csv(
            self.root / "final_allocation.csv",
            runner.final_allocation_rows(
                registry, self.basis, self.assignment))
        write_csv(
            self.root / "weight_cost_basis.csv",
            tuple({"module": module, "macs": cost}
                  for module, cost in self.basis.weight_macs))
        write_csv(
            self.root / "activation_cost_basis.csv",
            tuple({"module": owner[0], "kind": owner[1], "elements": cost}
                  for owner, cost in self.basis.activation_elements))
        write_json(
            self.root / "final_assignment.json",
            runner.assignment_payload(self.assignment))
        serialized = json.dumps(
            runner.assignment_payload(self.assignment), sort_keys=True)
        calibration = []
        for stage, count in (
                ("single_block", 151), ("joint", 128),
                ("demotion", 10), ("refinement", 128)):
            calibration.extend({
                "stage": stage,
                "local_round": 0,
                "config": "%s_%03d" % (stage, index),
                "assignment": serialized,
                "calibration_RMSE": 0.3,
                "boundary_RMSE": 0.5,
                "propagation_MSE": 0.001,
                "RMSE": 0.3,
                "nonfinite_ratio": 0.0,
                "nonpositive_ratio": 0.0,
                "coefficient_sum_max_error": 0.0,
                "contraction_violation_ratio": 0.0,
                "anchor_max_error": 0.0,
            } for index in range(count))
        write_csv(self.root / "calibration_metrics.csv", calibration)
        validation = []
        for config in plotter.VALIDATION_CONFIGS:
            validation.append({
                "stage": "validation",
                "local_round": 0,
                "config": config,
                "assignment": "FP32" if config == "FP32" else serialized,
                "RMSE": 0.2,
                "MAE": 0.1,
                "ABS_REL": 0.05,
                "IRMSE": 0.3,
                "flat_RMSE": 0.15,
                "boundary_RMSE": 0.25,
                "nonfinite_ratio": 0.0,
                "nonpositive_ratio": 0.0,
                "coefficient_sum_max_error": 0.0,
                "contraction_violation_ratio": 0.0,
                "anchor_max_error": 0.0,
            })
        write_csv(self.root / "validation_metrics.csv", validation)
        predictions = self.root / "predictions"
        for config in plotter.VALIDATION_CONFIGS:
            current = predictions / config
            current.mkdir(parents=True)
            for index in range(64):
                np.savez_compressed(
                    current / ("sample_%05d.npz" % index),
                    sample_index=np.asarray(index),
                    gt=np.ones((2, 2), dtype=np.float32),
                    pred=np.ones((2, 2), dtype=np.float32),
                    valid_gt=np.ones((2, 2), dtype=np.uint8),
                )
        budget = allocation.audit_budget(self.assignment, self.basis)
        manifest = {
            "protocol": {
                "beam_width": 512,
                "joint_measured_limit": 128,
                "local_round_limit": 3,
                "refinement_block_limit": 4,
                "refinement_width": 128,
                "refinement_measured_limit": 128,
            },
            "phase_counts": {
                "single_block": 151,
                "joint": 128,
                "local_rounds": 0,
                "local_candidates": 0,
                "demotion": 10,
                "refinement": 128,
                "validation": 4,
            },
            "refinement_blocks": ["stem", "encoder_layer1",
                                  "decoder_layer4", "initial_depth"],
            "budget": {
                "weight_numerator": budget.weight_numerator,
                "weight_denominator": budget.weight_denominator,
                "activation_numerator": budget.activation_numerator,
                "activation_denominator": budget.activation_denominator,
                "average_weight_bits": budget.average_weight_bits,
                "average_activation_bits": budget.average_activation_bits,
                "weight_feasible": True,
                "activation_feasible": True,
            },
            "artifacts": stem_runner._artifact_hashes(self.root),
        }
        write_json(self.root / "manifest.json", manifest)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_audit_accepts_complete_result(self):
        report = plotter.audit_result_root(self.root)

        self.assertEqual(report["prediction_samples"], 64)
        self.assertAlmostEqual(report["average_weight_bits"], 4.0)
        self.assertAlmostEqual(report["average_activation_bits"], 4.0)

    def test_audit_rejects_prediction_identity_difference(self):
        path = self.root / "predictions" / "FINAL" / "sample_00063.npz"
        path.unlink()
        np.savez_compressed(
            self.root / "predictions" / "FINAL" / "sample_00100.npz",
            sample_index=np.asarray(100),
            gt=np.ones((2, 2), dtype=np.float32),
            pred=np.ones((2, 2), dtype=np.float32),
            valid_gt=np.ones((2, 2), dtype=np.uint8),
        )
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"] = stem_runner._artifact_hashes(self.root)
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "prediction identities"):
            plotter.audit_result_root(self.root)

    def test_audit_rejects_artifact_hash_change(self):
        with (self.root / "validation_metrics.csv").open("a") as stream:
            stream.write("corrupt\n")

        with self.assertRaisesRegex(ValueError, "artifact hashes"):
            plotter.audit_result_root(self.root)


if __name__ == "__main__":
    unittest.main()
