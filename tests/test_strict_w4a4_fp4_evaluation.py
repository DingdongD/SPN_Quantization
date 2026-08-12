import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.strict_w4a4_fp4_evaluation import (
    METHOD_ORDER,
    PRIMARY_CONFIGS,
    STRESS_CONFIGS,
    analyze_result_root,
    paired_rmse_difference,
    performance_decision,
    validate_result_root,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def provenance(model):
    return {
        "model_class": "%s.Model" % model,
        "model_module": "%s.module" % model,
        "source_sha256": "%s-source" % model,
        "checkpoint_sha256": "%s-checkpoint" % model,
        "source_git_commit": "%s-commit" % model,
    }


def reconstruction(root, method, model):
    contract_root = root / "contracts" / method / model
    contract_root.mkdir(parents=True, exist_ok=True)
    contract_path = contract_root / "strict_deployment_contract.pt"
    contract_path.write_bytes(b"strict-contract")
    manifest_path = contract_root / "strict_reconstruction_manifest.json"
    manifest = {
        "strict": 1,
        "method": "%s_strict" % method,
        "model": model,
        "weight_bits": 4,
        "activation_bits": 0,
        "activation_manifest": [],
        "targets": ["encoder", "decoder", "propagation"],
        "deployment_contract": str(contract_path),
    }
    write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "method": "%s_strict" % method,
        "weight_bits": 4,
        "activation_bits": 0,
        "activation_policy": "evaluation_backend_owned",
        "targets": ["encoder", "decoder", "propagation"],
        "strict_deployment_contract": str(contract_path),
        "exact_weight_contract": 1,
    }


def sample_rows(model, configs, indices, method):
    method_offset = {
        "rtn": 0.02,
        "adaround": 0.01,
        "brecq": 0.00,
    }[method]
    config_rmse = {
        "FP32": 0.10,
        "FP4V_W4A4": 0.14,
        "FP4V_W4E2M1": 0.12,
        "FP4V_W4A8": 0.105,
        "HW_W4A4_full": 0.16,
    }
    rows = []
    for config in configs:
        for sample_rank, sample_index in enumerate(indices):
            offset = 0.0 if config == "FP32" else method_offset
            rmse = config_rmse[config] + offset + sample_rank * 0.001
            rows.append({
                "model": model,
                "config": config,
                "sample_index": sample_index,
                "RMSE": rmse,
                "MAE": rmse / 2.0,
                "ABS_REL": rmse / 10.0,
                "num_pixels": 20,
                "nonfinite_pixels": 0,
            })
    return rows


def prediction_payload(path, model, config, sample_index):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((2, 2), 1.0, dtype=np.float32)
    np.savez_compressed(
        str(path), sample_index=np.asarray(sample_index),
        model=np.asarray(model), config=np.asarray(config),
        gt=values, fp32=values, pred=values,
        valid_gt=np.ones_like(values, dtype=bool),
        nonfinite=np.zeros_like(values, dtype=bool))


def write_result(root, backend, method, model, configs, indices):
    model_root = root / backend / method / model
    is_primary = backend == "primary"
    metadata = {
        "model": model,
        "model_provenance": provenance(model),
        "calibration_indices": [101, 103],
        "evaluation_indices": list(indices),
        "evaluation_samples": len(indices),
        "configs": list(configs),
        "quant_backend": "fp4" if is_primary else "hardware",
        "quantization_execution": (
            "float_e2m1_qdq_integer_normalization_reference"
            if is_primary else "hardware_aligned_qdq"),
        "hardware_alignment": {
            "bias_contract": (
                "fp32_isolation" if is_primary
                else "int32 scale=sx*sw[o]"),
        },
    }
    if is_primary:
        metadata["fp4_validation"] = {
            "activation_format": "scaled_e2m1_rne",
            "bias_contract": "fp32_isolation",
            "propagation_signals": "a8",
        }
    if method != "rtn":
        metadata["reconstruction"] = reconstruction(root, method, model)
    write_json(model_root / "metadata.json", metadata)
    write_csv(
        model_root / "sample_metrics.csv",
        sample_rows(model, configs, indices, method))
    for config in configs:
        for sample_index in indices:
            prediction_payload(
                model_root / "predictions" / config /
                ("sample_%05d.npz" % sample_index),
                model, config, sample_index)
    if is_primary:
        write_csv(model_root / "semantic_a8_boundaries.csv", [{
            "model": model,
            "role": "guidance",
            "module": "%s.guidance" % model,
            "kind": "output",
            "bits": 8,
            "format": "uniform",
        }])
        write_csv(model_root / "fp4_manifest.csv", [
            {
                "model": model,
                "config": config,
                "module": "%s.encoder" % model,
                "kind": "input",
                "format": "e2m1" if config == "FP4V_W4E2M1" else "uniform",
                "bits": 8 if config == "FP4V_W4A8" else 4,
                "unsigned": 0,
                "codebook": "",
                "scale": 1.0,
                "channel_dim": "",
            }
            for config in PRIMARY_CONFIGS[1:]])
        write_csv(model_root / "layer_quantization_metrics.csv", [
            {
                "model": model,
                "config": config,
                "module": "%s.encoder" % model,
                "group": "encoder",
                "kind": "input",
                "error_sq": 2.0,
                "signal_sq": 20.0,
                "numel": 10,
                "zero_code_rate": 0.2,
                "saturation_rate": 0.01,
                "nonfinite_rate": 0.0,
            }
            for config in PRIMARY_CONFIGS[1:]])
        write_csv(model_root / "signal_metrics.csv", [
            {
                "model": model,
                "config": config,
                "sample_index": sample_index,
                "signal": "propagation_states",
                "iteration": iteration,
                "rmse": 0.01 * iteration,
            }
            for config in PRIMARY_CONFIGS[1:]
            for sample_index in indices
            for iteration in (1, 2)])


def write_result_root(root):
    indices = (3, 7)
    for method in ("rtn", "adaround", "brecq"):
        for model in ("cspn", "dyspn", "nlspn", "completionformer"):
            write_result(
                root, "primary", method, model, PRIMARY_CONFIGS, indices)
            write_result(
                root, "stress", method, model, STRESS_CONFIGS, indices)


class StrictW4A4FP4ContractTest(unittest.TestCase):
    def test_matrix_is_fixed_and_keeps_stress_separate(self):
        self.assertEqual(METHOD_ORDER, ("rtn", "adaround", "brecq"))
        self.assertEqual(
            PRIMARY_CONFIGS,
            ("FP32", "FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8"))
        self.assertEqual(STRESS_CONFIGS, ("FP32", "HW_W4A4_full"))

    def test_performance_decision_requires_all_three_conditions(self):
        accepted = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.08, rtn_rmse=1.09,
            nonfinite_samples=0, nonfinite_pixels=0)
        self.assertEqual(accepted["status"], "preserved")

        degraded = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.11, rtn_rmse=1.12,
            nonfinite_samples=0, nonfinite_pixels=0)
        self.assertEqual(
            degraded["status"], "rejected_fp32_degradation")

        regression = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.08, rtn_rmse=1.07,
            nonfinite_samples=0, nonfinite_pixels=0)
        self.assertEqual(regression["status"], "rejected_rtn_regression")

        invalid = performance_decision(
            fp32_rmse=1.0, quant_rmse=1.02, rtn_rmse=1.03,
            nonfinite_samples=1, nonfinite_pixels=10)
        self.assertEqual(invalid["status"], "rejected_nonfinite")

    def test_performance_decision_rejects_invalid_rmse_values(self):
        with self.assertRaisesRegex(ValueError, "finite positive RMSE"):
            performance_decision(
                fp32_rmse=0.0, quant_rmse=1.0, rtn_rmse=1.0,
                nonfinite_samples=0, nonfinite_pixels=0)


class StrictW4A4FP4ValidationTest(unittest.TestCase):
    def setUp(self):
        repository = Path(__file__).resolve().parents[1]
        self.directory = tempfile.TemporaryDirectory(dir=str(repository))
        self.root = Path(self.directory.name)
        write_result_root(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_complete_result_root_is_accepted(self):
        result = validate_result_root(self.root, expected_samples=2)

        self.assertEqual(set(result), {
            "cspn", "dyspn", "nlspn", "completionformer"})

    def test_checkpoint_mismatch_is_rejected(self):
        path = self.root / "primary" / "brecq" / "cspn" / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["model_provenance"]["checkpoint_sha256"] = "different"
        write_json(path, metadata)

        with self.assertRaisesRegex(
                ValueError, "model provenance mismatch"):
            validate_result_root(self.root, expected_samples=2)

    def test_integer_stress_nonfinite_metrics_are_retained(self):
        path = self.root / "stress" / "rtn" / "cspn" / \
            "sample_metrics.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows[-1]["RMSE"] = "inf"
        rows[-1]["MAE"] = "inf"
        rows[-1]["ABS_REL"] = "inf"
        rows[-1]["nonfinite_pixels"] = "4"
        write_csv(path, rows)

        validate_result_root(self.root, expected_samples=2)

    def test_primary_nonfinite_metrics_are_rejected(self):
        path = self.root / "primary" / "rtn" / "cspn" / \
            "sample_metrics.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows[-1]["RMSE"] = "inf"
        write_csv(path, rows)

        with self.assertRaisesRegex(ValueError, "nonfinite values"):
            validate_result_root(self.root, expected_samples=2)

    def test_manifest_accepts_declared_semantic_a8_boundaries(self):
        model_root = self.root / "primary" / "rtn" / "cspn"
        path = model_root / "fp4_manifest.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows.extend({
            "model": "cspn",
            "config": config,
            "module": "cspn.guidance",
            "kind": "output",
            "format": "uniform",
            "bits": 8,
            "unsigned": 0,
            "codebook": "",
            "scale": 1.0,
            "channel_dim": "",
        } for config in PRIMARY_CONFIGS[1:])
        write_csv(path, rows)

        validate_result_root(self.root, expected_samples=2)


class StrictW4A4FP4AggregationTest(unittest.TestCase):
    def setUp(self):
        repository = Path(__file__).resolve().parents[1]
        self.directory = tempfile.TemporaryDirectory(dir=str(repository))
        self.root = Path(self.directory.name)
        write_result_root(self.root)

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def select_one(rows, **expected):
        matches = [
            row for row in rows
            if all(row[key] == value for key, value in expected.items())]
        if len(matches) != 1:
            raise AssertionError("aggregate row lookup mismatch")
        return matches[0]

    def test_paired_bootstrap_is_deterministic(self):
        left = np.asarray([1.0, 1.2, 1.4, 1.6])
        right = np.asarray([0.6, 0.8, 1.0, 1.2])

        first = paired_rmse_difference(
            left, right, resamples=1000, seed=20260806)
        second = paired_rmse_difference(
            left, right, resamples=1000, seed=20260806)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], 0.4)
        self.assertEqual(first["samples"], 4)

    def test_aggregation_compares_each_method_with_same_format_rtn(self):
        tables = analyze_result_root(
            self.root, expected_samples=2,
            bootstrap_resamples=500, bootstrap_seed=20260806)

        row = self.select_one(
            tables["summary"], model="cspn", method="adaround",
            config="FP4V_W4E2M1")
        rtn = self.select_one(
            tables["summary"], model="cspn", method="rtn",
            config="FP4V_W4E2M1")
        self.assertAlmostEqual(
            row["delta_vs_rtn"], row["mean_rmse"] - rtn["mean_rmse"])
        self.assertEqual(row["status"], "rejected_fp32_degradation")

        paired = self.select_one(
            tables["paired"], model="cspn", method="brecq",
            comparison="e2m1_minus_a4")
        self.assertEqual(paired["samples"], 2)
        self.assertLess(paired["mean_difference"], 0.0)
        self.assertLessEqual(paired["ci_lower"], paired["mean_difference"])
        self.assertLessEqual(paired["mean_difference"], paired["ci_upper"])

    def test_cspn_brecq_nonfinite_stress_is_excluded_from_active_analysis(self):
        path = self.root / "stress" / "brecq" / "cspn" / \
            "sample_metrics.csv"
        rows = list(csv.DictReader(
            path.open(newline="", encoding="utf-8")))
        for row in rows:
            if row["config"] == "HW_W4A4_full":
                row["RMSE"] = "inf"
                row["MAE"] = "inf"
                row["ABS_REL"] = "inf"
                row["nonfinite_pixels"] = "69312"
        write_csv(path, rows)

        tables = analyze_result_root(
            self.root, expected_samples=2,
            bootstrap_resamples=500, bootstrap_seed=20260806)

        self.assertFalse(any(
            row["model"] == "cspn" and row["method"] == "brecq"
            for row in tables["stress"]))

    def test_aggregation_includes_activation_and_propagation_diagnostics(self):
        tables = analyze_result_root(
            self.root, expected_samples=2,
            bootstrap_resamples=500, bootstrap_seed=20260806)

        activation = self.select_one(
            tables["activation_groups"], model="nlspn",
            method="adaround", config="FP4V_W4A4", group="encoder")
        self.assertAlmostEqual(activation["sqnr_db"], 10.0)
        self.assertAlmostEqual(activation["zero_code_rate"], 0.2)
        propagation = self.select_one(
            tables["propagation_steps"], model="dyspn",
            method="brecq", config="FP4V_W4E2M1", iteration=2)
        self.assertAlmostEqual(propagation["mean_rmse"], 0.02)

    def test_analysis_writer_persists_all_tables_and_report(self):
        from scripts.analyze_strict_w4a4_fp4_evaluation import write_analysis

        output = self.root / "analysis"
        write_analysis(
            self.root, output, expected_samples=2,
            bootstrap_resamples=500, bootstrap_seed=20260806)

        expected = {
            "strict_w4a4_fp4_summary.csv",
            "strict_w4a4_fp4_paired.csv",
            "strict_w4a4_fp4_activation_groups.csv",
            "strict_w4a4_fp4_propagation_steps.csv",
            "strict_w4a4_integer_stress.csv",
            "strict_w4a4_fp4_report.md",
        }
        self.assertEqual(
            {path.name for path in output.iterdir()}, expected)
        report = (output / "strict_w4a4_fp4_report.md").read_text(
            encoding="utf-8")
        self.assertIn("rejected_fp32_degradation", report)

    def test_semantic_boundary_mismatch_is_rejected(self):
        path = self.root / "primary" / "adaround" / "nlspn" / \
            "semantic_a8_boundaries.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows[0]["bits"] = "4"
        write_csv(path, rows)

        with self.assertRaisesRegex(
                ValueError, "semantic A8 boundary mismatch"):
            validate_result_root(self.root, expected_samples=2)

    def test_missing_prediction_is_rejected(self):
        path = self.root / "primary" / "adaround" / "dyspn" / \
            "predictions" / "FP4V_W4E2M1" / "sample_00007.npz"
        path.unlink()

        with self.assertRaisesRegex(ValueError, "prediction payload"):
            validate_result_root(self.root, expected_samples=2)

    def test_all_models_must_share_calibration_indices(self):
        for backend in ("primary", "stress"):
            for method in ("rtn", "adaround", "brecq"):
                path = self.root / backend / method / "completionformer" / \
                    "metadata.json"
                metadata = json.loads(path.read_text(encoding="utf-8"))
                metadata["calibration_indices"] = [109, 111]
                write_json(path, metadata)

        with self.assertRaisesRegex(ValueError, "global calibration indices"):
            validate_result_root(self.root, expected_samples=2)


if __name__ == "__main__":
    unittest.main()
