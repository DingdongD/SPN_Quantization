import unittest
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_nyu_cspn_selective_w4a8 as runner
from spn_quant import cspn_encoder_prefix as prefix
from spn_quant.cspn_selective_w4a8 import build_stage1_candidates


def stable_union(sequences):
    values = []
    for sequence in sequences:
        for value in sequence:
            if value not in values:
                values.append(value)
    return tuple(values)


def registry():
    modules = stable_union(
        prefix.WEIGHT_MODULES_BY_UNIT[unit]
        for unit in prefix.ALL_UNIT_ORDER)
    owners = stable_union(
        prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
        for unit in prefix.ALL_UNIT_ORDER)
    return prefix.build_unit_registry(modules, owners)


def candidate(mask):
    return build_stage1_candidates(registry())[mask]


def configured_inputs(selected):
    weight_bits = dict(
        (name, 8 if name == runner.INITIAL_DEPTH_WEIGHT else 4)
        for name in stable_union(
            prefix.WEIGHT_MODULES_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER)
        if name != runner.STEM_WEIGHT_MODULE)
    specs = {}
    rotation_specs = {}
    for owner in stable_union(
            prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER):
        if owner == runner.STEM_INPUT_OWNER:
            continue
        spec = SimpleNamespace(
            bits=8 if owner in selected.activation_owners else 4)
        if owner[0].startswith("rotation."):
            rotation_specs[owner] = spec
        else:
            specs[owner] = spec
    stem_bits = 8 if runner.STEM_INPUT_OWNER in \
        selected.activation_owners else 4
    stem_contract = {
        "config": "STEM_W4A8" if stem_bits == 8 else "STRICT_W4A4",
        "weight_bits": 4,
        "activation_bits": stem_bits,
    }
    return weight_bits, specs, rotation_specs, stem_contract


class HardwareConfigurationTest(unittest.TestCase):
    def test_script_entrypoint_resolves_repository_imports(self):
        script = Path(runner.__file__).resolve()

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[2], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_primary_configuration_fixes_only_depth_head_weight_at_w8(self):
        selected = candidate(15)

        config = runner.hardware_configuration(selected)

        self.assertEqual(config["w_bits"], 4)
        self.assertEqual(config["a_bits"], 4)
        self.assertEqual(config["group_size"], 8)
        self.assertEqual(
            config["weight_bit_overrides"],
            ((runner.INITIAL_DEPTH_WEIGHT, 8),))
        self.assertNotIn(runner.STEM_INPUT_OWNER, config["promoted_owners"])
        self.assertIn(
            ("rotation.layer4_signed_skip", "boundary"),
            config["promoted_owners"])
        self.assertEqual(config["propagation"], runner.base.PROPAGATION_A8_Q13)

    def test_stem_configuration_depends_only_on_merged_input_owner(self):
        self.assertEqual(
            runner.stem_configuration(candidate(0)), "STRICT_W4A4")
        self.assertEqual(
            runner.stem_configuration(candidate(1)), "STEM_W4A8")

    def test_configured_precision_matches_exact_candidate(self):
        selected = candidate(15)
        inputs = configured_inputs(selected)

        runner.validate_configured_precision(selected, *inputs)

    def test_unrequested_w8_weight_fails(self):
        selected = candidate(0)
        weight_bits, specs, rotation_specs, stem_contract = \
            configured_inputs(selected)
        weight_bits["layer1.0.conv1"] = 8

        with self.assertRaisesRegex(RuntimeError, "W8 weight"):
            runner.validate_configured_precision(
                selected, weight_bits, specs, rotation_specs, stem_contract)

    def test_stage1_runtime_set_has_sixteen_primary_and_two_contexts(self):
        primary, contexts = runner.stage1_candidates(registry())

        self.assertEqual(len(primary), 16)
        self.assertEqual(len(contexts), 2)
        self.assertEqual(
            [value.name for value in contexts],
            ["CONTEXT_STRICT_W4A4", "CONTEXT_P3_T3_W8A8"])
        self.assertEqual(contexts[0].weight_modules, ())
        self.assertEqual(contexts[0].activation_owners, ())
        self.assertEqual(contexts[0].stem_config, "STRICT_W4A4")
        self.assertEqual(contexts[1].stem_config, "STEM_W8A8")
        self.assertIn("layer2.0.conv1", contexts[1].weight_modules)
        self.assertNotIn("layer3.0.conv1", contexts[1].weight_modules)


class CalibrationDiagnosticTest(unittest.TestCase):
    def test_demotion_summary_uses_propagation_and_downstream_error(self):
        selected = candidate(3)
        demotion = runner.build_single_demotions(selected)[0]
        anchor_rows = [
            {"block": "encoder_stem", "block_output_mse": 0.01,
             "error_energy": 1.0, "elements": 100},
            {"block": "encoder_layer1", "block_output_mse": 0.02,
             "error_energy": 4.0, "elements": 100},
            {"block": "propagation", "block_output_mse": 0.03,
             "error_energy": 3.0, "elements": 100},
        ]
        demotion_rows = [
            {"block": "encoder_stem", "block_output_mse": 0.02,
             "error_energy": 2.0, "elements": 100},
            {"block": "encoder_layer1", "block_output_mse": 0.04,
             "error_energy": 8.0, "elements": 100},
            {"block": "propagation", "block_output_mse": 0.05,
             "error_energy": 5.0, "elements": 100},
        ]

        row = runner.demotion_calibration_row(
            demotion, anchor_rows, demotion_rows,
            anchor_cost=0.2, demotion_cost=0.18)

        self.assertEqual(row["module"], demotion.owner[0])
        self.assertEqual(row["kind"], demotion.owner[1])
        self.assertAlmostEqual(row["propagation_mse"], 0.05)
        self.assertAlmostEqual(row["saved_cost"], 0.02)
        self.assertGreaterEqual(row["downstream_mse"], 0.04)
        self.assertGreaterEqual(row["anchor_downstream_mse"], 0.02)

    def test_candidate_aggregate_records_safety_maxima(self):
        selected = candidate(0)
        sample_rows = []
        for index in range(64):
            sample_rows.append({
                "config": selected.name,
                "sample_index": index,
                "RMSE": 0.17,
                "MAE": 0.08,
                "ABS_REL": 0.03,
                "IRMSE": 0.02,
                "flat_RMSE": 0.12,
                "boundary_RMSE": 0.4,
                "nonfinite_ratio": 0.0,
                "nonpositive_ratio": 0.0,
            })
        propagation_rows = [
            {"config": selected.name, "coefficient_sum_max_error": 0.0,
             "contraction_violation_rate": "", "anchor_max_error": ""},
            {"config": selected.name, "coefficient_sum_max_error": "",
             "contraction_violation_rate": 0.0, "anchor_max_error": 0.0},
        ]

        rows = runner.aggregate_candidate_metrics(
            sample_rows, propagation_rows, (selected,))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["samples"], 64)
        self.assertAlmostEqual(rows[0]["RMSE"], 0.17)
        self.assertEqual(rows[0]["coefficient_sum_max_error"], 0.0)
        self.assertEqual(rows[0]["contraction_violation_rate"], 0.0)
        self.assertEqual(rows[0]["anchor_max_error"], 0.0)


class OrchestrationTest(unittest.TestCase):
    def test_stage1_diagnostics_are_persisted_before_anchor_selection(self):
        root = Path("/unused/stage1.incomplete")
        matrix = {
            "result_rows": {
                "sample_rows": [{"config": "ACT_MASK_00"}],
                "propagation_rows": [{"config": "ACT_MASK_00"}],
            },
            "activation_basis": [{"module": "conv1_1"}],
        }
        aggregate = [{"config": "ACT_MASK_00", "RMSE": 0.3}]

        with patch.object(Path, "mkdir") as mkdir, \
                patch.object(runner, "write_csv") as write_csv:
            runner.write_stage1_diagnostics(root, matrix, aggregate)

        mkdir.assert_called_once_with(parents=True)
        self.assertEqual(write_csv.call_count, 4)
        self.assertEqual(
            [call.args[0].name for call in write_csv.call_args_list],
            ["stage1_sample_metrics_64.csv",
             "stage1_aggregate_metrics.csv",
             "stage1_propagation_metrics.csv",
             "activation_cost_basis.csv"])

    def test_runtime_matrix_uses_fresh_evaluator_for_every_candidate(self):
        selected = tuple(
            runner.runtime_primary(value)
            for value in build_stage1_candidates(registry())[:2])
        calls = []

        def evaluator(value, expected_registry):
            calls.append((value.name, expected_registry))
            result = {
                "sample_rows": [],
                "region_rows": [],
                "propagation_rows": [],
                "block_rows": [],
                "operation_rows": [{
                    "module": "conv1_1", "macs": 10,
                    "weight_elements": 4, "input_elements": 8,
                    "weight_bits": 4, "config": value.name,
                }],
                "layer_rows": [],
                "stem_rows": [],
                "activation_rows": [{
                    "module": "conv1_1", "kind": "input", "elements": 8,
                }],
            }
            return "registry", result

        matrix = runner.run_runtime_matrix(selected, evaluator)

        self.assertEqual([name for name, _ in calls],
                         [value.name for value in selected])
        self.assertIsNone(calls[0][1])
        self.assertEqual(calls[1][1], "registry")
        self.assertEqual(len(matrix["result_rows"]["operation_rows"]), 2)
        self.assertEqual(len(matrix["results_by_name"]), 2)

    def test_unrequested_a8_owner_fails(self):
        selected = candidate(0)
        weight_bits, specs, rotation_specs, stem_contract = \
            configured_inputs(selected)
        specs[("layer1.0.conv1", "input")].bits = 8

        with self.assertRaisesRegex(RuntimeError, "A8 activation"):
            runner.validate_configured_precision(
                selected, weight_bits, specs, rotation_specs, stem_contract)


if __name__ == "__main__":
    unittest.main()
