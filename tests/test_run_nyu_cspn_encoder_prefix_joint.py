import unittest
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from scripts import run_nyu_cspn_encoder_prefix_joint as runner
from spn_quant import cspn_encoder_prefix as prefix


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


def candidate(prefix_index=1, tail_index=1):
    return next(
        value for value in prefix.build_candidates(registry())
        if value.prefix_index == prefix_index and
        value.tail_index == tail_index)


class HardwareConfigurationTest(unittest.TestCase):
    def test_script_entrypoint_resolves_repository_imports(self):
        script = Path(runner.__file__).resolve()

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[2], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hardware_configuration_excludes_stem_owned_input_and_weight(self):
        selected = candidate(1, 1)

        config = runner.hardware_configuration(selected)

        self.assertEqual(config["w_bits"], 4)
        self.assertEqual(config["a_bits"], 4)
        self.assertEqual(config["group_size"], 8)
        self.assertEqual(config["weight_groups"], runner.base.ORDINARY_GROUPS)
        self.assertEqual(
            config["activation_groups"], runner.base.ORDINARY_GROUPS)
        self.assertNotIn(
            ("conv1_1", 8), config["weight_bit_overrides"])
        self.assertNotIn(
            ("conv1_1", "input"), config["promoted_owners"])
        self.assertIn(
            ("gud_up_proj_layer4.conv1", 8),
            config["weight_bit_overrides"])
        self.assertIn(
            ("boundary_controller.layer4_signed_skip", "boundary"),
            config["promoted_owners"])
        self.assertEqual(config["propagation"], runner.base.PROPAGATION_A8_Q13)

    def test_stem_configuration_matches_prefix(self):
        self.assertEqual(
            runner.stem_configuration(candidate(0, 0)), "STRICT_W4A4")
        self.assertEqual(
            runner.stem_configuration(candidate(1, 0)), "STEM_W8A8")

    def test_configured_precision_matches_candidate_union(self):
        selected = candidate(1, 1)
        generic_weights = {
            name: 8 if name in selected.weight_modules else 4
            for name in stable_union(
                prefix.WEIGHT_MODULES_BY_UNIT[unit]
                for unit in prefix.ALL_UNIT_ORDER)
            if name != "conv1_1"
        }
        specs = {}
        boundary_specs = {}
        for owner in stable_union(
                prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
                for unit in prefix.ALL_UNIT_ORDER):
            if owner == ("conv1_1", "input"):
                continue
            spec = SimpleNamespace(
                bits=8 if owner in selected.activation_owners else 4)
            if owner[0].startswith("boundary_controller."):
                boundary_specs[owner] = spec
            else:
                specs[owner] = spec
        stem_contract = {
            "config": "STEM_W8A8",
            "weight_bits": 8,
            "activation_bits": 8,
        }

        runner.validate_configured_precision(
            selected, generic_weights, specs, boundary_specs, stem_contract)

    def test_unrequested_generic_a8_owner_fails(self):
        selected = candidate(0, 0)
        owner = ("layer1.0.conv1", "input")

        with self.assertRaisesRegex(RuntimeError, "activation promotion"):
            runner.validate_configured_precision(
                selected,
                {"layer1.0.conv1": 4},
                {owner: SimpleNamespace(bits=8)},
                {},
                {"config": "STRICT_W4A4", "weight_bits": 4,
                 "activation_bits": 4})


class RegistryAndOperationBasisTest(unittest.TestCase):
    def test_registry_uses_only_exact_executed_candidate_sites(self):
        generic_modules = tuple(
            name for name in stable_union(
                prefix.WEIGHT_MODULES_BY_UNIT[unit]
                for unit in prefix.ALL_UNIT_ORDER)
            if name != "conv1_1")
        generic_owners = tuple(
            owner for owner in stable_union(
                prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
                for unit in prefix.ALL_UNIT_ORDER)
            if owner != ("conv1_1", "input") and
            not owner[0].startswith("boundary_controller."))
        instrumentor = SimpleNamespace(
            modules=dict((name, object()) for name in generic_modules),
            groups=dict((name, "encoder") for name in generic_modules),
            observers=dict(
                ((name, "input"), SimpleNamespace(observed=True))
                for name in generic_modules),
            activation_site_keys=lambda groups: generic_owners,
        )
        boundary_controller = SimpleNamespace(channels={
            "layer4_signed_skip": 64,
        })

        result = runner.candidate_registry_from_context(
            instrumentor, boundary_controller)

        self.assertEqual(
            len(stable_union(result.weights_by_unit.values())), 25)
        self.assertEqual(
            len(stable_union(result.activations_by_unit.values())), 49)

    def test_operation_modules_include_stem_exactly_once(self):
        instrumentor = SimpleNamespace(
            modules={
                "conv1_1": object(),
                "layer1.0.conv1": object(),
                "decoder": object(),
            },
            groups={
                "conv1_1": "encoder",
                "layer1.0.conv1": "encoder",
                "decoder": "decoder",
            },
            observers={
                ("conv1_1", "input"): SimpleNamespace(observed=True),
                ("layer1.0.conv1", "input"): SimpleNamespace(observed=True),
                ("decoder", "input"): SimpleNamespace(observed=True),
            },
        )

        names = runner.operation_module_names(instrumentor)

        self.assertEqual(names[0], "conv1_1")
        self.assertEqual(names.count("conv1_1"), 1)
        self.assertEqual(set(names[1:]), {"layer1.0.conv1", "decoder"})


class AggregationAndOutputTest(unittest.TestCase):
    def test_aggregate_requires_24_by_64_complete_samples(self):
        candidates = prefix.build_candidates(registry())
        rows = []
        for selected in candidates:
            for index in range(64):
                rows.append({
                    "config": selected.name,
                    "sample_index": index,
                    "RMSE": 1.0,
                    "MAE": 0.5,
                    "ABS_REL": 0.1,
                    "IRMSE": 0.2,
                    "flat_RMSE": 0.8,
                    "boundary_RMSE": 1.2,
                })

        aggregate = runner.aggregate_candidate_metrics(rows, candidates)

        self.assertEqual(len(aggregate), 24)
        self.assertEqual(aggregate[-1]["prefix_index"], 5)
        self.assertEqual(aggregate[-1]["tail_index"], 3)
        self.assertEqual(aggregate[-1]["samples"], 64)

    def test_duplicate_sample_identity_fails(self):
        selected = candidate(0, 0)
        rows = [{
            "config": selected.name,
            "sample_index": 0,
            "RMSE": 1.0,
            "MAE": 0.5,
            "ABS_REL": 0.1,
            "IRMSE": 0.2,
            "flat_RMSE": 0.8,
            "boundary_RMSE": 1.2,
        } for _ in range(64)]

        with self.assertRaisesRegex(ValueError, "unique"):
            runner.aggregate_candidate_metrics(rows, (selected,))

    def test_existing_output_directory_fails(self):
        path = Path(__file__).resolve().parent

        with self.assertRaises(FileExistsError):
            runner.validate_output_directory(path)

    def test_metrics_include_cost_and_paired_acceptance(self):
        candidates = prefix.build_candidates(registry())
        rows = []
        for selected in candidates:
            rmse = 1.0 if selected.name == "PREFIX_P0__TAIL_T0" else 0.9
            for index in range(64):
                rows.append({
                    "config": selected.name,
                    "sample_index": index,
                    "RMSE": rmse,
                    "MAE": 0.5,
                    "ABS_REL": 0.1,
                    "IRMSE": 0.2,
                    "flat_RMSE": 0.8,
                    "boundary_RMSE": 1.2,
                })
        aggregate = runner.aggregate_candidate_metrics(rows, candidates)
        weights = tuple({
            "module": name, "weight_elements": 1, "macs": 10,
        } for name in stable_union(
            prefix.WEIGHT_MODULES_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER))
        activations = tuple({
            "module": owner[0], "kind": owner[1], "elements": 1,
        } for owner in stable_union(
            prefix.ACTIVATION_OWNERS_BY_UNIT[unit]
            for unit in prefix.ALL_UNIT_ORDER))

        result = runner.enrich_aggregate_metrics(
            aggregate, rows, candidates, weights, activations)

        self.assertEqual(result[0]["rmse_delta"], 0.0)
        self.assertEqual(result[0]["wins"], 0)
        self.assertEqual(result[1]["wins"], 64)
        self.assertTrue(result[1]["accepted"])
        self.assertGreater(result[-1]["normalized_added_bit_cost"], 0.0)

    def test_operation_basis_rejects_changed_counts(self):
        expected = ({
            "module": "conv1_1", "macs": 10,
            "weight_elements": 2, "input_elements": 3,
        },)
        changed = ({
            "module": "conv1_1", "macs": 11,
            "weight_elements": 2, "input_elements": 3,
        },)

        with self.assertRaisesRegex(RuntimeError, "operation basis"):
            runner.validate_operation_basis(expected, changed)

    def test_candidate_contract_is_explicit(self):
        selected = candidate(2, 3)

        contract = runner.candidate_contract(selected)

        self.assertEqual(contract["prefix_index"], 2)
        self.assertEqual(contract["tail_index"], 3)
        self.assertEqual(contract["encoder_units"], ["stem", "encoder_layer1"])
        self.assertEqual(
            contract["tail_units"], ["decoder_layer4", "initial_depth"])
        self.assertIn("conv1_1", contract["weight_modules"])


class MatrixExecutionTest(unittest.TestCase):
    def test_matrix_runs_every_candidate_in_order_with_stable_bases(self):
        candidates = prefix.build_candidates(registry())
        calls = []
        operation_rows = [{
            "module": "conv1_1", "macs": 10,
            "weight_elements": 2, "input_elements": 3,
            "config": "ignored", "weight_bits": 4,
        }]
        activation_rows = [{
            "module": "conv1_1", "kind": "input", "elements": 4,
        }]

        def evaluator(selected, expected_registry):
            calls.append((selected.name, expected_registry))
            return registry(), {
                "sample_rows": [],
                "region_rows": [],
                "propagation_rows": [],
                "block_rows": [],
                "operation_rows": [dict(operation_rows[0])],
                "layer_rows": [],
                "stem_rows": [],
                "activation_rows": [dict(activation_rows[0])],
            }

        result = runner.run_candidate_matrix(candidates, evaluator)

        self.assertEqual([call[0] for call in calls], [
            selected.name for selected in candidates])
        self.assertIsNone(calls[0][1])
        self.assertEqual(calls[1][1], registry())
        self.assertEqual(len(result["results_by_name"]), 24)
        self.assertEqual(result["operation_basis"][0]["module"], "conv1_1")
        self.assertEqual(result["activation_basis"], activation_rows)

    def test_matrix_rejects_changed_activation_basis(self):
        candidates = prefix.build_candidates(registry())[:2]
        calls = 0

        def evaluator(selected, expected_registry):
            nonlocal calls
            calls += 1
            return registry(), {
                "sample_rows": [],
                "region_rows": [],
                "propagation_rows": [],
                "block_rows": [],
                "operation_rows": [{
                    "module": "conv1_1", "macs": 10,
                    "weight_elements": 2, "input_elements": 3,
                    "config": selected.name, "weight_bits": 4,
                }],
                "layer_rows": [],
                "stem_rows": [],
                "activation_rows": [{
                    "module": "conv1_1", "kind": "input",
                    "elements": 4 + calls,
                }],
            }

        with self.assertRaisesRegex(RuntimeError, "activation basis"):
            runner.run_candidate_matrix(candidates, evaluator)


if __name__ == "__main__":
    unittest.main()
