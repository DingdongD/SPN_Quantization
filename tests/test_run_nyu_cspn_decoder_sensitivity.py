import unittest
from types import SimpleNamespace

from scripts import run_nyu_cspn_decoder_sensitivity as runner
from spn_quant import cspn_sensitivity
from spn_quant.cspn_sensitivity import SensitivityCandidate


def candidate(name="candidate", mode="W8A8"):
    return SensitivityCandidate(
        name=name,
        stage="block",
        block="initial_depth",
        mode=mode,
        weight_modules=("gud_up_proj_layer5.conv1",)
        if mode in ("W8A4", "W8A8") else (),
        activation_owners=(("gud_up_proj_layer5.conv1", "input"),)
        if mode in ("W4A8", "W8A8") else (),
    )


class HardwareConfigurationTest(unittest.TestCase):
    def test_candidate_translates_to_strict_group8_with_explicit_promotions(self):
        config = runner.hardware_configuration(candidate())

        self.assertEqual(config["w_bits"], 4)
        self.assertEqual(config["a_bits"], 4)
        self.assertEqual(config["group_size"], 8)
        self.assertEqual(config["weight_groups"], runner.base.ORDINARY_GROUPS)
        self.assertEqual(
            config["activation_groups"], runner.base.ORDINARY_GROUPS)
        self.assertEqual(
            config["promoted_owners"],
            (("gud_up_proj_layer5.conv1", "input"),))
        self.assertEqual(
            config["weight_bit_overrides"],
            (("gud_up_proj_layer5.conv1", 8),))
        self.assertEqual(config["propagation"], runner.base.PROPAGATION_A8_Q13)

    def test_strict_candidate_has_no_promotions(self):
        strict = SensitivityCandidate(
            "STRICT_W4A4", "baseline", "all", "W4A4", (), ())

        config = runner.hardware_configuration(strict)

        self.assertEqual(config["promoted_owners"], ())
        self.assertEqual(config["weight_bit_overrides"], ())

    def test_registry_is_discovered_only_from_executed_official_sites(self):
        modules = tuple(
            name
            for block in cspn_sensitivity.BLOCK_ORDER
            for name in cspn_sensitivity.WEIGHT_MODULES_BY_BLOCK[block]
        )
        ordinary_owners = tuple(
            owner
            for block in cspn_sensitivity.BLOCK_ORDER
            for owner in cspn_sensitivity.ACTIVATION_OWNERS_BY_BLOCK[block]
            if not owner[0].startswith("rotation.")
        )
        instrumentor = SimpleNamespace(
            modules=dict((name, object()) for name in modules),
            groups=dict((name, "decoder") for name in modules),
            observers=dict(
                ((name, "input"), SimpleNamespace(observed=True))
                for name in modules),
            activation_site_keys=lambda groups: ordinary_owners,
        )
        rotation = SimpleNamespace(channels={
            "decoder_entry": 512,
            "layer4_signed_skip": 64,
        })

        result = runner.candidate_registry_from_context(
            instrumentor, rotation)

        self.assertEqual(
            sum(len(values) for values in result.weights_by_block.values()),
            16)
        self.assertEqual(
            sum(len(values) for values in result.activations_by_block.values()),
            30)

    def test_configured_precision_must_match_candidate_exactly(self):
        selected = candidate()
        specs = {
            ("gud_up_proj_layer5.conv1", "input"):
            SimpleNamespace(bits=8),
            ("other", "input"): SimpleNamespace(bits=4),
        }

        runner.validate_configured_precision(
            selected,
            {"gud_up_proj_layer5.conv1": 8, "other": 4},
            specs, {})

    def test_unrequested_a8_owner_fails(self):
        selected = candidate()
        specs = {
            ("gud_up_proj_layer5.conv1", "input"):
            SimpleNamespace(bits=8),
            ("other", "input"): SimpleNamespace(bits=8),
        }

        with self.assertRaisesRegex(RuntimeError, "activation promotion"):
            runner.validate_configured_precision(
                selected,
                {"gud_up_proj_layer5.conv1": 8, "other": 4},
                specs, {})


class ActivationCostRowsTest(unittest.TestCase):
    def test_calibration_observers_become_unique_per_inference_rows(self):
        ordinary = SimpleNamespace(
            activation_site_keys=lambda groups: (
                ("decoder", "input"), "decoder.relu#0"),
            channel_observers={
                ("decoder", "input"): SimpleNamespace(
                    minimum=[0, 0], scalar_count=300),
            },
            relu_channel_observers={
                "decoder.relu#0": SimpleNamespace(
                    minimum=[0, 0, 0], scalar_count=200),
            },
        )
        rotation = SimpleNamespace(
            observers={
                "decoder_entry": {
                    "identity": SimpleNamespace(scalar_count=400),
                },
            })

        rows = runner.activation_cost_rows(
            ordinary, rotation, calibration_samples=100,
            stem_input_elements=40)

        by_owner = {
            (row["module"], row["kind"]): row["elements"]
            for row in rows
        }
        self.assertEqual(by_owner[("decoder", "input")], 6)
        self.assertEqual(
            by_owner[("decoder.relu#0", "relu_output")], 6)
        self.assertEqual(
            by_owner[("rotation.decoder_entry", "boundary")], 4)
        self.assertEqual(by_owner[("conv1_1", "input")], 40)


class MetricAggregationTest(unittest.TestCase):
    def test_aggregate_requires_exact_64_unique_samples(self):
        rows = [{
            "config": "candidate",
            "sample_index": index,
            "RMSE": 1.0,
            "MAE": 0.5,
            "ABS_REL": 0.1,
            "IRMSE": 0.2,
            "flat_RMSE": 0.8,
            "boundary_RMSE": 1.2,
        } for index in range(64)]

        aggregate = runner.aggregate_candidate_metrics(rows, (candidate(),))

        self.assertEqual(aggregate[0]["samples"], 64)
        self.assertEqual(aggregate[0]["RMSE"], 1.0)

    def test_duplicate_sample_fails(self):
        rows = [{
            "config": "candidate",
            "sample_index": 0,
            "RMSE": 1.0,
            "MAE": 0.5,
            "ABS_REL": 0.1,
            "IRMSE": 0.2,
            "flat_RMSE": 0.8,
            "boundary_RMSE": 1.2,
        } for _ in range(64)]

        with self.assertRaisesRegex(ValueError, "unique"):
            runner.aggregate_candidate_metrics(rows, (candidate(),))


class OrchestrationTest(unittest.TestCase):
    def test_prediction_names_include_strict_best_stages_and_pareto_cumulative(self):
        aggregate = (
            {"config": "STRICT_W4A4", "stage": "baseline", "RMSE": 1.0},
            {"config": "block_a", "stage": "block", "RMSE": 0.9},
            {"config": "block_b", "stage": "block", "RMSE": 0.95},
            {"config": "site_a", "stage": "site", "RMSE": 0.85},
            {"config": "cum_a", "stage": "cumulative", "RMSE": 0.8,
             "normalized_added_bit_cost": 0.2},
        )
        pareto = (
            {"config": "STRICT_W4A4"},
            {"config": "block_a"},
            {"config": "cum_a"},
        )

        names = runner.prediction_candidate_names(aggregate, pareto)

        self.assertEqual(
            names, ("STRICT_W4A4", "block_a", "site_a", "cum_a"))

    def test_prediction_names_allow_empty_cumulative_stage(self):
        aggregate = (
            {"config": "STRICT_W4A4", "stage": "baseline", "RMSE": 1.0},
            {"config": "block_a", "stage": "block", "RMSE": 0.9},
            {"config": "site_a", "stage": "site", "RMSE": 0.85},
        )
        pareto = ({"config": "STRICT_W4A4"},)

        names = runner.prediction_candidate_names(aggregate, pareto)

        self.assertEqual(names, ("STRICT_W4A4", "block_a", "site_a"))


if __name__ == "__main__":
    unittest.main()
