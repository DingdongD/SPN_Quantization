import unittest

import torch
import torch.nn as nn

from scripts import run_nyu_cspn_activation_resolution as runner
from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor


class AttributionConfigurationTest(unittest.TestCase):
    def test_quantized_configs_share_propagation_contract(self):
        configs = runner.build_attribution_configurations()
        quantized = [row for row in configs if row["name"] != "FP32"]

        contracts = {
            tuple(sorted(row["propagation"].items()))
            for row in quantized
        }

        self.assertEqual(
            contracts, {tuple(sorted(runner.PROPAGATION_A8_Q13.items()))})
        self.assertEqual(
            [row["name"] for row in configs],
            ["FP32", "PA_ONLY", "W4_ONLY", "A4_ONLY", "W4A4_RTN"])

    def test_weight_and_activation_ownership_is_disjoint_when_required(self):
        configs = {
            row["name"]: row
            for row in runner.build_attribution_configurations()
        }

        self.assertNotEqual(configs["W4_ONLY"]["weight_groups"], set())
        self.assertEqual(configs["W4_ONLY"]["activation_groups"], set())
        self.assertEqual(configs["A4_ONLY"]["weight_groups"], set())
        self.assertNotEqual(configs["A4_ONLY"]["activation_groups"], set())
        self.assertEqual(
            configs["W4A4_RTN"]["weight_groups"],
            configs["W4A4_RTN"]["activation_groups"])

    def test_guidance_group_is_never_quantized(self):
        for config in runner.build_attribution_configurations():
            self.assertNotIn("propagation_head", config["weight_groups"])
            self.assertNotIn("propagation_head", config["activation_groups"])

    def test_interaction_delta_uses_pa_only_baseline(self):
        values = {
            "PA_ONLY": 1.0,
            "W4_ONLY": 1.2,
            "A4_ONLY": 1.5,
            "W4A4_RTN": 2.0,
        }

        self.assertAlmostEqual(runner.attribution_interaction(values), 0.3)

    def test_attribution_components_include_propagation_weight_activation(self):
        values = {
            "FP32": 0.1,
            "PA_ONLY": 0.2,
            "W4_ONLY": 0.4,
            "A4_ONLY": 0.7,
            "W4A4_RTN": 1.1,
        }

        components = runner.attribution_components(values)

        self.assertAlmostEqual(components["propagation"], 0.1)
        self.assertAlmostEqual(components["weight"], 0.2)
        self.assertAlmostEqual(components["activation"], 0.5)
        self.assertAlmostEqual(components["interaction"], 0.2)


class CalibrationSelectionTest(unittest.TestCase):
    def test_sensitive_sites_are_selected_from_calibration_only(self):
        rows = [
            {
                "split": "calibration", "site": "a",
                "zero_collapse_error_energy": 2.0,
            },
            {
                "split": "evaluation", "site": "b",
                "zero_collapse_error_energy": 100.0,
            },
        ]

        self.assertEqual(runner.select_candidate_sites(rows, 1), ("a",))

    def test_candidate_owner_deduplication_preserves_original_site(self):
        rows = [
            {"site": "relu#0", "module": "relu", "kind": "relu_output"},
            {"site": "relu#1", "module": "relu", "kind": "relu_output"},
            {"site": "conv#0", "module": "conv", "kind": "output"},
        ]

        selected = runner._candidate_owner_sites(
            rows, ("relu#0", "relu#1", "conv#0"))

        self.assertEqual(selected, (
            ("relu#0", ("relu", "relu_output")),
            ("conv#0", ("conv", "output")),
        ))

    def test_group_selection_uses_calibration_mse_then_sqnr(self):
        rows = [
            {
                "split": "calibration", "config": "g16",
                "block_output_mse": 1.0, "block_output_sqnr": 3.0,
            },
            {
                "split": "calibration", "config": "g8",
                "block_output_mse": 1.0, "block_output_sqnr": 4.0,
            },
            {
                "split": "evaluation", "config": "leak",
                "block_output_mse": 0.0, "block_output_sqnr": 100.0,
            },
        ]

        selected = runner.select_calibration_configuration(rows)

        self.assertEqual(selected["config"], "g8")

    def test_group_size_applies_only_to_divisible_sites(self):
        self.assertEqual(
            runner.site_granularity(channels=64, group_size=16), "group")
        self.assertEqual(
            runner.site_granularity(channels=40, group_size=16), "tensor")
        self.assertEqual(
            runner.site_granularity(channels=1, group_size=1), "channel")

    def test_group_config_names_declare_hybrid_tensor_sites(self):
        self.assertEqual(
            [row["name"] for row in runner.build_group_configurations()][:-1],
            ["W4A4_HYBRID_GROUP128", "W4A4_HYBRID_GROUP64",
             "W4A4_HYBRID_GROUP32", "W4A4_HYBRID_GROUP16",
             "W4A4_HYBRID_GROUP8"])

    def test_owner_maps_to_direct_downstream_block(self):
        self.assertEqual(
            runner.owner_block(("gud_up_proj_layer4.sc_conv1", "output")),
            "decoder_layer4")
        self.assertEqual(
            runner.owner_block(("rotation.layer4_signed_skip", "boundary")),
            "decoder_layer4")
        self.assertEqual(
            runner.owner_block(("layer3.1.relu#0", "relu_output")),
            "encoder_layer3")

    def test_empty_calibration_rows_fail_directly(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            runner.select_calibration_configuration([
                {
                    "split": "evaluation", "config": "g8",
                    "block_output_mse": 1.0,
                    "block_output_sqnr": 2.0,
                },
            ])

    def test_scale_selection_accepts_bounded_clipping(self):
        rows = [
            {
                "split": "calibration", "factor": 1.0,
                "block_output_mse": 2.0,
                "clipping_error_ratio": 0.0,
            },
            {
                "split": "calibration", "factor": 0.75,
                "block_output_mse": 1.0,
                "clipping_error_ratio": 0.2,
            },
        ]

        self.assertEqual(runner.select_activation_scale(rows)["factor"], 0.75)

    def test_scale_selection_rejects_clipping_dominated_candidate(self):
        rows = [
            {
                "split": "calibration", "factor": 1.0,
                "block_output_mse": 2.0,
                "clipping_error_ratio": 0.0,
            },
            {
                "split": "calibration", "factor": 0.5,
                "block_output_mse": 0.5,
                "clipping_error_ratio": 0.6,
            },
            {
                "split": "evaluation", "factor": 0.75,
                "block_output_mse": 0.1,
                "clipping_error_ratio": 0.0,
            },
        ]

        self.assertEqual(runner.select_activation_scale(rows)["factor"], 1.0)

    def test_decoder_merge_sites_ignore_encoder_and_depth_head(self):
        owners = (
            ("layer4.1.conv2", "output"),
            ("gud_up_proj_layer3.sc_conv1", "output"),
            ("gud_up_proj_layer5.conv1", "output"),
        )

        self.assertEqual(
            runner.decoder_merge_sites(owners),
            ("gud_up_proj_layer3::add#0",))

    def test_merge_extensions_share_one_calibration_base(self):
        base = runner._configuration(
            "W4A4_CHANNEL", runner.ORDINARY_GROUPS,
            runner.ORDINARY_GROUPS, runner.PROPAGATION_A8_Q13,
            granularity="channel", group_size=1)

        shared, residual = runner.build_merge_configurations(base)

        self.assertEqual(shared["name"], "W4A4_MERGE_SHARED")
        self.assertEqual(shared["merge_policy"], "shared")
        self.assertEqual(residual["name"], "W4A4_RESIDUAL")
        self.assertEqual(residual["merge_policy"], "residual")
        for field in (
                "weight_groups", "activation_groups", "granularity",
                "group_size", "propagation"):
            self.assertEqual(shared[field], residual[field])

    def test_merge_policy_enables_only_selected_adapter(self):
        class Adapter(object):
            def __init__(self):
                self.mode = "stale"
                self.resets = 0

            def disable(self):
                self.mode = "bypass"

            def reset_statistics(self):
                self.resets += 1

            def quantize(self):
                self.mode = "quantize"

        adapters = {"shared": Adapter(), "residual": Adapter()}
        config = runner._configuration(
            "W4A4_RESIDUAL", runner.ORDINARY_GROUPS,
            runner.ORDINARY_GROUPS, runner.PROPAGATION_A8_Q13,
            merge_policy="residual")

        active = runner.configure_merge_adapters(config, adapters)

        self.assertIs(active, adapters["residual"])
        self.assertEqual(adapters["shared"].mode, "bypass")
        self.assertEqual(adapters["shared"].resets, 0)
        self.assertEqual(adapters["residual"].mode, "quantize")
        self.assertEqual(adapters["residual"].resets, 1)

    def test_residual_adapter_keeps_unselected_structure_shared(self):
        adapters = runner.build_merge_adapters(
            nn.Identity(), ("decoder::add#0",))

        self.assertEqual(adapters["shared"].policy, "shared")
        self.assertEqual(adapters["residual"].policy, "shared")
        self.assertEqual(adapters["residual"].site_policies, {
            "decoder::add#0": "residual",
        })
        for policy in reversed(tuple(adapters)):
            adapters[policy].close()

    def test_evaluation_rejects_non_transferring_calibrated_scale(self):
        rows = [
            {"config": "W4A4_CHANNEL", "RMSE": 0.28},
            {"config": "W4A4_CALIBRATED_SCALE", "RMSE": 0.31},
        ]

        selected = runner.select_transferred_configuration(
            rows, "W4A4_CHANNEL", "W4A4_CALIBRATED_SCALE")

        self.assertEqual(selected["config"], "W4A4_CHANNEL")

    def test_evaluation_accepts_transferring_calibrated_scale(self):
        rows = [
            {"config": "W4A4_CHANNEL", "RMSE": 0.28},
            {"config": "W4A4_CALIBRATED_SCALE", "RMSE": 0.26},
        ]

        selected = runner.select_transferred_configuration(
            rows, "W4A4_CHANNEL", "W4A4_CALIBRATED_SCALE")

        self.assertEqual(
            selected["config"], "W4A4_CALIBRATED_SCALE")

    def test_prediction_configs_include_calibrated_scale_base(self):
        names = runner.prediction_configuration_names(
            "W4A4_CHANNEL", "W4A4_CALIBRATED_SCALE",
            "W4A4_SELECTIVE_W4A4_CHANNEL",
            ("W4A4_CALIBRATED_SCALE",))

        self.assertEqual(names, {
            "FP32", "W4A4_RTN", "W4A4_CHANNEL",
            "W4A4_SELECTIVE_W4A4_CHANNEL", "W4A4_CALIBRATED_SCALE",
        })


class ActivationSpecBuilderTest(unittest.TestCase):
    @staticmethod
    def _instrumentor():
        model = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False)).eval()
        instrumentor = HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        instrumentor.observe()
        model(torch.tensor(
            [[[[1.0]], [[2.0]], [[10.0]], [[20.0]]]]))
        instrumentor.freeze()
        return instrumentor

    def test_nondivisible_group_spec_remains_tensor(self):
        instrumentor = self._instrumentor()

        specs = runner.build_activation_specs(
            instrumentor, {"encoder"}, bits=4, group_size=16)

        self.assertTrue(all(
            spec.granularity == "tensor" for spec in specs.values()))
        instrumentor.close()

    def test_selective_group_and_a8_promotion_are_owner_scoped(self):
        instrumentor = self._instrumentor()

        specs = runner.build_activation_specs(
            instrumentor, {"encoder"}, bits=4, group_size=2,
            selected_owners=(("0", "input"),),
            promoted_owners=(("0", "output"),))

        self.assertEqual(specs[("0", "input")].granularity, "group")
        self.assertEqual(specs[("0", "input")].bits, 4)
        self.assertEqual(specs[("0", "output")].granularity, "tensor")
        self.assertEqual(specs[("0", "output")].bits, 8)
        instrumentor.close()

    def test_activation_maxima_follow_group_ranges_and_owner_factor(self):
        instrumentor = self._instrumentor()
        specs = runner.build_activation_specs(
            instrumentor, {"encoder"}, bits=4, group_size=2)

        maxima = runner.build_activation_maxima(
            instrumentor, specs,
            scale_factors=((('0', 'input'), 0.5),))

        torch.testing.assert_close(
            maxima[("0", "input")], torch.tensor([1.0, 10.0]))
        self.assertNotIn(("0", "output"), maxima)
        instrumentor.close()

    def test_strict_cspn_ownership_matches_official_rotation_path(self):
        self.assertEqual(runner.strict_owned_outputs(), {
            "conv1_1", "conv2", "gud_up_proj_layer5.conv1"})
        self.assertEqual(runner.strict_owned_inputs(), {
            "gud_up_proj_layer1.conv1",
            "gud_up_proj_layer1.sc_conv1",
            "gud_up_proj_layer4.conv1_1",
        })

    def test_rotation_group_sizes_are_declared_per_divisible_boundary(self):
        class Rotation(object):
            channels = {
                "decoder_entry": 512,
                "layer4_signed_skip": 64,
            }

        self.assertEqual(
            runner.build_rotation_group_sizes(Rotation(), 128),
            {"decoder_entry": 128, "layer4_signed_skip": None})
        self.assertEqual(
            runner.build_rotation_group_sizes(Rotation(), 16),
            {"decoder_entry": 16, "layer4_signed_skip": 16})
        self.assertEqual(
            runner.build_rotation_group_sizes(Rotation(), 1),
            {"decoder_entry": 1, "layer4_signed_skip": 1})

    def test_rotation_specs_follow_boundary_group_declarations(self):
        class Rotation(object):
            channels = {
                "decoder_entry": 512,
                "layer4_signed_skip": 64,
            }

        specs = runner.build_rotation_activation_specs(
            Rotation(), bits=4, group_size=128)

        self.assertEqual(
            specs[("rotation.decoder_entry", "boundary")].granularity,
            "group")
        self.assertEqual(
            specs[("rotation.layer4_signed_skip", "boundary")].granularity,
            "tensor")

    def test_rotation_specs_apply_owner_scoped_a8_promotion(self):
        class Rotation(object):
            channels = {
                "decoder_entry": 512,
                "layer4_signed_skip": 64,
            }

        specs = runner.build_rotation_activation_specs(
            Rotation(), bits=4, group_size=16,
            promoted_owners=(("rotation.decoder_entry", "boundary"),))

        self.assertEqual(
            specs[("rotation.decoder_entry", "boundary")].bits, 8)
        self.assertEqual(
            specs[("rotation.layer4_signed_skip", "boundary")].bits, 4)

    def test_rotation_boundary_rows_use_declared_activation_specs(self):
        config = runner._configuration(
            "W4A4_GROUP128", runner.ORDINARY_GROUPS,
            runner.ORDINARY_GROUPS, runner.PROPAGATION_A8_Q13,
            granularity="group", group_size=128)
        rotation_specs = {
            ("rotation.decoder_entry", "boundary"): runner.QuantSpec(
                bits=4, scheme="symmetric", granularity="group",
                axis=1, group_size=128, signed=True,
                preserve_zero=False),
        }

        rows = runner._annotate_activation_rows(
            [{
                "module": "rotation.decoder_entry",
                "kind": "boundary",
            }], config, {}, rotation_specs)

        self.assertEqual(rows[0]["bits"], 4)
        self.assertEqual(rows[0]["granularity"], "group")
        self.assertEqual(rows[0]["group_size"], 128)


class QuantizedConfigurationTest(unittest.TestCase):
    class Rotation(object):
        channels = {
            "decoder_entry": 512,
            "layer4_signed_skip": 64,
        }

        def __init__(self):
            self.disabled = 0
            self.calls = []

        def disable(self):
            self.disabled += 1

        def configure_specs(
                self, methods, bit_widths, group_sizes, scale_factors,
                quantize, absorb_weights):
            self.calls.append({
                "methods": methods,
                "bit_widths": bit_widths,
                "group_sizes": group_sizes,
                "scale_factors": scale_factors,
                "quantize": quantize,
                "absorb_weights": absorb_weights,
            })

    class Propagation(object):
        def __init__(self):
            self.disabled = 0
            self.config = None

        def disable(self):
            self.disabled += 1

        def configure(self, config):
            self.config = config

    def test_w4a4_configures_identity_owned_boundaries(self):
        instrumentor = ActivationSpecBuilderTest._instrumentor()
        rotation = self.Rotation()
        propagation = self.Propagation()
        config = runner._configuration(
            "W4A4_GROUP128", {"encoder"}, {"encoder"},
            runner.PROPAGATION_A8_Q13,
            granularity="group", group_size=128)

        runner._configure_quantized(
            config, instrumentor, rotation, propagation, {})

        self.assertEqual(rotation.disabled, 1)
        self.assertEqual(rotation.calls, [{
            "methods": {
                "decoder_entry": "identity",
                "layer4_signed_skip": "identity",
            },
            "bit_widths": {
                "decoder_entry": 4,
                "layer4_signed_skip": 4,
            },
            "group_sizes": {
                "decoder_entry": 128,
                "layer4_signed_skip": None,
            },
            "scale_factors": {
                "decoder_entry": 1.0,
                "layer4_signed_skip": 1.0,
            },
            "quantize": True,
            "absorb_weights": False,
        }])
        instrumentor.close()

    def test_fp32_disables_rotation_and_propagation(self):
        instrumentor = ActivationSpecBuilderTest._instrumentor()
        rotation = self.Rotation()
        propagation = self.Propagation()

        runner._configure_quantized(
            runner.build_attribution_configurations()[0],
            instrumentor, rotation, propagation, {})

        self.assertEqual(rotation.disabled, 1)
        self.assertEqual(rotation.calls, [])
        self.assertEqual(propagation.disabled, 1)
        instrumentor.close()

    def test_manifest_counts_rotation_sites_only_when_activation_is_quantized(self):
        instrumentor = ActivationSpecBuilderTest._instrumentor()
        rotation = self.Rotation()
        configurations = (
            runner._configuration("FP32", set(), set(), None),
            runner._configuration(
                "PA_ONLY", set(), set(), runner.PROPAGATION_A8_Q13),
            runner._configuration(
                "W4_ONLY", {"encoder"}, set(),
                runner.PROPAGATION_A8_Q13),
            runner._configuration(
                "A4_ONLY", set(), {"encoder"},
                runner.PROPAGATION_A8_Q13),
            runner._configuration(
                "W4A4_RTN", {"encoder"}, {"encoder"},
                runner.PROPAGATION_A8_Q13),
        )

        activation_rows = []
        for config in ("A4_ONLY", "W4A4_RTN"):
            for index in range(4):
                activation_rows.append({
                    "config": config,
                    "split": "calibration",
                    "site": "%s_%d" % (config, index),
                    "elements": 10,
                    "granularity": "tensor",
                })
        rows = runner._configuration_manifest(
            configurations, instrumentor, rotation, activation_rows)
        by_name = dict((row["config"], row) for row in rows)

        self.assertEqual(by_name["FP32"]["activation_sites"], 0)
        self.assertEqual(by_name["PA_ONLY"]["activation_sites"], 0)
        self.assertEqual(by_name["W4_ONLY"]["activation_sites"], 0)
        self.assertEqual(by_name["A4_ONLY"]["activation_sites"], 4)
        self.assertEqual(by_name["W4A4_RTN"]["activation_sites"], 4)
        self.assertEqual(
            by_name["W4A4_RTN"]["tensor_element_fraction"], 1.0)
        instrumentor.close()

    def test_activation_element_fractions_are_workload_weighted(self):
        rows = [
            {
                "config": "mixed", "split": "calibration",
                "site": "tensor", "elements": 10,
                "granularity": "tensor",
            },
            {
                "config": "mixed", "split": "calibration",
                "site": "group", "elements": 30,
                "granularity": "group",
            },
            {
                "config": "mixed", "split": "calibration",
                "site": "channel", "elements": 60,
                "granularity": "channel",
            },
        ]

        summary = runner.activation_granularity_summary(rows, "mixed")

        self.assertEqual(summary["activation_elements"], 100)
        self.assertAlmostEqual(summary["tensor_element_fraction"], 0.1)
        self.assertAlmostEqual(summary["group_element_fraction"], 0.3)
        self.assertAlmostEqual(summary["channel_element_fraction"], 0.6)


class BlockErrorAccumulatorTest(unittest.TestCase):
    def test_block_rows_and_aggregate_use_element_weighted_error(self):
        sites = (
            runner.BlockSite("small", "small"),
            runner.BlockSite("large", "large"),
        )
        accumulator = runner.BlockErrorAccumulator(sites)
        accumulator.update(
            {
                "small": torch.ones(1),
                "large": torch.ones(3),
            },
            {
                "small": torch.zeros(1),
                "large": torch.ones(3) * 3.0,
            })

        rows = {row["block"]: row for row in accumulator.rows()}
        aggregate = accumulator.aggregate()

        self.assertEqual(rows["small"]["block_output_mse"], 1.0)
        self.assertEqual(rows["large"]["block_output_mse"], 4.0)
        self.assertEqual(aggregate["block_output_mse"], 13.0 / 4.0)


class OutputCoverageTest(unittest.TestCase):
    def test_official_cspn_site_contract_is_explicit(self):
        class Instrumentor(object):
            def activation_site_keys(self, groups):
                self.groups = groups
                return tuple(runner.STRICT_ACTIVATION_OWNERS)

        class Rotation(object):
            channels = {
                "decoder_entry": 512,
                "layer4_signed_skip": 64,
            }

        instrumentor = Instrumentor()

        runner.validate_strict_site_contract(instrumentor, Rotation())

        self.assertEqual(instrumentor.groups, runner.ORDINARY_GROUPS)

    def test_official_cspn_site_contract_rejects_replaced_site(self):
        class Instrumentor(object):
            def activation_site_keys(self, groups):
                owners = set(runner.STRICT_ACTIVATION_OWNERS)
                owners.remove(("conv1_1", "input"))
                owners.add(("gud_up_proj_layer6", "output"))
                return tuple(owners)

        class Rotation(object):
            channels = {
                "decoder_entry": 512,
                "layer4_signed_skip": 64,
            }

        with self.assertRaisesRegex(RuntimeError, "missing"):
            runner.validate_strict_site_contract(Instrumentor(), Rotation())

    def test_sample_coverage_requires_every_fixed_index_once(self):
        rows = [
            {"config": "FP32", "sample_index": 3},
            {"config": "FP32", "sample_index": 5},
            {"config": "W4A4_RTN", "sample_index": 3},
            {"config": "W4A4_RTN", "sample_index": 5},
        ]

        runner.validate_sample_coverage(
            rows, ("FP32", "W4A4_RTN"), (3, 5))

    def test_sample_coverage_rejects_missing_index(self):
        rows = [
            {"config": "FP32", "sample_index": 3},
            {"config": "W4A4_RTN", "sample_index": 3},
            {"config": "W4A4_RTN", "sample_index": 5},
        ]

        with self.assertRaisesRegex(ValueError, "coverage"):
            runner.validate_sample_coverage(
                rows, ("FP32", "W4A4_RTN"), (3, 5))


if __name__ == "__main__":
    unittest.main()
