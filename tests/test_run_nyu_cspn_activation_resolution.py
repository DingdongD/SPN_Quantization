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

    def test_empty_calibration_rows_fail_directly(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            runner.select_calibration_configuration([
                {
                    "split": "evaluation", "config": "g8",
                    "block_output_mse": 1.0,
                    "block_output_sqnr": 2.0,
                },
            ])


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
