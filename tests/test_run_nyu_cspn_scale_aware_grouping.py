import unittest

import torch
import torch.nn as nn

from scripts import run_nyu_cspn_activation_resolution as base
from scripts import run_nyu_cspn_scale_aware_grouping as runner
from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor


class ScaleAwareRunnerContractTest(unittest.TestCase):
    @staticmethod
    def _calibrated_instrumentor():
        model = nn.Sequential(nn.Conv2d(16, 8, 1, bias=False)).eval()
        instrumentor = HardwareAlignedInstrumentor(
            model, lambda name, module: "encoder")
        values = torch.tensor([
            1.0, 100.0, 2.0, 90.0, 3.0, 80.0, 4.0, 70.0,
            5.0, 60.0, 6.0, 50.0, 7.0, 40.0, 8.0, 30.0,
        ]).reshape(1, 16, 1, 1)
        instrumentor.observe()
        model(values)
        instrumentor.freeze()
        specs = base.build_activation_specs(
            instrumentor, {"encoder"}, bits=4, group_size=8)
        return instrumentor, specs

    def test_method_matrix_is_exact_static_minmax_group8(self):
        methods = runner.build_methods()

        self.assertEqual(
            tuple(method["config"] for method in methods),
            runner.EXPECTED_CONFIGURATIONS)
        self.assertEqual(
            tuple(method["scale_aware"] for method in methods),
            (False, True))
        self.assertTrue(all(method["calibration"] == "minmax"
                            for method in methods))
        self.assertTrue(all(method["group_size"] == 8 for method in methods))
        self.assertTrue(all(method["dynamic"] is False for method in methods))

    def test_groupings_select_only_divisible_conv_inputs(self):
        instrumentor, specs = self._calibrated_instrumentor()

        groupings = runner.build_scale_aware_groupings(
            instrumentor, specs, group_size=8, epsilon=1e-12)

        self.assertEqual(set(groupings), {("0", "input")})
        self.assertEqual(
            groupings[("0", "input")].permutation.tolist(), [
                0, 2, 4, 6, 8, 10, 12, 14,
                15, 13, 11, 9, 7, 5, 3, 1,
            ])
        instrumentor.close()

    def test_grouping_rows_cover_each_channel_and_site(self):
        instrumentor, specs = self._calibrated_instrumentor()
        groupings = runner.build_scale_aware_groupings(
            instrumentor, specs, group_size=8, epsilon=1e-12)

        channel_rows, summary_rows = runner.grouping_rows(groupings)

        self.assertEqual(len(channel_rows), 16)
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(
            {row["group_size"] for row in channel_rows}, {8})
        self.assertEqual(
            {row["module"] for row in channel_rows}, {"0"})
        self.assertLess(
            summary_rows[0]["scale_aware_dispersion_mean"],
            summary_rows[0]["contiguous_dispersion_mean"])
        instrumentor.close()

    def test_configurations_change_only_input_permutation(self):
        instrumentor, specs = self._calibrated_instrumentor()
        groupings = runner.build_scale_aware_groupings(
            instrumentor, specs, group_size=8, epsilon=1e-12)

        configs = runner.build_configurations(groupings)

        self.assertEqual(configs[0]["activation_permutations"], ())
        self.assertEqual(
            set(dict(configs[1]["activation_permutations"])),
            {("0", "input")})
        for field in (
                "weight_groups", "activation_groups", "propagation",
                "group_size", "quantize_bias", "rotation_range_overrides"):
            self.assertEqual(configs[0][field], configs[1][field])
        instrumentor.close()

    def test_runtime_contract_requires_fixed_sample_counts(self):
        runner.validate_runtime_contract(
            calibration_samples=128, evaluation_indices=tuple(range(64)))

        with self.assertRaisesRegex(ValueError, "calibration"):
            runner.validate_runtime_contract(127, tuple(range(64)))
        with self.assertRaisesRegex(ValueError, "evaluation"):
            runner.validate_runtime_contract(128, tuple(range(63)))


if __name__ == "__main__":
    unittest.main()
