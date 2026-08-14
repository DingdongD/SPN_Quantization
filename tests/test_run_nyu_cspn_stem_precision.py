import unittest
from types import SimpleNamespace

import torch

from scripts import run_nyu_cspn_stem_precision as runner


def calibration_payload(indices=None):
    values = list(range(128)) if indices is None else list(indices)
    return {
        "indices": values,
        "count": len(values),
        "selection": "stratified",
    }


def evaluation_payload(indices=None):
    values = list(range(64)) if indices is None else list(indices)
    return {
        "evaluation_indices": values,
        "evaluation_samples": len(values),
        "seed": 20260812,
    }


class StemConfigurationTest(unittest.TestCase):
    def test_configuration_order_is_fixed(self):
        configurations = runner.build_configurations()

        self.assertEqual(
            [config.name for config in configurations],
            ["STRICT_W4A4", "STEM_W8A8",
             "STEM_FP16", "STEM_BRANCH_A4"])

    def test_only_w8a8_promotes_stem_relu_and_skip(self):
        by_name = {
            config.name: config for config in runner.build_configurations()
        }

        self.assertEqual(by_name["STEM_W8A8"].promoted_owners, (
            ("relu#0", "relu_output"),
            ("rotation.layer4_signed_skip", "boundary"),
        ))
        for name in ("STRICT_W4A4", "STEM_FP16", "STEM_BRANCH_A4"):
            self.assertEqual(by_name[name].promoted_owners, ())

    def test_stem_input_and_output_are_owned_once(self):
        ownership = runner.stem_ownership()

        self.assertIn("conv1_1", ownership.inputs)
        self.assertIn("conv1_1", ownership.outputs)
        self.assertEqual(len(ownership.inputs), len(set(ownership.inputs)))
        self.assertEqual(len(ownership.outputs), len(set(ownership.outputs)))

    def test_hardware_configuration_keeps_nonstem_group8_w4a4(self):
        for config in runner.build_configurations():
            hardware = runner.hardware_configuration(config)

            self.assertEqual(hardware["w_bits"], 4)
            self.assertEqual(hardware["a_bits"], 4)
            self.assertEqual(hardware["group_size"], 8)
            self.assertEqual(
                hardware["weight_groups"], runner.base.ORDINARY_GROUPS)
            self.assertEqual(
                hardware["activation_groups"], runner.base.ORDINARY_GROUPS)
            self.assertEqual(
                hardware["promoted_owners"], config.promoted_owners)

    def test_checkpoint_build_does_not_require_imagenet_pretraining(self):
        saved_args = SimpleNamespace(from_scratch=False)

        configured = runner.configure_checkpoint_build(saved_args)

        self.assertIs(configured, saved_args)
        self.assertTrue(configured.from_scratch)


class IndexProtocolTest(unittest.TestCase):
    def test_exact_128_train_and_64_validation_indices_are_loaded(self):
        protocol = runner.index_protocol(
            calibration_payload(), evaluation_payload())

        self.assertEqual(len(protocol.calibration_indices), 128)
        self.assertEqual(len(protocol.evaluation_indices), 64)
        self.assertEqual(protocol.seed, 20260812)

    def test_duplicate_calibration_index_fails(self):
        indices = list(range(127)) + [0]

        with self.assertRaisesRegex(ValueError, "unique"):
            runner.index_protocol(
                calibration_payload(indices), evaluation_payload())

    def test_wrong_calibration_count_fails(self):
        with self.assertRaisesRegex(ValueError, "128"):
            runner.index_protocol(
                calibration_payload(range(127)), evaluation_payload())

    def test_wrong_evaluation_count_fails(self):
        with self.assertRaisesRegex(ValueError, "64"):
            runner.index_protocol(
                calibration_payload(), evaluation_payload(range(63)))

    def test_negative_index_fails(self):
        indices = list(range(127)) + [-1]

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            runner.index_protocol(
                calibration_payload(indices), evaluation_payload())

    def test_same_numeric_index_across_train_and_validation_is_valid(self):
        protocol = runner.index_protocol(
            calibration_payload(), evaluation_payload())

        self.assertEqual(protocol.calibration_identities[0], ("train", 0))
        self.assertEqual(protocol.evaluation_identities[0], ("validation", 0))
        self.assertNotEqual(
            protocol.calibration_identities[0],
            protocol.evaluation_identities[0])

    def test_declared_count_must_match_index_list(self):
        payload = calibration_payload()
        payload["count"] = 127

        with self.assertRaisesRegex(ValueError, "declared"):
            runner.index_protocol(payload, evaluation_payload())


class AcceptanceTest(unittest.TestCase):
    def test_candidate_requires_lower_rmse_and_majority_wins(self):
        baseline = [
            {"sample_index": index, "RMSE": 1.0}
            for index in range(64)
        ]
        candidate = [
            {"sample_index": index, "RMSE": 0.9 if index < 33 else 1.0}
            for index in range(64)
        ]

        result = runner.acceptance(baseline, candidate)

        self.assertEqual(result["wins"], 33)
        self.assertTrue(result["accepted"])

    def test_lower_mean_without_majority_is_not_accepted(self):
        baseline = [
            {"sample_index": index, "RMSE": 1.0}
            for index in range(64)
        ]
        candidate = [
            {"sample_index": index,
             "RMSE": 0.0 if index < 32 else 1.5}
            for index in range(64)
        ]

        result = runner.acceptance(baseline, candidate)

        self.assertLess(result["candidate_rmse"], result["baseline_rmse"])
        self.assertEqual(result["wins"], 32)
        self.assertFalse(result["accepted"])


class MetricAggregationTest(unittest.TestCase):
    def test_aggregate_metrics_require_64_rows_per_configuration(self):
        rows = []
        for config_index, config in enumerate(runner.build_configurations()):
            for sample_index in range(64):
                rows.append({
                    "config": config.name,
                    "sample_index": sample_index,
                    "RMSE": 1.0 + config_index,
                    "MAE": 0.5 + config_index,
                    "ABS_REL": 0.1 + config_index,
                    "IRMSE": 0.2 + config_index,
                    "flat_RMSE": 0.8 + config_index,
                    "boundary_RMSE": 1.2 + config_index,
                })

        aggregate = runner.aggregate_metrics(
            rows, runner.build_configurations())

        self.assertEqual(len(aggregate), 4)
        self.assertEqual(aggregate[0]["samples"], 64)
        self.assertAlmostEqual(aggregate[2]["RMSE"], 3.0)

    def test_missing_sample_row_fails(self):
        rows = [{
            "config": "STRICT_W4A4",
            "sample_index": index,
            "RMSE": 1.0,
            "MAE": 1.0,
            "ABS_REL": 1.0,
            "IRMSE": 1.0,
            "flat_RMSE": 1.0,
            "boundary_RMSE": 1.0,
        } for index in range(63)]

        with self.assertRaisesRegex(ValueError, "64"):
            runner.aggregate_metrics(
                rows, (runner.build_configurations()[0],))

    def test_stem_statistics_are_partitioned_by_signal_domain(self):
        rows = [
            {"signal": "rgb_input"},
            {"signal": "depth_input"},
            {"signal": "rgb_partial"},
            {"signal": "depth_partial"},
            {"signal": "stem_output"},
        ]

        activation, partial = runner.partition_stem_statistics(rows)

        self.assertEqual(
            [row["signal"] for row in activation],
            ["rgb_input", "depth_input"])
        self.assertEqual(
            [row["signal"] for row in partial],
            ["rgb_partial", "depth_partial", "stem_output"])


class PrecisionCoverageTest(unittest.TestCase):
    def setUp(self):
        self.operations = (
            {
                "module": "conv1_1", "macs": 100,
                "weight_elements": 20, "input_elements": 40,
            },
            {
                "module": "layer1.0.conv1", "macs": 900,
                "weight_elements": 180, "input_elements": 360,
            },
        )

    def test_w8a8_reports_only_stem_as_promoted(self):
        config = runner.build_configurations()[1]

        rows = runner.precision_coverage(self.operations, config)
        by_format = {row["format"]: row for row in rows}

        self.assertAlmostEqual(by_format["W8A8"]["mac_fraction"], 0.1)
        self.assertAlmostEqual(by_format["W4A4"]["mac_fraction"], 0.9)
        self.assertEqual(by_format["W8A8"]["weight_elements"], 20)

    def test_fp16_reports_only_stem_as_float(self):
        config = runner.build_configurations()[2]

        rows = runner.precision_coverage(self.operations, config)
        by_format = {row["format"]: row for row in rows}

        self.assertAlmostEqual(by_format["FP16"]["mac_fraction"], 0.1)
        self.assertAlmostEqual(by_format["W4A4"]["mac_fraction"], 0.9)

    def test_strict_and_branch_a4_remain_all_w4a4(self):
        for config in (runner.build_configurations()[0],
                       runner.build_configurations()[3]):
            rows = runner.precision_coverage(self.operations, config)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["format"], "W4A4")
            self.assertAlmostEqual(rows[0]["mac_fraction"], 1.0)

    def test_operation_rows_must_contain_stem(self):
        with self.assertRaisesRegex(ValueError, "conv1_1"):
            runner.precision_coverage(self.operations[1:],
                                      runner.build_configurations()[0])

    def test_conv_operation_counter_records_first_stable_shape(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1_1 = torch.nn.Conv2d(4, 2, 3, padding=1)
                self.other = torch.nn.Conv2d(2, 3, 1)

            def forward(self, tensor):
                return self.other(self.conv1_1(tensor))

        model = Model().eval()
        counter = runner.ConvOperationCounter(
            model, ("conv1_1", "other"))
        model(torch.ones(1, 4, 5, 6))
        first = counter.rows()
        model(torch.ones(1, 4, 5, 6))
        second = counter.rows()
        counter.close()

        self.assertEqual(first, second)
        by_name = {row["module"]: row for row in first}
        self.assertEqual(by_name["conv1_1"]["input_elements"], 120)
        self.assertEqual(by_name["conv1_1"]["weight_elements"], 72)
        self.assertEqual(by_name["conv1_1"]["macs"], 2160)
        self.assertEqual(by_name["other"]["macs"], 180)

    def test_operation_modules_exclude_unexecuted_legacy_layers(self):
        instrumentor = SimpleNamespace(
            modules={"conv1_1": object(), "legacy": object()},
            groups={"conv1_1": "encoder", "legacy": "decoder"},
            observers={
                ("conv1_1", "input"): SimpleNamespace(observed=True),
                ("legacy", "input"): SimpleNamespace(observed=False),
            },
        )

        selected = runner.executed_operation_modules(instrumentor)

        self.assertEqual(selected, ("conv1_1",))


class PairedBoundaryCaptureTest(unittest.TestCase):
    def test_output_capture_requires_one_new_tensor_per_take(self):
        module = torch.nn.ReLU()
        capture = runner.TensorCapture.output(module)
        expected = module(torch.tensor([-1.0, 2.0]))

        value = capture.take()

        torch.testing.assert_close(value, expected)
        with self.assertRaisesRegex(RuntimeError, "capture"):
            capture.take()
        capture.close()

    def test_argument_capture_selects_skip_argument(self):
        class Merge(torch.nn.Module):
            def forward(self, main, skip):
                return main + skip

        module = Merge()
        capture = runner.TensorCapture.argument(module, 1)
        skip = torch.tensor([3.0])
        module(torch.tensor([1.0]), skip)

        value = capture.take()

        torch.testing.assert_close(value, skip)
        capture.close()

    def test_pair_error_accumulator_reports_finite_mse_and_sqnr(self):
        accumulator = runner.PairErrorAccumulator("skip4_input")
        accumulator.update(
            torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))
        accumulator.update(
            torch.tensor([2.0, 4.0]), torch.tensor([2.0, 3.0]))

        row = accumulator.row("STRICT_W4A4")

        self.assertEqual(row["signal"], "skip4_input")
        self.assertEqual(row["updates"], 2)
        self.assertAlmostEqual(row["mse"], 0.5)
        self.assertTrue(torch.isfinite(torch.tensor(row["sqnr_db"])))


if __name__ == "__main__":
    unittest.main()
