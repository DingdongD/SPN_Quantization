import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts import run_nyu_rtn_quantization as runner


class RTNExperimentRunnerTest(unittest.TestCase):
    def test_propagation_backend_has_cumulative_ablation_matrix(self):
        configs = runner.build_propagation_configurations([
            "encoder", "propagation_head",
        ])

        self.assertEqual([config["name"] for config in configs], [
            "FP32",
            "PA_Generic_W4A4",
            "PA_Constraint",
            "PA_OffsetA8",
            "PA_StateA8",
            "PA_W8A8",
        ])
        self.assertFalse(configs[1]["external_output_ownership"])
        self.assertIsNone(configs[1]["propagation"])
        self.assertEqual(configs[2]["propagation"]["confidence_bits"], 8)
        self.assertEqual(configs[2]["propagation"]["offset_bits"], 4)
        self.assertEqual(configs[3]["propagation"]["offset_bits"], 8)
        self.assertEqual(configs[4]["propagation"]["state_bits"], 8)
        self.assertEqual((configs[5]["w_bits"], configs[5]["a_bits"]),
                         (8, 8))

    def test_propagation_runtime_configuration_keeps_generic_official_loop(self):
        class Adapter(object):
            def __init__(self):
                self.action = None

            def disable(self):
                self.action = ("disable", None)

            def capture(self):
                self.action = ("capture", None)

            def configure(self, value):
                self.action = ("configure", value)

        configs = runner.build_propagation_configurations(["encoder"])
        adapter = Adapter()

        runner.configure_runtime_adapter(
            configs[1], adapter, propagation_backend=True,
            model_name="dyspn")
        self.assertEqual(adapter.action, ("disable", None))

        runner.configure_runtime_adapter(
            configs[1], adapter, propagation_backend=True,
            model_name="cspn")
        self.assertEqual(adapter.action, ("capture", None))

        runner.configure_runtime_adapter(
            configs[2], adapter, propagation_backend=True,
            model_name="cspn")
        self.assertEqual(adapter.action[0], "configure")
        self.assertEqual(adapter.action[1].affinity_bits, 4)
        self.assertEqual(adapter.action[1].confidence_bits, 8)

    def test_completionformer_bypass_states_are_restored_to_forward_order(self):
        class Adapter(object):
            def last_states(self):
                return []

        class Capture(object):
            model_name = "completionformer"
            current = {}

        first = torch.tensor([1.0])
        last = torch.tensor([18.0])
        output = {
            "pred": last,
            "pred_inter": [last, first],
        }

        signals = runner.model_signals(output, Adapter(), Capture())

        self.assertEqual(
            [float(value.item()) for value in signals["propagation_states"]],
            [1.0, 18.0])

    def test_propagation_metadata_does_not_claim_integer_sampling_or_mac(self):
        class Instrumentor(object):
            def externally_owned_outputs(self):
                return ["prop_layer.conv_offset_aff"]

        metadata = runner.propagation_metadata(Instrumentor())

        self.assertEqual(
            metadata["execution"],
            "integer_normalization_float_sampling_qdq_reference")
        self.assertEqual(
            metadata["propagation_accumulator"],
            "float_reference_after_integer_coefficient_qdq")

    def test_configurations_are_full_then_group_then_state_stress(self):
        configs = runner.build_configurations(["encoder", "decoder"])

        self.assertEqual(
            [config["name"] for config in configs],
            [
                "FP32",
                "W8A8_full",
                "W4A8_full",
                "W4A4_full",
                "W8A8_encoder_only",
                "W4A4_encoder_only",
                "W8A8_decoder_only",
                "W4A4_decoder_only",
                "W8A8_full_stateA8",
                "W4A4_full_stateA4",
            ],
        )

        w4a8 = dict((config["name"], config) for config in configs)["W4A8_full"]
        self.assertEqual((w4a8["w_bits"], w4a8["a_bits"]), (4, 8))
        self.assertTrue(runner.should_export_predictions("W4A8_full"))

    def test_replace_config_rows_preserves_unrelated_existing_results(self):
        existing = [
            {"config": "FP32", "value": "baseline"},
            {"config": "W8A8_full", "value": "old-w8"},
            {"config": "W4A8_full", "value": "stale"},
            {"config": "W4A4_full", "value": "old-w4"},
        ]

        rows = runner.replace_config_rows(
            existing, [{"config": "W4A8_full", "value": "fresh"}],
            {"W4A8_full"})

        self.assertEqual(rows, [
            {"config": "FP32", "value": "baseline"},
            {"config": "W8A8_full", "value": "old-w8"},
            {"config": "W4A4_full", "value": "old-w4"},
            {"config": "W4A8_full", "value": "fresh"},
        ])

    def test_merge_manifest_replaces_selected_configs_and_preserves_others(self):
        existing = [
            {"config": "MP_W4A4_base", "module": "old-base"},
            {"config": "MP_top4_A8", "module": "keep-top4"},
        ]
        replacement = [
            {"config": "MP_W4A4_base", "module": "fresh-base"},
            {"config": "MP_W8A8_full", "module": "fresh-w8"},
        ]

        rows = runner.merge_manifest_rows(
            existing, replacement, {"MP_W4A4_base", "MP_W8A8_full"})

        self.assertEqual(rows, [
            {"config": "MP_top4_A8", "module": "keep-top4"},
            {"config": "MP_W4A4_base", "module": "fresh-base"},
            {"config": "MP_W8A8_full", "module": "fresh-w8"},
        ])

    def test_explicit_prediction_exports_replace_legacy_name_defaults(self):
        selected = {"MP_W4A4_base"}

        self.assertTrue(runner.should_export_predictions(
            "MP_W4A4_base", selected))
        self.assertFalse(runner.should_export_predictions(
            "W4A4_full", selected))
        self.assertTrue(runner.should_export_predictions("W4A4_full", None))

    def test_hardware_backend_configs_are_full_model_only(self):
        configs = runner.build_hardware_configurations(["encoder", "decoder"])

        self.assertEqual([config["name"] for config in configs], [
            "FP32", "HW_W8A8_full", "HW_W4A8_full", "HW_W4A4_full",
        ])
        self.assertEqual(
            [(config["w_bits"], config["a_bits"]) for config in configs[1:]],
            [(8, 8), (4, 8), (4, 4)],
        )
        self.assertTrue(runner.should_export_predictions("HW_W4A8_full"))
        self.assertTrue(runner.should_export_predictions("HW_W4A4_full"))
        self.assertTrue(runner.should_export_predictions("HW_W8A8_full"))

    def test_outlier_configs_cover_activation_and_mitigation_ablations(self):
        configs = runner.build_outlier_configurations(["encoder", "decoder"])
        by_name = dict((config["name"], config) for config in configs)

        self.assertEqual([config["name"] for config in configs], [
            "FP32", "HW_W4A4_MinMax", "HW_W8A4_full",
            "HW_W4A4_SQ_A50",
        ])
        self.assertEqual(by_name["HW_W8A4_full"]["w_bits"], 8)
        self.assertEqual(by_name["HW_W8A4_full"]["a_bits"], 4)
        self.assertEqual(by_name["HW_W4A4_SQ_A50"]["smooth_alpha"], 0.5)

    def test_lognp_configs_isolate_tensor_channel_and_compensation_controls(self):
        configs = runner.build_lognp_configurations(["encoder", "decoder"])
        self.assertEqual([config["name"] for config in configs], [
            "FP32", "LOGNP_W8A4_tensor", "LOGNP_W8A4_channel",
            "LOGNP_W8A4_channel_bias", "LOGNP_W8A4_channel_weight",
            "LOGNP_W4A4_channel_weight", "W4A8_full",
        ])
        self.assertEqual(configs[1]["activation_mode"], "lognp")
        self.assertFalse(configs[1]["lognp_per_channel"])
        self.assertTrue(configs[2]["lognp_per_channel"])
        self.assertEqual(configs[3]["compensation_method"], "bias")
        self.assertEqual(configs[4]["compensation_method"], "weight")
        self.assertEqual((configs[5]["w_bits"], configs[5]["a_bits"]), (4, 4))
        self.assertNotIn("activation_mode", configs[-1])

    def test_instrumentor_options_include_activation_bit_overrides(self):
        config = {
            "activation_overrides": {("enc", "input"): 1.0},
            "activation_bit_overrides": {"enc": 8},
            "smooth_alpha": 0.5,
            "ignored": "value",
        }

        result = runner.instrumentor_options(config)

        self.assertEqual(result, {
            "activation_overrides": {("enc", "input"): 1.0},
            "activation_bit_overrides": {"enc": 8},
            "smooth_alpha": 0.5,
        })

    def test_instrumentor_options_include_lognp_contract(self):
        config = {
            "activation_mode": "lognp",
            "alpha_factor": 0.5,
            "max_z": 20.0,
            "lognp_per_channel": False,
            "compensation_method": "bias",
        }

        self.assertEqual(runner.instrumentor_options(config), {
            "activation_mode": "lognp",
            "alpha_factor": 0.5,
            "max_z": 20.0,
            "lognp_per_channel": False,
        })

    def test_lognp_compensation_modules_use_ranked_sensitive_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cspn"
            root.mkdir(parents=True)
            path = root / "layer_quantization_metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "config", "kind", "module", "group", "sqnr_db"])
                writer.writeheader()
                writer.writerows([
                    {"config": "MP_W4A4_base", "kind": "input",
                     "module": "site_b", "group": "decoder", "sqnr_db": "2"},
                    {"config": "MP_W4A4_base", "kind": "input",
                     "module": "site_a", "group": "encoder", "sqnr_db": "1"},
                    {"config": "MP_W4A4_base", "kind": "input",
                     "module": "not_in_model", "group": "encoder", "sqnr_db": "0"},
                ])

            result = runner.select_lognp_compensation_modules(
                "cspn", ["site_a", "site_b"], tmp, limit=4)

            self.assertEqual(result, ["site_a", "site_b"])

    def test_metric_csv_supplies_unique_sample_indices_in_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_index"])
                writer.writeheader()
                writer.writerows([
                    {"sample_index": "8"},
                    {"sample_index": "3"},
                    {"sample_index": "8"},
                ])

            indices = runner.load_sample_indices(path)

            self.assertEqual(indices, [8, 3])

    def test_aggregate_regions_uses_pixel_weighted_error(self):
        rows = [
            {"region": "all", "num_pixels": 1, "sum_sq": 4.0,
             "sum_abs": 2.0, "sum_abs_rel": 1.0},
            {"region": "all", "num_pixels": 3, "sum_sq": 12.0,
             "sum_abs": 6.0, "sum_abs_rel": 3.0},
        ]

        summary = runner.aggregate_region_rows(rows)

        self.assertEqual(summary[0]["num_pixels"], 4)
        self.assertAlmostEqual(summary[0]["RMSE"], 2.0)
        self.assertAlmostEqual(summary[0]["MAE"], 2.0)
        self.assertAlmostEqual(summary[0]["ABS_REL"], 1.0)

    def test_prediction_payload_preserves_gt_fp32_and_nonfinite_masks(self):
        payload = runner.prediction_payload(
            gt=np.array([[1.0, 0.0], [2.0, 3.0]], np.float32),
            fp32=np.array([[1.1, 4.0], [2.1, 3.1]], np.float32),
            pred=np.array([[1.2, 5.0], [np.nan, 2.5]], np.float32),
            sample_index=7, model="cspn", config="MP_W4A4_base",
            sparse=np.array([[0.0, 0.0], [2.0, 0.0]], np.float32))

        self.assertTrue(np.array_equal(
            payload["valid_gt"], [[True, False], [True, True]]))
        self.assertTrue(payload["nonfinite"][1, 0])
        self.assertAlmostEqual(float(payload["abs_err"][0, 0]), 0.2,
                               places=6)
        self.assertTrue(np.isnan(payload["abs_err"][1, 0]))
        self.assertTrue(np.allclose(payload["fp32"],
                                    [[1.1, 4.0], [2.1, 3.1]]))
        self.assertTrue(np.array_equal(payload["sparse"],
                                       [[0.0, 0.0], [2.0, 0.0]]))

    def test_prepare_prediction_dir_removes_only_stale_npz_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions" / "FP32"
            path.mkdir(parents=True)
            (path / "sample_00001.npz").write_bytes(b"stale")
            (path / "notes.txt").write_text("keep", encoding="utf-8")

            result = runner.prepare_prediction_dir(tmp, "FP32")

            self.assertEqual(result, path)
            self.assertFalse((path / "sample_00001.npz").exists())
            self.assertTrue((path / "notes.txt").exists())

    def test_persist_tables_writes_propagation_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner.persist_tables(
                tmp, [], [], [], [], [], [{
                    "model": "cspn",
                    "config": "PA_Constraint",
                    "sample_index": 7,
                    "signal": "affinity_constraints",
                    "iteration": 0,
                    "coefficient_sum_max_error": 0.0,
                }])

            rows = runner.read_csv(
                Path(tmp) / "propagation_quantization_metrics.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["config"], "PA_Constraint")
            self.assertEqual(rows[0]["signal"], "affinity_constraints")


if __name__ == "__main__":
    unittest.main()
