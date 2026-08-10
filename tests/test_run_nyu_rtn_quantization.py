import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from scripts import run_nyu_rtn_quantization as runner


class RTNExperimentRunnerTest(unittest.TestCase):
    def test_dyspn_propagation_inputs_follow_official_forward_order(self):
        class Propagation(torch.nn.Module):
            def forward(self, initial, guidance, sparse_depth,
                        confidence_logits):
                return initial + guidance + sparse_depth + confidence_logits

        module = Propagation()
        capture = runner.PropagationInputCapture("dyspn", module)
        inputs = tuple(torch.full((1,), float(index)) for index in range(4))

        module(*inputs)
        capture.close()

        self.assertEqual(float(capture.current["pred_init"]), 0.0)
        self.assertEqual(float(capture.current["guidance"]), 1.0)
        self.assertEqual(float(capture.current["sparse_depth"]), 2.0)
        self.assertEqual(float(capture.current["confidence_logits"]), 3.0)

    def test_calibration_dataset_reads_explicit_data_root(self):
        saved_args = SimpleNamespace(
            model="dyspn",
            train_list="train.csv",
            data_root="/datasets/nyu",
            n_sample=500,
            seed=123,
        )
        with mock.patch.object(runner.sweep, "NyuHdf5Dataset") as dataset, \
                mock.patch.object(
                    runner.sweep, "resolve_data_root",
                    side_effect=AssertionError("implicit data root resolution")):
            runner.calibration_dataset(saved_args)

        dataset.assert_called_once_with(
            csv_file="train.csv",
            root_dir="/datasets/nyu",
            split="train",
            n_sample=500,
            seed=123,
        )

    def test_fp4_backend_uses_propagation_adapter(self):
        self.assertIn("fp4", runner.QUANT_BACKENDS)
        self.assertTrue(runner.uses_propagation_adapter("fp4"))
        self.assertTrue(runner.uses_propagation_adapter("propagation"))
        self.assertFalse(runner.uses_propagation_adapter("hardware"))

    def test_completionformer_joint_backend_has_strict_ablation_order(self):
        configs = runner.build_completionformer_joint_configurations(
            ["encoder", "decoder"])

        self.assertEqual([config["name"] for config in configs], [
            "FP32",
            "JIQ_RTN_W4A4",
            "JIQ_Attention_W4A4",
            "JIQ_Concat_W4A4",
            "JIQ_Joint_W4A4",
            "JIQ_W4A8",
        ])
        required = {
            "attention_enabled", "concat_enabled", "qkv_bits",
            "concat_bits", "output_bits",
        }
        for config in configs[1:]:
            self.assertTrue(required.issubset(config))
            self.assertEqual(config["w_bits"], 4)
            self.assertIsNotNone(config["propagation"])
        self.assertEqual(
            (configs[-1]["a_bits"], configs[-1]["qkv_bits"],
             configs[-1]["concat_bits"], configs[-1]["output_bits"]),
            (8, 8, 8, 8))
        self.assertTrue(
            runner.uses_propagation_adapter("completionformer_joint"))

    def test_completionformer_joint_backend_rejects_other_models(self):
        with self.assertRaisesRegex(ValueError, "requires completionformer"):
            runner.validate_completionformer_joint_model(
                "completionformer_joint", "dyspn")

    def test_joint_ownership_is_selected_per_ablation(self):
        class JointAdapter(object):
            def attention_owned_outputs(self):
                return ["former.attn.q", "former.attn.kv"]

            def concat_owned_inputs(self):
                return ["former.concat_conv"]

            def concat_owned_outputs(self):
                return ["former.concat_conv"]

        adapter = JointAdapter()
        propagation = {"prop_layer.conv_offset_aff"}
        configs = runner.build_completionformer_joint_configurations([
            "encoder"])
        by_name = dict((config["name"], config) for config in configs)

        inputs, outputs = runner.completionformer_joint_ownership(
            by_name["JIQ_Attention_W4A4"], adapter, propagation)
        self.assertEqual(inputs, set())
        self.assertEqual(outputs, {
            "former.attn.q", "former.attn.kv",
            "prop_layer.conv_offset_aff",
        })

        inputs, outputs = runner.completionformer_joint_ownership(
            by_name["JIQ_Concat_W4A4"], adapter, propagation)
        self.assertEqual(inputs, {"former.concat_conv"})
        self.assertEqual(outputs, {
            "former.concat_conv", "prop_layer.conv_offset_aff",
        })

    def test_joint_calibration_replays_identical_indices_in_two_passes(self):
        events = []

        class Model(object):
            def __call__(self, sample):
                events.append(("forward", sample))

        class Instrumentor(object):
            def observe(self):
                events.append(("instrumentor", "observe"))

            def freeze(self):
                events.append(("instrumentor", "freeze"))

            def set_external_ownership(self, inputs, outputs):
                events.append(("ownership", inputs, outputs))

            def configure(self, w_bits, a_bits, groups, quantize_bias):
                events.append((
                    "instrumentor", "configure", w_bits, a_bits,
                    groups, quantize_bias))

            def disable(self):
                events.append(("instrumentor", "disable"))

        class PropagationAdapter(object):
            def observe(self):
                events.append(("propagation", "observe"))

            def freeze(self):
                events.append(("propagation", "freeze"))

            def configure(self, config):
                events.append(("propagation", "configure", config))

            def disable(self):
                events.append(("propagation", "disable"))

        class JointAdapter(object):
            def capture_targets(self):
                events.append(("joint", "capture_targets"))

            def observe_reconstruction(self):
                events.append(("joint", "observe_reconstruction"))

            def freeze(self):
                events.append(("joint", "freeze"))

        propagation = {
            "affinity_bits": 8,
            "confidence_bits": 8,
            "offset_bits": 8,
            "state_bits": 8,
            "coefficient_fraction_bits": 13,
        }
        with mock.patch.object(
                runner, "seeded_sample", side_effect=lambda dataset, index, seed: index), \
                mock.patch.object(
                    runner, "batch_from_sample", side_effect=lambda sample: sample), \
                mock.patch.object(
                    runner.sweep, "batch_to_model_input",
                    side_effect=lambda model, batch, device: ((batch,), None)):
            runner.calibrate_completionformer_joint(
                model=Model(),
                saved_args=SimpleNamespace(model="completionformer"),
                dataset=[0, 1, 2],
                indices=[2, 0],
                device=torch.device("cpu"),
                seed=17,
                instrumentor=Instrumentor(),
                propagation_adapter=PropagationAdapter(),
                joint_adapter=JointAdapter(),
                groups=["encoder"],
                propagation_outputs={"prop_layer.conv_offset_aff"},
                propagation_config=propagation)

        self.assertEqual(
            [event[1] for event in events if event[0] == "forward"],
            [2, 0, 2, 0])
        self.assertLess(
            events.index(("joint", "capture_targets")),
            events.index(("joint", "observe_reconstruction")))
        self.assertIn((
            "ownership", set(), {"prop_layer.conv_offset_aff"}), events)

    def test_joint_table_audit_rejects_duplicates_and_missing_sites(self):
        config = runner.build_completionformer_joint_configurations([
            "encoder"])[4]
        attention_manifest = [{
            "config": config["name"], "family": "attention",
            "module": "former.attn", "head": 0,
        }]
        concat_manifest = [{
            "config": config["name"], "family": "concat",
            "module": "former.concat_conv",
        }]
        search = [
            {
                "config": config["name"], "family": "attention",
                "module": "former.attn", "round": 0,
                "parameter": "q", "factor": 1.0,
            },
            {
                "config": config["name"], "family": "concat",
                "module": "former.concat_conv", "round": 0,
                "parameter": "output", "factor": 1.0,
            },
        ]
        attention_metrics = [{
            "config": config["name"], "family": "attention",
            "module": "former.attn", "updates": 64,
        }]
        concat_metrics = [{
            "config": config["name"], "family": "concat",
            "module": "former.concat_conv", "updates": 64,
        }]

        runner.validate_completionformer_joint_tables(
            configs=[config],
            attention_names=["former.attn"],
            concat_names=["former.concat_conv"],
            evaluation_updates=64,
            attention_manifest_rows=attention_manifest,
            concat_manifest_rows=concat_manifest,
            search_rows=search,
            attention_metric_rows=attention_metrics,
            concat_metric_rows=concat_metrics)

        with self.assertRaisesRegex(ValueError, "duplicate attention manifest"):
            runner.validate_completionformer_joint_tables(
                configs=[config],
                attention_names=["former.attn"],
                concat_names=["former.concat_conv"],
                evaluation_updates=64,
                attention_manifest_rows=attention_manifest * 2,
                concat_manifest_rows=concat_manifest,
                search_rows=search,
                attention_metric_rows=attention_metrics,
                concat_metric_rows=concat_metrics)

        with self.assertRaisesRegex(ValueError, "concat manifest modules"):
            runner.validate_completionformer_joint_tables(
                configs=[config],
                attention_names=["former.attn"],
                concat_names=["former.concat_conv"],
                evaluation_updates=64,
                attention_manifest_rows=attention_manifest,
                concat_manifest_rows=[],
                search_rows=search,
                attention_metric_rows=attention_metrics,
                concat_metric_rows=concat_metrics)

    def test_fp4_instrumentor_options_are_forwarded_explicitly(self):
        config = {
            "activation_format_overrides": {
                ("depth", "input"): "uniform",
            },
            "quantize_bias": False,
        }

        options = runner.instrumentor_options(config)

        self.assertEqual(
            options["activation_format_overrides"],
            {("depth", "input"): "uniform"})
        self.assertFalse(options["quantize_bias"])

    def test_fp4_metadata_declares_accuracy_only_contract(self):
        self.assertEqual(
            runner.fp4_validation_metadata(),
            {
                "activation_format": "scaled_e2m1_rne",
                "codebook": "0;+/-0.5;+/-1;+/-1.5;+/-2;+/-3;+/-4;+/-6",
                "scale_policy": "calibration_absmax_div_6_frozen",
                "bias_contract": "fp32_isolation",
                "propagation_signals": "a8",
                "native_fp4_execution": False,
                "execution":
                    "float_e2m1_qdq_integer_normalization_reference",
            })

    def test_fp4_runner_configs_apply_strict_semantic_overrides(self):
        modules = {
            "backbone.conv1_dep.0",
            "backbone.dep_dec0.0",
            "backbone.gd_dec0.0",
            "backbone.cf_dec0.0",
        }

        configs, semantic_rows = runner.build_fp4_runner_configurations(
            ["encoder", "depth_head"], "completionformer", modules)

        self.assertEqual(len(configs), 7)
        self.assertEqual(len(semantic_rows), 4)
        expected_key = ("backbone.dep_dec0.0", "output")
        for config in configs[1:]:
            self.assertEqual(
                config["activation_bit_overrides"][expected_key], 8)
            self.assertEqual(
                config["activation_format_overrides"][expected_key],
                "uniform")
        self.assertNotIn("activation_bit_overrides", configs[0])

    def test_dyspn_concat_consumers_are_resolved_strictly(self):
        modules = {
            "base.dec2.0",
            "base.dec3.0",
            "base.dec4.0",
            "base.gd_dec1_.0",
            "base.gd_dec0_dyspn_6_5.0",
        }

        resolved = runner.resolve_per_channel_activation_inputs(
            "dyspn", modules)

        self.assertEqual(resolved, modules)

    def test_missing_dyspn_concat_consumer_fails(self):
        modules = {
            "base.dec2.0",
            "base.dec3.0",
            "base.dec4.0",
            "base.gd_dec1_.0",
        }

        with self.assertRaisesRegex(
                RuntimeError, "dyspn concat boundary guidance_output"):
            runner.resolve_per_channel_activation_inputs("dyspn", modules)

    def test_dyspn_operator_contract_requires_official_grid_sample_mode(self):
        propagation = SimpleNamespace(mode="yx")
        model = SimpleNamespace(
            mode="dyspn", iteration=6, num_sample=5,
            _modules={"dyspn_6_5": propagation})

        contract = runner.validate_dyspn_operator_contract(model)

        self.assertEqual(contract, {
            "mode": "dyspn",
            "propagation_module": "dyspn_6_5",
            "coordinate_mode": "yx",
            "sampling_operator": "torch.nn.functional.grid_sample",
            "deform_conv_active": False,
        })

    def test_dyspn_operator_contract_rejects_deform_mode(self):
        model = SimpleNamespace(
            mode="deform_dyspn", iteration=6, num_sample=5, _modules={})

        with self.assertRaisesRegex(
                RuntimeError, "expected official mode=dyspn"):
            runner.validate_dyspn_operator_contract(model)

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
            "PA_W4A8",
            "PA_W8A8",
        ])
        self.assertFalse(configs[1]["external_output_ownership"])
        self.assertIsNone(configs[1]["propagation"])
        self.assertEqual(configs[2]["propagation"]["confidence_bits"], 8)
        self.assertEqual(configs[2]["propagation"]["offset_bits"], 4)
        self.assertEqual(configs[3]["propagation"]["offset_bits"], 8)
        self.assertEqual(configs[4]["propagation"]["state_bits"], 8)
        w4a8 = configs[5]
        self.assertEqual((w4a8["w_bits"], w4a8["a_bits"]), (4, 8))
        self.assertEqual(w4a8["propagation"], {
            "affinity_bits": 8,
            "confidence_bits": 8,
            "offset_bits": 8,
            "state_bits": 8,
            "coefficient_fraction_bits": 13,
        })
        self.assertEqual((configs[6]["w_bits"], configs[6]["a_bits"]),
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

    def test_append_identity_rejects_checkpoint_or_source_mismatch(self):
        existing = {
            "model": "cspn",
            "iteration": 24,
            "calibration_indices": [3, 7],
            "model_provenance": {
                "checkpoint_sha256": "old-checkpoint",
                "source_git_commit": "official-commit",
                "source_sha256": "official-source",
            },
        }
        current = {
            "checkpoint_sha256": "new-checkpoint",
            "source_git_commit": "official-commit",
            "source_sha256": "official-source",
        }

        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            runner.validate_append_identity(
                existing, "cspn", 24, [3, 7], current)

    def test_append_identity_accepts_exact_same_run_inputs(self):
        provenance = {
            "checkpoint_sha256": "checkpoint",
            "source_git_commit": "official-commit",
            "source_sha256": "official-source",
        }
        existing = {
            "model": "nlspn",
            "iteration": 18,
            "calibration_indices": [3, 7],
            "model_provenance": dict(provenance),
        }

        runner.validate_append_identity(
            existing, "nlspn", 18, [3, 7], provenance)

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
        self.assertTrue(all(
            config["quantize_bias"] for config in configs[1:]))
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
