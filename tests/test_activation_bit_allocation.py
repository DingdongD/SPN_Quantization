import unittest

from scripts import activation_bit_allocation as allocation


class SensitiveModuleSelectionTest(unittest.TestCase):
    def test_selects_unique_finite_lowest_sqnr_inputs(self):
        rows = [
            {"config": "HW_W4A4_MinMax", "kind": "input",
             "module": "enc.a", "group": "encoder", "sqnr_db": "4.0"},
            {"config": "HW_W4A4_MinMax", "kind": "input",
             "module": "enc.a", "group": "encoder", "sqnr_db": "3.0"},
            {"config": "HW_W4A4_MinMax", "kind": "input",
             "module": "head", "group": "depth_head", "sqnr_db": "5.0"},
            {"config": "HW_W4A4_MinMax", "kind": "input",
             "module": "zero", "group": "encoder", "sqnr_db": "inf"},
            {"config": "other", "kind": "input",
             "module": "ignored", "group": "encoder", "sqnr_db": "-10"},
        ]

        result = allocation.select_sensitive_modules(rows, limit=2)

        self.assertEqual(result, [
            {"module": "enc.a", "group": "encoder", "sqnr_db": 3.0},
            {"module": "head", "group": "depth_head", "sqnr_db": 5.0},
        ])


class MixedConfigurationTest(unittest.TestCase):
    def test_builds_site_group_topk_full_and_state_configurations(self):
        module_groups = {
            "enc.a": "encoder",
            "enc.b": "encoder",
            "depth": "depth_head",
            "guide": "propagation_head",
        }
        candidates = [
            {"module": "enc.a", "group": "encoder", "sqnr_db": 3.0},
            {"module": "depth", "group": "depth_head", "sqnr_db": 4.0},
        ]

        configs = allocation.build_mixed_configurations(
            module_groups, candidates)
        by_name = dict((config["name"], config) for config in configs)

        self.assertIn("FP32", by_name)
        self.assertEqual(by_name["MP_W4A4_base"]["activation_bit_overrides"], {})
        self.assertEqual((by_name["MP_W4A8_full"]["w_bits"],
                          by_name["MP_W4A8_full"]["a_bits"]), (4, 8))
        self.assertEqual((by_name["MP_W8A8_full"]["w_bits"],
                          by_name["MP_W8A8_full"]["a_bits"]), (8, 8))
        self.assertEqual(by_name["MP_W8A8_full"]["selection"], "full_w8a8")
        self.assertEqual(by_name["MP_site01_A8"]["activation_bit_overrides"],
                         {"enc.a": 8})
        self.assertEqual(by_name["MP_encoder_A8"]["activation_bit_overrides"],
                         {"enc.a": 8, "enc.b": 8})
        self.assertEqual(by_name["MP_heads_A8"]["activation_bit_overrides"],
                         {"depth": 8, "guide": 8})
        self.assertEqual(by_name["MP_top2_A8"]["activation_bit_overrides"],
                         {"enc.a": 8, "depth": 8})
        self.assertEqual(by_name["MP_W4A4_stateA16"]["state_bits"], 16)
        self.assertEqual(by_name["MP_W4A4_stateA8"]["state_bits"], 8)

    def test_config_manifest_rows_preserve_module_mapping(self):
        configs = allocation.build_mixed_configurations(
            {"enc.a": "encoder"},
            [{"module": "enc.a", "group": "encoder", "sqnr_db": 2.5}])

        rows = allocation.config_manifest_rows(configs)
        site = next(row for row in rows if row["config"] == "MP_site01_A8")

        self.assertEqual(site["module"], "enc.a")
        self.assertEqual(site["activation_bits"], 8)
        self.assertEqual(site["selection"], "single_site")


class AllocationAccountingTest(unittest.TestCase):
    def test_summary_uses_pooled_rmse_and_incremental_activation_bits(self):
        regional = [
            {"model": "dyspn", "config": "FP32", "region": "all",
             "RMSE": "0.2", "MAE": "0.1", "ABS_REL": "0.03"},
            {"model": "dyspn", "config": "MP_W4A4_base", "region": "all",
             "RMSE": "2.0", "MAE": "1.0", "ABS_REL": "0.5"},
            {"model": "dyspn", "config": "MP_site01_A8", "region": "all",
             "RMSE": "1.0", "MAE": "0.5", "ABS_REL": "0.2"},
        ]
        samples = [
            {"model": "dyspn", "config": config,
             "nonfinite_pixels": "0", "num_pixels": "100"}
            for config in ("FP32", "MP_W4A4_base", "MP_site01_A8")
        ]
        manifest = [
            {"config": "FP32", "selection": "fp32", "module": "",
             "activation_bits": "", "default_activation_bits": "",
             "weight_bits": "", "state_bits": ""},
            {"config": "MP_W4A4_base", "selection": "baseline", "module": "",
             "activation_bits": "4", "default_activation_bits": "4",
             "weight_bits": "4", "state_bits": ""},
            {"config": "MP_site01_A8", "selection": "single_site",
             "module": "enc.a", "activation_bits": "8",
             "default_activation_bits": "4", "weight_bits": "4",
             "state_bits": ""},
        ]
        layers = [
            {"model": "dyspn", "config": "MP_W4A4_base",
             "module": "enc.a", "kind": "input", "numel": "100"},
            {"model": "dyspn", "config": "MP_W4A4_base",
             "module": "enc.a", "kind": "output", "numel": "50"},
            {"model": "dyspn", "config": "MP_W4A4_base",
             "module": "enc.b", "kind": "input", "numel": "50"},
        ]

        rows = allocation.allocation_summary_rows(
            regional, samples, manifest, layers)
        result = dict((row["config"], row) for row in rows)

        self.assertEqual(result["MP_W4A4_base"]["activation_bit_traffic"], 800.0)
        self.assertEqual(result["MP_site01_A8"]["activation_bit_traffic"], 1400.0)
        self.assertEqual(result["MP_site01_A8"]["extra_activation_bits"], 600.0)
        self.assertAlmostEqual(result["MP_site01_A8"]["rmse_gain"], 1.0)
        self.assertAlmostEqual(result["MP_site01_A8"]["nonfinite_rate"], 0.0)

    def test_cspn_ranking_prioritizes_nonfinite_reduction(self):
        rows = [
            {"model": "cspn", "config": "low_rmse", "RMSE": 0.5,
             "nonfinite_rate": 0.8, "extra_activation_bits": 100.0},
            {"model": "cspn", "config": "stable", "RMSE": 1.0,
             "nonfinite_rate": 0.0, "extra_activation_bits": 200.0},
        ]

        ranked = allocation.rank_allocation_rows(rows)

        self.assertEqual([row["config"] for row in ranked],
                         ["stable", "low_rmse"])
        self.assertEqual([row["rank"] for row in ranked], [1, 2])


if __name__ == "__main__":
    unittest.main()
