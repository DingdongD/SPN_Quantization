import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from spn_quant.propagation.controller import PropagationQuantConfig


_CSPN_PATH = Path(__file__).resolve().parents[1] / "models" / "cspn.py"
_CSPN_SPEC = importlib.util.spec_from_file_location("official_cspn", _CSPN_PATH)
_CSPN_MODULE = importlib.util.module_from_spec(_CSPN_SPEC)
_CSPN_SPEC.loader.exec_module(_CSPN_MODULE)
Affinity_Propagate = _CSPN_MODULE.Affinity_Propagate


class ModulatedDeformConvFunction(object):
    @staticmethod
    def apply(confidence, offset, modulation, weight, bias, stride, padding,
              dilation, groups, deformable_groups, im2col_step):
        del offset, modulation, weight, bias, stride, padding, dilation
        del groups, deformable_groups, im2col_step
        return confidence


class ToyNLSPNModule(nn.Module):
    def __init__(self, affinity="TGASS", preserve_input=True):
        super(ToyNLSPNModule, self).__init__()
        self.args = SimpleNamespace(
            conf_prop=True,
            preserve_input=preserve_input,
            legacy=False,
        )
        self.affinity = affinity
        self.prop_time = 2
        self.ch_g = 1
        self.ch_f = 1
        self.k_f = 3
        self.num = 2
        self.idx_ref = 1
        self.aff_scale_const = nn.Parameter(
            torch.tensor([2.0 if affinity == "TC" else 1.0]),
            requires_grad=False)
        self.conv_offset_aff = nn.Conv2d(1, 3 * self.num, 1, bias=True)
        self.w_conf = nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.deformable_groups = 1
        self.im2col_step = 64
        nn.init.zeros_(self.conv_offset_aff.weight)
        with torch.no_grad():
            self.conv_offset_aff.bias.copy_(torch.tensor([
                0.2, -0.1, 0.1, -0.2, 2.0, -1.0,
            ]))

    def _sample_confidence_for_affinity(self, confidence, offset):
        del offset
        return confidence.repeat(1, self.num, 1, 1)

    def _propagate_once(self, feat, offset, affinity):
        del offset
        center = affinity[:, self.idx_ref:self.idx_ref + 1]
        neighbor = torch.cat((
            affinity[:, :self.idx_ref], affinity[:, self.idx_ref + 1:]), dim=1)
        return center * feat + neighbor.mean(dim=1, keepdim=True) * feat

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
                rgb=None):
        del guidance, confidence, feat_fix, rgb
        states = [feat_init for _ in range(self.prop_time)]
        offset = feat_init.repeat(1, 2 * (self.num + 1), 1, 1) * 0
        affinity = feat_init.repeat(1, self.num + 1, 1, 1) * 0
        return feat_init, states, offset, affinity, self.aff_scale_const.data


class ToyOfficialConfidenceSampler(ToyNLSPNModule):
    _sample_confidence_for_affinity = None

    def __init__(self):
        super(ToyOfficialConfidenceSampler, self).__init__()
        self.args.preserve_input = False
        self.num = 8
        self.idx_ref = 4
        self.conv_offset_aff = nn.Conv2d(1, 3 * self.num, 1, bias=True)
        nn.init.zeros_(self.conv_offset_aff.weight)
        nn.init.zeros_(self.conv_offset_aff.bias)


class ToyLegacyConfidenceSampler(ToyOfficialConfidenceSampler):
    def __init__(self):
        super(ToyLegacyConfidenceSampler, self).__init__()
        self.args.legacy = True


class ToyDySPNModule(nn.Module):
    def __init__(self, iteration=3, num=3):
        super(ToyDySPNModule, self).__init__()
        self.iteration = int(iteration)
        self.num = int(num)
        self.ch = self.iteration * self.num
        self.conv_offset_aff = nn.Conv2d(
            self.ch, 3 * self.ch, 1, bias=True)
        nn.init.zeros_(self.conv_offset_aff.weight)
        with torch.no_grad():
            values = torch.linspace(-1.0, 1.0, self.ch)
            self.conv_offset_aff.bias.copy_(torch.cat((
                torch.zeros(2 * self.ch), values)))

    def get_refgrid(self, batch, height, width, offset):
        del offset
        return torch.zeros(
            batch, self.iteration, self.num, height, width, 2)

    def forward(self, initial, guidance, sparse, confidence_logits):
        del guidance, confidence_logits
        states = [initial for _ in range(self.iteration)]
        offsets = tuple(torch.zeros(
            initial.shape[0], self.num, initial.shape[2], initial.shape[3], 2)
            for _ in range(self.iteration))
        affinity = tuple(torch.full(
            (initial.shape[0], 1, self.num, initial.shape[2], initial.shape[3]),
            1.0 / self.num) for _ in range(self.iteration))
        return {
            "pred": initial,
            "pred_init": initial,
            "list_feat": states,
            "offset": offsets,
            "aff": affinity,
        }


class CSPNPropagationAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.guidance = torch.randn(1, 8, 3, 4)
        self.initial = torch.rand(1, 1, 3, 4) * 2.0
        self.sparse = torch.zeros_like(self.initial)
        self.sparse[:, :, 1, 2] = 1.25

    def test_disabled_adapter_is_exact_official_bypass(self):
        from spn_quant.propagation.adapters import CSPNPropagationAdapter

        module = Affinity_Propagate(2, 3, "8sum").eval()
        expected = module(self.guidance, self.initial, self.sparse)
        adapter = CSPNPropagationAdapter(module)

        adapter.disable()
        actual = module(self.guidance, self.initial, self.sparse)

        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        adapter.close()

    def test_capture_mode_keeps_fp32_values_and_records_each_state(self):
        from spn_quant.propagation.adapters import CSPNPropagationAdapter

        module = Affinity_Propagate(2, 3, "8sum").eval()
        expected = module(self.guidance, self.initial, self.sparse)
        adapter = CSPNPropagationAdapter(module)

        adapter.capture()
        actual = module(self.guidance, self.initial, self.sparse)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(len(adapter.last_states()), 2)
        self.assertTrue(all(
            state.device.type == "cpu" for state in adapter.last_states()))
        adapter.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_training_capture_retains_detached_states_on_execution_device(self):
        from spn_quant.propagation.adapters import CSPNPropagationAdapter

        module = Affinity_Propagate(24, 3, "8sum").eval()
        adapter = CSPNPropagationAdapter(module)
        guidance = self.guidance.cuda()
        initial = self.initial.cuda()
        sparse = self.sparse.cuda()

        adapter.capture_training_states()
        with torch.no_grad():
            module(guidance, initial, sparse)
        states = adapter.last_states()

        self.assertEqual(len(states), 24)
        self.assertTrue(all(state.device == guidance.device for state in states))
        self.assertTrue(all(not state.requires_grad for state in states))

        adapter.capture()
        with torch.no_grad():
            module(guidance, initial, sparse)
        self.assertTrue(all(
            state.device.type == "cpu" for state in adapter.last_states()))
        adapter.close()

    def test_capture_mode_preserves_official_zero_denominator_nonfinite_mask(self):
        from spn_quant.propagation.adapters import CSPNPropagationAdapter

        module = Affinity_Propagate(2, 3, "8sum").eval()
        zero_guidance = torch.zeros_like(self.guidance)
        expected = module(zero_guidance, self.initial, self.sparse)
        adapter = CSPNPropagationAdapter(module)

        adapter.capture()
        actual = module(zero_guidance, self.initial, self.sparse)

        self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
        torch.testing.assert_close(
            actual, expected, equal_nan=True, atol=0.0, rtol=0.0)
        self.assertEqual(len(adapter.last_states()), 2)
        adapter.close()

    def test_quantized_cspn_preserves_constraint_and_official_anchor_value(self):
        from spn_quant.propagation.adapters import CSPNPropagationAdapter

        module = Affinity_Propagate(2, 3, "8sum").eval()
        adapter = CSPNPropagationAdapter(module)
        adapter.observe()
        module(self.guidance, self.initial, self.sparse)
        adapter.freeze()
        adapter.configure(PropagationQuantConfig(
            affinity_bits=4,
            confidence_bits=8,
            offset_bits=4,
            state_bits=4,
        ))

        output = module(self.guidance, self.initial, self.sparse)

        mask = self.sparse > 0
        torch.testing.assert_close(
            output[mask], self.initial[mask], atol=0.0, rtol=0.0)
        constraints = [
            row for row in adapter.statistics()
            if row["signal"] == "affinity_constraints"
        ]
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0]["coefficient_sum_max_error"], 0.0)
        self.assertEqual(constraints[0]["contraction_violation_rate"], 0.0)
        self.assertEqual(len(adapter.last_states()), 2)
        adapter.close()

    def test_float_cspn_state_dtype_is_explicit(self):
        from spn_quant.propagation.adapters import CSPNPropagationAdapter

        for state_dtype, expected_dtype in (
                ("fp32", torch.float32),
                ("bf16", torch.bfloat16),
                ("fp16", torch.float16)):
            module = Affinity_Propagate(2, 3, "8sum").eval()
            adapter = CSPNPropagationAdapter(module)
            adapter.observe()
            module(self.guidance, self.initial, self.sparse)
            adapter.freeze()
            adapter.configure_float(state_dtype)
            output = module(self.guidance, self.initial, self.sparse)

            self.assertEqual(output.dtype, expected_dtype)
            self.assertTrue(all(
                state.dtype == expected_dtype
                for state in adapter.last_states()))
            statistics = adapter.statistics()
            self.assertEqual(sum(
                row["signal"] == "state" for row in statistics),
                module.prop_time)
            self.assertEqual(sum(
                row["signal"] == "affinity_constraints"
                for row in statistics), 1)
            self.assertEqual(sum(
                row["signal"] == "anchor" for row in statistics),
                module.prop_time)
            adapter.close()

    def test_quantized_cspn_uses_int32_propagation_accumulators(self):
        from spn_quant.propagation.adapters import CSPNPropagationAdapter

        module = Affinity_Propagate(2, 3, "8sum").eval()
        adapter = CSPNPropagationAdapter(module)
        adapter.observe()
        module(self.guidance, self.initial, self.sparse)
        adapter.freeze()
        adapter.configure(PropagationQuantConfig(
            affinity_bits=8,
            confidence_bits=8,
            offset_bits=8,
            state_bits=8,
        ))

        module(self.guidance, self.initial, self.sparse)

        accumulator_rows = [
            row for row in adapter.statistics()
            if row["signal"] == "state_accumulator"
        ]
        self.assertEqual(len(accumulator_rows), module.prop_time)
        self.assertTrue(all(
            row["accumulator_dtype"] == "int32"
            for row in accumulator_rows))
        adapter.close()


class NLSPNPropagationAdapterTest(unittest.TestCase):
    def test_float_propagation_state_dtype_is_explicit(self):
        from spn_quant.propagation.adapters import NLSPNPropagationAdapter

        for state_dtype, expected_dtype in (
                ("fp32", torch.float32),
                ("bf16", torch.bfloat16),
                ("fp16", torch.float16)):
            module = ToyNLSPNModule(affinity="TGASS", preserve_input=False)
            adapter = NLSPNPropagationAdapter(module, model_name="nlspn")
            initial = torch.full((1, 1, 2, 3), 0.75)
            guidance = torch.ones(1, 1, 2, 3)
            confidence = torch.ones(1, 1, 2, 3)

            adapter.observe()
            module(initial, guidance, confidence, None)
            adapter.freeze()
            adapter.configure_float(state_dtype)
            result = module(initial, guidance, confidence, None)

            self.assertEqual(result[0].dtype, expected_dtype)
            self.assertEqual(len(adapter.last_states()), module.prop_time)
            self.assertTrue(all(
                state.dtype == expected_dtype
                for state in adapter.last_states()))
            statistics = adapter.statistics()
            self.assertEqual(sum(
                row["signal"] == "state" for row in statistics),
                module.prop_time)
            self.assertEqual(sum(
                row["signal"] == "affinity_constraints"
                for row in statistics), 1)
            adapter.close()

    def test_float_propagation_rejects_unknown_state_dtype(self):
        from spn_quant.propagation.adapters import NLSPNPropagationAdapter

        module = ToyNLSPNModule()
        adapter = NLSPNPropagationAdapter(module, model_name="nlspn")
        adapter.observe()
        initial = torch.full((1, 1, 2, 3), 0.75)
        guidance = torch.ones(1, 1, 2, 3)
        confidence = torch.ones(1, 1, 2, 3)
        module(initial, guidance, confidence, initial)
        adapter.freeze()
        with self.assertRaises(KeyError):
            adapter.configure_float("fp8")
        adapter.close()

    def _run_model(self, model_name):
        from spn_quant.propagation.adapters import NLSPNPropagationAdapter
        from spn_quant.propagation.fixed_point import Q13_ONE

        module = ToyNLSPNModule(affinity="TGASS", preserve_input=True)
        adapter = NLSPNPropagationAdapter(module, model_name=model_name)
        initial = torch.full((1, 1, 2, 3), 0.75)
        guidance = torch.ones(1, 1, 2, 3)
        confidence = torch.tensor([[[[0.0, 0.5, 1.0],
                                     [0.25, 0.75, 1.0]]]])
        sparse = torch.zeros_like(initial)
        sparse[:, :, 0, 1] = 1.5

        adapter.observe()
        module(initial, guidance, confidence, sparse)
        adapter.freeze()
        adapter.configure(PropagationQuantConfig(
            affinity_bits=4,
            confidence_bits=8,
            offset_bits=4,
            state_bits=4,
        ))
        result = module(initial, guidance, confidence, sparse)

        codes = adapter.last_coefficient_codes()
        self.assertTrue(bool(torch.all(codes.sum(dim=1) == Q13_ONE)))
        self.assertEqual(adapter.last_confidence_codes().min().item(), 0)
        self.assertEqual(adapter.last_confidence_codes().max().item(), 255)
        anchor_rows = [
            row for row in adapter.statistics()
            if row["signal"] == "anchor_injection"
        ]
        self.assertEqual(len(anchor_rows), module.prop_time)
        self.assertEqual(
            max(row["anchor_max_error"] for row in anchor_rows), 0.0)
        self.assertEqual(len(result[1]), module.prop_time)
        self.assertEqual(result[1][-1].data_ptr(), result[0].data_ptr())
        adapter.close()

    def test_nlspn_preserves_integer_constraints_and_anchor_injection(self):
        self._run_model("nlspn")

    def test_completionformer_preserves_integer_constraints_and_anchor_injection(self):
        self._run_model("completionformer")

    def test_tc_mode_uses_direct_q13_coefficients_for_both_model_variants(self):
        from spn_quant.propagation.adapters import NLSPNPropagationAdapter
        from spn_quant.propagation.fixed_point import Q13_ONE

        for model_name in ("nlspn", "completionformer"):
            module = ToyNLSPNModule(affinity="TC", preserve_input=False)
            adapter = NLSPNPropagationAdapter(module, model_name=model_name)
            initial = torch.full((1, 1, 2, 3), 0.75)
            guidance = torch.ones(1, 1, 2, 3)
            confidence = torch.ones(1, 1, 2, 3)
            adapter.observe()
            module(initial, guidance, confidence, None)
            adapter.freeze()
            adapter.configure(PropagationQuantConfig())

            module(initial, guidance, confidence, None)

            codes = adapter.last_coefficient_codes()
            self.assertTrue(bool(torch.all(codes.sum(dim=1) == Q13_ONE)))
            self.assertLessEqual(
                int(torch.cat((codes[:, :module.idx_ref],
                               codes[:, module.idx_ref + 1:]), dim=1)
                    .abs().sum(dim=1).max()), Q13_ONE)
            adapter.close()

    def test_official_confidence_sampler_is_resolved_from_bound_method_globals(self):
        from spn_quant.propagation.adapters import NLSPNPropagationAdapter

        module = ToyOfficialConfidenceSampler()
        adapter = NLSPNPropagationAdapter(module, model_name="nlspn")
        initial = torch.full((1, 1, 2, 3), 0.75)
        guidance = torch.ones(1, 1, 2, 3)
        confidence = torch.ones(1, 1, 2, 3)

        adapter.observe()
        output = module(initial, guidance, confidence, None)

        self.assertEqual(len(output[1]), module.prop_time)
        adapter.close()

    def test_legacy_confidence_sampling_updates_propagation_offsets(self):
        from spn_quant.propagation.adapters import NLSPNPropagationAdapter

        module = ToyLegacyConfidenceSampler()
        adapter = NLSPNPropagationAdapter(module, model_name="nlspn")
        initial = torch.full((1, 1, 2, 3), 0.75)
        guidance = torch.ones(1, 1, 2, 3)
        confidence = torch.ones(1, 1, 2, 3)

        adapter.capture()
        output = module(initial, guidance, confidence, None)

        offset = output[2]
        torch.testing.assert_close(offset[:, 0], torch.full_like(
            offset[:, 0], -1.0))
        torch.testing.assert_close(offset[:, 1], torch.full_like(
            offset[:, 1], -1.0))
        adapter.close()


class DySPNPropagationAdapterTest(unittest.TestCase):
    def test_deterministic_grid_sample_matches_official_cpu_gradients(self):
        from spn_quant.propagation.adapters import (
            deterministic_bilinear_grid_sample,
        )

        torch.manual_seed(11)
        source = torch.randn(2, 2, 4, 5, dtype=torch.float64)
        grid = torch.empty(2, 3, 4, 2, dtype=torch.float64).uniform_(
            -0.8, 0.8)
        gradient = torch.randn(2, 2, 3, 4, dtype=torch.float64)
        official_source = source.clone().requires_grad_(True)
        official_grid = grid.clone().requires_grad_(True)
        official = torch.nn.functional.grid_sample(
            official_source, official_grid, mode="bilinear",
            padding_mode="zeros", align_corners=False)
        official.backward(gradient)
        deterministic_source = source.clone().requires_grad_(True)
        deterministic_grid = grid.clone().requires_grad_(True)

        deterministic = deterministic_bilinear_grid_sample(
            deterministic_source, deterministic_grid)
        deterministic.backward(gradient)

        self.assertTrue(torch.equal(deterministic, official))
        torch.testing.assert_close(
            deterministic_source.grad, official_source.grad,
            rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            deterministic_grid.grad, official_grid.grad,
            rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_deterministic_grid_sample_has_repeatable_cuda_backward(self):
        from spn_quant.propagation.adapters import (
            deterministic_bilinear_grid_sample,
        )

        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        gradients = []
        try:
            for _ in range(2):
                source = torch.linspace(
                    -1.0, 1.0, 60, device="cuda").reshape(
                        1, 3, 4, 5).requires_grad_(True)
                grid = torch.linspace(
                    -0.8, 0.8, 24, device="cuda").reshape(
                        1, 3, 4, 2).requires_grad_(True)
                output = deterministic_bilinear_grid_sample(source, grid)
                output.square().sum().backward()
                gradients.append((source.grad.clone(), grid.grad.clone()))
        finally:
            torch.use_deterministic_algorithms(previous)

        self.assertTrue(torch.equal(gradients[0][0], gradients[1][0]))
        self.assertTrue(torch.equal(gradients[0][1], gradients[1][1]))

    def test_neighbor_grid_sample_matches_independent_official_calls(self):
        from spn_quant.propagation.adapters import (
            deterministic_bilinear_grid_sample_neighbors,
        )

        torch.manual_seed(13)
        source = torch.randn(2, 1, 4, 5, dtype=torch.float64)
        grids = torch.empty(2, 3, 3, 4, 2, dtype=torch.float64).uniform_(
            -0.8, 0.8)
        expected = torch.stack(tuple(
            torch.nn.functional.grid_sample(
                source, grids[:, neighbor], mode="bilinear",
                padding_mode="zeros", align_corners=False)
            for neighbor in range(grids.shape[1])), dim=1)

        actual = deterministic_bilinear_grid_sample_neighbors(source, grids)

        self.assertTrue(torch.equal(actual, expected))

    def test_statistics_are_scoped_to_the_current_forward(self):
        from spn_quant.propagation.adapters import DySPNPropagationAdapter

        module = ToyDySPNModule(iteration=3, num=3)
        adapter = DySPNPropagationAdapter(module)
        initial = torch.full((1, 1, 2, 3), 0.75)
        guidance = torch.ones(1, module.ch, 2, 3)
        sparse = torch.zeros_like(initial)
        confidence_logits = torch.zeros_like(initial)
        adapter.observe()
        module(initial, guidance, sparse, confidence_logits)
        adapter.freeze()
        adapter.configure(PropagationQuantConfig())

        module(initial, guidance, sparse, confidence_logits)
        first_count = len(adapter.statistics())
        module(initial, guidance, sparse, confidence_logits)
        second_count = len(adapter.statistics())

        self.assertGreater(first_count, 0)
        self.assertEqual(second_count, first_count)
        adapter.close()

    def test_dyspn_uses_exact_sum_lut_softmax_and_unsigned_confidence(self):
        from spn_quant.propagation.adapters import DySPNPropagationAdapter
        from spn_quant.propagation.fixed_point import Q13_ONE

        module = ToyDySPNModule(iteration=3, num=3)
        adapter = DySPNPropagationAdapter(module)
        initial = torch.full((1, 1, 2, 3), 0.75)
        guidance = torch.ones(1, module.ch, 2, 3)
        sparse = torch.tensor([[[[1.0, 0.0, 1.5],
                                 [0.0, 2.0, 0.0]]]])
        confidence_logits = torch.tensor([[[[-100.0, 0.0, 100.0],
                                             [-100.0, 0.0, 100.0]]]])
        adapter.observe()
        module(initial, guidance, sparse, confidence_logits)
        adapter.freeze()
        adapter.configure(PropagationQuantConfig(
            affinity_bits=4,
            confidence_bits=8,
            offset_bits=4,
            state_bits=4,
        ))

        result = module(initial, guidance, sparse, confidence_logits)

        codes = adapter.last_coefficient_codes()
        self.assertTrue(bool(torch.all(codes >= 0)))
        self.assertTrue(bool(torch.all(codes.sum(dim=2) == Q13_ONE)))
        self.assertEqual(adapter.last_confidence_codes().min().item(), 0)
        self.assertEqual(adapter.last_confidence_codes().max().item(), 255)
        anchor_rows = [
            row for row in adapter.statistics()
            if row["signal"] == "anchor_injection"
        ]
        self.assertEqual(len(anchor_rows), module.iteration)
        self.assertEqual(
            max(row["anchor_max_error"] for row in anchor_rows), 0.0)
        self.assertEqual(len(result["list_feat"]), module.iteration)
        self.assertEqual(
            result["list_feat"][-1].data_ptr(), result["pred"].data_ptr())
        adapter.close()

    def test_float_dyspn_state_dtype_is_explicit(self):
        from spn_quant.propagation.adapters import DySPNPropagationAdapter

        for state_dtype, expected_dtype in (
                ("fp32", torch.float32),
                ("bf16", torch.bfloat16),
                ("fp16", torch.float16)):
            module = ToyDySPNModule(iteration=3, num=3)
            adapter = DySPNPropagationAdapter(module)
            initial = torch.full((1, 1, 2, 3), 0.75)
            guidance = torch.ones(1, module.ch, 2, 3)
            sparse = torch.zeros_like(initial)
            confidence_logits = torch.zeros_like(initial)
            adapter.observe()
            module(initial, guidance, sparse, confidence_logits)
            adapter.freeze()
            adapter.configure_float(state_dtype)
            result = module(initial, guidance, sparse, confidence_logits)

            self.assertEqual(result["pred"].dtype, expected_dtype)
            self.assertTrue(all(
                state.dtype == expected_dtype
                for state in adapter.last_states()))
            statistics = adapter.statistics()
            self.assertEqual(sum(
                row["signal"] == "state" for row in statistics),
                module.iteration)
            self.assertEqual(sum(
                row["signal"] == "affinity_constraints"
                for row in statistics), 1)
            self.assertEqual(sum(
                row["signal"] == "anchor_injection"
                for row in statistics), module.iteration)
            adapter.close()

    def test_factory_selects_official_propagation_module_and_owned_output(self):
        from spn_quant.propagation.adapters import (
            DySPNPropagationAdapter,
            install_propagation_adapter,
            propagation_projection_outputs,
        )

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.iteration = 3
                self.num_sample = 3
                self.dyspn_3_3 = ToyDySPNModule(3, 3)

        model = Model()
        self.assertEqual(propagation_projection_outputs("dyspn", model), {
            "dyspn_3_3.conv_offset_aff",
        })
        adapter = install_propagation_adapter("dyspn", model)
        self.assertIsInstance(adapter, DySPNPropagationAdapter)
        adapter.close()

if __name__ == "__main__":
    unittest.main()
