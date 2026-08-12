import unittest

import torch
import torch.nn as nn

from spn_quant.runtime import EdgeQDQRuntime
from scripts.hardware_merge_adapters import (
    CallIndexedAddAdapter,
    CallIndexedConcatAdapter,
    SharedMergeQuantizer,
)


class SharedMergeQuantizerTest(unittest.TestCase):
    def test_unsigned_concat_branches_share_one_w4_scale(self):
        merge = SharedMergeQuantizer(unsigned=True)
        small = torch.tensor([0.0, 0.1])
        large = torch.tensor([0.0, 10.0])
        merge.observe((small, large))
        merge.freeze(bits=4)
        quantized = merge.quantize((small, large))
        self.assertAlmostEqual(merge.scale, 10.0 / 15.0)
        self.assertEqual(float(quantized[0][1]), 0.0)

    def test_independent_concat_preserves_small_branch_resolution(self):
        class Decoder(nn.Module):
            def _concat(self, left, right, dim=1):
                return torch.cat((left, right), dim=dim)

            def forward(self, left, right):
                return self._concat(left, right)

        model = Decoder()
        adapter = CallIndexedConcatAdapter(model, policy="independent")
        small = torch.tensor([[[[0.0, 0.1]]]])
        large = torch.tensor([[[[0.0, 10.0]]]])
        adapter.observe()
        model(small, large)
        adapter.freeze(bits=4)
        adapter.quantize()
        output = model(small, large)
        self.assertGreater(float(output[0, 0, 0, 1]), 0.0)
        manifest = adapter.manifest()[0]
        self.assertEqual(manifest["policy"], "independent")
        adapter.close()

    def test_grouped_concat_uses_distinct_channel_group_scales(self):
        class Decoder(nn.Module):
            def _concat(self, left, right, dim=1):
                return torch.cat((left, right), dim=dim)

            def forward(self, left, right):
                return self._concat(left, right)

        model = Decoder()
        adapter = CallIndexedConcatAdapter(
            model, policy="grouped", group_size=1)
        small = torch.tensor([[[[0.0, 0.1]]]])
        large = torch.tensor([[[[0.0, 10.0]]]])
        adapter.observe()
        model(small, large)
        adapter.freeze(bits=4)
        adapter.quantize()
        output = model(small, large)
        self.assertGreater(float(output[0, 0, 0, 1]), 0.0)
        manifest = adapter.manifest()[0]
        self.assertEqual(manifest["groups"], 2)
        adapter.close()

    def test_concat_calls_at_different_decoder_stages_get_distinct_scales(self):
        class Decoder(nn.Module):
            def _concat(self, left, right, dim=1):
                return torch.cat((left, right), dim=dim)

            def forward(self, left, right):
                first = self._concat(left, right)
                second = self._concat(left * 10.0, right * 10.0)
                return first, second

        model = Decoder()
        adapter = CallIndexedConcatAdapter(model)
        left = torch.ones(1, 1, 2, 2)
        right = torch.ones(1, 1, 2, 2) * 2.0
        adapter.observe()
        model(left, right)
        adapter.freeze(bits=4)
        rows = adapter.manifest()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["scale"], rows[1]["scale"])
        adapter.close()

    def test_add_adapter_quantizes_branches_before_wide_add(self):
        class Residual(nn.Module):
            def _add(self, left, right):
                return left + right

            def forward(self, left, right):
                return self._add(left, right)

        model = Residual()
        adapter = CallIndexedAddAdapter(model, policy="independent")
        left = torch.tensor([0.1])
        right = torch.tensor([10.0])
        adapter.observe()
        model(left, right)
        adapter.freeze(bits=4)
        adapter.quantize()
        output = model(left, right)
        self.assertGreater(float(output.item()), 10.0)
        self.assertEqual(adapter.manifest()[0]["operation"], "add")
        adapter.close()

    def test_residual_add_uses_a4_update_a8_base_and_a8_output(self):
        class Residual(nn.Module):
            def _add(self, update, base):
                return update + base

            def forward(self, update, base):
                return self._add(update, base)

        model = Residual()
        adapter = CallIndexedAddAdapter(model, policy="residual")
        update = torch.tensor([-0.2, 0.1])
        base = torch.tensor([10.0, 8.0])
        adapter.observe()
        model(update, base)
        adapter.freeze(bits=4)
        adapter.quantize()

        output = model(update, base)
        row = adapter.manifest()[0]

        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(row["policy"], "residual")
        self.assertEqual(row["branch_bits"], "4;8")
        self.assertEqual(row["output_bits"], 8)
        self.assertGreater(row["base_to_update_rms_ratio"], 1.0)
        self.assertLess(row["update_to_base_energy_ratio"], 1.0)
        self.assertIn("branch_new_zero_rates", row)
        adapter.close()

    def test_concat_output_is_reused_by_downstream_edge_runtime(self):
        class Decoder(nn.Module):
            def _concat(self, left, right, dim=1):
                return torch.cat((left, right), dim=dim)

            def forward(self, left, right):
                return self._concat(left, right)

        class Counting(object):
            def __init__(self):
                self.calls = 0

            def __call__(self, tensor):
                self.calls += 1
                return tensor

        runtime = EdgeQDQRuntime()
        model = Decoder()
        adapter = CallIndexedConcatAdapter(
            model, policy="independent", runtime=runtime, manage_runtime=False)
        left = torch.tensor([[[[0.1]]]])
        right = torch.tensor([[[[10.0]]]])
        adapter.observe()
        runtime.begin_forward()
        model(left, right)
        adapter.freeze(bits=4)
        adapter.quantize()
        runtime.begin_forward()
        output = model(left, right)
        quantizer = Counting()
        reused = runtime.process("consumer", output, quantizer)
        self.assertIs(reused, output)
        self.assertEqual(quantizer.calls, 0)
        adapter.close()

    def test_expected_call_count_fails_closed(self):
        class Decoder(nn.Module):
            def _concat(self, left, right, dim=1):
                return torch.cat((left, right), dim=dim)

            def forward(self, left, right):
                return self._concat(left, right)

        model = Decoder()
        adapter = CallIndexedConcatAdapter(model, expected_calls=2)
        adapter.observe()
        model(torch.ones(1, 1), torch.ones(1, 1))
        with self.assertRaises(RuntimeError):
            adapter.freeze(bits=4)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
