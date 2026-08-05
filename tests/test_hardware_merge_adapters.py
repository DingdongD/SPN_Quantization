import unittest

import torch
import torch.nn as nn

from scripts.hardware_merge_adapters import CallIndexedConcatAdapter, SharedMergeQuantizer


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
        self.assertEqual(merge.qparams(), {
            "bits": 4, "unsigned": True, "scale": 10.0 / 15.0,
            "zero_point": 0, "qmin": 0, "qmax": 15,
        })

    def test_signed_add_branches_share_symmetric_scale(self):
        merge = SharedMergeQuantizer(unsigned=False)
        first = torch.tensor([-2.0, 1.0])
        second = torch.tensor([-7.0, 3.0])
        merge.observe((first, second))
        merge.freeze(bits=4)

        quantized = merge.quantize((first, second))

        self.assertEqual(merge.scale, 1.0)
        torch.testing.assert_close(quantized[0], first)
        torch.testing.assert_close(quantized[1], second)

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
        adapter.quantize()
        output = model(left, right)
        self.assertEqual(output[0].shape[1], 2)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
