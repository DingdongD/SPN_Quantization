import unittest

import torch
import torch.nn as nn

from scripts import propagation_quantization as propagation


class ToyNLSPN(nn.Module):
    def __init__(self, iterations=3):
        super().__init__()
        self.prop_time = iterations

    def _propagate_once(self, feat, offset, affinity):
        return feat * affinity + offset

    def forward(self, feat, offset, affinity):
        states = []
        for _ in range(self.prop_time):
            feat = self._propagate_once(feat, offset, affinity)
            states.append(feat)
        return feat, states


class PropagationQuantizationTest(unittest.TestCase):
    def test_disabled_adapter_is_exact_bypass(self):
        module = ToyNLSPN()
        sample = torch.tensor([0.25, 0.75])
        expected = module(sample, torch.tensor(0.1), torch.tensor(0.9))[0]
        adapter = propagation.NLSPNStateAdapter(module)

        adapter.disable()
        actual = module(sample, torch.tensor(0.1), torch.tensor(0.9))[0]

        self.assertTrue(torch.equal(actual, expected))
        adapter.close()

    def test_state_quantization_is_applied_after_every_iteration(self):
        module = ToyNLSPN(iterations=3)
        adapter = propagation.NLSPNStateAdapter(module)
        sample = torch.tensor([0.25, 0.75])
        offset = torch.tensor(0.1)
        affinity = torch.tensor(0.9)

        adapter.observe()
        module(sample, offset, affinity)
        adapter.freeze()
        adapter.configure(bits=2)
        quantized, states = module(sample, offset, affinity)
        rows = adapter.statistics()
        adapter.disable()
        fp32, _ = module(sample, offset, affinity)

        self.assertFalse(torch.equal(quantized, fp32))
        self.assertEqual(len(states), 3)
        self.assertEqual([row["iteration"] for row in rows], [1, 2, 3])
        self.assertTrue(all(row["numel"] > 0 for row in rows))
        adapter.close()

    def test_capture_mode_records_unquantized_iteration_states(self):
        module = ToyNLSPN(iterations=2)
        adapter = propagation.NLSPNStateAdapter(module)

        adapter.capture()
        module(torch.tensor([1.0]), torch.tensor(0.5), torch.tensor(0.5))

        self.assertEqual(len(adapter.last_states()), 2)
        self.assertAlmostEqual(float(adapter.last_states()[0]), 1.0)
        self.assertAlmostEqual(float(adapter.last_states()[1]), 1.0)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
