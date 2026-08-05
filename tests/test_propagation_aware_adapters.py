import unittest
import importlib.util
from pathlib import Path

import torch

from spn_quant.propagation.controller import PropagationQuantConfig


_CSPN_PATH = Path(__file__).resolve().parents[1] / "models" / "cspn.py"
_CSPN_SPEC = importlib.util.spec_from_file_location("official_cspn", _CSPN_PATH)
_CSPN_MODULE = importlib.util.module_from_spec(_CSPN_SPEC)
_CSPN_SPEC.loader.exec_module(_CSPN_MODULE)
Affinity_Propagate = _CSPN_MODULE.Affinity_Propagate


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


if __name__ == "__main__":
    unittest.main()
