import math
import unittest

import torch
import torch.nn as nn

from scripts import rtn_quantization as rtn


class RTNQuantizationTest(unittest.TestCase):
    def test_weight_quantization_is_symmetric_per_output_channel(self):
        weight = torch.tensor([
            [[[1.0, -0.5], [0.25, 0.0]]],
            [[[8.0, -4.0], [2.0, 0.0]]],
        ])

        quantized, scale = rtn.symmetric_weight_qdq(weight, bits=4)

        self.assertEqual(tuple(scale.shape), (2, 1, 1, 1))
        self.assertTrue(torch.allclose(scale[:, 0, 0, 0], torch.tensor([1.0 / 7.0, 8.0 / 7.0])))
        self.assertEqual(float(quantized[0].abs().max()), 1.0)
        self.assertEqual(float(quantized[1].abs().max()), 8.0)

    def test_activation_observer_freezes_asymmetric_range_including_zero(self):
        observer = rtn.MinMaxObserver()
        observer.update(torch.tensor([1.0, 2.0, 3.0]))

        quantizer = observer.quantizer(bits=4)

        self.assertEqual(quantizer.qmin, -8)
        self.assertEqual(quantizer.qmax, 7)
        self.assertEqual(quantizer.minimum, 0.0)
        self.assertEqual(quantizer.maximum, 3.0)
        self.assertGreater(quantizer.scale, 0.0)

    def test_activation_quantizer_counts_clipped_values(self):
        quantizer = rtn.AffineTensorQuantizer(bits=4, minimum=-1.0, maximum=1.0)
        stats = rtn.QuantizationStats()

        output = quantizer(torch.tensor([-2.0, -0.25, 0.25, 2.0]), stats)

        self.assertGreaterEqual(float(output.min()), quantizer.representable_min - 1e-6)
        self.assertLessEqual(float(output.max()), quantizer.representable_max + 1e-6)
        self.assertEqual(stats.numel, 4)
        self.assertEqual(stats.saturated, 2)
        self.assertTrue(math.isfinite(stats.sqnr_db))

    def test_instrumentor_bypass_reproduces_fp32_exactly(self):
        torch.manual_seed(4)
        model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU(), nn.Conv2d(2, 1, 1)).eval()
        sample = torch.randn(1, 1, 3, 3)
        expected = model(sample).clone()
        instrumentor = rtn.RTNInstrumentor(model, lambda name, module: "feature")

        instrumentor.disable()
        actual = model(sample)

        self.assertTrue(torch.equal(actual, expected))
        instrumentor.close()

    def test_instrumentor_observes_then_quantizes_selected_group(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU(), nn.Conv2d(2, 1, 1)).eval()
        groups = {"0": "encoder", "2": "head"}
        instrumentor = rtn.RTNInstrumentor(model, lambda name, module: groups[name])
        calibration = torch.linspace(-1.0, 1.0, 9).reshape(1, 1, 3, 3)
        evaluation = calibration * 1.7

        instrumentor.observe()
        model(calibration)
        instrumentor.freeze()
        instrumentor.configure(w_bits=4, a_bits=4, enabled_groups={"head"})
        quantized = model(evaluation)
        rows = instrumentor.statistics()
        instrumentor.disable()
        fp32 = model(evaluation)

        self.assertFalse(torch.equal(quantized, fp32))
        self.assertEqual(instrumentor.module_groups(), {"0": "encoder", "2": "head"})
        self.assertTrue(any(row["module"] == "2" and row["kind"] == "output" for row in rows))
        self.assertFalse(any(row["module"] == "0" and row["kind"] == "output" for row in rows))
        instrumentor.close()


if __name__ == "__main__":
    unittest.main()
