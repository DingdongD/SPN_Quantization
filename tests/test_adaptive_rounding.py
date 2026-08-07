import unittest

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
    AdaptiveRoundingParametrization,
    CosineTemperatureDecay,
    LinearTemperatureDecay,
    select_weight_modules,
)


class AdaptiveRoundingTest(unittest.TestCase):
    def test_soft_rounding_has_gradients_and_hard_weight_is_on_grid(self):
        torch.manual_seed(1)
        module = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        config = AdaptiveRoundingConfig(bits=4)
        rounding = AdaptiveRoundingParametrization(module, module.weight, config)
        quantized = rounding(module.weight)
        quantized.sum().backward()
        self.assertIsNotNone(rounding.alpha.grad)
        self.assertGreater(float(rounding.alpha.grad.abs().sum()), 0.0)

        rounding.soft_targets = False
        hard = rounding(module.weight)
        scale = rounding.scale
        codes = hard / scale
        torch.testing.assert_close(codes, torch.round(codes))
        self.assertLessEqual(float(codes.detach().abs().max()), 7.0)

    def test_controller_hardens_conv_linear_and_transposed_conv(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 3, 3, padding=1)
                self.up = nn.ConvTranspose2d(3, 2, 2, stride=2)
                self.fc = nn.Linear(8, 4)

            def forward(self, x):
                y = self.up(self.conv(x))
                return self.fc(y[:, :, :2, :2].reshape(x.shape[0], -1))

        model = Model()
        controller = AdaptiveRoundingController(model)
        controller.install(("conv", "up", "fc"))
        self.assertTrue(parametrize.is_parametrized(model.conv, "weight"))
        output = model(torch.randn(2, 2, 2, 2))
        self.assertEqual(tuple(output.shape), (2, 4))
        manifest = controller.harden()
        self.assertEqual(len(manifest), 3)
        self.assertFalse(parametrize.is_parametrized(model.conv, "weight"))
        self.assertTrue(all(row["hardened"] == 1 for row in manifest))

    def test_regex_selection_and_temperature_schedule(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU(), nn.Conv2d(2, 2, 1))
        self.assertEqual(select_weight_modules(model, (r"^0$",)), ["0"])
        schedule = LinearTemperatureDecay(10, warmup_fraction=0.2,
                                          beta_start=20.0, beta_end=2.0)
        self.assertIsNone(schedule(0))
        self.assertAlmostEqual(schedule(9), 2.0)

        cosine = CosineTemperatureDecay(
            10, warmup_fraction=0.2,
            beta_start=20.0, beta_end=2.0)
        self.assertIsNone(cosine(0))
        self.assertAlmostEqual(cosine(2), 20.0)
        self.assertAlmostEqual(cosine(9), 2.0)
        self.assertGreater(cosine(5), schedule(5))


if __name__ == "__main__":
    unittest.main()
