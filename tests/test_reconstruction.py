import unittest

import torch
import torch.nn as nn

from spn_quant.adaptive_rounding import AdaptiveRoundingConfig
from spn_quant.reconstruction import (
    CalibrationRecord,
    ModuleIOCache,
    ReconstructionConfig,
    SemanticBlockReconstructor,
    reconstruction_loss,
)


class ReconstructionTest(unittest.TestCase):
    def test_fisher_loss_weights_sensitive_elements(self):
        reference = torch.tensor([0.0, 0.0])
        candidate = torch.tensor([1.0, 1.0])
        gradient = torch.tensor([1.0, 10.0])
        mse = reconstruction_loss(reference, candidate, mode="mse")
        fisher = reconstruction_loss(reference, candidate, gradient, mode="fisher")
        self.assertAlmostEqual(float(mse), 1.0)
        self.assertGreater(float(fisher), float(mse))

    def test_module_io_cache_captures_gradients(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Linear(3, 2)

            def forward(self, x):
                return self.block(x)

        model = Model()
        cache = ModuleIOCache(model, model.block)
        x = torch.randn(4, 3)
        records = cache.capture((x,), loss_closure=lambda output: output.pow(2).mean())
        cache.close()
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0].gradients)
        self.assertEqual(tuple(records[0].reference.shape), (4, 2))

    def test_adaround_layer_reconstruction_reduces_rtn_error(self):
        torch.manual_seed(14)
        teacher = nn.Linear(6, 4, bias=False)
        latent = torch.randn(256, 1)
        inputs = torch.cat([
            latent + 0.05 * torch.randn_like(latent)
            for _ in range(6)
        ], dim=1)
        with torch.no_grad():
            teacher.weight.uniform_(-0.8, 0.8)
        records = [
            CalibrationRecord((inputs[index:index + 32],),
                              teacher(inputs[index:index + 32]).detach())
            for index in range(0, 256, 32)
        ]
        config = ReconstructionConfig(
            steps=400, learning_rate=1.0e-3,
            round_loss_weight=1.0e-4, warmup_fraction=0.2,
            hard_eval_interval=10, seed=14)
        reconstructor = SemanticBlockReconstructor(
            teacher, AdaptiveRoundingConfig(bits=2), config)
        result = reconstructor.fit(records, harden=True)
        self.assertLess(result.after_loss, result.before_loss * 0.75)
        self.assertEqual(len(result.weight_manifest), 1)

    def test_brecq_block_learns_weights_and_activation_scale(self):
        torch.manual_seed(11)
        block = nn.Sequential(
            nn.Conv2d(2, 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(3, 2, 1, bias=False),
        )
        inputs = [torch.randn(4, 2, 4, 4) for _ in range(5)]
        records = [CalibrationRecord((x,), block(x).detach()) for x in inputs]
        config = ReconstructionConfig(
            steps=80, learning_rate=1.0e-2,
            activation_learning_rate=1.0e-3,
            activation_bits=4, qdrop_probability=0.1,
            round_loss_weight=1.0e-3, seed=3)
        reconstructor = SemanticBlockReconstructor(
            block, AdaptiveRoundingConfig(bits=4), config)
        result = reconstructor.fit(records, harden=True)
        reconstructor.close()
        self.assertTrue(torch.isfinite(torch.tensor(result.after_loss)))
        self.assertEqual(len(result.weight_manifest), 2)
        self.assertEqual(len(result.activation_manifest), 2)


if __name__ == "__main__":
    unittest.main()
