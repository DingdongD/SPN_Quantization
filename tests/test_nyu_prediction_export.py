import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts import export_nyu_predictions as export


class PredictionExportTest(unittest.TestCase):
    def test_prepare_args_preserves_training_tf32_mode(self):
        saved = SimpleNamespace(allow_tf32=True)
        cli = SimpleNamespace(device="cuda:0")

        prepared = export.prepare_args(saved, cli)

        self.assertTrue(prepared.allow_tf32)

    def test_raw_rgb_is_not_denormalized_for_visualization(self):
        rgb = torch.full((3, 2, 2), 0.5)

        image = export.rgb_for_visualization(rgb)

        self.assertEqual(image.shape, (2, 2, 3))
        self.assertTrue(np.allclose(image, 0.5))


if __name__ == "__main__":
    unittest.main()
