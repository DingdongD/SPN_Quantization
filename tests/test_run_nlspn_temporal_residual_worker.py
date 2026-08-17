import numpy as np
import torch

from scripts import run_nlspn_temporal_residual_worker as worker


class FakeNLSPN(torch.nn.Module):
    def forward(self, sample):
        depth = sample["dep"] + 1.0
        _, _, height, width = depth.shape
        return {
            "pred": depth,
            "pred_init": depth - 0.1,
            "guidance": depth.repeat(1, 8, 1, 1),
            "confidence": torch.full_like(depth, 0.75),
            "offset": depth.repeat(1, 16, 1, 1),
            "aff": depth.repeat(1, 9, 1, 1),
        }


def test_predict_baseline_exports_required_fields():
    rgb = np.zeros((2, 3, 4, 5), dtype=np.float32)
    sparse = np.zeros((2, 4, 5), dtype=np.float32)
    result, seconds = worker.predict_baseline(
        FakeNLSPN(), rgb, sparse, torch.device("cpu"))
    assert seconds >= 0.0
    assert set(result) == {
        "pred", "pred_init", "guidance", "confidence", "offset", "aff"}
    assert result["pred"].shape == (2, 4, 5)
    assert result["guidance"].shape == (2, 8, 4, 5)
    assert all(np.isfinite(value).all() for value in result.values())
