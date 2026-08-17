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


class FakePropLayer(torch.nn.Module):
    def forward(self, feat_init, guidance, confidence, feat_fix, rgb):
        result = feat_init + guidance[:, :1] * 0.0
        return result, [result], None, None, torch.tensor(1.0)


class FakePropagationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prop_layer = FakePropLayer()


def test_reconstruct_pair_uses_sparse_signed_residual_and_original_prop():
    model = FakePropagationModel()
    previous_depth = torch.full((1, 1, 2, 3), 2.0)
    current_sparse = torch.tensor([[[[0.0, 1.5, 0.0],
                                     [3.0, 0.0, 0.0]]]])
    flow = torch.zeros((1, 2, 2, 3))
    guidance = torch.zeros((1, 8, 2, 3))
    confidence = torch.ones((1, 1, 2, 3))
    reconstructed, dense_residual = worker.reconstruct_pair(
        model, previous_depth, current_sparse, flow,
        guidance, confidence, torch.zeros((1, 3, 2, 3)))
    np.testing.assert_allclose(
        dense_residual.numpy()[0, 0],
        [[0.0, -0.5, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        reconstructed.numpy()[0, 0],
        [[2.0, 1.5, 2.0], [3.0, 2.0, 2.0]])
