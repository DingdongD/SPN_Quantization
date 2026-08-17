from pathlib import Path

import pytest
import torch

from scripts import raft_small_compat as compat


def test_raft_small_matches_official_parameter_contract():
    model = compat.raft_small()
    weights = torch.load(
        str(compat.DEFAULT_WEIGHT_PATH), map_location="cpu")
    assert sum(parameter.numel() for parameter in model.parameters()) == 990162
    assert set(model.state_dict()) == set(weights)


def test_prepare_pair_pads_normalizes_and_preserves_direction():
    previous = torch.zeros(1, 3, 228, 304)
    current = torch.ones(1, 3, 228, 304)
    image1, image2 = compat.prepare_backward_pair(current, previous)
    assert image1.shape == image2.shape == (1, 3, 232, 304)
    assert torch.all(image1 == 1.0)
    assert torch.all(image2 == -1.0)


def test_load_official_weights_is_strict(tmp_path):
    bad = tmp_path / "bad.pth"
    torch.save({}, str(bad))
    with pytest.raises(RuntimeError):
        compat.load_official_weights(compat.raft_small(), bad)


class FakeRAFT(torch.nn.Module):
    def forward(self, current, previous, num_flow_updates=12):
        assert current.shape == previous.shape == (1, 3, 232, 304)
        assert num_flow_updates == 12
        return [torch.ones(1, 2, 232, 304)]


def test_predict_backward_flow_crops_back_to_nlspn_geometry():
    flow = compat.predict_backward_flow(
        FakeRAFT(),
        torch.zeros(1, 3, 228, 304),
        torch.zeros(1, 3, 228, 304),
    )
    assert flow.shape == (1, 2, 228, 304)
    assert torch.all(flow == 1.0)
