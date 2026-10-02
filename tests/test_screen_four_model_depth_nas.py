import pytest
import torch.nn as nn

from scripts import screen_four_model_depth_nas as screen


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.stage1 = nn.Sequential(
            nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4))
        self.encoder.stage2 = nn.ModuleList(
            [nn.Linear(4, 4), nn.Linear(4, 4)])


def test_apply_depths_preserves_container_type_and_restores_full_stage():
    model = ToyModel()
    paths = ("encoder.stage1", "encoder.stage2")
    full = screen.stage_modules(model, paths)

    screen.apply_depths(model, paths, full, (1, 1))

    assert isinstance(model.encoder.stage1, nn.Sequential)
    assert isinstance(model.encoder.stage2, nn.ModuleList)
    assert len(model.encoder.stage1) == 1
    assert len(model.encoder.stage2) == 1

    screen.apply_depths(model, paths, full, (3, 2))
    assert len(model.encoder.stage1) == 3
    assert len(model.encoder.stage2) == 2


def test_apply_depths_rejects_zero_or_excess_depth():
    model = ToyModel()
    paths = ("encoder.stage1", "encoder.stage2")
    full = screen.stage_modules(model, paths)

    with pytest.raises(ValueError, match="invalid depth"):
        screen.apply_depths(model, paths, full, (0, 1))
    with pytest.raises(ValueError, match="invalid depth"):
        screen.apply_depths(model, paths, full, (3, 3))
