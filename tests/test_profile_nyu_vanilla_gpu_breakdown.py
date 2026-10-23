import pytest
import torch

from scripts.profile_nyu_vanilla_gpu_breakdown import resolve_module


class Nested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.propagation = torch.nn.Identity()


def test_resolve_module_supports_dotted_paths():
    model = Nested()
    assert resolve_module(model, "block.propagation") is model.block.propagation


def test_resolve_module_rejects_missing_target():
    with pytest.raises(RuntimeError, match="no propagation module"):
        resolve_module(Nested(), "block.missing")
