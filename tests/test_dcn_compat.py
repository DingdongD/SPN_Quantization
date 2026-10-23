import importlib.util
from pathlib import Path

import torch


_PATH = Path(__file__).resolve().parents[1] / "models" / "DCN.py"
_SPEC = importlib.util.spec_from_file_location("dcn_compat", _PATH)
DCN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(DCN)


def test_dcn_forward_maps_legacy_signature_to_torchvision(monkeypatch):
    captured = {}
    expected = torch.ones(1, 3, 4, 5)

    def fake_deform_conv2d(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(DCN, "deform_conv2d", fake_deform_conv2d)
    inputs = torch.zeros(1, 2, 4, 5)
    weight = torch.zeros(3, 2, 3, 3)
    bias = torch.zeros(3)
    offset = torch.zeros(1, 18, 4, 5)
    mask = torch.ones(1, 9, 4, 5)

    actual = DCN.modulated_deform_conv_forward(
        inputs, weight, bias, offset, mask,
        3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 64)

    assert actual is expected
    assert captured == {
        "input": inputs,
        "offset": offset,
        "weight": weight,
        "bias": bias,
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "mask": mask,
    }
