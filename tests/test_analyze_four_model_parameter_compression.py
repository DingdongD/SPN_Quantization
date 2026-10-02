from types import SimpleNamespace

import pytest
import torch

from scripts import analyze_four_model_parameter_compression as compression


parameter_storage = compression.parameter_storage


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.nn.Linear(4, 2, bias=True)
        self.prop_layer = torch.nn.Linear(2, 1, bias=True)


def test_parameter_storage_separates_integer_fp16_and_fp32_parameters():
    model = Toy()
    contract = SimpleNamespace(search_units=(
        SimpleNamespace(name="encoder", members=("quant",)),))

    result = parameter_storage(
        model, contract,
        {"weight_bits": {"encoder": 4}}, "nlspn")

    # quant.weight: 8x4b; prop_layer weight+bias: 3x16b;
    # quant.bias: 2x32b.
    assert result["parameters"] == 13
    assert result["packed_weight_bits"] == 144
    assert result["parameter_weighted_bits"] == pytest.approx(144 / 13)
    assert result["parameter_counts_by_bits"] == {
        "4": 8, "16": 3, "32": 2}


def test_optional_structured_candidate_is_applied(monkeypatch):
    calls = []
    monkeypatch.setattr(
        compression.structured, "apply_structured_candidate",
        lambda model, model_name, candidate_id: calls.append(
            (model, model_name, candidate_id)) or {"ratio": 0.625})
    model = object()

    assert compression.apply_structured_candidate(
        model, "nlspn", "bridge_62p5pct") == {"ratio": 0.625}
    assert calls == [(model, "nlspn", "bridge_62p5pct")]
    assert compression.apply_structured_candidate(model, "nlspn", None) == {}
