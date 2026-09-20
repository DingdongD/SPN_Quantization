import copy

import torch
import torch.nn as nn

from spn_quant.nas.weights import transfer_prefix_state


class TransferTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 3, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.extra = nn.Parameter(torch.full((2,), -7.0))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))


def _source_state():
    return {
        "conv.weight": torch.arange(5 * 4 * 3 * 3, dtype=torch.float32).reshape(5, 4, 3, 3),
        "bn.weight": torch.arange(5, dtype=torch.float32) + 1,
        "bn.bias": torch.arange(5, dtype=torch.float32) + 10,
        "bn.running_mean": torch.arange(5, dtype=torch.float32) + 20,
        "bn.running_var": torch.arange(5, dtype=torch.float32) + 30,
        "bn.num_batches_tracked": torch.tensor(9, dtype=torch.long),
        "counter": torch.tensor(4, dtype=torch.long),
        "unexpected": torch.ones(1),
    }


def test_prefix_transfer_copies_overlaps_and_reports_missing_keys():
    target = TransferTarget()
    source = _source_state()
    original_extra = target.extra.detach().clone()

    report = transfer_prefix_state(target, source)

    assert torch.equal(target.conv.weight, source["conv.weight"][:3])
    assert torch.equal(target.bn.weight, source["bn.weight"][:3])
    assert int(target.counter.item()) == 4
    assert torch.equal(target.extra, original_extra)
    assert "conv.weight" in report["partial"]
    assert "counter" in report["copied"]
    assert report["missing"] == ["extra"]
    assert report["unexpected"] == ["unexpected"]


def test_prefix_transfer_skips_rank_or_dtype_mismatch_without_mutating_source():
    target = TransferTarget()
    source = _source_state()
    source["conv.weight"] = torch.ones(3, dtype=torch.float32)
    source["bn.weight"] = source["bn.weight"].double()
    original_source = copy.deepcopy(source)
    original_conv = target.conv.weight.detach().clone()

    first = transfer_prefix_state(target, source)
    second_target = TransferTarget()
    second_target.load_state_dict(TransferTarget().state_dict())
    second = transfer_prefix_state(second_target, source)

    assert torch.equal(target.conv.weight, original_conv)
    assert "conv.weight" in first["skipped"]
    assert "bn.weight" in first["skipped"]
    assert first == second
    assert all(torch.equal(source[key], original_source[key]) for key in source)
