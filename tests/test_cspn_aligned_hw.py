from pathlib import Path
import sys

import pytest
import torch

MODELS = Path(__file__).resolve().parents[1] / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from cspn_aligned_hw import FastAffinityPropagate


def test_bf16_state_rounds_each_propagation_iteration():
    module = FastAffinityPropagate(1, 3)
    guidance = torch.zeros(1, 8, 2, 2)
    depth = torch.tensor([[[[1.001, 2.003], [3.007, 4.009]]]])

    fp32 = module(guidance, depth)
    module.configure_state_dtype("bf16")
    bf16 = module(guidance, depth)

    assert torch.equal(fp32, depth)
    assert torch.equal(bf16, depth.to(torch.bfloat16).float())
    assert not torch.equal(bf16, fp32)


def test_propagation_state_dtype_rejects_unknown_format():
    module = FastAffinityPropagate(1, 3)

    with pytest.raises(ValueError, match="state dtype"):
        module.configure_state_dtype("tf32")
