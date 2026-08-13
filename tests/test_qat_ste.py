import pytest
import torch

from spn_quant.qat.ste import hard_forward_proxy, round_ste


def test_hard_forward_proxy_returns_hard_and_routes_proxy_gradient():
    hard = torch.tensor([1.0, -2.0])
    proxy = torch.tensor([0.25, 0.5], requires_grad=True)

    output = hard_forward_proxy(hard, proxy)

    assert torch.equal(output, hard)
    output.sum().backward()
    assert torch.equal(proxy.grad, torch.ones_like(proxy))


def test_round_ste_matches_round_and_has_identity_gradient():
    value = torch.tensor([-1.6, -0.4, 0.4, 1.6], requires_grad=True)

    output = round_ste(value)

    assert torch.equal(output, torch.round(value.detach()))
    output.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_hard_forward_proxy_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        hard_forward_proxy(
            torch.zeros(2), torch.zeros(3, requires_grad=True))


@pytest.mark.parametrize("function", (hard_forward_proxy, round_ste))
def test_ste_rejects_nonfinite_values(function):
    value = torch.tensor([float("nan")], requires_grad=True)
    with pytest.raises(FloatingPointError, match="non-finite"):
        if function is hard_forward_proxy:
            function(torch.zeros_like(value), value)
        else:
            function(value)
