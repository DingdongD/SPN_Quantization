import pytest
import torch
import torch.nn as nn

from spn_quant.hawq_trace import (
    HutchinsonTraceConfig,
    estimate_block_traces,
    estimate_block_trace_samples,
    masked_curvature_loss,
)


def test_masked_curvature_loss_uses_depth_and_boundary_mse():
    prediction = torch.tensor([[[[1.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)

    loss = masked_curvature_loss(
        prediction, target, valid, 1.0, 0.25, 0.5)

    assert float(loss) == 3.0


def test_hutchinson_trace_matches_diagonal_quadratic_hessian():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[1.0, 2.0]]))
    config = HutchinsonTraceConfig(probes=8, seed=7)

    def loss_fn():
        return (model.weight.square() * torch.tensor([[2.0, 5.0]])).sum()

    result = estimate_block_traces(
        (("linear", model.weight),), loss_fn, config)

    assert result[0].block == "linear"
    assert result[0].mean == 14.0
    assert result[0].normalized_mean == 7.0


def test_hutchinson_seed_is_reproducible():
    def estimate():
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.copy_(torch.tensor([[1.0, -2.0]]))
        return estimate_block_traces(
            (("linear", model.weight),),
            lambda: (model.weight.square() *
                     torch.tensor([[2.0, 5.0]])).sum(),
            HutchinsonTraceConfig(8, 11))[0]

    first = estimate()
    second = estimate()

    assert first.estimates == second.estimates
    assert first.mean == second.mean


def test_negative_final_trace_is_rejected():
    model = nn.Linear(1, 1, bias=False)

    with pytest.raises(ValueError, match="negative mean"):
        estimate_block_traces(
            (("linear", model.weight),),
            lambda: -model.weight.square().sum(),
            HutchinsonTraceConfig(2, 3))


def test_signed_probe_samples_are_available_for_cross_batch_aggregation():
    model = nn.Linear(1, 1, bias=False)

    result = estimate_block_trace_samples(
        (("linear", model.weight),),
        lambda: -model.weight.square().sum(),
        HutchinsonTraceConfig(2, 3),
    )

    assert result == (("linear", (-2.0, -2.0)),)


def test_block_trace_excludes_cross_block_hessian_terms():
    first = nn.Parameter(torch.tensor([1.0]))
    second = nn.Parameter(torch.tensor([1.0]))

    result = estimate_block_trace_samples(
        (("first", first), ("second", second)),
        lambda: first.square().sum() + second.square().sum() +
        4.0 * (first * second).sum(),
        HutchinsonTraceConfig(8, 3),
    )

    assert result == (
        ("first", (2.0,) * 8),
        ("second", (2.0,) * 8),
    )


def test_trace_rejects_duplicate_block_names():
    model = nn.Linear(2, 1, bias=False)

    with pytest.raises(ValueError, match="duplicates"):
        estimate_block_traces(
            (("linear", model.weight), ("linear", model.weight)),
            lambda: model.weight.square().sum(),
            HutchinsonTraceConfig(2, 3))
