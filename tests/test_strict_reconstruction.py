import torch
import torch.nn as nn

from spn_quant.adaptive_rounding import AdaptiveRoundingConfig
from spn_quant.strict_reconstruction import (
    StrictBlockReconstructor,
    StrictCalibrationRecord,
    StrictReconstructionConfig,
    strict_reconstruction_loss,
)


def test_strict_reconstruction_hardens_and_exports_contracts():
    torch.manual_seed(3)
    block = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    inputs = torch.randn(16, 4)
    reference = block(inputs).detach()
    records = [
        StrictCalibrationRecord(
            inputs=(inputs[index:index + 1],),
            reference=reference[index:index + 1])
        for index in range(16)
    ]
    reconstructor = StrictBlockReconstructor(
        block,
        AdaptiveRoundingConfig(bits=4),
        StrictReconstructionConfig(
            beta_schedule="cosine",
            steps=20,
            batch_size=4,
            learning_rate=1.0e-2,
            round_loss_weight=1.0e-4),
        contract_prefix="block")

    result = reconstructor.fit(records)

    assert result.after_loss >= 0.0
    assert len(result.weight_contracts) == 2
    assert all(
        name.startswith("block.")
        for name in result.weight_contracts)
    assert not any(
        "parametrizations.weight" in name
        for name, _ in block.named_parameters())


def test_strict_reconstruction_retains_rtn_when_final_hard_state_is_worse():
    class ControlledReconstructor(StrictBlockReconstructor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.evaluations = iter((1.0, 2.0, 1.0))

        def _evaluate(self, records):
            del records
            return next(self.evaluations)

    block = nn.Linear(2, 2, bias=False)
    records = [StrictCalibrationRecord(
        inputs=(torch.ones(1, 2),),
        reference=torch.zeros(1, 2),
    )]
    reconstructor = ControlledReconstructor(
        block,
        AdaptiveRoundingConfig(bits=4),
        StrictReconstructionConfig(
            beta_schedule="cosine",
            steps=1,
            batch_size=1,
            round_loss_weight=0.0),
    )

    result = reconstructor.fit(records)

    assert result.retained_rtn
    assert result.before_loss == result.after_loss


def test_fisher_diagonal_and_full_losses_are_supported():
    reference = torch.zeros(2, 3, 2, 2)
    candidate = torch.ones_like(reference)
    gradient = torch.ones_like(reference) * 2.0

    diagonal = strict_reconstruction_loss(
        reference, candidate, gradient,
        mode="fisher_diag")
    full = strict_reconstruction_loss(
        reference, candidate, gradient,
        mode="fisher_full")

    assert float(diagonal.item()) > 0.0
    assert float(full.item()) > 0.0


def test_mse_loss_matches_official_channel_sum_reduction():
    reference = torch.zeros(2, 3, 2, 2)
    candidate = torch.ones_like(reference)

    loss = strict_reconstruction_loss(
        reference, candidate, mode="mse")

    assert float(loss.item()) == 3.0


def test_fisher_loss_requires_gradients():
    reference = torch.zeros(1, 2)
    candidate = torch.ones_like(reference)

    try:
        strict_reconstruction_loss(
            reference, candidate,
            gradients=None,
            mode="fisher_diag")
    except ValueError as error:
        assert "requires output gradients" in str(error)
    else:
        raise AssertionError("missing Fisher gradients were accepted")


def test_strict_config_rejects_unknown_loss():
    try:
        StrictReconstructionConfig(
            round_loss_weight=1.0e-3,
            beta_schedule="cosine",
            loss="fisher")
    except ValueError as error:
        assert "unknown strict reconstruction loss" in str(error)
    else:
        raise AssertionError("non-canonical Fisher mode was accepted")
