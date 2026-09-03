import pytest
import torch

from spn_quant.task_sensitivity import (
    candidate_sensitivity_scores,
    gradient_weighted_error,
)


def test_candidate_sensitivity_scores_cover_exact_bit_levels():
    gradient = torch.tensor([2.0, -1.0])
    reference = torch.tensor([4.0, -2.0])
    quantized = {
        4: torch.tensor([3.0, -1.0]),
        6: torch.tensor([3.5, -1.5]),
        8: torch.tensor([3.9, -1.9]),
    }

    scores = candidate_sensitivity_scores(
        gradient, reference, quantized)

    assert tuple(scores) == (4, 6, 8)
    assert scores[4] == pytest.approx(3.0)
    assert scores[8] == pytest.approx(0.3)


def test_candidate_sensitivity_scores_reject_incomplete_levels():
    tensor = torch.ones(2)

    with pytest.raises(ValueError, match="score levels"):
        candidate_sensitivity_scores(
            tensor, tensor, {4: tensor, 8: tensor})


def test_candidate_sensitivity_scores_matches_task_gradient_formula():
    gradient = torch.tensor([2.0, -3.0])
    reference = torch.tensor([4.0, -5.0])
    quantized = {4: torch.tensor([3.0, -4.0]),
                 6: torch.tensor([4.0, -5.0]),
                 8: torch.tensor([4.0, -5.0])}

    scores = candidate_sensitivity_scores(gradient, reference, quantized)

    assert scores[4] == pytest.approx(
        gradient_weighted_error(gradient, reference, quantized[4]))
