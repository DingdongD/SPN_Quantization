import math

import pytest
import torch
import torch.nn as nn

from scripts.run_nyu_qdrop_reconstruction import (
    build_calibration_split,
    merge_contracts,
    parse_args,
    resolve_execution_order,
    select_probability_candidate,
    validate_phase_seed,
)


def test_calibration_split_is_unique_deterministic_and_eval_disjoint():
    evaluation = tuple(range(64))
    first = build_calibration_split(
        dataset_size=2000,
        calibration_samples=1024,
        reconstruction_samples=896,
        validation_samples=128,
        evaluation_indices=evaluation,
        seed=61,
    )
    second = build_calibration_split(
        dataset_size=2000,
        calibration_samples=1024,
        reconstruction_samples=896,
        validation_samples=128,
        evaluation_indices=evaluation,
        seed=61,
    )

    assert first == second
    assert len(first.calibration) == 1024
    assert len(first.reconstruction) == 896
    assert len(first.validation) == 128
    assert len(set(first.calibration)) == 1024
    assert set(first.reconstruction).isdisjoint(first.validation)
    assert set(first.calibration).isdisjoint(evaluation)


def test_probability_selection_orders_by_validation_loss_then_probability():
    rows = (
        {"probability": 0.75, "validation_loss": 0.3,
         "finite": 1, "failed_targets": 0},
        {"probability": 0.5, "validation_loss": 0.2,
         "finite": 1, "failed_targets": 0},
        {"probability": 0.25, "validation_loss": 0.2,
         "finite": 1, "failed_targets": 0},
    )

    selected = select_probability_candidate(
        rows, expected_probabilities=(0.25, 0.5, 0.75))

    assert selected["probability"] == 0.25


@pytest.mark.parametrize(
    "rows",
    (
        (
            {"probability": 0.25, "validation_loss": 0.2,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.5, "validation_loss": 0.3,
             "finite": 1, "failed_targets": 0},
        ),
        (
            {"probability": 0.25, "validation_loss": 0.2,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.5, "validation_loss": math.nan,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.75, "validation_loss": 0.3,
             "finite": 1, "failed_targets": 0},
        ),
        (
            {"probability": 0.25, "validation_loss": 0.2,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.5, "validation_loss": 0.3,
             "finite": 0, "failed_targets": 0},
            {"probability": 0.75, "validation_loss": 0.4,
             "finite": 1, "failed_targets": 1},
        ),
    ),
)
def test_probability_selection_fails_closed(rows):
    with pytest.raises(ValueError):
        select_probability_candidate(
            rows, expected_probabilities=(0.25, 0.5, 0.75))


def test_runner_requires_every_path_model_phase_and_seed():
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args([
        "--config", "qdrop.json",
        "--run-dir", "run",
        "--checkpoint", "best.pt",
        "--data-root", "data",
        "--model", "dyspn",
        "--phase", "formal",
        "--seed", "1005",
        "--out-dir", "output",
    ])
    assert args.model == "dyspn"
    assert args.phase == "formal"
    assert args.seed == 1005


class BranchedModel(nn.Module):
    def __init__(self):
        super(BranchedModel, self).__init__()
        self.left = nn.Conv2d(1, 1, 1)
        self.right = nn.Conv2d(1, 1, 1)
        self.output = nn.Conv2d(1, 1, 1)
        self.unused = nn.Conv2d(1, 1, 1)

    def forward(self, value):
        right = self.right(value)
        left = self.left(value)
        return self.output(left + right)


def test_execution_order_comes_from_forward_not_manifest_sorting():
    model = BranchedModel().eval()

    order = resolve_execution_order(
        model,
        targets=("left", "output", "right"),
        model_args=(torch.ones(1, 1, 2, 2),),
    )

    assert order == ("right", "left", "output")


def test_execution_order_requires_every_target_once():
    model = BranchedModel().eval()

    with pytest.raises(RuntimeError, match="exactly once"):
        resolve_execution_order(
            model,
            targets=("left", "right", "output", "unused"),
            model_args=(torch.ones(1, 1, 2, 2),),
        )


def test_contract_merge_rejects_cross_block_overlap():
    combined = {}
    merge_contracts(combined, {"conv": {"bits": 4}}, "weight")

    with pytest.raises(RuntimeError, match="duplicate QDrop weight"):
        merge_contracts(combined, {"conv": {"bits": 4}}, "weight")


def test_formal_phase_requires_a_configured_seed():
    validate_phase_seed("formal", 1005, (1005, 1006, 1007))
    validate_phase_seed("probability-search", 61, (1005, 1006, 1007))

    with pytest.raises(ValueError, match="formal QDrop seed"):
        validate_phase_seed("formal", 61, (1005, 1006, 1007))
