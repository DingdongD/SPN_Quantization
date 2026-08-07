import torch.nn as nn

from scripts.run_nyu_strict_reconstruction import (
    method_beta_schedule,
    method_round_loss_weight,
    method_steps,
    method_warmup_fraction,
    maximum_primary_fold_error,
    parse_args,
    select_targets,
)


def test_fold_validation_uses_primary_depth_output_error():
    teacher = {
        "max_abs_error": 4.0,
        "primary_max_abs_error": 1.0e-6,
    }
    student = {
        "max_abs_error": 6.0,
        "primary_max_abs_error": 2.0e-6,
    }

    assert maximum_primary_fold_error(teacher, student) == 2.0e-6


def test_runner_accepts_explicit_data_root():
    args = parse_args([
        "--run-dir", "run",
        "--method", "adaround_strict",
        "--data-root", "dataset",
    ])

    assert args.data_root == "dataset"


def test_adaround_accepts_one_weight_block_with_post_activation():
    model = nn.Module()
    model.head = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
    )

    targets = select_targets(
        model, ["head"], [], "adaround_strict")

    assert targets == ["head"]


def test_adaround_rejects_block_with_multiple_weights():
    model = nn.Module()
    model.head = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 1, 1),
    )

    try:
        select_targets(
            model, ["head"], [], "adaround_strict")
    except TypeError as error:
        assert "exactly one supported weight" in str(error)
    else:
        raise AssertionError("multi-weight AdaRound block was accepted")


def test_reconstruction_method_uses_official_round_loss_weight():
    assert method_round_loss_weight("adaround_strict", None) == 1.0e-2
    assert method_round_loss_weight("brecq_strict", None) == 1.0e-2
    assert method_round_loss_weight("adaround_strict", 0.25) == 0.25


def test_reconstruction_method_uses_reference_schedule_defaults():
    assert method_steps("adaround_strict", None) == 15000
    assert method_steps("brecq_strict", None) == 20000
    assert method_warmup_fraction("adaround_strict", None) == 0.2
    assert method_warmup_fraction("brecq_strict", None) == 0.0
    assert method_beta_schedule("adaround_strict", None) == "cosine"
    assert method_beta_schedule("brecq_strict", None) == "linear"
