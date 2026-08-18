import json
from pathlib import Path

import pytest
import torch.nn as nn

from scripts.run_nyu_strict_reconstruction import (
    method_beta_schedule,
    method_round_loss_weight,
    method_steps,
    method_warmup_fraction,
    maximum_primary_fold_error,
    load_persisted_protocol,
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


def test_runner_requires_persisted_protocol_paths():
    args = parse_args([
        "--run-dir", "run",
        "--method", "adaround_strict",
        "--data-root", "dataset",
        "--calibration-indices", "calibration_indices.json",
        "--calibration-metadata", "calibration_metadata.json",
        "--evaluation-protocol", "evaluation_protocol.json",
    ])

    assert args.data_root == "dataset"
    assert args.calibration_indices == "calibration_indices.json"


def _write_protocol(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    calibration_indices = tmp_path / "calibration_indices.json"
    calibration_indices.write_text(json.dumps({
        "indices": list(range(128)),
        "count": 128,
        "selection": "stratified",
    }), encoding="utf-8")
    from spn_quant.deployment_contract import file_sha256
    calibration_metadata = tmp_path / "calibration_metadata.json"
    calibration_metadata.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_root": str(data_root),
    }), encoding="utf-8")
    evaluation_protocol = tmp_path / "evaluation_protocol.json"
    evaluation_protocol.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "seed": 20260812,
        "evaluation_samples": 64,
        "evaluation_indices": list(range(64)),
    }), encoding="utf-8")
    return checkpoint, data_root, calibration_indices, \
        calibration_metadata, evaluation_protocol


def test_persisted_protocol_preserves_exact_calibration_order(tmp_path):
    checkpoint, data_root, indices, metadata, evaluation = \
        _write_protocol(tmp_path)

    protocol, provenance = load_persisted_protocol(
        calibration_indices_path=indices,
        calibration_metadata_path=metadata,
        evaluation_protocol_path=evaluation,
        checkpoint=checkpoint,
        data_root=data_root,
        seed=20260812,
    )

    assert protocol.calibration_indices == tuple(range(128))
    assert protocol.evaluation_indices == tuple(range(64))
    assert protocol.seed == 20260812
    assert provenance["checkpoint_sha256"]
    assert provenance["calibration_indices_sha256"]


def test_persisted_protocol_rejects_changed_seed(tmp_path):
    checkpoint, data_root, indices, metadata, evaluation = \
        _write_protocol(tmp_path)

    with pytest.raises(ValueError, match="seed"):
        load_persisted_protocol(
            calibration_indices_path=indices,
            calibration_metadata_path=metadata,
            evaluation_protocol_path=evaluation,
            checkpoint=checkpoint,
            data_root=data_root,
            seed=17,
        )


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
