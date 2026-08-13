import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts import train_nyu_cspn_group_a4_qat as runner


def _training_values():
    return {
        "epochs": 30,
        "patience": 6,
        "min_relative_improvement": 0.001,
        "batch_size": 4,
        "val_batch_size": 1,
        "workers": 2,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "seed": 20260812,
    }


def test_training_config_requires_every_field():
    values = _training_values()
    assert runner.TrainingConfig(**values).epochs == 30
    del values["patience"]
    with pytest.raises(TypeError):
        runner.TrainingConfig(**values)


def test_tracker_stops_after_six_insignificant_epochs():
    tracker = runner.QATConvergenceTracker(
        max_epochs=30, patience=6, min_relative_improvement=0.001)
    values = (0.3000, 0.2999, 0.29985, 0.29982,
              0.29981, 0.29980, 0.29979)

    stopped = [tracker.update(epoch + 1, value)
               for epoch, value in enumerate(values)]

    assert stopped[-1]
    assert tracker.reason == "validation_plateau"
    assert tracker.best_rmse == min(values)
    assert tracker.best_epoch == len(values)


def test_load_calibration_metadata_requires_stratified_128(tmp_path):
    path = tmp_path / "metadata.json"
    payload = {
        "calibration_indices": list(range(128)),
        "evaluation_indices": list(range(64)),
        "calibration_source": {
            "type": "index_file",
            "path": "indices.json",
            "sha256": "abc",
            "selection": "32_tail_96_kmedoids",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    metadata = runner.load_calibration_metadata(path)

    assert metadata.calibration_indices == tuple(range(128))
    assert metadata.evaluation_indices == tuple(range(64))

    payload["calibration_source"]["selection"] = "random"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stratified"):
        runner.load_calibration_metadata(path)


def test_resume_contract_rejects_changed_mode():
    contract = runner.checkpoint_contract(
        mode="static", calibration_indices=tuple(range(128)),
        owner_manifest=("owner",))
    with pytest.raises(ValueError, match="quantization mode"):
        runner.validate_resume_contract(
            contract, mode="dynamic", calibration_indices=tuple(range(128)),
            owner_manifest=("owner",))


def test_resume_contract_rejects_changed_calibration_indices():
    contract = runner.checkpoint_contract(
        mode="static", calibration_indices=tuple(range(128)),
        owner_manifest=("owner",))
    changed = tuple(range(127)) + (256,)
    with pytest.raises(ValueError, match="calibration indices"):
        runner.validate_resume_contract(
            contract, mode="static", calibration_indices=changed,
            owner_manifest=("owner",))


def test_set_qat_train_mode_freezes_batchnorm():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())

    runner.set_qat_train_mode(model)

    assert model.training
    assert not model[1].training
    assert all(not parameter.requires_grad
               for parameter in model[1].parameters())
    assert model[0].weight.requires_grad


def test_assert_finite_parameters_rejects_nan():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="parameter"):
        runner.assert_finite_parameters(model)


def _cli_values():
    return [
        "--mode", "static",
        "--checkpoint", "/models/best.pt",
        "--data-root", "/data/nyu",
        "--calibration-metadata", "/runs/metadata.json",
        "--output-root", "/runs/qat",
        "--device", "cuda:0",
        "--epochs", "30",
        "--patience", "6",
        "--min-relative-improvement", "0.001",
        "--batch-size", "4",
        "--val-batch-size", "1",
        "--workers", "2",
        "--learning-rate", "0.001",
        "--momentum", "0.9",
        "--weight-decay", "0.0001",
        "--seed", "20260812",
        "--max-train-samples", "0",
        "--max-val-samples", "0",
        "--fold-max-error", "0.001",
        "--log-interval", "100",
    ]


def test_parse_args_requires_explicit_experiment_values():
    args = runner.parse_args(_cli_values())

    assert args.mode == "static"
    assert args.max_train_samples == 0
    assert args.resume is None

    missing = _cli_values()
    position = missing.index("--fold-max-error")
    del missing[position:position + 2]
    with pytest.raises(SystemExit):
        runner.parse_args(missing)


def test_owner_manifest_is_stable_and_complete():
    manifest = {
        "weight_modules": ["decoder", "encoder"],
        "ordinary_owners": ["('conv', 'input')", "relu#0"],
        "structural_owners": ["decoder_entry", "layer4_signed_skip"],
    }

    owners = runner.owner_manifest_tuple(manifest)

    assert owners == (
        "weight:decoder", "weight:encoder",
        "activation:('conv', 'input')", "activation:relu#0",
        "structural:decoder_entry", "structural:layer4_signed_skip",
    )


def test_checkpoint_reconstruction_skips_pretrained_initialization(
        monkeypatch):
    payload = {"args": {
        "model": "cspn",
        "from_scratch": False,
    }}
    monkeypatch.setattr(runner.torch, "load", lambda *args, **kwargs: payload)
    cli = Namespace(
        data_root="/data/nyu", device="cuda:0", seed=7,
        batch_size=4, val_batch_size=1, workers=2,
        max_train_samples=8, max_val_samples=4)

    saved = runner.saved_checkpoint_args(Path("unused.pt"), cli)

    assert saved.from_scratch
