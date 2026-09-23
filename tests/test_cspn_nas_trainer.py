import argparse
import json
from pathlib import Path

import torch

from scripts import train_nyu_iteration_sweep as trainer
from spn_quant.nas.spec import EncoderSpec


def _args(**overrides):
    values = {
        "model": "cspn",
        "iteration": 24,
        "cspn_backbone": "resnet18",
        "from_scratch": True,
        "cspn_encoder_spec": "",
        "cspn_control_checkpoint": "",
        "split_manifest": "",
        "train_full_data": False,
        "run_name": "",
        "train_list": "train.csv",
        "eval_list": "val.csv",
        "data_root": ".",
        "n_sample": 500,
        "seed": 123,
        "max_train_samples": 0,
        "max_val_samples": 0,
        "batch_size": 1,
        "val_batch_size": 1,
        "workers": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_cspn_loads_nas_spec_and_control_checkpoint(tmp_path):
    spec_path = tmp_path / "candidate.json"
    spec = EncoderSpec(32, (32, 64, 128, 256), (1, 1, 1, 0))
    spec_path.write_text(json.dumps(spec.to_dict()))

    control, _ = trainer.build_cspn(_args(), torch.device("cpu"))
    checkpoint = tmp_path / "control.pt"
    torch.save({"net": control.state_dict()}, checkpoint)

    model, metadata = trainer.build_cspn(
        _args(
            cspn_encoder_spec=str(spec_path),
            cspn_control_checkpoint=str(checkpoint),
        ),
        torch.device("cpu"),
    )

    assert model.encoder_spec == spec
    assert metadata["architecture"] == "CSPN encoder NAS"
    assert metadata["encoder_spec"]["slug"] == spec.slug
    assert metadata["control_checkpoint_sha256"]
    assert metadata["weight_transfer"]["partial"]


def test_legacy_cspn_build_path_is_unchanged():
    model, metadata = trainer.build_cspn(_args(), torch.device("cpu"))

    assert type(model).__name__ == "ResNet"
    assert metadata["architecture"] == "CSPN resnet18"
    assert "encoder_spec" not in metadata


def test_split_manifest_uses_train_list_for_search_train_and_dev(tmp_path, monkeypatch):
    instances = []

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, csv_file, root_dir, split, n_sample, seed):
            self.csv_file = csv_file
            self.split = split
            instances.append(self)

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"index": index}

    monkeypatch.setattr(trainer, "CspnOfficialDataset", FakeDataset)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "train_indices": [1, 3],
        "dev_indices": [0, 2],
    }))

    trainloader, devloader = trainer.make_loaders(
        _args(split_manifest=str(manifest)))

    assert [dataset.csv_file for dataset in instances] == ["train.csv", "train.csv"]
    assert [dataset.split for dataset in instances] == ["train", "val"]
    assert trainloader.dataset.indices == [1, 3]
    assert devloader.dataset.indices == [0, 2]


def test_full_training_keeps_all_samples_and_monitors_dev_subset(
        tmp_path, monkeypatch):
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, csv_file, root_dir, split, n_sample, seed):
            pass

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"index": index}

    monkeypatch.setattr(trainer, "CspnOfficialDataset", FakeDataset)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "train_indices": [1, 3],
        "dev_indices": [0, 2],
    }))

    trainloader, devloader = trainer.make_loaders(
        _args(split_manifest=str(manifest), train_full_data=True, seed=77))

    assert len(trainloader.dataset) == 4
    assert devloader.dataset.indices == [0, 2]
    assert trainloader.generator.initial_seed() == 77


def test_run_name_overrides_default_directory_name():
    assert trainer.resolve_run_name(_args(run_name="nas-candidate")) == "nas-candidate"
    assert trainer.resolve_run_name(_args()) == "cspn_iter24"
