import argparse
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_nlspn_scene_finetune_worker as worker


def _cli(tmp_path):
    return argparse.Namespace(
        data_root=str(tmp_path / "data"),
        train_manifest=str(tmp_path / "train.csv"),
        val_manifest=str(tmp_path / "val.csv"),
        test_manifest=str(tmp_path / "test.csv"),
        source_checkpoint=str(tmp_path / "source.pt"),
        source_args=str(tmp_path / "args.json"),
        window_manifest=str(tmp_path / "windows.csv"),
        raw_output=str(tmp_path / "raw"), device="cpu", seed=2026,
        resume=None)


def _fake_dependencies(
        calls, stage2_values=(0.9995, 0.9994, 0.9993, 0.9992)):
    values = iter((1.2, 1.1, 1.0) + tuple(stage2_values))

    def configure(model, stage):
        calls.append("configure_stage:{}".format(stage))
        return [{"name": "stage{}".format(stage), "params": [], "lr": 1e-4}]

    def train(model, loader, optimizer, accumulation, clip):
        calls.append("train_epoch:{}".format(loader))
        return {"loss": 1.0}

    def evaluate(model, loader):
        calls.append("evaluate_val")
        return {"pooled_rmse": next(values)}

    return {
        "prepare_fn": lambda cli: calls.append("prepare") or {
            "model": object(), "train_loader_factory": lambda epoch: (
                "stage1" if epoch <= 3 else "stage2"),
            "val_loader": object(), "context": {"accumulation_steps": 1}},
        "baseline_val_fn": lambda prepared: calls.append("baseline_val"),
        "configure_stage_fn": configure,
        "optimizer_factory": lambda groups: object(),
        "train_epoch_fn": train,
        "evaluate_model_fn": evaluate,
        "record_epoch_fn": lambda *args: calls.append("record_epoch"),
        "save_latest_fn": lambda *args: calls.append("save_latest"),
        "save_best_fn": lambda *args: calls.append("save_best"),
        "select_best_fn": lambda *args: calls.append("select_best") or object(),
        "load_test_fn": lambda *args: calls.append("load_test") or object(),
        "evaluate_test_pair_fn": lambda *args: calls.append(
            "evaluate_test_pair") or {"test_metric_row_count": 16000},
        "finalize_fn": lambda *args: calls.append("finalize") or {
            "stage1_epochs": 3},
    }


def test_worker_never_reads_test_before_best_selection(tmp_path):
    calls = []
    result = worker.run_training(_cli(tmp_path), **_fake_dependencies(calls))
    assert calls.index("select_best") < calls.index("load_test")
    assert calls.count("evaluate_test_pair") == 1
    assert result["stage1_epochs"] == 3


def test_worker_applies_stage_one_then_stage_two(tmp_path):
    calls = []
    worker.run_training(_cli(tmp_path), **_fake_dependencies(calls))
    stages = [item for item in calls if item.startswith("configure_stage")]
    assert stages == ["configure_stage:1", "configure_stage:2"]
    assert calls.count("train_epoch:stage1") == 3
    assert 1 <= calls.count("train_epoch:stage2") <= 15


def test_training_failure_never_starts_test(tmp_path):
    calls = []
    dependencies = _fake_dependencies(calls)
    dependencies["train_epoch_fn"] = lambda *args: (
        (_ for _ in ()).throw(RuntimeError("nonfinite loss")))
    with pytest.raises(RuntimeError, match="nonfinite loss"):
        worker.run_training(_cli(tmp_path), **dependencies)
    assert "load_test" not in calls


def test_stage_two_stops_after_four_nonsignificant_epochs(tmp_path):
    calls = []
    values = (0.9995, 0.9994, 0.9993, 0.9992, 0.9991)
    worker.run_training(
        _cli(tmp_path), **_fake_dependencies(calls, stage2_values=values))
    assert calls.count("train_epoch:stage2") == 4


class TinyDepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self, sample):
        return {"pred": sample["dep"] * self.scale}


def _microbatches(count):
    return [
        {"rgb": torch.zeros(1, 3, 2, 2),
         "dep": torch.ones(1, 1, 2, 2),
         "gt": torch.full((1, 1, 2, 2), 2.0)}
        for _ in range(count)]


def test_train_epoch_steps_partial_final_accumulation_group(monkeypatch):
    model = TinyDepthModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = []
    original = optimizer.step

    def counted_step(*args, **kwargs):
        steps.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)
    result = worker.train_epoch(
        model, _microbatches(5), optimizer, accumulation_steps=2,
        clip_norm=1.0)
    assert len(steps) == 3
    assert result["microbatches"] == 5
    assert result["optimizer_steps"] == 3
    assert model.bn.training is False


def test_validate_test_rows_requires_exact_paired_geometry():
    rows = []
    for variant in ("generic", "specialized"):
        for scene, count in (("room3", 4000), ("room7", 4000)):
            rows.extend({"variant": variant, "scene": scene, "frame_id": frame}
                        for frame in range(1, count + 1))
    worker.validate_test_row_identities(rows)
    with pytest.raises(ValueError, match="16000"):
        worker.validate_test_row_identities(rows[:-1])


def test_load_held_out_windows_filters_exact_thirty(tmp_path):
    path = tmp_path / "windows.csv"
    fields = ("selection_seed", "scene", "stratum", "window_id",
              "start_frame", "end_frame", "frame_ids")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in ("room3", "room7", "room4"):
            for index in range(15):
                start = index * 5 + 1
                writer.writerow({
                    "selection_seed": 2026, "scene": scene,
                    "stratum": ("low", "medium", "high")[index % 3],
                    "window_id": "{}/{}".format(scene, index),
                    "start_frame": start, "end_frame": start + 4,
                    "frame_ids": json.dumps(list(range(start, start + 5)))})
    windows = worker.load_held_out_windows(path)
    assert len(windows) == 30
    assert {row["scene"] for row in windows} == {"room3", "room7"}
    assert sum(len(row["frame_ids"]) for row in windows) == 150


def test_parser_exposes_only_fixed_formal_policy_defaults():
    parser = worker.make_parser()
    destinations = {action.dest for action in parser._actions}
    assert "stage1_epochs" not in destinations
    assert "stage2_epochs" not in destinations
    assert "learning_rate" not in destinations
    assert parser.get_default("seed") == 2026
    assert worker.RAW_ARTIFACTS == (
        "best.pt", "latest.pt", "specialized_args.json",
        "epoch_metrics.csv", "baseline_val_frame_metrics.csv",
        "test_frame_metrics.csv", "window_predictions.npz",
        "worker_metadata.json")


def test_install_torchvision_dcn_backend_replaces_only_operator_symbol():
    original = object()
    module = SimpleNamespace(ModulatedDeformConvFunction=original)
    result = worker.install_torchvision_dcn_backend(module=module)
    assert result == "torchvision.ops.deform_conv2d"
    assert module.ModulatedDeformConvFunction is worker.TorchvisionDCNFunction
    assert module.ModulatedDeformConvFunction is not original


def test_install_torchvision_dcn_backend_rejects_wrong_model_module():
    with pytest.raises(RuntimeError, match="operator symbol"):
        worker.install_torchvision_dcn_backend(module=SimpleNamespace())
