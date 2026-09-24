import csv
import inspect
from pathlib import Path

import pytest
import torch

from scripts import evaluate_cspn_nas_official as evaluator


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        depth = torch.full((1, 2, 2), float(index + 1))
        return {
            "rgbd": torch.cat((torch.zeros(3, 2, 2), depth), dim=0),
            "depth": depth,
            "cspn_preprocessed": True,
        }


class _SparseDepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, value):
        self.batch_sizes.append(value.shape[0])
        return value[:, 3:4]


def test_evaluate_dataset_writes_one_finite_row_per_sample(tmp_path):
    output = tmp_path / "metrics.csv"

    rows = evaluator.evaluate_dataset(
        _SparseDepthModel(), _Dataset(), torch.device("cpu"),
        seed=17, output=output)

    assert [row["sample_id"] for row in rows] == [0, 1]
    assert all(row["seed"] == 17 for row in rows)
    assert all(row["RMSE"] == pytest.approx(0.0) for row in rows)
    assert all(row["DELTA1.25"] == pytest.approx(1.0) for row in rows)
    with output.open(newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == 2
    assert set(written[0]) == set(evaluator.FIELDNAMES)


def test_evaluate_dataset_batches_inference_but_keeps_sample_rows(tmp_path):
    model = _SparseDepthModel()

    rows = evaluator.evaluate_dataset(
        model, _Dataset(), torch.device("cpu"), seed=17,
        output=tmp_path / "metrics.csv", batch_size=2, workers=0)

    assert model.batch_sizes == [2]
    assert [row["sample_id"] for row in rows] == [0, 1]


def test_parse_run_rejects_invalid_seed_assignment():
    with pytest.raises(ValueError, match="SEED=RUN_DIR"):
        evaluator.parse_run("not-an-assignment")


def test_checkpoint_defaults_to_best_but_is_configurable():
    signature = inspect.signature(evaluator.load_model)

    assert signature.parameters["checkpoint"].default == "best.pt"
    args = evaluator._parser().parse_args([
        "--run", "1=run", "--eval-list", "val.csv",
        "--data-root", ".", "--output", "metrics.csv",
        "--checkpoint", "last.pt",
    ])
    assert args.checkpoint == "last.pt"


def test_cspn_steps_can_override_checkpoint_iteration():
    signature = inspect.signature(evaluator.load_model)

    assert signature.parameters["cspn_steps"].default is None
    args = evaluator._parser().parse_args([
        "--run", "1=run", "--eval-list", "val.csv",
        "--data-root", ".", "--output", "metrics.csv",
        "--cspn-steps", "12",
    ])
    assert args.cspn_steps == 12


def test_cspn_steps_must_be_positive():
    parser = evaluator._parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--run", "1=run", "--eval-list", "val.csv",
            "--data-root", ".", "--output", "metrics.csv",
            "--cspn-steps", "0",
        ])


def test_evaluation_batch_options_are_configurable():
    args = evaluator._parser().parse_args([
        "--run", "1=run", "--eval-list", "val.csv",
        "--data-root", ".", "--output", "metrics.csv",
        "--batch-size", "4", "--workers", "2",
    ])

    assert args.batch_size == 4
    assert args.workers == 2
