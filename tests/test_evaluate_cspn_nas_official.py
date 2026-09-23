import csv
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
    def forward(self, value):
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


def test_parse_run_rejects_invalid_seed_assignment():
    with pytest.raises(ValueError, match="SEED=RUN_DIR"):
        evaluator.parse_run("not-an-assignment")

