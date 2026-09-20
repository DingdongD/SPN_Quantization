import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import run_cspn_encoder_nas as runner
from spn_quant.nas.spec import EncoderSpec


def _prepare_args(tmp_path, **overrides):
    train_list = tmp_path / "train.csv"
    train_list.write_text(
        "Name\n" + "\n".join(f"data/{index:05d}.h5" for index in range(10)))
    checkpoint = tmp_path / "control.pt"
    checkpoint.write_bytes(b"checkpoint")
    values = {
        "train_list": str(train_list),
        "control_checkpoint": str(checkpoint),
        "output_root": str(tmp_path / "experiment"),
        "seed": 7,
        "dev_count": 2,
        "max_candidates": 4,
        "smoke_candidates": 0,
        "data_root": "/data",
        "python": "python",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_prepare_writes_immutable_split_candidates_and_commands(tmp_path):
    args = _prepare_args(tmp_path)

    summary = runner.prepare_experiment(args)
    repeated = runner.prepare_experiment(args)

    root = Path(args.output_root)
    split = json.loads((root / "split.json").read_text())
    rows = list(csv.DictReader((root / "candidates.csv").open()))
    commands = (root / "train_commands.sh").read_text()
    assert summary == repeated
    assert len(split["train_indices"]) == 8
    assert len(split["dev_indices"]) == 2
    assert len(rows) == 4
    assert all((root / row["spec_path"]).exists() for row in rows)
    assert all(int(row["full_parameters"]) > int(row["encoder_parameters"])
               for row in rows)
    assert "--split-manifest" in commands
    assert "--cspn-encoder-spec" in commands

    with pytest.raises(ValueError, match="identity mismatch"):
        runner.prepare_experiment(_prepare_args(tmp_path, seed=8))


def test_analytical_parameter_count_matches_constructed_models():
    for spec in (
        EncoderSpec.r18(),
        EncoderSpec(32, (32, 64, 128, 256), (1, 1, 1, 0)),
    ):
        estimated = runner.parameter_counts(spec)
        actual = runner.constructed_parameter_counts(spec)
        assert estimated == actual


def test_rank_results_writes_deterministic_survivors(tmp_path):
    results = tmp_path / "results.csv"
    with results.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("candidate", "rmse", "parameters", "latency_ms"))
        writer.writeheader()
        for index in range(12):
            writer.writerow({
                "candidate": f"c{index:02d}",
                "rmse": 0.14 + index * 0.001,
                "parameters": 100 - index,
                "latency_ms": 2.0 + index * 0.01,
            })
    output = tmp_path / "selected.json"

    selected = runner.rank_results(results, output)

    assert len(selected) == 8
    assert json.loads(output.read_text()) == selected


def test_benchmark_records_busy_gpu_failure(tmp_path, monkeypatch):
    args = _prepare_args(tmp_path, max_candidates=1)
    runner.prepare_experiment(args)
    monkeypatch.setattr(runner, "gpu_environment", lambda index: {
        "index": index,
        "processes": [{"pid": 999999, "used_memory_mib": 10}],
    })

    with pytest.raises(RuntimeError, match="other compute processes"):
        runner.benchmark_candidates(
            Path(args.output_root), device="cuda:0", limit=1,
            warmup=1, iterations=1, repeats=1)

    failure = json.loads(
        (Path(args.output_root) / "benchmark_failure.json").read_text())
    assert failure["status"] == "failed_preflight"
    assert "other compute processes" in failure["reason"]
