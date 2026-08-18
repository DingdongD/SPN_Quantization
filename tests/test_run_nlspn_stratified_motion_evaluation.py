from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_nlspn_stratified_motion_evaluation as launcher


def _cli(tmp_path):
    return Namespace(
        data_root=str(tmp_path / "data"),
        checkpoint=str(tmp_path / "best.pt"),
        args_json=str(tmp_path / "args.json"),
        formal_dir=str(tmp_path / "formal"),
        target_root=str(tmp_path / "target"),
        staging_root=str(tmp_path / "staging"),
        raft_weights=str(tmp_path / "raft.pt"),
        device="cuda:2",
        seed=2026,
        bootstrap_replicates=2000,
    )


def test_build_worker_command_uses_one_legacy_process(tmp_path):
    command, environment = launcher.build_worker_command(
        tmp_path / "data", tmp_path / "manifest.json",
        tmp_path / "best.pt", tmp_path / "args.json",
        tmp_path / "formal", tmp_path / "staging",
        tmp_path / "raft.pt", "cuda:2", 2026)
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--seed") + 1] == "2026"
    assert str(launcher.REPO_ROOT) in environment["PYTHONPATH"]


@pytest.mark.parametrize("existing", ("target", "staging"))
def test_run_evaluation_refuses_existing_output_before_discovery(
        tmp_path, existing):
    cli = _cli(tmp_path)
    Path(getattr(cli, existing + "_root")).mkdir()
    calls = []

    def discover(_root, _seed):
        calls.append("discover")
        return []

    with pytest.raises(FileExistsError, match=existing):
        launcher.run_evaluation(cli, discover_fn=discover)
    assert calls == []


def test_compare_rows_accepts_csv_strings_and_rejects_numeric_change():
    expected = [{"method": "full", "rmse": 1.25, "count": 3,
                 "passes": True}]
    actual = [{"method": "full", "rmse": "1.25", "count": "3",
               "passes": "True"}]
    launcher.compare_rows(actual, expected, ("method",), "summary")
    actual[0]["rmse"] = "1.251"
    with pytest.raises(RuntimeError, match="summary.*rmse"):
        launcher.compare_rows(actual, expected, ("method",), "summary")


def test_preflight_requires_immutable_inputs(tmp_path):
    cli = _cli(tmp_path)
    with pytest.raises(FileNotFoundError, match="data root"):
        launcher.preflight(cli)
    Path(cli.data_root).mkdir()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        launcher.preflight(cli)

