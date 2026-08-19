import argparse
from pathlib import Path

import pytest

from scripts import run_nlspn_scene_finetune as launcher


def _cli(tmp_path):
    return argparse.Namespace(
        data_root=str(tmp_path / "data"),
        source_checkpoint=str(tmp_path / "best.pt"),
        source_args=str(tmp_path / "args.json"),
        motion_root=str(tmp_path / "motion"),
        target_root=str(tmp_path / "target"),
        staging_root=str(tmp_path / "staging"),
        device="cuda:2", seed=2026, resume=None)


def test_worker_command_uses_legacy_environment(tmp_path):
    cli = _cli(tmp_path)
    staging = Path(cli.staging_root)
    manifests = {name: staging / (name + "_manifest.csv")
                 for name in ("train", "val", "test")}
    command, environment = launcher.build_worker_command(cli, manifests)
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--seed") + 1] == "2026"
    assert str(launcher.REPO_ROOT) in environment["PYTHONPATH"]


@pytest.mark.parametrize("name", ("target_root", "staging_root"))
def test_existing_output_is_rejected_before_scan(tmp_path, name):
    cli = _cli(tmp_path)
    Path(getattr(cli, name)).mkdir()
    calls = []
    with pytest.raises(FileExistsError):
        launcher.run(cli, manifest_builder=lambda *_: calls.append("scan"))
    assert calls == []


def test_preflight_rejects_nonsibling_outputs(tmp_path):
    cli = _cli(tmp_path)
    cli.staging_root = str(tmp_path / "nested/staging")
    with pytest.raises(ValueError, match="siblings"):
        launcher.preflight(cli, cuda_available_fn=lambda _: True,
                           free_bytes_fn=lambda _: 30 * 1024 ** 3,
                           validate_source=False)


def test_preflight_rejects_nonformal_seed(tmp_path):
    cli = _cli(tmp_path)
    cli.seed = 7
    with pytest.raises(ValueError, match="2026"):
        launcher.preflight(cli, cuda_available_fn=lambda _: True,
                           free_bytes_fn=lambda _: 30 * 1024 ** 3,
                           validate_source=False)


def test_preflight_rejects_low_disk_and_unavailable_cuda(tmp_path):
    cli = _cli(tmp_path)
    for path in (Path(cli.data_root), Path(cli.motion_root)):
        path.mkdir()
    Path(cli.source_checkpoint).touch(); Path(cli.source_args).touch()
    with pytest.raises(RuntimeError, match="20 GB"):
        launcher.preflight(cli, cuda_available_fn=lambda _: True,
                           free_bytes_fn=lambda _: 19 * 1024 ** 3,
                           validate_source=False)
    with pytest.raises(RuntimeError, match="CUDA"):
        launcher.preflight(cli, cuda_available_fn=lambda _: False,
                           free_bytes_fn=lambda _: 30 * 1024 ** 3,
                           validate_source=False)


def test_free_bytes_accepts_not_yet_created_output_parents(tmp_path):
    value = launcher._free_bytes(tmp_path / "missing/parent/staging")
    assert value > 0


def test_run_uses_exact_failure_safe_orchestration_order(tmp_path):
    cli = _cli(tmp_path)
    calls = []

    def preflight(*args, **kwargs):
        calls.append("preflight")

    def build(*args):
        calls.append("build_manifests")
        return {"train": [], "val": [], "test": []}

    def write(manifests, directory):
        calls.append("write_manifests")
        return {name: Path(directory) / (name + "_manifest.csv")
                for name in manifests}

    result = launcher.run(
        cli, preflight_fn=preflight, manifest_builder=build,
        manifest_writer=write,
        snapshot_fn=lambda *args: calls.append("snapshot_inputs") or {"x": "y"},
        worker_runner=lambda *args: calls.append("worker"),
        replace_log_fn=lambda *args: calls.append("replace_log"),
        finalizer=lambda *args, **kwargs: calls.append("finalize_artifacts") or {},
        validator=lambda *args, **kwargs: calls.append(
            "validate_staging" if len([x for x in calls if x.startswith("validate")]) == 0
            else "validate_target") or {"complete": True},
        recheck_fn=lambda *args: calls.append("recheck_inputs"),
        promoter=lambda *args: calls.append("promote"))
    assert calls == [
        "preflight", "build_manifests", "write_manifests", "snapshot_inputs",
        "worker", "replace_log", "finalize_artifacts", "validate_staging",
        "recheck_inputs", "promote", "validate_target", "recheck_inputs"]
    assert result["target_root"] == str(Path(cli.target_root).resolve())


@pytest.mark.parametrize("failure", ("worker", "finalizer", "validator", "recheck"))
def test_failure_never_promotes_target(tmp_path, failure):
    cli = _cli(tmp_path)
    promoted = []

    def fail_if(name, result=None):
        if failure == name:
            raise RuntimeError(name)
        return result

    with pytest.raises(RuntimeError, match=failure):
        launcher.run(
            cli, preflight_fn=lambda *_: None,
            manifest_builder=lambda *_: {"train": [], "val": [], "test": []},
            manifest_writer=lambda manifests, directory: {
                name: Path(directory) / (name + ".csv") for name in manifests},
            snapshot_fn=lambda *_: {},
            worker_runner=lambda *_: fail_if("worker"),
            replace_log_fn=lambda *_: None,
            finalizer=lambda *args, **kwargs: fail_if("finalizer", {}),
            validator=lambda *args, **kwargs: fail_if("validator", {}),
            recheck_fn=lambda *_: fail_if("recheck"),
            promoter=lambda *_: promoted.append(True))
    assert not promoted
    assert not Path(cli.target_root).exists()


def test_post_promotion_recheck_uses_target_manifest_paths(tmp_path):
    cli = _cli(tmp_path)
    checked = []

    def write(manifests, directory):
        Path(directory).mkdir(parents=True)
        paths = {}
        for name in manifests:
            path = Path(directory) / (name + ".csv")
            path.write_text(name)
            paths[name] = path
        return paths

    def finalize(staging, manifests, **kwargs):
        for name, path in manifests.items():
            (Path(staging) / (name + "_manifest.csv")).write_bytes(path.read_bytes())

    def recheck(cli, manifests, expected):
        assert all(Path(path).is_file() for path in manifests.values())
        checked.append({name: Path(path) for name, path in manifests.items()})

    def promote(staging, target, validator):
        Path(staging).replace(target)

    launcher.run(
        cli, preflight_fn=lambda *_: None,
        manifest_builder=lambda *_: {"train": [], "val": [], "test": []},
        manifest_writer=write, snapshot_fn=lambda *_: {},
        worker_runner=lambda *_: None, replace_log_fn=lambda *_: None,
        finalizer=finalize, validator=lambda *_: {"complete": True},
        recheck_fn=recheck, promoter=promote)
    assert checked[0]["train"].parent == Path(cli.staging_root)
    assert checked[1]["train"].parent == Path(cli.target_root)
    assert not (Path(cli.target_root) / "manifests").exists()
