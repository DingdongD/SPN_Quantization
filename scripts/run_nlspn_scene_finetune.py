#!/usr/bin/env python3
"""Stage, run, validate, and atomically publish NLSPN scene fine-tuning."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from scripts import nlspn_scene_finetune_artifacts as artifacts
from scripts import nlspn_scene_finetune_data as data
from scripts import nlspn_validated_output_promotion as promotion
from scripts import run_nlspn_scene_finetune_worker as worker


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_PATH = REPO_ROOT / "scripts/run_nlspn_scene_finetune_worker.py"
DEFAULT_TARGET = Path(
    "/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1")
DEFAULT_STAGING = DEFAULT_TARGET.with_name(
    "full_rmse_scene_disjoint_v1_staging")
DEFAULT_SOURCE = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS = DEFAULT_SOURCE.with_name("args.json")
DEFAULT_MOTION_ROOT = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "cross_scene_motion_stratified_5x3")
DEFAULT_DATA = Path("/workspace/VoxelNet/train")


def _cuda_available(device):
    parsed = torch.device(device)
    if parsed.type != "cuda" or not torch.cuda.is_available():
        return False
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    return 0 <= index < torch.cuda.device_count()


def _free_bytes(path):
    candidate = Path(path).resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(str(path))
        candidate = parent
    return shutil.disk_usage(str(candidate)).free


def _validate_source_checkpoint(path):
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("net"), dict):
        raise ValueError("source checkpoint must contain dictionary-valued net")


def _validate_motion_manifest(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 90:
        raise ValueError("motion manifest must contain exactly 90 windows")
    if {row.get("scene") for row in rows} != {
            "bedroom_ir", "livingroom_ir", "room3", "room4", "room6", "room7"}:
        raise ValueError("motion manifest scene set is invalid")


def preflight(cli, cuda_available_fn=_cuda_available,
              free_bytes_fn=_free_bytes, validate_source=True):
    target = Path(cli.target_root).resolve()
    staging = Path(cli.staging_root).resolve()
    if target.exists():
        raise FileExistsError("target output already exists: {}".format(target))
    if staging.exists() and not cli.resume:
        raise FileExistsError("staging output already exists: {}".format(staging))
    if target == staging or target.parent != staging.parent:
        raise ValueError("target and staging outputs must be distinct siblings")
    if int(cli.seed) != 2026:
        raise ValueError("formal seed must be exactly 2026")
    source = Path(cli.source_checkpoint).resolve()
    for output in (target, staging):
        if source == output or output in source.parents:
            raise ValueError("source checkpoint resolves inside an output path")
    if cli.resume:
        resume = Path(cli.resume).resolve()
        if staging != resume and staging not in resume.parents:
            raise ValueError("resume checkpoint must be inside staging output")
        if not resume.is_file():
            raise FileNotFoundError(str(resume))
    required_dirs = (Path(cli.data_root), Path(cli.motion_root))
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(str(path))
    for path in (Path(cli.source_checkpoint), Path(cli.source_args)):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    if int(free_bytes_fn(staging)) < 20 * 1024 ** 3:
        raise RuntimeError("formal run requires at least 20 GB free disk")
    if not cuda_available_fn(cli.device):
        raise RuntimeError("requested CUDA device is unavailable")
    motion_manifest = Path(cli.motion_root) / "selected_windows.csv"
    if not motion_manifest.is_file():
        raise FileNotFoundError(str(motion_manifest))
    if validate_source:
        _validate_source_checkpoint(cli.source_checkpoint)
        with Path(cli.source_args).open("r", encoding="utf-8") as stream:
            args = json.load(stream)
        if not isinstance(args, dict) or args.get("model") != "nlspn":
            raise ValueError("source args are not NLSPN")
        _validate_motion_manifest(motion_manifest)
    return {"target": target, "staging": staging}


def build_worker_command(cli, manifests):
    command = [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH),
        "--data-root", str(Path(cli.data_root).resolve()),
        "--train-manifest", str(Path(manifests["train"]).resolve()),
        "--val-manifest", str(Path(manifests["val"]).resolve()),
        "--test-manifest", str(Path(manifests["test"]).resolve()),
        "--source-checkpoint", str(Path(cli.source_checkpoint).resolve()),
        "--source-args", str(Path(cli.source_args).resolve()),
        "--window-manifest", str(
            (Path(cli.motion_root) / "selected_windows.csv").resolve()),
        "--raw-output", str((Path(cli.staging_root) / "raw").resolve()),
        "--device", str(cli.device), "--seed", str(int(cli.seed)),
    ]
    if cli.resume:
        command.extend(["--resume", str(Path(cli.resume).resolve())])
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else "")
    environment.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    return command, environment


def snapshot_inputs(cli, manifests):
    paths = {
        "source_checkpoint": Path(cli.source_checkpoint),
        "source_args": Path(cli.source_args),
        "motion_manifest": Path(cli.motion_root) / "selected_windows.csv",
        "train_manifest": manifests["train"],
        "val_manifest": manifests["val"],
        "test_manifest": manifests["test"],
    }
    return {name: worker.file_sha256(path) for name, path in paths.items()}


def recheck_inputs(cli, manifests, expected):
    actual = snapshot_inputs(cli, manifests)
    if actual != expected:
        changed = sorted(name for name in expected if expected.get(name) != actual.get(name))
        raise RuntimeError("immutable inputs changed: {}".format(changed))
    return actual


def stream_worker(cli, manifests):
    command, environment = build_worker_command(cli, manifests)
    log_path = Path(cli.staging_root) / "worker.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("legacy training worker failed with exit code {}".format(
            return_code))


def _replace_log(cli):
    path = Path(cli.staging_root) / "worker.log"
    if not path.is_file():
        raise RuntimeError("streamed worker log is missing")


def run(cli, preflight_fn=preflight, manifest_builder=data.build_manifests,
        manifest_writer=data.write_manifests, snapshot_fn=snapshot_inputs,
        worker_runner=stream_worker, replace_log_fn=_replace_log,
        finalizer=artifacts.write_final_artifacts,
        validator=artifacts.validate_final_tree,
        recheck_fn=recheck_inputs,
        promoter=promotion.promote_new_validated_output):
    preflight_fn(cli)
    staging = Path(cli.staging_root).resolve()
    target = Path(cli.target_root).resolve()
    if not staging.exists():
        staging.mkdir(parents=True)
    manifests = manifest_builder(cli.data_root)
    manifest_paths = manifest_writer(manifests, staging / "manifests")
    input_digests = snapshot_fn(cli, manifest_paths)
    (staging / "raw").mkdir(exist_ok=True)
    worker_runner(cli, manifest_paths)
    replace_log_fn(cli)
    finalizer(staging, manifest_paths, source_digests=input_digests)
    temporary_manifests = staging / "manifests"
    if temporary_manifests.is_dir():
        shutil.rmtree(str(temporary_manifests))
    staging_manifest_paths = {
        split: staging / (split + "_manifest.csv")
        for split in ("train", "val", "test")}
    staging_metadata = validator(staging)
    recheck_fn(cli, staging_manifest_paths, input_digests)
    promoter(staging, target, validator)
    target_metadata = validator(target)
    target_manifest_paths = {
        split: target / (split + "_manifest.csv")
        for split in ("train", "val", "test")}
    recheck_fn(cli, target_manifest_paths, input_digests)
    metadata = target_metadata or staging_metadata or {}
    return {
        "target_root": str(target),
        "selected_epoch": metadata.get("worker", {}).get("selected_epoch"),
        "generic_rmse": metadata.get("aggregate", {}).get(
            "variants", {}).get("generic", {}).get("pooled_rmse"),
        "specialized_rmse": metadata.get("aggregate", {}).get(
            "variants", {}).get("specialized", {}).get("pooled_rmse"),
        "success_gate": metadata.get("success_gate"),
        "report": str(target / "report.md"),
    }


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run formal scene-specific NLSPN fine-tuning")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--source-checkpoint", default=str(DEFAULT_SOURCE))
    parser.add_argument("--source-args", default=str(DEFAULT_ARGS))
    parser.add_argument("--motion-root", default=str(DEFAULT_MOTION_ROOT))
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET))
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume")
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    result = run(cli)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
