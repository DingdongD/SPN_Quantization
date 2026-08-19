#!/usr/bin/env python3
"""Python 3.7 worker for scene-specific full NLSPN fine-tuning."""

from __future__ import print_function

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torchvision.ops import deform_conv2d

from scripts import nlspn_scene_finetune_core as core
from scripts import nlspn_scene_finetune_data as data
from scripts import run_spn_sequence_worker as sequence_worker


RAW_ARTIFACTS = (
    "best.pt", "latest.pt", "specialized_args.json",
    "epoch_metrics.csv", "baseline_val_frame_metrics.csv",
    "test_frame_metrics.csv", "window_predictions.npz",
    "worker_metadata.json")


class TorchvisionDCNFunction(object):
    """Drop-in execution backend for the NLSPN DCNv2 operator."""

    @staticmethod
    def apply(input_tensor, offset, mask, weight, bias, stride, padding,
              dilation, groups, deformable_groups, im2col_step):
        inferred_groups = input_tensor.shape[1] // weight.shape[1]
        kernel_elements = weight.shape[2] * weight.shape[3]
        inferred_deformable_groups = offset.shape[1] // (2 * kernel_elements)
        if int(groups) != int(inferred_groups):
            raise RuntimeError("DCN group contract mismatch")
        if int(deformable_groups) != int(inferred_deformable_groups):
            raise RuntimeError("DCN deformable-group contract mismatch")
        return deform_conv2d(
            input_tensor, offset, weight, bias, stride, padding, dilation,
            mask=mask)


def install_torchvision_dcn_backend(model=None, module=None):
    if module is None:
        if model is None:
            raise ValueError("model or module is required")
        module = sys.modules.get(model.__class__.__module__)
    if module is None or not hasattr(module, "ModulatedDeformConvFunction"):
        raise RuntimeError("NLSPN model module is missing DCN operator symbol")
    module.ModulatedDeformConvFunction = TorchvisionDCNFunction
    return "torchvision.ops.deform_conv2d"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _load_rows(path, split, root):
    rows = data._read_manifest(path, split)
    data._require_paths_within_root(rows, Path(root).resolve())
    return rows


def _read_args(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("model") != "nlspn":
        raise ValueError("source args must describe nlspn")
    return value


def _contract_sha256(objects):
    payload = "\n".join(inspect.getsource(item) for item in objects)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_device(batch, device):
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name in ("rgb", "dep", "gt", "valid")
    }


def _finite_gradients(model):
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters())


def _optimizer_step(model, optimizer, clip_norm, microbatches, target_steps):
    if microbatches < target_steps:
        scale = float(target_steps) / float(microbatches)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    if not _finite_gradients(model):
        raise RuntimeError("nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
    if not math.isfinite(float(norm)):
        raise RuntimeError("nonfinite gradient norm")
    optimizer.step()
    optimizer.zero_grad()


def train_epoch(model, loader, optimizer, accumulation_steps, clip_norm):
    accumulation_steps = int(accumulation_steps)
    if accumulation_steps <= 0:
        raise ValueError("accumulation steps must be positive")
    model.train()
    core.keep_batch_norm_frozen(model)
    device = next(model.parameters()).device
    optimizer.zero_grad()
    total_loss = 0.0
    total_samples = 0
    microbatches = 0
    pending = 0
    optimizer_steps = 0
    for batch in loader:
        sample = _to_device(batch, device)
        output = model(sample)
        prediction = output["pred"] if isinstance(output, dict) else output
        loss = core.masked_l1_l2(prediction, sample["gt"], 10.0)
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite loss")
        batch_size = int(sample["gt"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        (loss / accumulation_steps).backward()
        microbatches += 1
        pending += 1
        if pending == accumulation_steps:
            _optimizer_step(
                model, optimizer, clip_norm, pending, accumulation_steps)
            optimizer_steps += 1
            pending = 0
    if pending:
        _optimizer_step(model, optimizer, clip_norm, pending, accumulation_steps)
        optimizer_steps += 1
    if total_samples == 0:
        raise RuntimeError("training loader is empty")
    return {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "microbatches": microbatches,
        "optimizer_steps": optimizer_steps,
    }


def _metric_row(variant, scene, frame_id, raw):
    count = int(raw["valid_pixel_count"])
    row = {
        "variant": variant,
        "scene": str(scene),
        "frame_id": int(frame_id),
        "squared_error_sum": float(raw["squared_error_sum"]),
        "absolute_error_sum": float(raw["absolute_error_sum"]),
        "abs_rel_sum": float(raw["abs_rel_sum"]),
        "valid_pixels": count,
        "rmse": math.sqrt(float(raw["squared_error_sum"]) / count),
        "mae": float(raw["absolute_error_sum"]) / count,
        "abs_rel": float(raw["abs_rel_sum"]) / count,
    }
    for name in core.DEPTH_BAND_NAMES:
        short = name.replace("band_", "")
        row["band_{}_squared_error_sum".format(short)] = float(
            raw[name + "_squared_error_sum"])
        row["band_{}_absolute_error_sum".format(short)] = float(
            raw[name + "_absolute_error_sum"])
        row["band_{}_valid_pixels".format(short)] = int(
            raw[name + "_valid_pixel_count"])
    return row


def metric_fields():
    fields = [
        "variant", "scene", "frame_id", "squared_error_sum",
        "absolute_error_sum", "abs_rel_sum", "valid_pixels",
        "rmse", "mae", "abs_rel"]
    for name in core.DEPTH_BAND_NAMES:
        short = name.replace("band_", "")
        fields.extend([
            "band_{}_squared_error_sum".format(short),
            "band_{}_absolute_error_sum".format(short),
            "band_{}_valid_pixels".format(short)])
    return tuple(fields)


def evaluate_model(model, loader, variant="model", return_rows=False):
    model.eval()
    device = next(model.parameters()).device
    accumulator = core.MetricAccumulator()
    rows = []
    with torch.no_grad():
        for batch in loader:
            sample = _to_device(batch, device)
            output = model(sample)
            prediction = output["pred"] if isinstance(output, dict) else output
            scenes = list(batch["scene"])
            frame_ids = batch["frame_id"].tolist()
            for index, (scene, frame_id) in enumerate(zip(scenes, frame_ids)):
                raw = core.frame_error_sums(
                    prediction[index:index + 1], sample["gt"][index:index + 1])
                accumulator.add_frame(
                    str(scene), prediction[index:index + 1],
                    sample["gt"][index:index + 1])
                if return_rows:
                    rows.append(_metric_row(variant, scene, frame_id, raw))
    metrics = accumulator.finalize()
    return (metrics, rows) if return_rows else metrics


def validate_test_row_identities(rows):
    if len(rows) != 16000:
        raise ValueError("test metrics must contain exactly 16000 rows")
    identities = [(row["variant"], row["scene"], int(row["frame_id"]))
                  for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("test metrics contain duplicate identities")
    expected = set()
    for variant in ("generic", "specialized"):
        for scene in ("room3", "room7"):
            for frame_id in range(1, data.SCENE_FRAME_COUNTS[scene] + 1):
                expected.add((variant, scene, frame_id))
    if set(identities) != expected:
        raise ValueError("test metric identities differ from exact geometry")


def load_held_out_windows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if row.get("scene") not in ("room3", "room7"):
            continue
        if int(row.get("selection_seed", -1)) != 2026:
            raise ValueError("window selection seed must be 2026")
        frame_ids = json.loads(row["frame_ids"])
        if (len(frame_ids) != 5 or
                any(right != left + 1
                    for left, right in zip(frame_ids, frame_ids[1:]))):
            raise ValueError("window frame IDs must be five consecutive IDs")
        converted = dict(row)
        converted["frame_ids"] = [int(value) for value in frame_ids]
        converted["start_frame"] = int(row["start_frame"])
        converted["end_frame"] = int(row["end_frame"])
        selected.append(converted)
    if len(selected) != 30:
        raise ValueError("held-out window manifest must contain exactly 30 windows")
    if {row["scene"] for row in selected} != {"room3", "room7"}:
        raise ValueError("held-out windows must contain room3 and room7")
    return selected


def _loader(dataset, batch_size, shuffle=False, seed=2026):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=4, pin_memory=True, generator=generator,
        persistent_workers=False)


def _build_source_model(args, checkpoint, device):
    model, metadata = sequence_worker.build_model("nlspn", args, device)
    backend = install_torchvision_dcn_backend(model=model)
    sequence_worker.load_checkpoint_strict(model, checkpoint)
    metadata = dict(metadata)
    metadata["dcn_backend"] = backend
    return model, metadata


def _probe_batch(cli, args, rows):
    dataset = data.SceneDepthDataset(rows[:12], cli.data_root, "train", cli.seed)

    def attempt(batch_size):
        def run():
            model, _ = _build_source_model(
                args, cli.source_checkpoint, torch.device(cli.device))
            groups = core.configure_stage(model, 1)
            optimizer = torch.optim.Adam([
                {"params": group["params"], "lr": group["lr"]}
                for group in groups])
            loader = _loader(dataset, batch_size, False, cli.seed)
            train_epoch(model, [next(iter(loader))], optimizer, 1, 1.0)
            del optimizer
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return core.run_probe_attempt(run)

    return core.probe_physical_batch(attempt, 12)


def prepare_training(cli):
    raw = Path(cli.raw_output)
    raw.mkdir(parents=True, exist_ok=True)
    metadata_path = raw / "worker_metadata.json"
    _atomic_json(metadata_path, {"complete": False, "seed": int(cli.seed)})
    if int(cli.seed) != 2026:
        raise ValueError("formal split seed must be 2026")
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    rows = {
        "train": _load_rows(cli.train_manifest, "train", cli.data_root),
        "val": _load_rows(cli.val_manifest, "val", cli.data_root),
    }
    args = _read_args(cli.source_args)
    model, model_metadata = _build_source_model(args, cli.source_checkpoint, device)
    meta = {
        "source_checkpoint_sha256": file_sha256(cli.source_checkpoint),
        "train_manifest_sha256": file_sha256(cli.train_manifest),
        "val_manifest_sha256": file_sha256(cli.val_manifest),
        "test_manifest_sha256": file_sha256(cli.test_manifest),
        "preprocessing_sha256": _contract_sha256((
            data.preprocess_arrays, data.build_sparse, data.augment_arrays)),
        "split_seed": int(cli.seed),
        "model_state_schema_sha256": core.model_state_schema_sha256(model),
        "stage_configuration_sha256": _contract_sha256((core.configure_stage,)),
    }
    batch = _probe_batch(cli, args, rows["train"])
    train_dataset = data.SceneDepthDataset(
        rows["train"], cli.data_root, "train", cli.seed)
    val_dataset = data.SceneDepthDataset(
        rows["val"], cli.data_root, "val", cli.seed)

    def train_loader_factory(epoch):
        train_dataset.set_epoch(epoch)
        return _loader(
            train_dataset, batch["physical_batch_size"], True,
            cli.seed + int(epoch))

    context = dict(batch)
    context.update({
        "raw": raw, "args": args, "meta": meta,
        "source_checkpoint": str(Path(cli.source_checkpoint).resolve()),
        "source_args": str(Path(cli.source_args).resolve()),
        "device": str(device), "model_metadata": model_metadata,
        "epoch_rows": [], "started_at": time.time(),
    })
    resume_checkpoint = None
    if cli.resume:
        resume_checkpoint = torch.load(str(cli.resume), map_location="cpu")
        core.validate_resume(resume_checkpoint, meta)
        core.load_net_strict(model, resume_checkpoint)
    context["resume_checkpoint"] = resume_checkpoint
    return {
        "model": model,
        "train_loader_factory": train_loader_factory,
        "val_loader": _loader(val_dataset, 1, False, cli.seed),
        "context": context,
        "cli": cli,
    }


def baseline_validation(prepared):
    metrics, rows = evaluate_model(
        prepared["model"], prepared["val_loader"], "generic", True)
    _atomic_csv(
        prepared["context"]["raw"] / "baseline_val_frame_metrics.csv",
        rows, metric_fields())
    prepared["context"]["baseline_val"] = metrics
    return metrics


def _record_epoch(prepared, stage, epoch, train_metrics, val_metrics, groups):
    row = {
        "stage": int(stage), "epoch": int(epoch),
        "train_loss": float(train_metrics["loss"]),
        "val_rmse": float(val_metrics["pooled_rmse"]),
        "val_mae": float(val_metrics["pooled_mae"]),
        "val_abs_rel": float(val_metrics["pooled_abs_rel"]),
        "learning_rates": json.dumps(
            {group["name"]: group["lr"] for group in groups}, sort_keys=True),
    }
    prepared["context"]["epoch_rows"].append(row)
    _atomic_csv(
        prepared["context"]["raw"] / "epoch_metrics.csv",
        prepared["context"]["epoch_rows"], tuple(row))
    print(json.dumps(row, sort_keys=True), flush=True)


def _checkpoint(prepared, optimizer, tracker, stage, epoch, val_metrics):
    return core.build_checkpoint(
        prepared["model"], epoch, stage, optimizer.state_dict(),
        tracker.state_dict(), val_metrics, prepared["context"]["args"],
        prepared["context"]["meta"])


def _save_latest(prepared, optimizer, tracker, stage, epoch, val_metrics):
    core.atomic_save_checkpoint(
        _checkpoint(prepared, optimizer, tracker, stage, epoch, val_metrics),
        prepared["context"]["raw"] / "latest.pt",
        protected_path=prepared["context"]["source_checkpoint"])


def _save_best(prepared, optimizer, tracker, stage, epoch, val_metrics):
    core.atomic_save_checkpoint(
        _checkpoint(prepared, optimizer, tracker, stage, epoch, val_metrics),
        prepared["context"]["raw"] / "best.pt",
        protected_path=prepared["context"]["source_checkpoint"])


def _select_best(prepared, tracker):
    path = prepared["context"]["raw"] / "best.pt"
    checkpoint = torch.load(str(path), map_location="cpu")
    core.validate_resume(checkpoint, prepared["context"]["meta"])
    model, _ = _build_source_model(
        prepared["context"]["args"], prepared["context"]["source_checkpoint"],
        torch.device(prepared["context"]["device"]))
    core.load_net_strict(model, checkpoint)
    model.eval()
    prepared["context"]["selected_epoch"] = int(checkpoint["epoch"])
    return model


def _load_test(prepared):
    cli = prepared["cli"]
    rows = _load_rows(cli.test_manifest, "test", cli.data_root)
    dataset = data.SceneDepthDataset(rows, cli.data_root, "test", cli.seed)
    return {
        "loader": _loader(dataset, 1, False, cli.seed),
        "windows": load_held_out_windows(cli.window_manifest),
    }


def _paired_test(prepared, specialized, test_bundle):
    context = prepared["context"]
    generic, _ = _build_source_model(
        context["args"], context["source_checkpoint"],
        torch.device(context["device"]))
    generic.eval()
    specialized.eval()
    rows = []
    selected_keys = {
        (window["scene"], frame_id)
        for window in test_bundle["windows"] for frame_id in window["frame_ids"]}
    captures = {}
    with torch.no_grad():
        for batch in test_bundle["loader"]:
            sample = _to_device(batch, torch.device(context["device"]))
            generic_output = generic(sample)
            specialized_output = specialized(sample)
            generic_pred = generic_output["pred"]
            specialized_pred = specialized_output["pred"]
            scene = str(batch["scene"][0])
            frame_id = int(batch["frame_id"][0])
            for variant, prediction in (
                    ("generic", generic_pred),
                    ("specialized", specialized_pred)):
                raw = core.frame_error_sums(prediction, sample["gt"])
                rows.append(_metric_row(variant, scene, frame_id, raw))
            key = (scene, frame_id)
            if key in selected_keys:
                captures[key] = {
                    "rgb": sample["rgb"][0].cpu().numpy(),
                    "sparse": sample["dep"][0, 0].cpu().numpy(),
                    "gt": sample["gt"][0, 0].cpu().numpy(),
                    "valid": sample["valid"][0, 0].cpu().numpy(),
                    "generic": generic_pred[0, 0].cpu().numpy(),
                    "specialized": specialized_pred[0, 0].cpu().numpy(),
                }
    validate_test_row_identities(rows)
    if set(captures) != selected_keys:
        raise RuntimeError("window capture identities are incomplete")
    _atomic_csv(context["raw"] / "test_frame_metrics.csv", rows, metric_fields())

    ordered = [(window["scene"], frame_id)
               for window in test_bundle["windows"]
               for frame_id in window["frame_ids"]]
    np.savez_compressed(
        str(context["raw"] / "window_predictions.npz"),
        scenes=np.asarray([item[0] for item in ordered]),
        frame_ids=np.asarray([item[1] for item in ordered], dtype=np.int32),
        rgb=np.stack([captures[item]["rgb"] for item in ordered]),
        sparse=np.stack([captures[item]["sparse"] for item in ordered]),
        gt=np.stack([captures[item]["gt"] for item in ordered]),
        valid=np.stack([captures[item]["valid"] for item in ordered]),
        generic=np.stack([captures[item]["generic"] for item in ordered]),
        specialized=np.stack([captures[item]["specialized"] for item in ordered]))
    return {"test_metric_row_count": len(rows), "window_count": 30}


def _finalize(prepared, test_result, stage1_epochs, stage2_epochs):
    context = prepared["context"]
    args = dict(context["args"])
    args.update({
        "source_checkpoint": context["source_checkpoint"],
        "selected_epoch": context["selected_epoch"],
        "stage1_epochs": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "seed": 2026,
    })
    _atomic_json(context["raw"] / "specialized_args.json", args)
    missing = [name for name in RAW_ARTIFACTS
               if not (context["raw"] / name).is_file()]
    if missing:
        raise RuntimeError("worker raw artifacts are missing {}".format(missing))
    metadata = {
        "complete": True,
        "seed": 2026,
        "stage1_epochs": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "selected_epoch": context["selected_epoch"],
        "physical_batch_size": context["physical_batch_size"],
        "accumulation_steps": context["accumulation_steps"],
        "test_metric_row_count": test_result["test_metric_row_count"],
        "window_count": test_result["window_count"],
        "elapsed_seconds": time.time() - context["started_at"],
        "meta": context["meta"],
    }
    _atomic_json(context["raw"] / "worker_metadata.json", metadata)
    return metadata


def run_training(
        cli, prepare_fn=prepare_training, baseline_val_fn=baseline_validation,
        configure_stage_fn=core.configure_stage,
        optimizer_factory=None, train_epoch_fn=train_epoch,
        evaluate_model_fn=evaluate_model, record_epoch_fn=_record_epoch,
        save_latest_fn=_save_latest, save_best_fn=_save_best,
        select_best_fn=_select_best, load_test_fn=_load_test,
        evaluate_test_pair_fn=_paired_test, finalize_fn=_finalize):
    if optimizer_factory is None:
        optimizer_factory = lambda groups: torch.optim.Adam([
            {"params": group["params"], "lr": group["lr"]}
            for group in groups])
    prepared = prepare_fn(cli)
    baseline_val_fn(prepared)
    model = prepared["model"]
    context = prepared["context"]
    tracker = core.ValidationTracker(patience=4, min_relative_gain=0.001)
    resume = context.get("resume_checkpoint")
    if resume:
        tracker = core.ValidationTracker.from_state_dict(resume["tracker"])
    stage_counts = {1: 0, 2: 0}
    for stage, epoch_values in ((1, range(1, 4)), (2, range(4, 19))):
        groups = configure_stage_fn(model, stage)
        optimizer = optimizer_factory(groups)
        if resume and int(resume["stage"] if "stage" in resume else
                          resume["scheduler"]["stage"]) == stage:
            optimizer.load_state_dict(resume["optimizer"])
        if stage == 2:
            tracker.nonsignificant_epochs = 0
            tracker.significant_best_rmse = tracker.best_rmse
            tracker.significant_best_epoch = tracker.best_epoch
        for epoch in epoch_values:
            if resume and epoch <= int(resume["epoch"]):
                continue
            train_metrics = train_epoch_fn(
                model, prepared["train_loader_factory"](epoch), optimizer,
                context["accumulation_steps"], 1.0)
            val_metrics = evaluate_model_fn(model, prepared["val_loader"])
            decision = tracker.update(epoch, val_metrics["pooled_rmse"])
            record_epoch_fn(
                prepared, stage, epoch, train_metrics, val_metrics, groups)
            save_latest_fn(
                prepared, optimizer, tracker, stage, epoch, val_metrics)
            if decision["save_best"]:
                save_best_fn(
                    prepared, optimizer, tracker, stage, epoch, val_metrics)
            stage_counts[stage] += 1
            if stage == 2 and decision["stop"]:
                break
    specialized = select_best_fn(prepared, tracker)
    test_bundle = load_test_fn(prepared)
    test_result = evaluate_test_pair_fn(prepared, specialized, test_bundle)
    return finalize_fn(
        prepared, test_result, stage_counts[1], stage_counts[2])


def make_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune scene-specialized full NLSPN")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-args", required=True)
    parser.add_argument("--window-manifest", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume")
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    torch.set_num_threads(1)
    result = run_training(cli)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
