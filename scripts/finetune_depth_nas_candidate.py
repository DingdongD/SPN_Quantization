#!/usr/bin/env python3
"""Fine-tune one inherited-weight depth-NAS candidate on NYU."""

from __future__ import annotations

import argparse
from argparse import Namespace
import csv
import json
from pathlib import Path
import random
import sys
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import screen_four_model_depth_nas as nas  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402


def _subset(dataset, count: int, seed: int):
    if count <= 0 or count >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def _training_args(model_name: str, learning_rate: float) -> Namespace:
    recipe = sweep.MODEL_RECIPES[model_name]
    return Namespace(
        model=model_name,
        loss=recipe["loss"],
        warm_up=False,
        lr=float(learning_rate),
        momentum=0.9,
        weight_decay=float(recipe["weight_decay"]),
        log_interval=100,
    )


def _write_history(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, model_name: str, candidate_id: str,
        device: str, output: Path, epochs: int, train_samples: int,
        batch_size: int, learning_rate: float, seed: int,
        initial_checkpoint: Path | None = None) -> Path:
    if output.exists():
        raise FileExistsError("fine-tune output already exists: %s" % output)
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    payload = source["models"][model_name]
    runtime = NYUModelRuntime.from_args(quant._runtime_args(payload, device))
    quant.configure_runtime_execution(runtime)
    try:
        model = runtime.build_model(runtime.device)
        paths = nas.STAGE_PATHS[model_name]
        full_stages = nas.stage_modules(model, paths)
        depths = dict(nas.CANDIDATE_DEPTHS[model_name])[candidate_id]
        nas.apply_depths(model, paths, full_stages, depths)
        if initial_checkpoint is not None:
            initial = torch.load(str(initial_checkpoint), map_location=runtime.device)
            if initial["model"] != model_name or \
                    initial["candidate_id"] != candidate_id:
                raise ValueError("initial checkpoint identity mismatch")
            model.load_state_dict(initial["net"], strict=True)
        trainset = _subset(runtime.build_dataset("train"), train_samples, seed)
        valset = Subset(runtime.build_dataset("val"), [
            int(value) for value in payload["evaluation_indices"]])
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2,
            pin_memory=True, drop_last=True)
        valloader = DataLoader(
            valset, batch_size=1, shuffle=False, num_workers=2,
            pin_memory=True, drop_last=False)
        args = _training_args(model_name, learning_rate)
        optimizer, scheduler = sweep.make_optimizer_scheduler(args, model)
        output.mkdir(parents=True)
        history = []
        best_rmse = float("inf")
        for epoch in range(1, epochs + 1):
            train_metrics = sweep.train_one_epoch(
                args, model, trainloader, optimizer, runtime.device, epoch)
            val_metrics = sweep.evaluate(
                args, model, valloader, runtime.device)
            if model_name == "cspn":
                scheduler.step(val_metrics["MAE"])
            else:
                scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_rmse_m": train_metrics["RMSE"],
                "validation_loss": val_metrics["loss"],
                "validation_mean_sample_rmse_m": val_metrics["RMSE"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if float(val_metrics["RMSE"]) < best_rmse:
                best_rmse = float(val_metrics["RMSE"])
                torch.save({
                    "net": model.state_dict(),
                    "model": model_name,
                    "candidate_id": candidate_id,
                    "depths": list(depths),
                    "epoch": epoch,
                    "validation": val_metrics,
                    "source_checkpoint": str(runtime.checkpoint),
                }, str(output / "best.pt"))
        _write_history(output / "history.csv", history)
        (output / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "model": model_name,
            "candidate_id": candidate_id,
            "depths": list(depths),
            "epochs": epochs,
            "train_samples": len(trainset),
            "validation_indices": list(payload["evaluation_indices"]),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "best_mean_sample_rmse_m": best_rmse,
            "source_checkpoint": str(runtime.checkpoint),
            "initial_checkpoint": "" if initial_checkpoint is None else
                str(initial_checkpoint.resolve()),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output / "manifest.json"
    finally:
        runtime.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=nas.MODEL_ORDER, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.candidate, args.device,
              args.output, args.epochs, args.train_samples,
              args.batch_size, args.learning_rate, args.seed,
              args.initial_checkpoint))


if __name__ == "__main__":
    main()
