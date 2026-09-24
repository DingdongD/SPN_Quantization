#!/usr/bin/env python3
"""Evaluate trained CSPN NAS checkpoints on the official NYU validation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "models") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "models"))

from cspn_encoder_nas import build_cspn_nas
from scripts import train_nyu_iteration_sweep as trainer
from spn_quant.nas.spec import EncoderSpec


FIELDNAMES = ("seed", "sample_id", "RMSE", "MAE", "ABS_REL", "DELTA1.25")


def parse_run(value: str) -> tuple[int, Path]:
    try:
        seed_value, run_value = value.split("=", 1)
        seed = int(seed_value)
    except (TypeError, ValueError) as error:
        raise ValueError("run must have form SEED=RUN_DIR") from error
    if not run_value:
        raise ValueError("run must have form SEED=RUN_DIR")
    return seed, Path(run_value)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _torch_load(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    run_dir: Path,
    device: torch.device,
    checkpoint: str = "best.pt",
    cspn_steps: int | None = None,
):
    checkpoint_path = Path(run_dir) / checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError("missing best checkpoint: %s" % checkpoint_path)
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    try:
        spec = EncoderSpec.from_dict(checkpoint["meta"]["encoder_spec"])
        saved_iteration = int(checkpoint["args"]["iteration"])
        state = dict(checkpoint["net"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint lacks CSPN NAS identity: %s" % checkpoint_path) from error
    iteration = saved_iteration if cspn_steps is None else int(cspn_steps)
    if iteration <= 0:
        raise ValueError("cspn_steps must be positive")
    model = build_cspn_nas(spec, cspn_step=iteration)
    dynamic_key = "post_process_layer.sum_conv.weight"
    state.pop(dynamic_key, None)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if key != dynamic_key]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: missing=%s unexpected=%s" %
            (missing, list(incompatible.unexpected_keys)))
    return model.to(device).eval(), checkpoint, checkpoint_path


def _write_rows(output: Path, rows: Iterable[dict]) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def evaluate_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    *,
    seed: int,
    output: Path,
    batch_size: int = 1,
    workers: int = 0,
) -> list[dict]:
    rows = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
        drop_last=False)
    processed = 0
    with torch.inference_mode():
        for batch in loader:
            model_args, gt = trainer.batch_to_model_input("cspn", batch, device)
            pred = trainer.extract_pred(model(*model_args))
            for local_index in range(int(gt.shape[0])):
                sample_id = processed + local_index
                metric = trainer.evaluate_error(
                    gt[local_index:local_index + 1],
                    pred[local_index:local_index + 1])
                row = {
                    "seed": int(seed),
                    "sample_id": int(sample_id),
                    **{key: float(metric[key])
                       for key in ("RMSE", "MAE", "ABS_REL", "DELTA1.25")},
                }
                if not all(math.isfinite(row[key]) for key in FIELDNAMES[2:]):
                    raise RuntimeError(
                        "non-finite metric for sample %d" % sample_id)
                rows.append(row)
            processed += int(gt.shape[0])
            if processed % 50 < int(gt.shape[0]) or processed == len(dataset):
                print("seed=%d sample=%d/%d" %
                      (seed, processed, len(dataset)), flush=True)
    _write_rows(Path(output), rows)
    return rows


def evaluate_run(
    seed: int,
    run_dir: Path,
    *,
    eval_list: Path,
    data_root: Path,
    device: torch.device,
    output: Path,
    checkpoint: str = "best.pt",
    cspn_steps: int | None = None,
    batch_size: int = 1,
    workers: int = 0,
) -> dict:
    model, payload, checkpoint_path = load_model(
        run_dir, device, checkpoint=checkpoint, cspn_steps=cspn_steps)
    saved_seed = int(payload["args"]["seed"])
    if saved_seed != int(seed):
        raise ValueError(
            "assigned seed %d does not match checkpoint seed %d" %
            (seed, saved_seed))
    dataset = trainer.CspnOfficialDataset(
        csv_file=str(eval_list), root_dir=str(data_root), split="val",
        n_sample=int(payload["args"].get("n_sample", 500)), seed=seed)
    rows = evaluate_dataset(
        model, dataset, device, seed=seed, output=output,
        batch_size=batch_size, workers=workers)
    return {
        "seed": seed,
        "run_dir": str(Path(run_dir).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "eval_list": str(Path(eval_list).resolve()),
        "eval_list_sha256": _sha256(Path(eval_list)),
        "samples": len(rows),
        "cspn_steps": int(
            payload["args"]["iteration"] if cspn_steps is None else cspn_steps),
        "batch_size": int(batch_size),
        "workers": int(workers),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="training seed and run directory as SEED=RUN_DIR")
    parser.add_argument("--eval-list", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--cspn-steps", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--workers", type=nonnegative_int, default=0)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    output = Path(args.output)
    all_rows = []
    metadata = []
    for value in args.run:
        seed, run_dir = parse_run(value)
        seed_output = output.with_name("%s.seed%d.csv" % (output.stem, seed))
        metadata.append(evaluate_run(
            seed, run_dir, eval_list=Path(args.eval_list),
            data_root=Path(args.data_root), device=device,
            output=seed_output, checkpoint=args.checkpoint,
            cspn_steps=args.cspn_steps, batch_size=args.batch_size,
            workers=args.workers))
        with seed_output.open(newline="", encoding="utf-8") as stream:
            all_rows.extend(csv.DictReader(stream))
    _write_rows(output, all_rows)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
