#!/usr/bin/env python3
"""AdaRound/BRECQ reconstruction for one NYU depth-completion checkpoint.

The command reconstructs explicitly selected layers or semantic blocks, hardens
learned W4 rounding decisions, exports learned activation ranges, and writes a
checkpoint that can be evaluated by the existing edge-aware quantization path.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    build_model,
    load_run_args,
    prepare_args,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    batch_from_sample,
    calibration_dataset,
    evaluation_dataset,
    seeded_sample,
)
from spn_quant.adaptive_rounding import (  # noqa: E402
    AdaptiveRoundingConfig,
    is_supported_weight_module,
)
from spn_quant.reconstruction import (  # noqa: E402
    ModuleIOCache,
    ReconstructionConfig,
    SemanticBlockReconstructor,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def module_at(model: nn.Module, name: str) -> nn.Module:
    try:
        return dict(model.named_modules())[name]
    except KeyError:
        raise KeyError("unknown reconstruction target: %s" % name)


def supported_weight_count(module: nn.Module) -> int:
    count = int(is_supported_weight_module(module))
    count += sum(1 for name, child in module.named_modules()
                 if name and is_supported_weight_module(child))
    return count


def list_targets(model: nn.Module) -> List[Dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        weights = supported_weight_count(module)
        if weights:
            rows.append({
                "module": name,
                "type": type(module).__name__,
                "supported_weights": weights,
                "adaround": int(is_supported_weight_module(module)),
                "brecq": 1,
            })
    return rows


def select_targets(model: nn.Module, exact: Sequence[str],
                   patterns: Sequence[str], method: str) -> List[str]:
    names = dict(model.named_modules())
    selected = []
    for name in exact:
        if name not in names:
            raise KeyError("unknown reconstruction target: %s" % name)
        selected.append(name)
    compiled = [re.compile(pattern) for pattern in patterns]
    for name, module in model.named_modules():
        if not name or not any(pattern.search(name) for pattern in compiled):
            continue
        if method == "adaround" and not is_supported_weight_module(module):
            continue
        if method == "brecq" and supported_weight_count(module) == 0:
            continue
        selected.append(name)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("no reconstruction targets selected")
    for name in selected:
        module = names[name]
        if method == "adaround" and not is_supported_weight_module(module):
            raise TypeError("AdaRound target must be Conv/ConvTranspose/Linear: %s" % name)
        if method == "brecq" and supported_weight_count(module) == 0:
            raise TypeError("BRECQ target contains no supported weights: %s" % name)
    return selected


def validate_non_overlapping_targets(targets: Sequence[str]) -> None:
    for index, left in enumerate(targets):
        for right in targets[index + 1:]:
            if left.startswith(right + ".") or right.startswith(left + "."):
                raise ValueError(
                    "overlapping reconstruction targets are not supported: %s, %s" %
                    (left, right))


def prefix_name(block_name: str, local_name: str) -> str:
    if not local_name:
        return block_name
    return "%s.%s" % (block_name, local_name) if block_name else local_name


def capture_records(model: nn.Module, target: nn.Module, saved_args: Any,
                    dataset: Any, indices: Sequence[int], device: torch.device,
                    seed: int, loss_mode: str, expected_calls: int):
    cache = ModuleIOCache(model, target, expected_calls=expected_calls)
    records = []
    try:
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            batch = batch_from_sample(sample)
            model_args, gt = sweep.batch_to_model_input(
                saved_args.model, batch, device)
            if loss_mode == "fisher":
                def loss_closure(output, target_depth=gt):
                    prediction = sweep.extract_pred(output)
                    return sweep.compute_loss(saved_args, prediction, target_depth)
            else:
                loss_closure = None
            records.extend(cache.capture(model_args, loss_closure=loss_closure))
            print("capture target=%s sample=%d/%d index=%05d" % (
                target.__class__.__name__, rank, len(indices), index), flush=True)
    finally:
        cache.close()
    return records


def evaluate_model(model: nn.Module, saved_args: Any, dataset: Any,
                   indices: Sequence[int], device: torch.device,
                   seed: int) -> List[Dict[str, Any]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for index in indices:
            sample = seeded_sample(dataset, index, seed)
            batch = batch_from_sample(sample)
            model_args, gt = sweep.batch_to_model_input(
                saved_args.model, batch, device)
            prediction = sweep.extract_pred(model(*model_args))
            metric = sweep.evaluate_error(gt_depth=gt, pred_depth=prediction)
            rows.append({
                "sample_index": int(index),
                "RMSE": float(metric["RMSE"]),
                "MAE": float(metric["MAE"]),
                "ABS_REL": float(metric["ABS_REL"]),
                "finite": int(torch.isfinite(prediction).all().item()),
            })
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--method", choices=("adaround", "brecq"), required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--target-regex", action="append", default=[])
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=0,
                        help="0 disables learned activation reconstruction")
    parser.add_argument("--weight-clip-ratio", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--activation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--round-loss-weight", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--beta-start", type=float, default=20.0)
    parser.add_argument("--beta-end", type=float, default=2.0)
    parser.add_argument("--hard-eval-interval", type=int, default=50)
    parser.add_argument("--loss", choices=("mse", "fisher"), default="mse")
    parser.add_argument("--qdrop-probability", type=float, default=0.0)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--expected-calls", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out-dir", default="profile_logs/nyu_reconstruction")
    parser.add_argument("--save-checkpoint", default="reconstructed.pt")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    saved_args = prepare_args(load_run_args(run_dir), args)
    if saved_args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for %s" % saved_args.model)
    device = torch.device(saved_args.device)
    model, meta = build_model(saved_args, checkpoint, device)
    model.eval()

    if args.list_targets:
        rows = list_targets(model)
        for row in rows:
            print("{module:70s} {type:24s} weights={supported_weights} "
                  "adaround={adaround} brecq={brecq}".format(**row))
        return

    targets = select_targets(
        model, args.target, args.target_regex, args.method)
    validate_non_overlapping_targets(targets)
    dataset = calibration_dataset(saved_args)
    calibration_count = min(int(args.calibration_samples), len(dataset))
    rng = np.random.RandomState(args.seed)
    calibration_indices = rng.choice(
        len(dataset), calibration_count, replace=False).tolist()

    out_dir = Path(args.out_dir) / saved_args.model / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    weight_rows = []
    activation_rows = []
    histories = {}
    active_reconstructors = []

    for target_index, target_name in enumerate(targets, 1):
        target = module_at(model, target_name)
        records = capture_records(
            model, target, saved_args, dataset, calibration_indices,
            device, args.seed, args.loss, args.expected_calls)
        activation_bits = int(args.a_bits) if int(args.a_bits) > 0 else None
        reconstruction_config = ReconstructionConfig(
            steps=args.steps,
            learning_rate=args.learning_rate,
            activation_learning_rate=args.activation_learning_rate,
            round_loss_weight=args.round_loss_weight,
            warmup_fraction=args.warmup_fraction,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            loss=args.loss,
            activation_bits=activation_bits,
            qdrop_probability=args.qdrop_probability,
            seed=args.seed + target_index,
            hard_eval_interval=args.hard_eval_interval,
        )
        weight_config = AdaptiveRoundingConfig(
            bits=args.w_bits, clip_ratio=args.weight_clip_ratio)
        reconstructor = SemanticBlockReconstructor(
            target, weight_config, reconstruction_config)
        result = reconstructor.fit(records, harden=True)
        active_reconstructors.append(reconstructor)

        summary = result.manifest()
        summary.update({
            "target": target_name,
            "method": args.method,
            "model": saved_args.model,
        })
        summary_rows.append(summary)
        for row in result.weight_manifest:
            row = dict(row)
            row["target"] = target_name
            row["module"] = prefix_name(target_name, row.get("module", ""))
            weight_rows.append(row)
        for row in result.activation_manifest:
            row = dict(row)
            row["target"] = target_name
            row["site"] = prefix_name(target_name, row.get("site", ""))
            activation_rows.append(row)
        histories[target_name] = result.history
        print("reconstructed %d/%d target=%s before=%.8f after=%.8f" % (
            target_index, len(targets), target_name,
            result.before_loss, result.after_loss), flush=True)

    checkpoint_path = out_dir / args.save_checkpoint
    torch.save({
        "net": model.state_dict(),
        "source_checkpoint": str(checkpoint),
        "method": args.method,
        "targets": targets,
        "weight_manifest": weight_rows,
        "activation_manifest": activation_rows,
        "architecture": meta,
    }, checkpoint_path)
    write_csv(out_dir / "reconstruction_summary.csv", summary_rows)
    write_csv(out_dir / "weight_rounding_manifest.csv", weight_rows)
    write_csv(out_dir / "activation_reconstruction_manifest.csv", activation_rows)
    write_json(out_dir / "reconstruction_history.json", histories)
    write_json(out_dir / "reconstruction_manifest.json", {
        "model": saved_args.model,
        "method": args.method,
        "source_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint_path),
        "targets": targets,
        "weight_bits": args.w_bits,
        "activation_bits": args.a_bits,
        "calibration_indices": calibration_indices,
        "weight_manifest": weight_rows,
        "activation_manifest": activation_rows,
    })

    if int(args.eval_samples) > 0:
        valset = evaluation_dataset(saved_args)
        count = min(int(args.eval_samples), len(valset))
        eval_indices = list(range(count))
        eval_rows = evaluate_model(
            model, saved_args, valset, eval_indices, device, args.seed)
        write_csv(out_dir / "evaluation_metrics.csv", eval_rows)
        finite = all(row["finite"] for row in eval_rows)
        mean_rmse = sum(row["RMSE"] for row in eval_rows) / max(len(eval_rows), 1)
        print("evaluation samples=%d finite=%s mean_RMSE=%.6f" % (
            len(eval_rows), finite, mean_rmse), flush=True)
    for reconstructor in active_reconstructors:
        reconstructor.close()
    print("saved reconstructed checkpoint: %s" % checkpoint_path, flush=True)


if __name__ == "__main__":
    main()
