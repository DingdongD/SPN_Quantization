#!/usr/bin/env python3
"""Export NYU validation predictions from one trained sweep checkpoint."""

from __future__ import print_function

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_nyu_iteration_sweep as sweep  # noqa: E402


def torch_load(path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def load_run_args(run_dir):
    args_path = Path(run_dir) / "args.json"
    if not args_path.exists():
        raise FileNotFoundError("missing args.json in %s" % run_dir)
    data = json.loads(args_path.read_text(encoding="utf-8"))
    return argparse.Namespace(**data)


def metric_np(gt, pred):
    valid = gt > 1e-4
    if not np.any(valid):
        return {"RMSE": float("nan"), "MAE": float("nan"),
                "ABS_REL": float("nan"), "MSE": float("nan")}
    diff = np.abs(pred[valid] - gt[valid])
    mse = float(np.mean(diff ** 2))
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(diff)),
        "ABS_REL": float(np.mean(diff / np.maximum(gt[valid], 1e-6))),
        "MSE": mse,
    }


def rgb_for_visualization(rgb):
    return np.clip(rgb.detach().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)


def write_csv(path, rows):
    fieldnames = [
        "label", "model", "iteration", "sample_index", "RMSE", "MAE",
        "ABS_REL", "MSE", "npz_path",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((k, row.get(k, "")) for k in fieldnames))


def prepare_args(saved_args, cli_args):
    saved_args.device = cli_args.device
    saved_args.workers = 0
    saved_args.batch_size = 1
    saved_args.val_batch_size = 1
    saved_args.max_train_samples = 0
    saved_args.max_val_samples = 0
    saved_args.cudnn_benchmark = False
    return saved_args


def build_model(args, checkpoint, device):
    model, meta = sweep.BUILDERS[args.model](args, device)
    state = torch_load(checkpoint, map_location="cpu")
    state_dict = state["net"] if isinstance(state, dict) and "net" in state else state
    if args.model == "cspn":
        state_dict = dict((k, v) for k, v in state_dict.items()
                          if k != "post_process_layer.sum_conv.weight")
    model.load_state_dict(state_dict, strict=False if args.model == "cspn" else True)
    model.eval()
    return model, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--label", default="")
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--out-dir", default="profile_logs/nyu_prediction_comparison/predictions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))

    saved_args = prepare_args(load_run_args(run_dir), args)
    label = args.label or "%s_iter%d" % (saved_args.model, saved_args.iteration)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if saved_args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for %s" % saved_args.model)
    device = torch.device(saved_args.device)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = saved_args.allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = saved_args.allow_tf32

    model, meta = build_model(saved_args, checkpoint, device)
    dataset = sweep.NyuHdf5Dataset(
        csv_file=saved_args.eval_list,
        root_dir=str(sweep.resolve_data_root(saved_args)),
        split="val",
        n_sample=saved_args.n_sample,
        seed=saved_args.seed,
    )

    rows = []
    with torch.no_grad():
        for sample_index in args.sample_indices:
            np.random.seed(args.seed + sample_index)
            torch.manual_seed(args.seed + sample_index)
            sample = dataset[sample_index]
            batch = {key: value.unsqueeze(0) for key, value in sample.items()}
            model_args, gt = sweep.batch_to_model_input(saved_args.model, batch, device)
            pred = sweep.extract_pred(model(*model_args)).detach().cpu()[0, 0].numpy()

            rgb = rgb_for_visualization(sample["rgbd"][:3])
            sparse = sample["rgbd"][3].numpy()
            gt_np = sample["depth"][0].numpy()
            err = np.abs(pred - gt_np) * (gt_np > 1e-4)
            metrics = metric_np(gt_np, pred)

            npz_name = "sample_%05d_%s.npz" % (sample_index, label)
            npz_path = out_dir / npz_name
            np.savez_compressed(
                str(npz_path),
                rgb=rgb.astype(np.float32),
                sparse=sparse.astype(np.float32),
                gt=gt_np.astype(np.float32),
                pred=pred.astype(np.float32),
                abs_err=err.astype(np.float32),
                valid=(gt_np > 1e-4),
                label=np.array(label),
                model=np.array(saved_args.model),
                iteration=np.array(saved_args.iteration),
                sample_index=np.array(sample_index),
                metrics=np.array(json.dumps(metrics)),
                meta=np.array(json.dumps(meta)),
            )
            row = {
                "label": label,
                "model": saved_args.model,
                "iteration": saved_args.iteration,
                "sample_index": sample_index,
                "npz_path": str(npz_path),
            }
            row.update(metrics)
            rows.append(row)
            print("%s sample=%05d rmse=%.5f mae=%.5f" % (
                label, sample_index, metrics["RMSE"], metrics["MAE"]), flush=True)

    write_csv(out_dir / ("%s_metrics.csv" % label), rows)
    print("saved %d prediction files to %s" % (len(rows), out_dir), flush=True)


if __name__ == "__main__":
    main()
