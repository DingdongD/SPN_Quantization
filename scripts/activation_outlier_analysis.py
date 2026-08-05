#!/usr/bin/env python3
"""Profile activation tails and channel-localized outliers."""

from __future__ import division

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PERCENTILES = (
    ("p75", 0.75),
    ("p90", 0.90),
    ("p99", 0.99),
    ("p99_9", 0.999),
    ("p99_99", 0.9999),
)


def per_update_budget(capacity, updates, maximum=8192):
    if capacity <= 0 or updates <= 0 or maximum <= 0:
        raise ValueError("capacity, updates, and maximum must be positive")
    return min(int(maximum), max(1, (int(capacity) + int(updates) - 1)
                                    // int(updates)))


def _safe_ratio(numerator, denominator):
    if denominator > 0.0:
        return float(numerator) / float(denominator)
    return float("inf") if numerator > 0.0 else 1.0


class BoundedActivationSampler(object):
    """Deterministically samples values while retaining exact channel maxima."""

    def __init__(self, capacity=1000000, per_update=8192):
        self.capacity = int(capacity)
        self.per_update = int(per_update)
        if self.capacity <= 0 or self.per_update <= 0:
            raise ValueError("capacity and per_update must be positive")
        self.chunks = []
        self.retained_values = 0
        self.total_values = 0
        self.updates = 0
        self.channel_absmax = None

    def _sample(self, flat, count):
        if count >= flat.numel():
            return flat
        stride = float(flat.numel()) / float(count)
        offset = (self.updates * 0.6180339887498949) % 1.0
        indices = torch.floor(
            (torch.arange(count, device=flat.device, dtype=torch.float64) + offset)
            * stride).to(torch.long)
        return flat.index_select(0, indices.clamp(max=flat.numel() - 1))

    def update(self, tensor, channel_axis=1):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        detached = tensor.detach().abs()
        self.total_values += int(detached.numel())
        axis = int(channel_axis)
        if axis < 0:
            axis += detached.ndim
        if axis < 0 or axis >= detached.ndim:
            raise ValueError("invalid channel axis")
        channel_max = detached.movedim(axis, 0).reshape(
            detached.shape[axis], -1).amax(dim=1).float().cpu()
        if self.channel_absmax is None:
            self.channel_absmax = channel_max
        else:
            if self.channel_absmax.shape != channel_max.shape:
                raise ValueError("channel count changed between updates")
            self.channel_absmax = torch.maximum(self.channel_absmax, channel_max)

        remaining = self.capacity - self.retained_values
        if remaining > 0:
            count = min(self.per_update, remaining, int(detached.numel()))
            sampled = self._sample(detached.reshape(-1), count).float().cpu()
            self.chunks.append(sampled)
            self.retained_values += int(sampled.numel())
        self.updates += 1

    def values(self):
        if not self.chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(self.chunks)

    def statistics(self):
        values = self.values()
        if values.numel() == 0 or self.channel_absmax is None:
            raise RuntimeError("activation sampler has no observations")
        channel = self.channel_absmax
        quantiles = torch.tensor(
            [value for _, value in PERCENTILES], dtype=torch.float32)
        measured = torch.quantile(values, quantiles)
        row = dict((name, float(value.item()))
                   for (name, _), value in zip(PERCENTILES, measured))
        row.update({
            "minimum": float(values.min().item()),
            "maximum": float(channel.max().item()),
            "retained_values": int(values.numel()),
            "total_values": self.total_values,
            "updates": self.updates,
        })
        row["max_over_p99_99"] = _safe_ratio(row["maximum"], row["p99_99"])
        row["p99_99_over_p99"] = _safe_ratio(row["p99_99"], row["p99"])
        for name in ("p99", "p99_9", "p99_99"):
            row["fraction_above_%s" % name] = float(
                (values > row[name]).float().mean().item())

        channel_quantiles = torch.quantile(
            channel, torch.tensor([0.5, 0.99], dtype=torch.float32))
        row.update({
            "channels": int(channel.numel()),
            "channel_absmax_median": float(channel_quantiles[0].item()),
            "channel_absmax_p99": float(channel_quantiles[1].item()),
            "channel_absmax_max": float(channel.max().item()),
        })
        row["channel_max_over_median"] = _safe_ratio(
            row["channel_absmax_max"], row["channel_absmax_median"])
        row["channel_p99_over_median"] = _safe_ratio(
            row["channel_absmax_p99"], row["channel_absmax_median"])
        return row


class ActivationOutlierProfiler(object):
    def __init__(self, model, group_fn, capacity=1000000, per_update=8192):
        self.model = model
        self.capacity = int(capacity)
        self.per_update = int(per_update)
        self.samplers = {}
        self.call_counts = {}
        self.modules = {}
        self.groups = {}
        self.handles = [model.register_forward_pre_hook(self._reset_calls)]

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            group = group_fn(name, module)
            if group is None:
                continue
            self.modules[name] = module
            self.groups[name] = str(group)
            self.handles.append(
                module.register_forward_pre_hook(self._make_input_hook(name)))
            self.handles.append(
                module.register_forward_hook(self._make_output_hook(name)))

    def _reset_calls(self, module, inputs):
        del module, inputs
        self.call_counts = {}

    def _next_index(self, name, kind):
        key = (name, kind)
        index = self.call_counts.get(key, 0)
        self.call_counts[key] = index + 1
        return index

    def _sampler(self, name, index, kind):
        key = (name, index, kind)
        if key not in self.samplers:
            self.samplers[key] = BoundedActivationSampler(
                capacity=self.capacity, per_update=self.per_update)
        return self.samplers[key]

    def _channel_axis(self, module):
        return 1 if isinstance(module, nn.Conv2d) else -1

    def _make_input_hook(self, name):
        def hook(module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                index = self._next_index(name, "input")
                self._sampler(name, index, "input").update(
                    inputs[0], channel_axis=self._channel_axis(module))
            return None
        return hook

    def _make_output_hook(self, name):
        def hook(module, inputs, output):
            del inputs
            if torch.is_tensor(output):
                index = self._next_index(name, "output")
                self._sampler(name, index, "output").update(
                    output, channel_axis=self._channel_axis(module))
            return None
        return hook

    def rows(self):
        rows = []
        for (name, index, kind), sampler in sorted(self.samplers.items()):
            row = {
                "module": name,
                "site": "%s#%d" % (name, index),
                "call_index": index,
                "group": self.groups[name],
                "kind": kind,
            }
            row.update(sampler.statistics())
            rows.append(row)
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def occupancy_rows(layer_rows, config="HW_W4A4_full"):
    selected = [row for row in layer_rows if row.get("config") == config]
    if not selected:
        return []
    model = selected[0]["model"]
    measures = {
        "parameter_elements": lambda row: row.get("kind") == "weight",
        "activation_elements": lambda row: row.get("kind") in ("input", "output"),
        "boundaries": lambda row: row.get("kind") in ("input", "output"),
    }
    output = []
    for measure, include in measures.items():
        totals = defaultdict(float)
        for row in selected:
            if include(row):
                value = 1.0 if measure == "boundaries" else float(row["numel"])
                totals[row["group"]] += value
        denominator = sum(totals.values())
        for group, value in sorted(totals.items()):
            output.append({
                "model": model,
                "config": config,
                "measure": measure,
                "group": group,
                "value": value,
                "share": value / denominator if denominator else 0.0,
            })
    return output


def group_summary_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["kind"])].append(row)
    output = []
    for (group, kind), points in sorted(grouped.items()):
        output.append({
            "group": group,
            "kind": kind,
            "sites": len(points),
            "median_max_over_p99_99": float(np.median([
                row["max_over_p99_99"] for row in points])),
            "maximum_max_over_p99_99": max(
                row["max_over_p99_99"] for row in points),
            "median_p99_99_over_p99": float(np.median([
                row["p99_99_over_p99"] for row in points])),
            "median_channel_max_over_median": float(np.median([
                row["channel_max_over_median"] for row in points])),
            "maximum_channel_max_over_median": max(
                row["channel_max_over_median"] for row in points),
        })
    return output


def _read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _save_channel_maxima(path, profiler):
    arrays = {}
    index_rows = []
    for array_index, ((name, call_index, kind), sampler) in enumerate(
            sorted(profiler.samplers.items())):
        key = "site_%05d" % array_index
        arrays[key] = sampler.channel_absmax.numpy().astype(np.float32)
        index_rows.append({
            "key": key,
            "module": name,
            "site": "%s#%d" % (name, call_index),
            "call_index": call_index,
            "kind": kind,
            "channels": int(sampler.channel_absmax.numel()),
        })
    np.savez_compressed(str(path), **arrays)
    _write_csv(Path(path).with_name("channel_absmax_index.csv"), index_rows)


def main():
    from scripts import train_nyu_iteration_sweep as sweep
    from scripts.export_nyu_predictions import build_model, load_run_args, prepare_args
    from scripts.hardware_aligned_quantization import prepare_hardware_model
    from scripts.nyu_quantization_analysis import classify_module
    from scripts.run_nyu_rtn_quantization import (
        batch_from_sample,
        calibration_dataset,
        seeded_sample,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--out-dir", default="profile_logs/nyu_activation_outliers")
    parser.add_argument("--hardware-root",
                        default="profile_logs/nyu_hardware_aligned_quantization")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=1000000)
    parser.add_argument("--per-update", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    saved_args = prepare_args(load_run_args(run_dir), args)
    device = torch.device(saved_args.device)
    model, architecture = build_model(saved_args, checkpoint, device)
    dataset = calibration_dataset(saved_args)
    count = min(args.calibration_samples, len(dataset))
    indices = np.random.RandomState(args.seed).choice(
        len(dataset), count, replace=False).tolist()

    first = seeded_sample(dataset, indices[0], args.seed)
    first_args, _ = sweep.batch_to_model_input(
        saved_args.model, batch_from_sample(first), device)
    excluded = [("conv1_1", "bn1")] if saved_args.model == "cspn" else []
    preparation = prepare_hardware_model(
        model, first_args, excluded_pairs=excluded)
    per_update = args.per_update or per_update_budget(args.capacity, count)
    profiler = ActivationOutlierProfiler(
        model,
        lambda name, module: classify_module(saved_args.model, name, module),
        capacity=args.capacity,
        per_update=per_update,
    )

    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, args.seed)
            model_args, _ = sweep.batch_to_model_input(
                saved_args.model, batch_from_sample(sample), device)
            model(*model_args)
            if rank % 16 == 0 or rank == count:
                print("profile %d/%d" % (rank, count), flush=True)

    rows = profiler.rows()
    for row in rows:
        row["model"] = saved_args.model
    model_out = Path(args.out_dir) / saved_args.model
    model_out.mkdir(parents=True, exist_ok=True)
    _write_csv(model_out / "activation_percentiles.csv", rows)
    _write_csv(model_out / "group_outlier_summary.csv", group_summary_rows(rows))
    _save_channel_maxima(model_out / "channel_absmax.npz", profiler)

    layer_rows = _read_csv(
        Path(args.hardware_root) / saved_args.model /
        "layer_quantization_metrics.csv")
    occupancy = occupancy_rows(layer_rows)
    _write_csv(model_out / "occupancy.csv", occupancy)
    _write_json(model_out / "metadata.json", {
        "model": saved_args.model,
        "checkpoint": str(checkpoint),
        "architecture": architecture,
        "seed": args.seed,
        "calibration_samples": count,
        "calibration_indices": indices,
        "capacity": args.capacity,
        "per_update": per_update,
        "sites": len(rows),
        "folded_pairs": preparation["folded_pairs"],
        "unfolded_fanout_pairs": preparation["unfolded_fanout_pairs"],
        "folded_fp32_max_abs_error": preparation["max_abs_error"],
    })
    profiler.close()
    print("model=%s sites=%d out=%s" % (
        saved_args.model, len(rows), model_out), flush=True)


if __name__ == "__main__":
    main()
