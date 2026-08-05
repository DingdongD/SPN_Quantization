#!/usr/bin/env python3
"""Model grouping and information-damage metrics for NYU RTN experiments."""

from __future__ import division

import math

import numpy as np
import torch


MODULE_GROUP_ORDER = [
    "encoder",
    "attention",
    "decoder",
    "depth_head",
    "propagation_head",
]


def classify_module(model_name, name, module=None):
    del module
    if model_name == "cspn":
        if name.startswith("gud_up_proj_layer6"):
            return "propagation_head"
        if name.startswith("gud_up_proj_layer5") or name == "conv3":
            return "depth_head"
        if name.startswith("gud_up_proj_layer") or name.startswith("up_proj_layer"):
            return "decoder"
        return "encoder"

    if model_name == "dyspn":
        if ".conv_offset_aff" in name or "base.gd_dec0" in name:
            return "propagation_head"
        if name.startswith("base.conv6") or name.startswith("base.gd_dec1"):
            return "decoder"
        return "encoder"

    if model_name == "nlspn":
        if name.startswith("prop_layer") or name.startswith("gd_dec") or name.startswith("cf_dec"):
            return "propagation_head"
        if name.startswith("id_dec"):
            return "depth_head"
        if name.startswith("conv6"):
            return "decoder"
        return "encoder"

    if model_name == "completionformer":
        if name.startswith("prop_layer") or name.startswith("backbone.gd_dec") \
                or name.startswith("backbone.cf_dec"):
            return "propagation_head"
        if name.startswith("backbone.dep_dec"):
            return "depth_head"
        if name.startswith("backbone.dec"):
            return "decoder"
        if ".attn." in name or ".mlp." in name:
            return "attention"
        return "encoder"

    raise ValueError("unknown model: %s" % model_name)


def group_manifest(model_name, model):
    manifest = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            manifest[name] = classify_module(model_name, name, module)
    return manifest


def _flatten_tensor(value):
    if torch.is_tensor(value):
        return value.detach().double().reshape(-1)
    if isinstance(value, (list, tuple)):
        tensors = [_flatten_tensor(item) for item in value if item is not None]
        if not tensors:
            return torch.empty(0, dtype=torch.double)
        return torch.cat(tensors)
    raise TypeError("expected tensor or tensor sequence, got %s" % type(value).__name__)


def tensor_metrics(reference, candidate):
    reference = _flatten_tensor(reference)
    candidate = _flatten_tensor(candidate)
    if reference.numel() != candidate.numel():
        raise ValueError("tensor sizes differ: %d != %d" % (
            reference.numel(), candidate.numel()))
    if reference.numel() == 0:
        raise ValueError("cannot compare empty tensors")
    difference = candidate - reference
    error_sq = float(torch.sum(difference * difference).item())
    signal_sq = float(torch.sum(reference * reference).item())
    reference_norm = math.sqrt(signal_sq)
    candidate_sq = float(torch.sum(candidate * candidate).item())
    denominator = reference_norm * math.sqrt(candidate_sq)
    cosine = float(torch.sum(reference * candidate).item()) / denominator if denominator else 1.0
    if error_sq == 0.0:
        sqnr = float("inf")
    elif signal_sq == 0.0:
        sqnr = float("-inf")
    else:
        sqnr = 10.0 * math.log10(signal_sq / error_sq)
    return {
        "numel": int(reference.numel()),
        "mse": error_sq / reference.numel(),
        "rmse": math.sqrt(error_sq / reference.numel()),
        "mae": float(torch.mean(torch.abs(difference)).item()),
        "sqnr_db": sqnr,
        "cosine": cosine,
        "sign_flip_rate": float(torch.mean((reference * candidate < 0).double()).item()),
        "zeroed_rate": float(torch.mean(((reference != 0) & (candidate == 0)).double()).item()),
    }


def affinity_metrics(reference, candidate, channel_dim=1):
    metrics = tensor_metrics(reference, candidate)
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        return metrics
    reference_winner = torch.argmax(torch.abs(reference), dim=channel_dim)
    candidate_winner = torch.argmax(torch.abs(candidate), dim=channel_dim)
    metrics["dominant_neighbor_change_rate"] = float(
        torch.mean((reference_winner != candidate_winner).double()).item())
    return metrics


def offset_metrics(reference, candidate, channel_dim=1):
    metrics = tensor_metrics(reference, candidate)
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        return metrics
    channels = reference.shape[channel_dim]
    if channels % 2 != 0:
        raise ValueError("offset channel dimension must contain x/y pairs")
    moved_ref = torch.movedim(reference.detach().double(), channel_dim, -1)
    moved_candidate = torch.movedim(candidate.detach().double(), channel_dim, -1)
    pairs_ref = moved_ref.reshape(*moved_ref.shape[:-1], channels // 2, 2)
    pairs_candidate = moved_candidate.reshape(*moved_candidate.shape[:-1], channels // 2, 2)
    endpoint = torch.sqrt(torch.sum((pairs_candidate - pairs_ref) ** 2, dim=-1))
    metrics["endpoint_error"] = float(endpoint.mean().item())
    metrics["endpoint_error_max"] = float(endpoint.max().item())
    return metrics


def _depth_row(region, gt, pred, mask):
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "region": region,
            "num_pixels": 0,
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "sum_abs_rel": 0.0,
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "ABS_REL": float("nan"),
        }
    difference = np.abs(pred[mask] - gt[mask]).astype(np.float64)
    sum_sq = float(np.sum(difference ** 2))
    sum_abs = float(np.sum(difference))
    sum_abs_rel = float(np.sum(difference / np.maximum(gt[mask], 1e-6)))
    return {
        "region": region,
        "num_pixels": count,
        "sum_sq": sum_sq,
        "sum_abs": sum_abs,
        "sum_abs_rel": sum_abs_rel,
        "RMSE": math.sqrt(sum_sq / count),
        "MAE": sum_abs / count,
        "ABS_REL": sum_abs_rel / count,
    }


def depth_boundary_mask(gt, valid, threshold):
    boundary = np.zeros_like(valid, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] \
        & (np.abs(gt[:, 1:] - gt[:, :-1]) > threshold)
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    vertical = valid[1:, :] & valid[:-1, :] \
        & (np.abs(gt[1:, :] - gt[:-1, :]) > threshold)
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def regional_depth_metrics(gt, pred, sparse, edge_threshold=0.1):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    sparse = np.asarray(sparse)
    if gt.shape != pred.shape or gt.shape != sparse.shape:
        raise ValueError("gt, pred, and sparse must have identical shapes")
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-4)
    boundary = depth_boundary_mask(gt, valid, edge_threshold)
    sparse_anchor = valid & (sparse > 1e-4)
    regions = [
        ("all", valid),
        ("boundary", valid & boundary),
        ("smooth", valid & ~boundary),
        ("sparse_anchor", sparse_anchor),
        ("holes", valid & ~sparse_anchor),
        ("near_0_2m", valid & (gt < 2.0)),
        ("mid_2_5m", valid & (gt >= 2.0) & (gt < 5.0)),
        ("far_5_10m", valid & (gt >= 5.0) & (gt <= 10.0)),
    ]
    return [_depth_row(name, gt, pred, mask) for name, mask in regions]


def extract_output_signals(output):
    if torch.is_tensor(output):
        return {"pred": output}
    if not isinstance(output, dict):
        raise TypeError("model output must be a tensor or dict")
    signals = {}
    aliases = {
        "pred": "pred",
        "pred_init": "pred_init",
        "guidance": "guidance",
        "offset": "offset",
        "aff": "affinity",
        "confidence": "confidence",
        "list_feat": "propagation_states",
        "pred_inter": "propagation_states",
    }
    for source, target in aliases.items():
        value = output.get(source)
        if value is not None:
            signals[target] = value
    return signals


def compare_output_signals(reference, candidate):
    rows = []
    for name in sorted(set(reference) & set(candidate)):
        reference_value = reference[name]
        candidate_value = candidate[name]
        if name == "propagation_states":
            for iteration, (ref_state, candidate_state) in enumerate(
                    zip(reference_value, candidate_value), 1):
                row = tensor_metrics(ref_state, candidate_state)
                row.update({"signal": name, "iteration": iteration})
                rows.append(row)
            continue
        if name == "affinity":
            row = affinity_metrics(reference_value, candidate_value)
        elif name == "offset" and torch.is_tensor(reference_value) \
                and torch.is_tensor(candidate_value):
            row = offset_metrics(reference_value, candidate_value, channel_dim=1)
        else:
            row = tensor_metrics(reference_value, candidate_value)
        row.update({"signal": name, "iteration": 0})
        rows.append(row)
    return rows
