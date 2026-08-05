#!/usr/bin/env python3
"""Quantization primitives for activation-outlier mitigation experiments."""

from __future__ import division

import csv
from pathlib import Path

import numpy as np
import torch


def load_percentile_overrides(root, percentile):
    path = Path(root) / "activation_percentiles.csv"
    overrides = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["module"], row["kind"])
            value = float(row[percentile])
            overrides[key] = max(value, overrides.get(key, 0.0))
    return overrides


def load_input_channel_maxima(root):
    root = Path(root)
    arrays = np.load(str(root / "channel_absmax.npz"), allow_pickle=False)
    maxima = {}
    with (root / "channel_absmax_index.csv").open(
            "r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] != "input":
                continue
            value = torch.from_numpy(arrays[row["key"]].copy()).float()
            previous = maxima.get(row["module"])
            if previous is not None:
                if previous.shape != value.shape:
                    raise ValueError("input channels changed for %s" % row["module"])
                value = torch.maximum(previous, value)
            maxima[row["module"]] = value
    arrays.close()
    return maxima


def _weight_input_absmax(weight):
    if weight.ndim < 2:
        raise ValueError("weight must have input and output dimensions")
    return weight.detach().abs().movedim(1, 0).reshape(
        weight.shape[1], -1).amax(dim=1)


def smoothquant_scale(weight, activation_absmax, alpha, epsilon=1e-8):
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("SmoothQuant alpha must be in [0, 1]")
    activation_absmax = torch.as_tensor(
        activation_absmax, device=weight.device, dtype=weight.dtype).reshape(-1)
    if activation_absmax.numel() != weight.shape[1]:
        raise ValueError("activation maxima do not match weight input channels")
    activation_absmax = activation_absmax.clamp_min(float(epsilon))
    weight_absmax = _weight_input_absmax(weight).clamp_min(float(epsilon))
    return activation_absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)


def apply_input_scale_to_weight(weight, scale):
    scale = torch.as_tensor(scale, device=weight.device, dtype=weight.dtype)
    if scale.numel() != weight.shape[1]:
        raise ValueError("scale does not match weight input channels")
    shape = [1, weight.shape[1]] + [1] * (weight.ndim - 2)
    return weight * scale.reshape(shape)


def clipped_symmetric_weight_qdq(weight, bits, clip_ratio=1.0):
    if bits < 2:
        raise ValueError("weight bits must be at least 2")
    clip_ratio = float(clip_ratio)
    if clip_ratio <= 0.0 or clip_ratio > 1.0:
        raise ValueError("clip ratio must be in (0, 1]")
    qmax = 2 ** (int(bits) - 1) - 1
    flat = weight.reshape(weight.shape[0], -1)
    threshold = flat.abs().max(dim=1)[0] * clip_ratio
    threshold = torch.where(threshold > 0, threshold, torch.ones_like(threshold))
    shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
    threshold = threshold.reshape(shape)
    scale = threshold / float(qmax)
    clipped = weight.clamp(-threshold, threshold)
    codes = torch.round(clipped / scale).clamp(-qmax, qmax)
    return codes * scale, scale

