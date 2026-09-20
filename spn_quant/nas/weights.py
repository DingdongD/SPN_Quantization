"""Deterministic prefix-channel transfer for reduced encoder candidates."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


def transfer_prefix_state(
    target: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, list[str]]:
    target_state = target.state_dict()
    updated = {key: value.detach().clone() for key, value in target_state.items()}
    report = {
        "copied": [],
        "partial": [],
        "skipped": [],
        "missing": [],
        "unexpected": sorted(set(source_state) - set(target_state)),
    }

    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            report["missing"].append(key)
            continue
        if source_value.dtype != target_value.dtype or source_value.ndim != target_value.ndim:
            report["skipped"].append(key)
            continue
        if source_value.shape == target_value.shape:
            updated[key] = source_value.detach().clone()
            report["copied"].append(key)
            continue
        if source_value.ndim == 0:
            report["skipped"].append(key)
            continue
        slices = tuple(
            slice(0, min(source_size, target_size))
            for source_size, target_size in zip(source_value.shape, target_value.shape)
        )
        value = target_value.detach().clone()
        value[slices] = source_value.detach()[slices]
        updated[key] = value
        report["partial"].append(key)

    target.load_state_dict(updated, strict=True)
    for key in ("copied", "partial", "skipped", "missing"):
        report[key].sort()
    return report
