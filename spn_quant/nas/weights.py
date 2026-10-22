"""Deterministic prefix-channel transfer for reduced encoder candidates."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


def _copy_overlap(
    key: str,
    target_value: torch.Tensor,
    source_value: torch.Tensor,
) -> torch.Tensor:
    value = target_value.detach().clone()
    if (key.endswith(".conv1_1.weight") and target_value.ndim == 4 and
            target_value.shape[1] % 2 == 0 and source_value.shape[1] % 2 == 0):
        output_channels = min(target_value.shape[0], source_value.shape[0])
        target_half = target_value.shape[1] // 2
        source_half = source_value.shape[1] // 2
        half_channels = min(target_half, source_half)
        value[:output_channels, :half_channels] = \
            source_value[:output_channels, :half_channels]
        value[:output_channels,
              target_half:target_half + half_channels] = \
            source_value[:output_channels,
                         source_half:source_half + half_channels]
        return value
    slices = tuple(
        slice(0, min(source_size, target_size))
        for source_size, target_size in zip(source_value.shape, target_value.shape)
    )
    value[slices] = source_value.detach()[slices]
    return value


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
        updated[key] = _copy_overlap(key, target_value, source_value)
        report["partial"].append(key)

    target.load_state_dict(updated, strict=True)
    for key in ("copied", "partial", "skipped", "missing"):
        report[key].sort()
    return report
