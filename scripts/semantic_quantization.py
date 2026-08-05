#!/usr/bin/env python3
"""Helpers for exporting semantic quantization-site manifests."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch.nn as nn

from spn_quant import QuantSiteRegistry, QuantSpec, build_module_site_registry


def build_semantic_registry(model: nn.Module, example_args: Sequence[Any],
                            group_fn: Callable[[str, nn.Module], Optional[str]],
                            default_spec: Optional[QuantSpec] = None
                            ) -> QuantSiteRegistry:
    kwargs = {}
    if default_spec is not None:
        kwargs["default_spec"] = default_spec
    return build_module_site_registry(model, example_args, group_fn, **kwargs)


def write_site_manifest(path: str, registry: QuantSiteRegistry) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = registry.manifest()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination
