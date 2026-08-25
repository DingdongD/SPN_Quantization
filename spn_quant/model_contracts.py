"""Strict quantization ownership contracts for selected SPN models."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Tuple

import torch.nn as nn

from spn_quant.adapters.completionformer import CompletionFormerSemanticAdapter
from spn_quant.adapters.dyspn import DySPNSemanticAdapter
from spn_quant.adapters.nlspn import NLSPNSemanticAdapter
from spn_quant.qdrop_targets import (
    QDropTargetPlan,
    resolve_qdrop_targets,
)


@dataclass(frozen=True)
class QuantizationBlock:
    name: str
    weight_modules: Tuple[str, ...]
    activation_owners: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class SearchTopology:
    prefix_groups: Tuple[Tuple[str, ...], ...]
    tail_groups: Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class QuantizationModelContract:
    model_name: str
    blocks: Tuple[QuantizationBlock, ...]
    prefix_groups: Tuple[Tuple[str, ...], ...]
    tail_groups: Tuple[Tuple[str, ...], ...]
    protected_roles: Tuple[str, ...]
    attention_edges: Tuple[str, ...]
    concat_edges: Tuple[str, ...]
    protected_modules: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model quantization contract requires a model name")
        if not self.blocks:
            raise ValueError("model quantization contract requires blocks")
        if len(set(self.block_names)) != len(self.block_names):
            raise ValueError("duplicate quantization block name")
        for block in self.blocks:
            if not block.name:
                raise ValueError("quantization block requires a name")
            if not block.weight_modules:
                raise ValueError("empty weight block: %s" % block.name)
        if len(set(self.weight_modules)) != len(self.weight_modules):
            raise ValueError("weight module belongs to multiple weight blocks")
        owners = tuple(owner for block in self.blocks
                       for owner, role in block.activation_owners)
        if len(set(owners)) != len(owners):
            raise ValueError("duplicate activation owner")
        protected = set(self.protected_roles)
        generic_roles = tuple(role for block in self.blocks
                              for owner, role in block.activation_owners)
        protected_generic = sorted(protected.intersection(generic_roles))
        if protected_generic:
            raise ValueError(
                "protected semantic role assigned to generic block: %s" %
                protected_generic)
        if set(self.weight_modules).intersection(self.protected_modules):
            raise ValueError("protected module assigned to generic block")
        self._validate_groups(self.prefix_groups, "prefix")
        self._validate_groups(self.tail_groups, "tail")
        self._validate_edges(self.attention_edges, "attention")
        self._validate_edges(self.concat_edges, "concat")

    @property
    def block_names(self) -> Tuple[str, ...]:
        return tuple(block.name for block in self.blocks)

    @property
    def weight_modules(self) -> Tuple[str, ...]:
        return tuple(name for block in self.blocks for name in block.weight_modules)

    @property
    def search_topology(self) -> SearchTopology:
        return SearchTopology(self.prefix_groups, self.tail_groups)

    def _validate_groups(self, groups: Tuple[Tuple[str, ...], ...],
                         name: str) -> None:
        known = set(self.block_names)
        for group in groups:
            if not group:
                raise ValueError("empty %s search group" % name)
            unknown = sorted(set(group) - known)
            if unknown:
                raise ValueError("unknown %s search blocks: %s" %
                                 (name, unknown))

    def _validate_edges(self, edges: Tuple[str, ...], name: str) -> None:
        if len(set(edges)) != len(edges):
            raise ValueError("duplicate %s edge" % name)
        owners = set(owner for block in self.blocks
                     for owner, role in block.activation_owners)
        unknown = sorted(set(edges) - owners)
        if unknown:
            raise ValueError("unknown %s edge owners: %s" % (name, unknown))


ADAPTERS = {
    "completionformer": CompletionFormerSemanticAdapter,
    "dyspn": DySPNSemanticAdapter,
    "nlspn": NLSPNSemanticAdapter,
}

def _is_under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _matches(block_name: str, patterns: Tuple[str, ...]) -> bool:
    return any(re.search(pattern, block_name) is not None for pattern in patterns)


def _resolve_groups(block_names: Tuple[str, ...],
                    pattern_groups: Tuple[Tuple[str, ...], ...],
                    cumulative: bool) -> Tuple[Tuple[str, ...], ...]:
    groups = []
    selected = ()
    for patterns in pattern_groups:
        current = tuple(name for name in block_names if _matches(name, patterns))
        if not current:
            raise ValueError("required contract block is empty: %s" % patterns)
        selected = selected + current if cumulative else current
        groups.append(selected)
    return tuple(groups)


def _protected_modules(model_name: str, model: nn.Module) -> Tuple[str, ...]:
    if model_name == "dyspn":
        return tuple(name for name, _ in model.named_modules()
                     if name.startswith("dyspn_"))
    return tuple(name for name, _ in model.named_modules()
                 if _is_under(name, "prop_layer"))


def _activation_owners(plan: QDropTargetPlan, block_name: str) \
        -> Tuple[Tuple[str, str], ...]:
    return tuple((site.site, site.role) for site in plan.activation_sites
                 if site.owner_name == block_name)


def _build_blocks(plan: QDropTargetPlan,
                  modules: Dict[str, nn.Module]) -> Tuple[QuantizationBlock, ...]:
    blocks = []
    for block_name in plan.blocks:
        weights = tuple(name for name in modules
                        if _is_under(name, block_name))
        if not weights:
            raise ValueError("required contract block is empty: %s" % block_name)
        blocks.append(QuantizationBlock(
            block_name, weights, _activation_owners(plan, block_name)))
    return tuple(blocks)


def _edge_names(plan: QDropTargetPlan, kind: str) -> Tuple[str, ...]:
    return tuple(site.site for site in plan.activation_sites
                 if site.owner_kind == kind)


def build_model_quantization_contract(model_name: str,
                                      model: nn.Module) -> QuantizationModelContract:
    """Resolve one complete generic-quantization contract without fallbacks."""
    adapter = ADAPTERS[model_name]
    manifest = adapter.module_manifest(model)
    modules = {row["name"]: row["module"] for row in manifest}
    if not modules:
        raise ValueError("model quantization contract has no supported modules")
    plan = resolve_qdrop_targets(model_name, model)
    blocks = _build_blocks(plan, modules)
    block_names = tuple(block.name for block in blocks)
    prefix_groups = _resolve_groups(
        block_names, adapter.CONTRACT_PREFIX_GROUP_PATTERNS, True)
    tail_groups = _resolve_groups(
        block_names, adapter.CONTRACT_TAIL_GROUP_PATTERNS, False)
    return QuantizationModelContract(
        model_name=model_name,
        blocks=blocks,
        prefix_groups=prefix_groups,
        tail_groups=tail_groups,
        protected_roles=adapter.CONTRACT_PROTECTED_ROLES,
        attention_edges=_edge_names(plan, "attention_qkv"),
        concat_edges=_edge_names(plan, "concat_input"),
        protected_modules=_protected_modules(model_name, model),
    )
