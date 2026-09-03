"""Strict quantization ownership contracts for selected SPN models."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Tuple

import torch.nn as nn

from spn_quant.adapters.completionformer import CompletionFormerSemanticAdapter
from spn_quant.adapters.cspn import CSPNSemanticAdapter
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
class PrecisionSearchUnit:
    name: str
    members: Tuple[str, ...]
    activation_owners: Tuple[Tuple[str, str], ...]
    kind: str
    minimum_weight_bits: int
    minimum_activation_bits: int
    allow_fp16: bool
    scale_policy: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("precision search unit requires a name")
        if not self.members:
            raise ValueError("precision search unit requires weight members")
        if self.minimum_weight_bits not in (4, 6, 8):
            raise ValueError("minimum weight bits must be 4, 6, or 8")
        if self.minimum_activation_bits not in (4, 6, 8):
            raise ValueError("minimum activation bits must be 4, 6, or 8")
        if self.scale_policy not in (
                "branch_independent", "dynamic_group8", "static_tensor"):
            raise ValueError("unsupported precision-unit scale policy")


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
    module_roles: Tuple[Tuple[str, str], ...]
    search_units: Tuple[PrecisionSearchUnit, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model quantization contract requires a model name")
        if not self.blocks:
            raise ValueError("model quantization contract requires blocks")
        if not self.protected_roles:
            raise ValueError("model quantization contract requires protected roles")
        if len(set(self.protected_roles)) != len(self.protected_roles):
            raise ValueError("duplicate protected role")
        role_names = tuple(name for name, role in self.module_roles)
        if len(set(role_names)) != len(role_names):
            raise ValueError("duplicate semantic module role")
        if len(set(self.protected_modules)) != len(self.protected_modules):
            raise ValueError("duplicate protected module")
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
        protected_semantic_modules = set(
            name for name, role in self.module_roles if role in protected)
        missing_protected_modules = sorted(
            protected_semantic_modules - set(self.protected_modules))
        if missing_protected_modules:
            raise ValueError(
                "protected semantic modules are not protected: %s" %
                missing_protected_modules)
        generic_protected_modules = sorted(
            protected_semantic_modules.intersection(self.weight_modules))
        if generic_protected_modules:
            raise ValueError(
                "protected semantic module assigned to generic block: %s" %
                generic_protected_modules)
        self._validate_groups(self.prefix_groups, "prefix")
        self._validate_groups(self.tail_groups, "tail")
        self._validate_edges(self.attention_edges, "attention")
        self._validate_edges(self.concat_edges, "concat")
        if self.search_units:
            self._validate_search_units()

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

    def _validate_search_units(self) -> None:
        names = tuple(unit.name for unit in self.search_units)
        if len(set(names)) != len(names):
            raise ValueError("duplicate precision search unit")
        members = tuple(member for unit in self.search_units
                        for member in unit.members)
        if len(set(members)) != len(members):
            raise ValueError("weight module belongs to multiple search units")
        if set(members) != set(self.weight_modules):
            raise ValueError("search-unit weight coverage differs from contract")
        owners = tuple(owner for unit in self.search_units
                       for owner in unit.activation_owners)
        if len(set(owners)) != len(owners):
            raise ValueError("activation owner belongs to multiple search units")
        expected_owners = tuple(owner for block in self.blocks
                                for owner in block.activation_owners)
        if set(owners) != set(expected_owners):
            raise ValueError(
                "search-unit activation coverage differs from contract")


ADAPTERS = {
    "completionformer": CompletionFormerSemanticAdapter,
    "cspn": CSPNSemanticAdapter,
    "dyspn": DySPNSemanticAdapter,
    "nlspn": NLSPNSemanticAdapter,
}


def propagation_owned_modules(contract: QuantizationModelContract) \
        -> Tuple[str, ...]:
    if not isinstance(contract, QuantizationModelContract):
        raise TypeError("propagation ownership requires a model contract")
    protected_roles = set(contract.protected_roles)
    semantic_modules = tuple(
        name for name, role in contract.module_roles
        if role in protected_roles)
    missing = sorted(set(semantic_modules) - set(contract.protected_modules))
    if missing:
        raise ValueError(
            "propagation semantic modules are not protected: %s" % missing)
    return tuple(contract.protected_modules)


def validate_propagation_ownership(contract: QuantizationModelContract,
                                   model: nn.Module) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("propagation ownership requires nn.Module")
    owned = propagation_owned_modules(contract)
    modules = dict(model.named_modules())
    missing = sorted(set(owned) - set(modules))
    if missing:
        raise KeyError("missing propagation modules: %s" % missing)
    overlap = sorted(set(owned).intersection(contract.weight_modules))
    if overlap:
        raise ValueError(
            "propagation modules assigned to ordinary quantization: %s" %
            overlap)

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


def _protected_modules(model_name: str, model: nn.Module,
                       module_roles: Tuple[Tuple[str, str], ...],
                       protected_roles: Tuple[str, ...]) -> Tuple[str, ...]:
    semantic = tuple(name for name, role in module_roles
                     if role in protected_roles)
    if model_name == "dyspn":
        propagation = tuple(name for name, _ in model.named_modules()
                            if name.startswith("dyspn_"))
    else:
        propagation = tuple(name for name, _ in model.named_modules()
                            if _is_under(name, "prop_layer"))
    names = semantic + propagation
    return tuple(name for index, name in enumerate(names)
                 if name not in names[:index])


def _activation_owners(plan: QDropTargetPlan, block_name: str) \
        -> Tuple[Tuple[str, str], ...]:
    return tuple((site.site, site.role) for site in plan.activation_sites
                 if site.owner_name == block_name)


def _generic_plan(plan: QDropTargetPlan, modules: Dict[str, nn.Module],
                  protected_modules: Tuple[str, ...]) -> QDropTargetPlan:
    generic_blocks = []
    for block_name in plan.blocks:
        owned_modules = tuple(name for name in modules
                              if _is_under(name, block_name))
        if not owned_modules:
            raise ValueError("required contract block is empty: %s" % block_name)
        if any(name not in protected_modules for name in owned_modules):
            generic_blocks.append(block_name)
    generic_block_names = tuple(generic_blocks)
    activation_sites = tuple(
        site for site in plan.activation_sites
        if site.owner_name in generic_block_names)
    return QDropTargetPlan(
        model=plan.model,
        blocks=generic_block_names,
        activation_sites=activation_sites,
        excluded_sites=plan.excluded_sites,
    )


def _build_blocks(plan: QDropTargetPlan, modules: Dict[str, nn.Module],
                  protected_modules: Tuple[str, ...]) \
        -> Tuple[QuantizationBlock, ...]:
    blocks = []
    for block_name in plan.blocks:
        weights = tuple(name for name in modules
                        if _is_under(name, block_name) and
                        name not in protected_modules)
        if not weights:
            raise ValueError("required contract block is empty: %s" % block_name)
        blocks.append(QuantizationBlock(
            block_name, weights, _activation_owners(plan, block_name)))
    return tuple(blocks)


def _edge_names(plan: QDropTargetPlan, kind: str) -> Tuple[str, ...]:
    return tuple(site.site for site in plan.activation_sites
                 if site.owner_kind == kind)


def _activation_member(site: str, role: str,
                       weight_modules: Tuple[str, ...]) -> str:
    if site.startswith("activation::") and site.endswith("::input"):
        member = site[len("activation::"):-len("::input")]
        if member not in weight_modules:
            raise ValueError("activation owner has no weight member: %s" % site)
        return member
    if site.startswith("attention::"):
        attention = site.split("::")[1]
        suffix = ".q" if role == "attention_q" else ".kv"
        member = attention + suffix
        if member not in weight_modules:
            raise ValueError("attention owner has no weight member: %s" % site)
        return member
    if site.startswith("concat::"):
        member = site.split("::")[1]
        if member not in weight_modules:
            raise ValueError("concat owner has no weight member: %s" % site)
        return member
    raise ValueError("unsupported activation owner: %s" % site)


def _resolve_search_units(
        adapter, blocks: Tuple[QuantizationBlock, ...],
        weight_modules: Tuple[str, ...]) -> Tuple[PrecisionSearchUnit, ...]:
    rules = adapter.CONTRACT_SEARCH_UNIT_RULES
    if not rules:
        raise ValueError("model adapter has no precision search-unit rules")
    member_units = {}
    resolved = []
    for rule in rules:
        if len(rule) != 7:
            raise ValueError("precision search-unit rule must have seven fields")
        name, patterns, kind, minimum_weight_bits, \
            minimum_activation_bits, allow_fp16, scale_policy = rule
        members = tuple(
            member for member in weight_modules
            if member not in member_units and _matches(member, patterns))
        if not members:
            raise ValueError("required precision search unit is empty: %s" % name)
        for member in members:
            member_units[member] = name
        resolved.append((
            name, members, kind, minimum_weight_bits,
            minimum_activation_bits, allow_fp16, scale_policy,
        ))
    uncovered = tuple(member for member in weight_modules
                      if member not in member_units)
    if uncovered:
        raise ValueError("unowned precision search weights: %s" %
                         list(uncovered))
    activation_by_unit = dict((row[0], []) for row in resolved)
    for block in blocks:
        for site, role in block.activation_owners:
            member = _activation_member(site, role, weight_modules)
            activation_by_unit[member_units[member]].append((site, role))
    return tuple(PrecisionSearchUnit(
        name=row[0],
        members=row[1],
        activation_owners=tuple(activation_by_unit[row[0]]),
        kind=row[2],
        minimum_weight_bits=int(row[3]),
        minimum_activation_bits=int(row[4]),
        allow_fp16=bool(row[5]),
        scale_policy=row[6],
    ) for row in resolved)


def build_model_quantization_contract(model_name: str,
                                      model: nn.Module) -> QuantizationModelContract:
    """Resolve one complete generic-quantization contract without fallbacks."""
    adapter = ADAPTERS[model_name]
    manifest = adapter.module_manifest(model)
    modules = {row["name"]: row["module"] for row in manifest}
    module_roles = tuple((row["name"], row["role"]) for row in manifest)
    if not modules:
        raise ValueError("model quantization contract has no supported modules")
    plan = resolve_qdrop_targets(model_name, model)
    protected_modules = _protected_modules(
        model_name, model, module_roles, adapter.CONTRACT_PROTECTED_ROLES)
    plan = _generic_plan(plan, modules, protected_modules)
    blocks = _build_blocks(plan, modules, protected_modules)
    block_names = tuple(block.name for block in blocks)
    weight_modules = tuple(name for block in blocks
                           for name in block.weight_modules)
    search_units = _resolve_search_units(adapter, blocks, weight_modules)
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
        protected_modules=protected_modules,
        module_roles=module_roles,
        search_units=search_units,
    )
