"""Explicit QDrop block ownership for the four supported SPN models."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch.nn as nn


SUPPORTED_MODELS = (
    "completionformer",
    "cspn",
    "dyspn",
    "nlspn",
)

EXCLUDED_PROPAGATION_SITES = tuple(sorted((
    "signal::affinity",
    "signal::affinity_logits",
    "signal::attention_probability",
    "signal::center_affinity",
    "signal::confidence",
    "signal::confidence_logits",
    "signal::gate",
    "signal::normalization",
    "signal::normalization_denominator",
    "signal::offset",
    "signal::offset_mask",
    "signal::propagation_iteration",
    "signal::propagation_state",
    "signal::rgb_input",
    "signal::sparse_anchor",
    "signal::sparse_anchor_mask",
    "signal::sparse_depth_input",
)))

WEIGHT_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)

INACTIVE_WEIGHT_ROOTS = {
    "completionformer": (),
    "cspn": (
        "conv3", "up_proj_layer1", "up_proj_layer2",
        "up_proj_layer3", "up_proj_layer4"),
    "dyspn": (),
    "nlspn": (),
}


@dataclass(frozen=True)
class QDropActivationSite:
    site: str
    owner_name: str
    owner_kind: str
    role: str
    signed: bool
    symmetric: bool

    def __post_init__(self):
        if not self.site:
            raise ValueError("QDrop activation site cannot be empty")
        if not self.owner_name:
            raise ValueError("QDrop activation owner cannot be empty")
        if not self.owner_kind:
            raise ValueError("QDrop activation owner kind cannot be empty")
        if not self.role:
            raise ValueError("QDrop activation role cannot be empty")
        if self.symmetric and not self.signed:
            raise ValueError("unsigned QDrop activation sites must be affine")


@dataclass(frozen=True)
class QDropTargetPlan:
    model: str
    blocks: tuple[str, ...]
    activation_sites: tuple[QDropActivationSite, ...]
    excluded_sites: tuple[str, ...]

    def __post_init__(self):
        if self.model not in SUPPORTED_MODELS:
            raise ValueError("unsupported QDrop model: %s" % self.model)
        if not self.blocks:
            raise ValueError("QDrop target plan has no blocks")
        if tuple(sorted(set(self.blocks))) != self.blocks:
            raise ValueError("QDrop blocks must be sorted and unique")
        for index, left in enumerate(self.blocks):
            for right in self.blocks[index + 1:]:
                if left.startswith(right + ".") or right.startswith(left + "."):
                    raise ValueError(
                        "overlapping QDrop blocks: %s, %s" % (left, right))
        site_names = tuple(site.site for site in self.activation_sites)
        if len(set(site_names)) != len(site_names):
            raise ValueError("duplicate QDrop activation site")
        unknown_owners = sorted(
            site.owner_name for site in self.activation_sites
            if site.owner_name not in self.blocks)
        if unknown_owners:
            raise ValueError(
                "QDrop activation sites have unknown owners: %s" %
                unknown_owners)
        if tuple(sorted(set(self.excluded_sites))) != self.excluded_sites:
            raise ValueError("QDrop excluded sites must be sorted and unique")
        collisions = propagation_collisions(self)
        if collisions:
            raise ValueError(
                "QDrop activation sites collide with propagation ownership: %s" %
                list(collisions))


def _module_map(model):
    return dict(model.named_modules())


def _require_modules(modules, names):
    missing = sorted(name for name in names if name not in modules)
    if missing:
        raise KeyError("missing required QDrop model roots: %s" % missing)


def _supported_weight_count(module):
    return sum(
        int(isinstance(child, WEIGHT_TYPES))
        for child in module.modules())


def _is_under(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def _is_propagation_name(model_name, name):
    if model_name == "cspn":
        return _is_under(name, "post_process_layer")
    if model_name == "dyspn":
        return name.startswith("dyspn_")
    if model_name in ("nlspn", "completionformer"):
        return _is_under(name, "prop_layer")
    raise KeyError(model_name)


def _is_inactive_weight_name(model_name, name):
    return any(
        _is_under(name, root)
        for root in INACTIVE_WEIGHT_ROOTS[model_name])


def _stage_blocks(modules, pattern, class_names):
    expression = re.compile(pattern)
    return [
        name for name, module in modules.items()
        if expression.fullmatch(name) and
        type(module).__name__ in class_names and
        _supported_weight_count(module) > 0
    ]


def _weighted_roots(modules, names):
    return [
        name for name in names
        if name in modules and _supported_weight_count(modules[name]) > 0
    ]


def _cspn_blocks(model, modules):
    del model
    required = (
        "conv1_1", "layer1", "layer2", "layer3", "layer4",
        "conv2", "up_proj_layer1", "up_proj_layer2",
        "up_proj_layer3", "up_proj_layer4", "conv3",
        "gud_up_proj_layer1", "gud_up_proj_layer2",
        "gud_up_proj_layer3", "gud_up_proj_layer4",
        "gud_up_proj_layer5", "gud_up_proj_layer6",
        "post_process_layer",
    )
    _require_modules(modules, required)
    blocks = _weighted_roots(modules, (
        "conv1_1", "conv2",
        "gud_up_proj_layer1", "gud_up_proj_layer2",
        "gud_up_proj_layer3", "gud_up_proj_layer4",
        "gud_up_proj_layer5", "gud_up_proj_layer6",
    ))
    blocks.extend(_stage_blocks(
        modules, r"layer[1-4]\.[0-9]+", ("BasicBlock", "Bottleneck")))
    return tuple(sorted(blocks))


def _dyspn_blocks(model, modules):
    del model
    required = (
        "base", "base.conv1_rgb", "base.conv1_dep",
        "base.conv2", "base.conv3", "base.conv4", "base.conv5",
        "base.conv6", "base.gd_dec1_",
    )
    _require_modules(modules, required)
    propagation_roots = [
        name for name in modules
        if name.startswith("dyspn_") and "." not in name]
    if not propagation_roots:
        raise KeyError("missing required QDrop model roots: ['dyspn_*']")
    blocks = _weighted_roots(modules, (
        "base.conv1_rgb", "base.conv1_dep", "base.conv6",
    ))
    blocks.extend(_stage_blocks(
        modules, r"base\.conv[2-5]\.[0-9]+",
        ("BasicBlock", "Bottleneck", "StoDepth_BasicBlock",
         "StoDepth_SE_BasicBlock", "StoDepth_Bottleneck")))
    decoder = re.compile(r"base\.(?:dec[2-5]_?|gd_dec[01][^.]*)")
    blocks.extend(
        name for name, module in modules.items()
        if decoder.fullmatch(name) and _supported_weight_count(module) > 0)
    return tuple(sorted(set(blocks)))


def _nlspn_blocks(model, modules):
    del model
    required = (
        "conv1_rgb", "conv1_dep", "conv2", "conv3", "conv4",
        "conv5", "conv6", "dec5", "dec4", "dec3", "dec2",
        "id_dec1", "id_dec0", "gd_dec1", "gd_dec0", "prop_layer",
    )
    _require_modules(modules, required)
    blocks = _weighted_roots(modules, (
        "conv1_rgb", "conv1_dep", "conv6",
        "dec5", "dec4", "dec3", "dec2",
        "id_dec1", "id_dec0", "gd_dec1", "gd_dec0",
        "cf_dec1", "cf_dec0",
    ))
    blocks.extend(_stage_blocks(
        modules, r"conv[2-5]\.[0-9]+", ("BasicBlock", "Bottleneck")))
    return tuple(sorted(blocks))


def _completionformer_blocks(model, modules):
    del model
    required = (
        "backbone", "backbone.conv1", "backbone.former",
        "backbone.dec6", "backbone.dec5", "backbone.dec4",
        "backbone.dec3", "backbone.dec2", "backbone.dep_dec1",
        "backbone.dep_dec0", "backbone.gd_dec1", "backbone.gd_dec0",
        "prop_layer",
    )
    _require_modules(modules, required)
    blocks = _weighted_roots(modules, (
        "backbone.conv1_rgb", "backbone.conv1_dep", "backbone.conv1",
        "backbone.dec6", "backbone.dec5", "backbone.dec4",
        "backbone.dec3", "backbone.dec2", "backbone.dep_dec1",
        "backbone.dep_dec0", "backbone.gd_dec1", "backbone.gd_dec0",
        "backbone.cf_dec1", "backbone.cf_dec0",
    ))
    blocks.extend(_stage_blocks(
        modules, r"backbone\.former\.embed_layer[12]\.[0-9]+",
        ("BasicBlock", "Bottleneck")))
    blocks.extend(_stage_blocks(
        modules, r"backbone\.former\.block[1-4]\.[0-9]+", ("Block",)))
    patch = re.compile(r"backbone\.former\.patch_embed[1-4]")
    blocks.extend(
        name for name, module in modules.items()
        if patch.fullmatch(name) and _supported_weight_count(module) > 0)
    return tuple(sorted(set(blocks)))


RESOLVERS = {
    "completionformer": _completionformer_blocks,
    "cspn": _cspn_blocks,
    "dyspn": _dyspn_blocks,
    "nlspn": _nlspn_blocks,
}


INITIAL_INPUT_ROOTS = {
    "completionformer": (
        "backbone.conv1_rgb", "backbone.conv1_dep", "backbone.conv1"),
    "cspn": ("conv1_1",),
    "dyspn": ("base.conv1_rgb", "base.conv1_dep"),
    "nlspn": ("conv1_rgb", "conv1_dep"),
}


def _owner_for(name, blocks):
    owners = [block for block in blocks if _is_under(name, block)]
    if len(owners) != 1:
        raise RuntimeError(
            "QDrop activation owner is not unique for %s: %s" %
            (name, owners))
    return owners[0]


def _is_initial_input(model_name, name):
    return any(
        _is_under(name, root)
        for root in INITIAL_INPUT_ROOTS[model_name])


def _generic_activation_sites(model_name, modules, blocks):
    sites = []
    for name, module in modules.items():
        if not isinstance(module, WEIGHT_TYPES) or \
                _is_propagation_name(model_name, name) or \
                _is_inactive_weight_name(model_name, name) or \
                _is_initial_input(model_name, name):
            continue
        if model_name == "completionformer" and \
                (name.endswith(".attn.q") or name.endswith(".attn.kv") or
                 name.endswith(".concat_conv")):
            continue
        owner = _owner_for(name, blocks)
        sites.append(QDropActivationSite(
            site="activation::%s::input" % name,
            owner_name=owner,
            owner_kind="module_input",
            role="module_input",
            signed=True,
            symmetric=True,
        ))
    return sites


def _completionformer_joint_sites(modules, blocks):
    sites = []
    for name, module in modules.items():
        if not name.startswith("backbone.former"):
            continue
        if type(module).__name__ == "Attention":
            owner = _owner_for(name, blocks)
            for role in ("q", "k", "v"):
                sites.append(QDropActivationSite(
                    site="attention::%s::%s" % (name, role),
                    owner_name=owner,
                    owner_kind="attention_qkv",
                    role="attention_%s" % role,
                    signed=True,
                    symmetric=True,
                ))
        if name.endswith(".concat_conv") and isinstance(module, nn.Conv2d):
            owner = _owner_for(name, blocks)
            for role in ("transformer", "cnn"):
                sites.append(QDropActivationSite(
                    site="concat::%s::%s_input" % (name, role),
                    owner_name=owner,
                    owner_kind="concat_input",
                    role="concat_%s_input" % role,
                    signed=True,
                    symmetric=True,
                ))
    return sites


def _validate_weight_ownership(model, plan):
    unsupported = []
    for name, module in model.named_modules():
        if not isinstance(module, WEIGHT_TYPES) or \
                _is_propagation_name(plan.model, name) or \
                _is_inactive_weight_name(plan.model, name):
            continue
        owners = [block for block in plan.blocks if _is_under(name, block)]
        if len(owners) != 1:
            unsupported.append(name)
    if unsupported:
        raise TypeError(
            "unsupported QDrop weight modules outside explicit model patterns: %s" %
            sorted(unsupported))


def propagation_collisions(plan):
    excluded = set(plan.excluded_sites)
    return tuple(sorted(
        site.site for site in plan.activation_sites
        if site.site in excluded))


def all_eligible_supported_weights_are_owned(model, plan):
    for name, module in model.named_modules():
        if not isinstance(module, WEIGHT_TYPES) or \
                _is_propagation_name(plan.model, name) or \
                _is_inactive_weight_name(plan.model, name):
            continue
        owners = [block for block in plan.blocks if _is_under(name, block)]
        if len(owners) != 1:
            return False
    return True


def resolve_qdrop_targets(model_name, model):
    if model_name not in RESOLVERS:
        raise KeyError(model_name)
    if not isinstance(model, nn.Module):
        raise TypeError("QDrop target resolution requires nn.Module")
    modules = _module_map(model)
    blocks = RESOLVERS[model_name](model, modules)
    ownership_plan = QDropTargetPlan(
        model=model_name,
        blocks=blocks,
        activation_sites=(),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    _validate_weight_ownership(model, ownership_plan)
    sites = _generic_activation_sites(model_name, modules, blocks)
    if model_name == "completionformer":
        sites.extend(_completionformer_joint_sites(modules, blocks))
    plan = QDropTargetPlan(
        model=model_name,
        blocks=blocks,
        activation_sites=tuple(sorted(sites, key=lambda site: site.site)),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    return plan
