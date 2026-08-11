import pytest
import torch.nn as nn

from spn_quant.qdrop_targets import (
    QDropActivationSite,
    QDropTargetPlan,
    all_eligible_supported_weights_are_owned,
    propagation_collisions,
    resolve_qdrop_targets,
)


class BasicBlock(nn.Module):
    def __init__(self, channels=4):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)


class WeightedBlock(nn.Module):
    def __init__(self, channels=4):
        super(WeightedBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()


class Attention(nn.Module):
    def __init__(self, channels=4):
        super(Attention, self).__init__()
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, 2 * channels)
        self.proj = nn.Linear(channels, channels)


class Block(nn.Module):
    def __init__(self, channels=4):
        super(Block, self).__init__()
        self.attn = Attention(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )
        self.resblock = BasicBlock(channels)
        self.concat_conv = nn.Conv2d(2 * channels, channels, 3, padding=1)


class PropagationBlock(nn.Module):
    def __init__(self):
        super(PropagationBlock, self).__init__()
        self.conv_offset_aff = nn.Conv2d(4, 27, 3, padding=1)


def make_toy_cspn():
    model = nn.Module()
    model.conv1_1 = nn.Conv2d(4, 4, 3, padding=1)
    for index in range(1, 5):
        setattr(model, "layer%d" % index, nn.Sequential(BasicBlock()))
    model.conv2 = nn.Conv2d(4, 4, 3, padding=1)
    for index in range(1, 5):
        setattr(model, "up_proj_layer%d" % index, WeightedBlock())
    model.conv3 = nn.Conv2d(4, 1, 3, padding=1)
    for index in range(1, 7):
        setattr(model, "gud_up_proj_layer%d" % index, WeightedBlock())
    model.post_process_layer = PropagationBlock()
    return model


def make_toy_dyspn():
    model = nn.Module()
    model.base = nn.Module()
    model.base.conv1_rgb = WeightedBlock()
    model.base.conv1_dep = WeightedBlock()
    for index in range(2, 6):
        setattr(model.base, "conv%d" % index, nn.Sequential(BasicBlock()))
    model.base.conv6 = WeightedBlock()
    for index in range(2, 6):
        setattr(model.base, "dec%d" % index, WeightedBlock())
    model.base.gd_dec1_ = WeightedBlock()
    model.base.gd_dec0_dyspn_6_5 = WeightedBlock()
    model.dyspn_6_5 = PropagationBlock()
    return model


def make_toy_nlspn():
    model = nn.Module()
    model.conv1_rgb = WeightedBlock()
    model.conv1_dep = WeightedBlock()
    for index in range(2, 6):
        setattr(model, "conv%d" % index, nn.Sequential(BasicBlock()))
    model.conv6 = WeightedBlock()
    for index in range(2, 6):
        setattr(model, "dec%d" % index, WeightedBlock())
    for name in ("id_dec1", "id_dec0", "gd_dec1", "gd_dec0",
                 "cf_dec1", "cf_dec0"):
        setattr(model, name, WeightedBlock())
    model.prop_layer = PropagationBlock()
    return model


def make_toy_completionformer():
    model = nn.Module()
    model.backbone = nn.Module()
    for name in ("conv1_rgb", "conv1_dep", "conv1"):
        setattr(model.backbone, name, WeightedBlock())
    model.backbone.former = nn.Module()
    model.backbone.former.embed_layer1 = nn.Sequential(BasicBlock())
    model.backbone.former.embed_layer2 = nn.Sequential(BasicBlock())
    model.backbone.former.patch_embed1 = WeightedBlock()
    model.backbone.former.block1 = nn.ModuleList((Block(),))
    for index in range(2, 7):
        setattr(model.backbone, "dec%d" % index, WeightedBlock())
    for name in ("dep_dec1", "dep_dec0", "gd_dec1", "gd_dec0",
                 "cf_dec1", "cf_dec0"):
        setattr(model.backbone, name, WeightedBlock())
    model.prop_layer = PropagationBlock()
    return model


def test_cspn_targets_are_nonoverlapping_and_exclude_propagation():
    model = make_toy_cspn()
    plan = resolve_qdrop_targets("cspn", model)

    assert "conv1_1" in plan.blocks
    assert "layer1.0" in plan.blocks
    assert "gud_up_proj_layer6" in plan.blocks
    assert "conv3" not in plan.blocks
    assert all(not name.startswith("up_proj_layer") for name in plan.blocks)
    assert all(not name.startswith("post_process_layer")
               for name in plan.blocks)
    assert "signal::affinity" in plan.excluded_sites
    assert "signal::sparse_depth_input" in plan.excluded_sites
    assert propagation_collisions(plan) == ()
    assert all_eligible_supported_weights_are_owned(model, plan)


def test_dyspn_targets_exclude_propagation_modules():
    model = make_toy_dyspn()
    plan = resolve_qdrop_targets("dyspn", model)

    assert "base.conv2.0" in plan.blocks
    assert "base.gd_dec1_" in plan.blocks
    assert all(not name.startswith("dyspn_") for name in plan.blocks)
    assert "signal::propagation_state" in plan.excluded_sites
    assert all_eligible_supported_weights_are_owned(model, plan)


def test_nlspn_targets_exclude_confidence_and_propagation():
    model = make_toy_nlspn()
    plan = resolve_qdrop_targets("nlspn", model)

    assert "conv2.0" in plan.blocks
    assert "id_dec0" in plan.blocks
    assert "cf_dec0" in plan.blocks
    assert all(not name.startswith("prop_layer") for name in plan.blocks)
    assert "signal::confidence" in plan.excluded_sites
    assert "signal::confidence_logits" in plan.excluded_sites
    assert all_eligible_supported_weights_are_owned(model, plan)


def test_completionformer_targets_include_attention_and_concat():
    model = make_toy_completionformer()
    plan = resolve_qdrop_targets("completionformer", model)

    assert "backbone.former.block1.0" in plan.blocks
    assert any(site.owner_kind == "attention_qkv"
               for site in plan.activation_sites)
    assert any(site.owner_kind == "concat_input"
               for site in plan.activation_sites)
    assert "signal::attention_probability" in plan.excluded_sites
    assert all(not name.startswith("prop_layer") for name in plan.blocks)
    assert all_eligible_supported_weights_are_owned(model, plan)


def test_every_propagation_only_role_is_excluded():
    plan = resolve_qdrop_targets("nlspn", make_toy_nlspn())
    required = {
        "signal::rgb_input",
        "signal::sparse_depth_input",
        "signal::sparse_anchor",
        "signal::sparse_anchor_mask",
        "signal::affinity_logits",
        "signal::affinity",
        "signal::center_affinity",
        "signal::confidence_logits",
        "signal::confidence",
        "signal::gate",
        "signal::normalization",
        "signal::normalization_denominator",
        "signal::attention_probability",
        "signal::offset",
        "signal::offset_mask",
        "signal::propagation_state",
        "signal::propagation_iteration",
    }
    assert required <= set(plan.excluded_sites)


def test_duplicate_activation_ownership_is_rejected():
    site = QDropActivationSite(
        site="activation::conv::input",
        owner_name="conv",
        owner_kind="module_input",
        role="module_input",
        signed=True,
        symmetric=True,
    )
    with pytest.raises(ValueError, match="duplicate QDrop activation site"):
        QDropTargetPlan(
            model="cspn",
            blocks=("conv",),
            activation_sites=(site, site),
            excluded_sites=("signal::affinity",),
        )


def test_overlapping_blocks_are_rejected():
    with pytest.raises(ValueError, match="overlapping QDrop blocks"):
        QDropTargetPlan(
            model="cspn",
            blocks=("layer1", "layer1.0"),
            activation_sites=(),
            excluded_sites=("signal::affinity",),
        )


def test_missing_required_model_root_is_rejected():
    model = make_toy_completionformer()
    del model._modules["backbone"]

    with pytest.raises(KeyError, match="backbone"):
        resolve_qdrop_targets("completionformer", model)


def test_unsupported_weight_module_outside_explicit_patterns_is_rejected():
    model = make_toy_nlspn()
    model.rogue = nn.Conv2d(4, 4, 1)

    with pytest.raises(TypeError, match="unsupported QDrop weight module.*rogue"):
        resolve_qdrop_targets("nlspn", model)


def test_unknown_model_is_rejected_without_generic_dispatch():
    with pytest.raises(KeyError):
        resolve_qdrop_targets("unknown", nn.Module())
