import pytest
import torch.nn as nn

from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
    build_model_quantization_contract,
)


class BasicBlock(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)


class WeightedBlock(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)


class Attention(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, 2 * channels)
        self.proj = nn.Linear(channels, channels)


class Block(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
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
        super().__init__()
        self.conv_offset_aff = nn.Conv2d(4, 27, 3, padding=1)


def make_dyspn_model():
    model = nn.Module()
    model.base = nn.Module()
    model.base.conv1_rgb = WeightedBlock()
    model.base.conv1_dep = WeightedBlock()
    for index in range(2, 6):
        setattr(model.base, "conv%d" % index, nn.Sequential(BasicBlock()))
        setattr(model.base, "dec%d" % index, WeightedBlock())
    model.base.conv6 = WeightedBlock()
    model.base.gd_dec1_ = WeightedBlock()
    model.base.gd_dec0_dyspn_6_5 = WeightedBlock()
    model.dyspn_6_5 = PropagationBlock()
    return model


def make_nlspn_model():
    model = nn.Module()
    model.conv1_rgb = WeightedBlock()
    model.conv1_dep = WeightedBlock()
    for index in range(2, 6):
        setattr(model, "conv%d" % index, nn.Sequential(BasicBlock()))
        setattr(model, "dec%d" % index, WeightedBlock())
    model.conv6 = WeightedBlock()
    for name in ("id_dec1", "id_dec0", "gd_dec1", "gd_dec0",
                 "cf_dec1", "cf_dec0"):
        setattr(model, name, WeightedBlock())
    model.prop_layer = PropagationBlock()
    return model


def make_completionformer_model():
    model = nn.Module()
    model.backbone = nn.Module()
    for name in ("conv1_rgb", "conv1_dep", "conv1"):
        setattr(model.backbone, name, WeightedBlock())
    model.backbone.former = nn.Module()
    model.backbone.former.embed_layer1 = nn.Sequential(BasicBlock())
    model.backbone.former.embed_layer2 = nn.Sequential(BasicBlock())
    for index in range(1, 5):
        setattr(model.backbone.former, "patch_embed%d" % index, WeightedBlock())
        setattr(model.backbone.former, "block%d" % index,
                nn.ModuleList((Block(),)))
    for index in range(2, 7):
        setattr(model.backbone, "dec%d" % index, WeightedBlock())
    for name in ("dep_dec1", "dep_dec0", "gd_dec1", "gd_dec0",
                 "cf_dec1", "cf_dec0"):
        setattr(model.backbone, name, WeightedBlock())
    model.prop_layer = PropagationBlock()
    return model


def test_dyspn_contract_protects_dcn_and_propagation_signals():
    contract = build_model_quantization_contract("dyspn", make_dyspn_model())

    assert "offset" in contract.protected_roles
    assert "affinity" in contract.protected_roles
    assert "guidance_logits" in contract.protected_roles
    assert all("conv_offset_aff" not in name for name in contract.weight_modules)
    assert all("gd_dec0_dyspn" not in name for name in contract.weight_modules)
    assert contract.prefix_groups
    assert contract.tail_groups


def test_nlspn_contract_excludes_propagation_projection():
    contract = build_model_quantization_contract("nlspn", make_nlspn_model())
    module_roles = dict(contract.module_roles)

    assert "confidence" in contract.protected_roles
    assert "guidance_logits" in contract.protected_roles
    assert module_roles["cf_dec0.conv"] == "confidence"
    assert module_roles["gd_dec0.conv"] == "guidance_logits"
    assert "cf_dec0.conv" in contract.protected_modules
    assert "gd_dec0.conv" in contract.protected_modules
    assert all(not name.startswith("prop_layer")
               for name in contract.weight_modules)
    assert all(not name.startswith(("cf_dec", "gd_dec0"))
               for name in contract.weight_modules)
    assert contract.attention_edges == ()
    assert contract.concat_edges == ()


def test_completionformer_contract_has_attention_and_concat_edges():
    contract = build_model_quantization_contract(
        "completionformer", make_completionformer_model())

    assert contract.attention_edges
    assert contract.concat_edges
    assert "guidance_logits" in contract.protected_roles
    assert any("backbone.former.block4.0" in group
               for group in contract.prefix_groups)
    assert all("prop_layer.conv_offset_aff" not in name
               for name in contract.weight_modules)
    assert all(not name.startswith(("backbone.cf_dec", "backbone.gd_dec0"))
               for name in contract.weight_modules)


def test_contract_rejects_empty_weight_block():
    with pytest.raises(ValueError, match="empty weight block"):
        QuantizationModelContract(
            model_name="dyspn",
            blocks=(QuantizationBlock("encoder", (), ()),),
            prefix_groups=(),
            tail_groups=(),
            protected_roles=("offset",),
            attention_edges=(),
            concat_edges=(),
            protected_modules=(),
            module_roles=(),
        )


def test_contract_rejects_duplicate_weight_module():
    with pytest.raises(ValueError, match="multiple weight blocks"):
        QuantizationModelContract(
            model_name="dyspn",
            blocks=(
                QuantizationBlock("encoder", ("base.conv1",), ()),
                QuantizationBlock("decoder", ("base.conv1",), ()),
            ),
            prefix_groups=(),
            tail_groups=(),
            protected_roles=("offset",),
            attention_edges=(),
            concat_edges=(),
            protected_modules=(),
            module_roles=(),
        )


def test_contract_rejects_duplicate_activation_owner():
    with pytest.raises(ValueError, match="duplicate activation owner"):
        QuantizationModelContract(
            model_name="dyspn",
            blocks=(
                QuantizationBlock(
                    "encoder", ("base.conv1",),
                    (("activation::base.conv1", "module_input"),)),
                QuantizationBlock(
                    "decoder", ("base.dec2",),
                    (("activation::base.conv1", "module_input"),)),
            ),
            prefix_groups=(),
            tail_groups=(),
            protected_roles=("offset",),
            attention_edges=(),
            concat_edges=(),
            protected_modules=(),
            module_roles=(),
        )


def test_contract_rejects_empty_protected_roles():
    with pytest.raises(ValueError, match="protected roles"):
        QuantizationModelContract(
            model_name="dyspn",
            blocks=(QuantizationBlock("encoder", ("base.conv1",), ()),),
            prefix_groups=(),
            tail_groups=(),
            protected_roles=(),
            attention_edges=(),
            concat_edges=(),
            protected_modules=(),
            module_roles=(),
        )


def test_contract_rejects_duplicate_protected_roles():
    with pytest.raises(ValueError, match="duplicate protected role"):
        QuantizationModelContract(
            model_name="dyspn",
            blocks=(QuantizationBlock("encoder", ("base.conv1",), ()),),
            prefix_groups=(),
            tail_groups=(),
            protected_roles=("offset", "offset"),
            attention_edges=(),
            concat_edges=(),
            protected_modules=(),
            module_roles=(),
        )


def test_contract_rejects_protected_semantic_module_in_generic_block():
    with pytest.raises(ValueError, match="protected module assigned"):
        QuantizationModelContract(
            model_name="nlspn",
            blocks=(QuantizationBlock("confidence", ("cf_dec0.conv",), ()),),
            prefix_groups=(),
            tail_groups=(),
            protected_roles=("confidence",),
            attention_edges=(),
            concat_edges=(),
            protected_modules=("cf_dec0.conv",),
            module_roles=(("cf_dec0.conv", "confidence"),),
        )


def test_contract_rejects_protected_role_in_generic_block():
    with pytest.raises(ValueError, match="protected semantic role"):
        QuantizationModelContract(
            model_name="dyspn",
            blocks=(QuantizationBlock(
                "encoder", ("base.conv1",),
                (("activation::base.conv1", "offset"),)),),
            prefix_groups=(),
            tail_groups=(),
            protected_roles=("offset",),
            attention_edges=(),
            concat_edges=(),
            protected_modules=(),
            module_roles=(),
        )
