import ast
import json
from pathlib import Path

import torch

from spn_quant.model_contracts import build_model_quantization_contract


BASELINE_ROOT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines")
DYSPN_ROOT = Path("/workspace/external_depth_completion_models/DySPN")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
COMPLETIONFORMER_ROOT = Path("/workspace/CompletionFormer")


def checkpoint(name):
    return torch.load(
        BASELINE_ROOT / name / "best.pt", map_location="cpu",
        weights_only=False)


def checkpoint_module_names(payload):
    return tuple(name.rsplit(".", 1)[0] for name in payload["net"])


def assert_module_roots(module_names, roots):
    for root in roots:
        assert any(name == root or name.startswith(root + ".")
                   for name in module_names)


def class_names(path):
    tree = ast.parse(path.read_text())
    return tuple(node.name for node in tree.body if isinstance(node, ast.ClassDef))


def test_official_dyspn_builder_strictly_loads_selected_checkpoint(monkeypatch):
    monkeypatch.syspath_prepend(str(DYSPN_ROOT))
    from DySPN.base import Model

    payload = checkpoint("dyspn_iter6")
    args = json.loads((BASELINE_ROOT / "dyspn_iter6" / "args.json").read_text())
    model = Model(
        iteration=args["iteration"],
        num_neighbor=args["dyspn_neighbors"],
        mode="dyspn",
        res=args["dyspn_resnet"],
        bm=args["dyspn_basemodel"],
        stodepth=True,
    )

    assert type(model).__name__ == "Model"
    model.load_state_dict(payload["net"], strict=True)
    contract = build_model_quantization_contract("dyspn", model)
    assert_module_roots(tuple(name for name, _ in model.named_modules()), (
        "base.conv1_rgb", "base.conv1_dep", "base.conv2", "base.conv3",
        "base.conv4", "base.conv5", "base.conv6", "base.dec5",
        "base.dec4", "base.dec3", "base.dec2", "base.gd_dec1_",
        "base.gd_dec0_dyspn_6_5", "dyspn_6_5.conv_offset_aff",
    ))
    assert "base.gd_dec0_dyspn_6_5.0" not in contract.weight_modules
    assert "dyspn_6_5.conv_offset_aff" not in contract.weight_modules


def test_official_nlspn_checkpoint_matches_official_class_and_module_tree():
    payload = checkpoint("nlspn_iter18")
    modules = checkpoint_module_names(payload)
    source = NLSPN_ROOT / "src" / "model" / "nlspnmodel.py"

    assert "NLSPNModel" in class_names(source)
    assert payload["meta"]["architecture"] == "NLSPN resnet34"
    assert payload["args"]["nlspn_network"] == "resnet34"
    assert_module_roots(modules, (
        "conv1_rgb", "conv1_dep", "conv2", "conv3", "conv4", "conv5",
        "conv6", "dec5", "dec4", "dec3", "dec2", "id_dec1", "id_dec0",
        "gd_dec1", "gd_dec0", "cf_dec1", "cf_dec0", "prop_layer",
    ))


def test_official_completionformer_checkpoint_matches_class_and_module_tree():
    payload = checkpoint("completionformer_iter18")
    modules = checkpoint_module_names(payload)
    source = COMPLETIONFORMER_ROOT / "src" / "model" / "completionformer.py"

    assert "CompletionFormer" in class_names(source)
    assert payload["meta"]["architecture"] == "CompletionFormer"
    assert payload["args"]["completionformer_model"] == "CompletionFormer"
    assert_module_roots(modules, (
        "backbone.conv1_rgb", "backbone.conv1_dep", "backbone.conv1",
        "backbone.former.embed_layer1", "backbone.former.embed_layer2",
        "backbone.former.patch_embed1", "backbone.former.patch_embed2",
        "backbone.former.patch_embed3", "backbone.former.patch_embed4",
        "backbone.former.block1", "backbone.former.block2",
        "backbone.former.block3", "backbone.former.block4", "backbone.dec6",
        "backbone.dec5", "backbone.dec4", "backbone.dec3", "backbone.dec2",
        "backbone.dep_dec1", "backbone.dep_dec0", "backbone.gd_dec1",
        "backbone.gd_dec0", "backbone.cf_dec1", "backbone.cf_dec0",
        "prop_layer",
    ))
