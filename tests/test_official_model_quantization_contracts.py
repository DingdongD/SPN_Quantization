import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from spn_quant.model_contracts import build_model_quantization_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines")
DYSPN_ROOT = Path("/workspace/external_depth_completion_models/DySPN")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
COMPLETIONFORMER_ROOT = Path("/workspace/CompletionFormer")
CSPN_ROOT = Path("/workspace/CSPN/cspn_pytorch")
TORCH_LIB = Path(
    "/opt/conda/envs/completionformer-py37/lib/python3.7/site-packages/torch/lib")


def _run_official_contract(model_name, source_root, checkpoint_name):
    source = source_root / "src"
    deformconv = source / "model" / "deformconv"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((
        str(REPO_ROOT), str(source), str(deformconv)))
    environment["LD_LIBRARY_PATH"] = os.pathsep.join((
        str(TORCH_LIB), environment["LD_LIBRARY_PATH"]))
    program = """
import json
from argparse import Namespace
from pathlib import Path

import torch

from spn_quant.model_contracts import build_model_quantization_contract

model_name = {model_name!r}
checkpoint_root = Path({checkpoint_root!r})
args = json.loads((checkpoint_root / "args.json").read_text())

if model_name == "nlspn":
    from model.nlspnmodel import NLSPNModel

    model = NLSPNModel(Namespace(
        network=args["nlspn_network"],
        from_scratch=args["from_scratch"],
        prop_time=args["iteration"],
        prop_kernel=3,
        conf_prop=True,
        affinity="TGASS",
        affinity_gamma=0.5,
        preserve_input=False,
        legacy=False,
        lr=args["lr"],
    ))
elif model_name == "completionformer":
    from model.completionformer import CompletionFormer

    model = CompletionFormer(Namespace(
        model=args["completionformer_model"],
        from_scratch=args["from_scratch"],
        prop_time=args["iteration"],
        prop_kernel=3,
        conf_prop=True,
        affinity="TGASS",
        affinity_gamma=0.5,
        preserve_input=False,
        legacy=False,
        max_depth=10.0,
    ))
else:
    raise KeyError(model_name)

payload = torch.load(str(checkpoint_root / "best.pt"), map_location="cpu")
model.load_state_dict(payload["net"], strict=True)
contract = build_model_quantization_contract(model_name, model)
print(json.dumps({{
    "class_name": type(model).__name__,
    "module_names": tuple(name for name, _ in model.named_modules()),
    "block_count": len(contract.blocks),
    "weight_count": len(contract.weight_modules),
    "attention_edges": len(contract.attention_edges),
    "concat_edges": len(contract.concat_edges),
    "weight_modules": contract.weight_modules,
    "search_units": tuple({{
        "name": unit.name,
        "members": unit.members,
        "kind": unit.kind,
        "minimum_activation_bits": unit.minimum_activation_bits,
        "allow_fp16": unit.allow_fp16,
    }} for unit in contract.search_units),
}}))
""".format(
        model_name=model_name,
        checkpoint_root=str(BASELINE_ROOT / checkpoint_name),
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(source),
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def _assert_module_roots(module_names, roots):
    for root in roots:
        assert any(name == root or name.startswith(root + ".")
                   for name in module_names)


def test_official_cspn_builder_strictly_loads_checkpoint_and_contract(
        monkeypatch):
    monkeypatch.syspath_prepend(str(CSPN_ROOT / "models"))
    import torch_resnet_cspn_nyu

    model = torch_resnet_cspn_nyu.resnet18(
        pretrained=False,
        cspn_config={"step": 24, "kernel": 3, "norm_type": "8sum"},
    )
    payload = torch.load(
        str(BASELINE_ROOT / "cspn_iter24" / "best.pt"), map_location="cpu")
    state = dict(payload["net"])
    state.pop("post_process_layer.sum_conv.weight", None)
    model.load_state_dict(state, strict=True)
    contract = build_model_quantization_contract("cspn", model)
    units = dict((unit.name, unit) for unit in contract.search_units)

    assert type(model).__name__ == "ResNet"
    assert len(contract.weight_modules) == 37
    assert units["initial_depth"].members == ("gud_up_proj_layer5.conv1",)
    assert units["initial_depth"].allow_fp16 is True
    assert all("gud_up_proj_layer6" not in name
               for name in contract.weight_modules)
    assert all("post_process_layer" not in name
               for name in contract.weight_modules)


def test_official_dyspn_builder_strictly_loads_checkpoint_and_contract(monkeypatch):
    monkeypatch.syspath_prepend(str(DYSPN_ROOT))
    from DySPN.base import Model

    checkpoint_root = BASELINE_ROOT / "dyspn_iter6"
    args = json.loads((checkpoint_root / "args.json").read_text())
    payload = torch.load(str(checkpoint_root / "best.pt"), map_location="cpu")
    model = Model(
        iteration=args["iteration"],
        num_neighbor=args["dyspn_neighbors"],
        mode="dyspn",
        res=args["dyspn_resnet"],
        bm=args["dyspn_basemodel"],
        stodepth=True,
    )

    model.load_state_dict(payload["net"], strict=True)
    contract = build_model_quantization_contract("dyspn", model)

    assert type(model).__name__ == "Model"
    assert contract.blocks
    _assert_module_roots(tuple(name for name, _ in model.named_modules()), (
        "base.conv1_rgb", "base.conv1_dep", "base.conv2", "base.conv3",
        "base.conv4", "base.conv5", "base.conv6", "base.dec5",
        "base.dec4", "base.dec3", "base.dec2", "base.gd_dec1_",
        "base.gd_dec0_dyspn_6_5", "dyspn_6_5.conv_offset_aff",
    ))
    assert all("base.gd_dec0_dyspn_6_5" not in name
               for name in contract.weight_modules)
    assert all("dyspn_6_5.conv_offset_aff" not in name
               for name in contract.weight_modules)


def test_official_nlspn_builder_strictly_loads_checkpoint_and_contract():
    assert sys.version_info[:2] == (3, 7)

    result = _run_official_contract("nlspn", NLSPN_ROOT, "nlspn_iter18")

    assert result["class_name"] == "NLSPNModel"
    assert result["block_count"] == 26
    assert result["weight_count"] == 45
    assert result["attention_edges"] == 0
    assert result["concat_edges"] == 0
    _assert_module_roots(result["module_names"], (
        "conv1_rgb", "conv1_dep", "conv2", "conv3", "conv4", "conv5",
        "conv6", "dec5", "dec4", "dec3", "dec2", "id_dec1", "id_dec0",
        "gd_dec1", "gd_dec0", "cf_dec1", "cf_dec0", "prop_layer",
    ))
    assert all(not name.startswith(("cf_dec", "gd_dec0"))
               for name in result["weight_modules"])
    units = dict((row["name"], row) for row in result["search_units"])
    assert units["early_boundary"]["members"] == [
        "conv2.0.conv1", "conv2.0.conv2", "conv3.0.downsample.0"]
    assert units["early_boundary"]["allow_fp16"] is True
    assert units["initial_depth"]["allow_fp16"] is True


def test_official_completionformer_builder_strictly_loads_checkpoint_and_contract():
    assert sys.version_info[:2] == (3, 7)

    result = _run_official_contract(
        "completionformer", COMPLETIONFORMER_ROOT, "completionformer_iter18")

    assert result["class_name"] == "CompletionFormer"
    assert result["block_count"] == 38
    assert result["weight_count"] == 244
    assert result["attention_edges"] == 48
    assert result["concat_edges"] == 32
    _assert_module_roots(result["module_names"], (
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
    assert all(not name.startswith(("backbone.cf_dec", "backbone.gd_dec0"))
               for name in result["weight_modules"])
    units = dict((row["name"], row) for row in result["search_units"])
    assert units["attention_qkv"]["minimum_activation_bits"] == 8
    assert all(name.endswith((".attn.q", ".attn.kv"))
               for name in units["attention_qkv"]["members"])
