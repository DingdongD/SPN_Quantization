from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts import export_compressed_spn_onnx as exporter
from spn_quant.compressed_checkpoint import CompressedCheckpoint
from spn_quant.rk3588_compressed_candidates import CompressedCandidate


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)


def _checkpoint(tmp_path: Path, state) -> CompressedCheckpoint:
    return CompressedCheckpoint(
        source_path=(tmp_path / "compressed.pt").resolve(),
        model="dyspn",
        net=state,
        net_sha256="abc",
        architecture={
            "nas_candidate": "drop_s4",
            "nas_depths": (3, 4, 6, 2),
            "structured_candidate": "pruned",
        },
        quantization={"propagation_dtype": "bf16"},
        model_build={
            "saved_args": {"model": "dyspn"},
            "checkpoint_architecture": "Tiny",
        },
        calibration={}, evidence={}, provenance={})


def test_reconstruct_compressed_model_applies_declared_architecture(tmp_path):
    source = _Tiny()
    with torch.no_grad():
        source.projection.weight.fill_(2.0)
        source.projection.bias.fill_(3.0)
    checkpoint = _checkpoint(tmp_path, source.state_dict())
    calls = []

    def builder(args: Namespace, device: torch.device):
        assert args.model == "dyspn"
        return _Tiny().to(device), {"architecture": "Tiny"}

    def apply_depths(model, candidate, depths):
        calls.append(("depth", candidate, tuple(depths)))

    def apply_structured(model, candidate):
        calls.append(("structured", candidate))

    model = exporter.reconstruct_compressed_model(
        checkpoint, torch.device("cpu"), builder=builder,
        apply_depths=apply_depths, apply_structured=apply_structured)

    assert calls == [
        ("depth", "drop_s4", (3, 4, 6, 2)),
        ("structured", "pruned"),
    ]
    assert torch.equal(model.projection.weight, source.projection.weight)
    assert torch.equal(model.projection.bias, source.projection.bias)
    assert not model.training


def test_prediction_wrapper_normalizes_model_interfaces():
    class DySPN(torch.nn.Module):
        def forward(self, rgb, depth):
            return {"pred": rgb[:, :1] + depth}

    class CompletionFormer(torch.nn.Module):
        def forward(self, sample):
            return {"pred": sample["rgb"][:, :1] + sample["dep"]}

    rgb = torch.ones((1, 3, 2, 2))
    depth = torch.full((1, 1, 2, 2), 2.0)

    for name, model in (("dyspn", DySPN()),
                        ("completionformer", CompletionFormer())):
        output = exporter.Prediction(model, name)(rgb, depth)
        torch.testing.assert_close(output, torch.full_like(depth, 3.0))


def test_prediction_wrapper_supports_cspn_and_nlspn_interfaces():
    class CSPN(torch.nn.Module):
        def forward(self, rgbd):
            return rgbd[:, :1] + rgbd[:, 3:4]

    class NLSPN(torch.nn.Module):
        def forward(self, sample):
            return {"pred": sample["rgb"][:, :1] + sample["dep"]}

    rgb = torch.ones((1, 3, 2, 2))
    depth = torch.full((1, 1, 2, 2), 2.0)
    rgbd = torch.cat((rgb, depth), dim=1)

    torch.testing.assert_close(
        exporter.Prediction(CSPN(), "cspn")(rgbd),
        torch.full_like(depth, 3.0))
    torch.testing.assert_close(
        exporter.Prediction(NLSPN(), "nlspn")(rgb, depth),
        torch.full_like(depth, 3.0))


def test_reconstruct_inventory_model_uses_model_specific_restore_order(tmp_path):
    source = _Tiny()
    with torch.no_grad():
        source.projection.weight.fill_(4.0)
        source.projection.bias.fill_(5.0)
    state_path = tmp_path / "state.pt"
    torch.save({"net": source.state_dict(), "args": {"model": "cspn"}},
               state_path)

    def candidate(model):
        return CompressedCandidate(
            model=model, nas_candidate="drop_s4", nas_depths=(2, 1),
            structured_candidate=("depth_pruned_drop_s4" if model == "cspn"
                                  else "pruned"),
            state_source=state_path, model_build_source=state_path,
            quantization_source=state_path,
            research_precision_policy="W4", rk3588_execution_precisions=(
                "fp16", "w8a8"), propagation_lowering="test",
            standalone_checkpoint=False)

    for model_name, expected in (
            ("cspn", ["build", "load", "depth"]),
            ("nlspn", ["build", "depth", "structured", "load"])):
        calls = []

        def builder(args, device):
            calls.append("build")
            return _Tiny().to(device), {"architecture": "Tiny"}

        def apply_depths(model, candidate_id, depths):
            calls.append("depth")

        def apply_structured(model, candidate_id):
            calls.append("structured")

        restored = exporter.reconstruct_inventory_model(
            candidate(model_name), torch.device("cpu"),
            saved_args={"model": model_name}, state=source.state_dict(),
            builder=builder, apply_depths=apply_depths,
            apply_structured=apply_structured,
            load_state=lambda model, state: (
                calls.append("load"), model.load_state_dict(state, strict=True)))

        assert calls == expected
        assert not restored.training


def test_load_fixed_inputs_requires_two_fixed_float32_tensors(tmp_path: Path):
    path = tmp_path / "inputs.npz"
    np.savez(path,
             input_0=np.ones((1, 3, 228, 304), dtype=np.float32),
             input_1=np.ones((1, 1, 228, 304), dtype=np.float32))

    rgb, depth = exporter.load_fixed_inputs(path, torch.device("cpu"))

    assert tuple(rgb.shape) == (1, 3, 228, 304)
    assert tuple(depth.shape) == (1, 1, 228, 304)
    assert rgb.dtype == torch.float32
    assert depth.dtype == torch.float32


def test_load_fixed_inputs_supports_cspn_single_tensor(tmp_path: Path):
    path = tmp_path / "cspn_inputs.npz"
    np.savez(path, input_0=np.ones((1, 4, 228, 304), dtype=np.float32))

    inputs = exporter.load_fixed_inputs(path, torch.device("cpu"), "cspn")

    assert len(inputs) == 1
    assert tuple(inputs[0].shape) == (1, 4, 228, 304)


def test_pretrained_downloads_are_suppressed_and_restored():
    import torch.utils.model_zoo as model_zoo

    original_model_zoo = model_zoo.load_url
    original_hub = torch.hub.load_state_dict_from_url
    with exporter.without_pretrained_downloads():
        assert model_zoo.load_url("unused") == {}
        assert torch.hub.load_state_dict_from_url("unused") == {}
    assert model_zoo.load_url is original_model_zoo
    assert torch.hub.load_state_dict_from_url is original_hub
