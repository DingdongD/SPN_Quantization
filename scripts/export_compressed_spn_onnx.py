#!/usr/bin/env python3
"""Export a standalone structurally compressed SPN checkpoint to ONNX."""

from __future__ import annotations

import argparse
from argparse import Namespace
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spn_quant.compressed_checkpoint import (  # noqa: E402
    CompressedCheckpoint,
    load_compressed_checkpoint,
)
from spn_quant.rk3588_compressed_candidates import (  # noqa: E402
    CompressedCandidate,
    candidate_inventory,
)


SUPPORTED_MODELS = ("cspn", "dyspn", "nlspn", "completionformer")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def without_pretrained_downloads():
    """Build checkpoint-defined topology without fetching initialization."""
    import torch.utils.model_zoo as model_zoo

    original_model_zoo = model_zoo.load_url
    original_hub = torch.hub.load_state_dict_from_url
    model_zoo.load_url = lambda *args, **kwargs: {}
    torch.hub.load_state_dict_from_url = lambda *args, **kwargs: {}
    try:
        yield
    finally:
        model_zoo.load_url = original_model_zoo
        torch.hub.load_state_dict_from_url = original_hub


def _official_dependencies(model_name: str):
    from scripts import screen_four_model_depth_nas as nas
    from scripts import screen_structured_channel_nas as structured
    from scripts import train_nyu_iteration_sweep as sweep

    builder = sweep.BUILDERS[model_name]

    def apply_depths(model, candidate, expected_depths):
        paths = nas.STAGE_PATHS[model_name]
        stages = nas.stage_modules(model, paths)
        configured = tuple(dict(
            nas.CANDIDATE_DEPTHS[model_name])[candidate])
        if configured != tuple(expected_depths):
            raise ValueError("checkpoint NAS depths differ from candidate")
        nas.apply_depths(model, paths, stages, configured)

    def apply_structured(model, candidate):
        structured.apply_structured_candidate(model, model_name, candidate)

    return builder, apply_depths, apply_structured


def reconstruct_compressed_model(
        checkpoint: CompressedCheckpoint, device: torch.device,
        builder: Optional[Callable] = None,
        apply_depths: Optional[Callable] = None,
        apply_structured: Optional[Callable] = None) -> torch.nn.Module:
    """Rebuild declared topology and strictly restore compressed weights."""
    supplied = (builder, apply_depths, apply_structured)
    if any(value is None for value in supplied):
        if not all(value is None for value in supplied):
            raise ValueError("reconstruction dependencies must be supplied together")
        builder, apply_depths, apply_structured = _official_dependencies(
            checkpoint.model)

    args = Namespace(**dict(checkpoint.model_build["saved_args"]))
    with without_pretrained_downloads():
        model, metadata = builder(args, device)
    expected_architecture = str(
        checkpoint.model_build["checkpoint_architecture"])
    if str(metadata.get("architecture")) != expected_architecture:
        raise ValueError("standalone model architecture mismatch")
    architecture = checkpoint.architecture
    apply_depths(
        model, str(architecture["nas_candidate"]),
        tuple(architecture["nas_depths"]))
    apply_structured(model, str(architecture["structured_candidate"]))
    model.load_state_dict(checkpoint.net, strict=True)
    model.eval()
    return model


def reconstruct_inventory_model(
        candidate: CompressedCandidate, device: torch.device,
        saved_args, state, builder: Optional[Callable] = None,
        apply_depths: Optional[Callable] = None,
        apply_structured: Optional[Callable] = None,
        load_state: Optional[Callable] = None) -> torch.nn.Module:
    """Restore a non-standalone inventory candidate in the required order."""
    supplied = (builder, apply_depths, apply_structured)
    if any(value is None for value in supplied):
        if not all(value is None for value in supplied):
            raise ValueError("reconstruction dependencies must be supplied together")
        builder, apply_depths, apply_structured = _official_dependencies(
            candidate.model)
    if load_state is None:
        def load_state(model, payload):
            payload = dict(payload)
            if candidate.model == "cspn":
                fixed_key = "post_process_layer.sum_conv.weight"
                fixed = payload.pop(fixed_key, None)
                if fixed is not None and (tuple(fixed.shape) != (1, 8, 1, 1, 1)
                                          or not bool(torch.all(fixed == 1))):
                    raise ValueError("invalid CSPN fixed sum kernel")
            incompatible = model.load_state_dict(payload, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "checkpoint mismatch: missing=%s unexpected=%s" %
                    (incompatible.missing_keys, incompatible.unexpected_keys))

    build_args = dict(saved_args)
    build_args["model"] = candidate.model
    build_args["from_scratch"] = True
    with without_pretrained_downloads():
        model, _ = builder(Namespace(**build_args), device)

    if candidate.model == "cspn":
        load_state(model, state)
        apply_depths(
            model, candidate.nas_candidate, candidate.nas_depths)
    else:
        apply_depths(
            model, candidate.nas_candidate, candidate.nas_depths)
        apply_structured(model, candidate.structured_candidate)
        load_state(model, state)
    model.eval()
    return model


def _torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_inventory_model(candidate: CompressedCandidate,
                         device: torch.device) -> torch.nn.Module:
    """Load either a standalone or recovered inventory model."""
    if candidate.standalone_checkpoint:
        checkpoint = load_compressed_checkpoint(
            candidate.state_source, expected_model=candidate.model)
        return reconstruct_compressed_model(checkpoint, device)
    state_payload = _torch_load(candidate.state_source)
    build_payload = _torch_load(candidate.model_build_source)
    return reconstruct_inventory_model(
        candidate, device, saved_args=build_payload["args"],
        state=state_payload["net"])


class Prediction(torch.nn.Module):
    """Normalize official model interfaces to two tensors and one prediction."""

    def __init__(self, model: torch.nn.Module, model_name: str):
        super().__init__()
        if model_name not in SUPPORTED_MODELS:
            raise ValueError("unsupported compressed export model: %s" % model_name)
        self.model = model
        self.model_name = model_name

    def forward(self, *inputs):
        if self.model_name == "cspn":
            return self.model(inputs[0])
        rgb, depth = inputs
        if self.model_name == "dyspn":
            output = self.model(rgb, depth)
            return output["pred"] if isinstance(output, dict) else output
        return self.model({"rgb": rgb, "dep": depth})["pred"]


def load_fixed_inputs(path: Path, device: torch.device,
                      model_name: str = "dyspn") -> Tuple[torch.Tensor, ...]:
    with np.load(str(path), allow_pickle=False) as payload:
        if model_name == "cspn":
            if set(payload.files) != {"input_0"}:
                raise ValueError("CSPN fixed input archive must contain input_0")
            rgbd = np.asarray(payload["input_0"])
            if rgbd.dtype != np.float32 or \
                    tuple(rgbd.shape) != (1, 4, 228, 304):
                raise ValueError("fixed CSPN deployment input is invalid")
            return (torch.from_numpy(rgbd.copy()).to(device),)
        if set(payload.files) != {"input_0", "input_1"}:
            raise ValueError("fixed input archive must contain input_0 and input_1")
        rgb = np.asarray(payload["input_0"])
        depth = np.asarray(payload["input_1"])
    if rgb.dtype != np.float32 or depth.dtype != np.float32:
        raise ValueError("fixed deployment inputs must be float32")
    if tuple(rgb.shape) != (1, 3, 228, 304) or \
            tuple(depth.shape) != (1, 1, 228, 304):
        raise ValueError("fixed deployment input shapes are invalid")
    return (torch.from_numpy(rgb.copy()).to(device),
            torch.from_numpy(depth.copy()).to(device))


def export(args: argparse.Namespace) -> Path:
    if args.model not in SUPPORTED_MODELS:
        raise ValueError("unsupported compressed export model")
    device = torch.device(args.device)
    candidate = None
    checkpoint = None
    if args.inventory_root is not None:
        candidate = candidate_inventory(args.inventory_root)[args.model]
        model = load_inventory_model(candidate, device)
        source_path = candidate.state_source
        net_hash = None
        quant_candidate = candidate.research_precision_policy
        propagation_dtype = "bf16" if "BF16" in \
            candidate.research_precision_policy else "fp16"
        architecture = {
            "nas_candidate": candidate.nas_candidate,
            "nas_depths": candidate.nas_depths,
            "structured_candidate": candidate.structured_candidate,
        }
    else:
        checkpoint = load_compressed_checkpoint(
            args.checkpoint, expected_model=args.model)
        model = reconstruct_compressed_model(checkpoint, device)
        source_path = checkpoint.source_path
        net_hash = checkpoint.net_sha256
        quant_candidate = checkpoint.quantization.get("candidate_id")
        propagation_dtype = checkpoint.quantization.get(
            "propagation_dtype")
        architecture = dict(checkpoint.architecture)
    inputs = load_fixed_inputs(args.fixed_inputs, device, args.model)
    wrapper = Prediction(model, args.model).eval()
    with torch.no_grad():
        reference = wrapper(*inputs).detach().cpu().numpy()
    stabilization_count = 0
    export_opset = 17
    if args.model in ("nlspn", "completionformer"):
        from scripts import export_completionformer_rknn_fp16_stable as stable

        if args.model == "completionformer":
            stabilization_count = stable.stabilize_completionformer_module(
                model, scale=float(args.completionformer_scale))
        dcn_module = ("model.modulated_deform_conv_func"
                      if args.model == "nlspn" else "model.nlspn_module")
        stable.install_native_dcn_symbolic(dcn_module)
        # PyTorch 1.10 in the verified CompletionFormer environment supports
        # opset 14. The DCN lowering raises the final standard graph to 16.
        export_opset = 14

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_names = ["rgbd"] if args.model == "cspn" else ["rgb", "dep"]
    torch.onnx.export(
        wrapper, inputs, str(args.output),
        input_names=input_names, output_names=["pred"],
        opset_version=export_opset, do_constant_folding=True)
    np.save(str(args.output) + ".reference.npy", reference)

    import onnx

    graph = onnx.load(str(args.output), load_external_data=False)
    op_counts = {}
    for node in graph.graph.node:
        key = (node.domain + "::" if node.domain else "") + node.op_type
        op_counts[key] = op_counts.get(key, 0) + 1
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "candidate": "nas_structured_pruned_fp16_topology",
        "execution_precision": "fp16_after_rknn_lowering",
        "low_bit_policy_executed": False,
        "low_bit_policy_reason": (
            "checkpoint carries W4/W6/W8 policy and scales; this export "
            "isolates topology compression before RKNN-supported quantization"),
        "input_shapes": [list(value.shape) for value in inputs],
        "output_shape": list(reference.shape),
        "propagation_semantics": "unchanged",
        "checkpoint": str(source_path),
        "checkpoint_sha256": file_sha256(source_path),
        "net_sha256": net_hash,
        "onnx": str(args.output.resolve()),
        "onnx_sha256": file_sha256(args.output),
        "onnx_opset": export_opset,
        "operator_counts": op_counts,
        "completionformer_stabilization_count": stabilization_count,
        "architecture": architecture,
        "quantization": {
            "candidate_id": quant_candidate,
            "propagation_dtype": propagation_dtype,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return args.manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--inventory-root", type=Path)
    parser.add_argument("--fixed-inputs", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--completionformer-scale", type=float, default=0.5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    manifest = export(_parser().parse_args(argv))
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
