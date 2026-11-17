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


SUPPORTED_MODELS = ("dyspn", "completionformer")


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


class Prediction(torch.nn.Module):
    """Normalize official model interfaces to two tensors and one prediction."""

    def __init__(self, model: torch.nn.Module, model_name: str):
        super().__init__()
        if model_name not in SUPPORTED_MODELS:
            raise ValueError("unsupported compressed export model: %s" % model_name)
        self.model = model
        self.model_name = model_name

    def forward(self, rgb, depth):
        if self.model_name == "dyspn":
            output = self.model(rgb, depth)
            return output["pred"] if isinstance(output, dict) else output
        return self.model({"rgb": rgb, "dep": depth})["pred"]


def load_fixed_inputs(
        path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    with np.load(str(path), allow_pickle=False) as payload:
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
    checkpoint = load_compressed_checkpoint(
        args.checkpoint, expected_model=args.model)
    device = torch.device(args.device)
    model = reconstruct_compressed_model(checkpoint, device)
    rgb, depth = load_fixed_inputs(args.fixed_inputs, device)
    wrapper = Prediction(model, args.model).eval()
    with torch.no_grad():
        reference = wrapper(rgb, depth).detach().cpu().numpy()
    stabilization_count = 0
    export_opset = 17
    if args.model == "completionformer":
        from scripts import export_completionformer_rknn_fp16_stable as stable

        stabilization_count = stable.stabilize_completionformer_module(
            model, scale=float(args.completionformer_scale))
        stable.install_native_dcn_symbolic()
        # PyTorch 1.10 in the verified CompletionFormer environment supports
        # opset 14. The DCN lowering raises the final standard graph to 16.
        export_opset = 14

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (rgb, depth), str(args.output),
        input_names=["rgb", "dep"], output_names=["pred"],
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
        "input_shapes": [list(rgb.shape), list(depth.shape)],
        "output_shape": list(reference.shape),
        "propagation_semantics": "unchanged",
        "checkpoint": str(checkpoint.source_path),
        "checkpoint_sha256": file_sha256(checkpoint.source_path),
        "net_sha256": checkpoint.net_sha256,
        "onnx": str(args.output.resolve()),
        "onnx_sha256": file_sha256(args.output),
        "onnx_opset": export_opset,
        "operator_counts": op_counts,
        "completionformer_stabilization_count": stabilization_count,
        "architecture": dict(checkpoint.architecture),
        "quantization": {
            "candidate_id": checkpoint.quantization.get("candidate_id"),
            "propagation_dtype": checkpoint.quantization.get(
                "propagation_dtype"),
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
    parser.add_argument("--checkpoint", type=Path, required=True)
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
