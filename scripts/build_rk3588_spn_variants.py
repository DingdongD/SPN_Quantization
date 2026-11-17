#!/usr/bin/env python3
"""Build auditable legacy and coarse-packed RK3588 SPN graphs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import onnx
from onnx import numpy_helper

from spn_quant.rknn_graph import (
    decompose_convtranspose_for_rknn,
    decompose_modulated_deform_conv_for_rknn,
    pack_grid_sample_groups_for_rknn,
    replace_sign_for_rknn,
)


@dataclass(frozen=True)
class Variant:
    name: str
    pack_grid_samples: bool


def variant_plan():
    return (
        Variant("compressed_fp16_legacy_grid", False),
        Variant("compressed_fp16_packed_grid", True),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_graph(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    counts: Dict[str, int] = {}
    deformable_count = 0
    for node in model.graph.node:
        name = (node.domain + "::" if node.domain else "") + node.op_type
        counts[name] = counts.get(name, 0) + 1
        deformable_count += int(
            node.domain == "com.opencorex"
            and node.op_type == "ModulatedDeformConv2D")
    initializer_names = {item.name for item in model.graph.initializer}
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "bytes": Path(path).stat().st_size,
        "inputs": [value.name for value in model.graph.input
                   if value.name not in initializer_names],
        "outputs": [value.name for value in model.graph.output],
        "operator_counts": dict(sorted(counts.items())),
        "deformable_count": deformable_count,
        "grid_sample_count": counts.get("GridSample", 0),
        "conv_transpose_count": counts.get("ConvTranspose", 0),
        "sign_count": counts.get("Sign", 0),
    }


def validate_variant_pair(legacy: dict, packed: dict) -> None:
    if legacy["inputs"] != packed["inputs"] or \
            legacy["outputs"] != packed["outputs"]:
        raise ValueError("A/B graph interfaces differ")
    if legacy["deformable_count"] or packed["deformable_count"]:
        raise ValueError("native deformable operators remain after lowering")
    if packed["grid_sample_count"] >= legacy["grid_sample_count"]:
        raise ValueError("packed rewrite did not reduce GridSample count")


def _convtranspose_overrides(source: Path, model_name: str) -> dict:
    model = onnx.load(str(source))
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shapes = {
        value.name: [int(dim.dim_value) for dim in
                     value.type.tensor_type.shape.dim]
        for value in (list(inferred.graph.input)
                      + list(inferred.graph.value_info)
                      + list(inferred.graph.output))
        if value.type.HasField("tensor_type")
    }
    initializers = {
        value.name: numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    transposed = [node for node in model.graph.node
                  if node.op_type == "ConvTranspose"]
    known_spatial = {
        "completionformer": ((7, 9), (14, 19), (28, 38),
                             (57, 76), (114, 152)),
    }.get(model_name)
    overrides = {}
    for index, node in enumerate(transposed):
        shape = list(shapes.get(node.input[0], []))
        weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
        if len(shape) != 4:
            shape = [0, 0, 0, 0]
        shape[0] = shape[0] or 1
        if not shape[1] and weight is not None:
            shape[1] = int(weight.shape[0])
        if (not shape[2] or not shape[3]) and known_spatial is not None:
            shape[2:] = list(known_spatial[index])
        if not all(shape):
            raise ValueError(
                "cannot establish ConvTranspose input shape for %s" %
                (node.name or node.input[0]))
        overrides[node.input[0]] = shape
    return overrides


def rewrite_variant(source: Path, output: Path, model_name: str,
                    pack_grid_samples: bool) -> dict:
    """Apply exact admission rewrites and preserve every intermediate graph."""
    source = Path(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    native = audit_graph(source)
    current = source
    steps = {}

    lowered = output.with_name(output.stem + ".01_propagation.onnx")
    if native["deformable_count"]:
        steps["deformable"] = decompose_modulated_deform_conv_for_rknn(
            current, lowered, pack_grid_samples=pack_grid_samples)
        current = lowered
    elif pack_grid_samples:
        steps["packed_groups"] = pack_grid_sample_groups_for_rknn(
            current, lowered)
        current = lowered

    current_audit = audit_graph(current)
    if current_audit["sign_count"]:
        sign_safe = output.with_name(output.stem + ".02_sign.onnx")
        steps["sign"] = replace_sign_for_rknn(current, sign_safe)
        current = sign_safe

    current_audit = audit_graph(current)
    if current_audit["conv_transpose_count"]:
        deconv_safe = output
        steps["conv_transpose"] = decompose_convtranspose_for_rknn(
            current, deconv_safe,
            input_shape_overrides=_convtranspose_overrides(
                current, model_name))
        current = deconv_safe
    elif current != output:
        # ONNX load/save gives a deterministic standalone artifact without
        # relying on shell copy commands.
        model = onnx.load(str(current))
        onnx.save(model, str(output))
        current = output

    final = audit_graph(current)
    if final["deformable_count"] or final["conv_transpose_count"] or \
            final["sign_count"]:
        raise ValueError("unsupported operators remain in RKNN-safe graph")
    return {"steps": steps, "native": native, "safe": final}


def compile_rknn(source: Path, output: Path) -> dict:
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        calls = (
            ("config", lambda: rknn.config(
                target_platform="rk3588", float_dtype="float16",
                optimization_level=3)),
            ("load_onnx", lambda: rknn.load_onnx(model=str(source))),
            ("build", lambda: rknn.build(do_quantization=False)),
            ("export", lambda: rknn.export_rknn(str(output))),
        )
        status = {}
        for name, call in calls:
            status[name] = int(call())
            if status[name] != 0:
                raise RuntimeError("RKNN %s failed with %d" %
                                   (name, status[name]))
    finally:
        rknn.release()
    return {
        "status": status,
        "path": str(Path(output).resolve()),
        "sha256": file_sha256(output),
        "bytes": Path(output).stat().st_size,
    }


def build(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for variant in variant_plan():
        safe = args.output_dir / (variant.name + ".onnx")
        record = rewrite_variant(
            args.source, safe, args.model, variant.pack_grid_samples)
        if not args.skip_rknn:
            record["rknn"] = compile_rknn(
                safe, args.output_dir / (variant.name + ".rknn"))
        records[variant.name] = record
    validate_variant_pair(
        records[variant_plan()[0].name]["safe"],
        records[variant_plan()[1].name]["safe"])
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "target": "rk3588",
        "execution_precision": "fp16",
        "source": audit_graph(args.source),
        "variants": records,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return args.manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dyspn", "completionformer"),
                        required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skip-rknn", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(build(_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
