#!/usr/bin/env python3
"""Export the vanilla-equivalent FP16-stabilized CompletionFormer RKNN graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch


def stabilize_completionformer_module(model, scale: float = 0.5) -> int:
    """Apply the audited algebraically equivalent FP16 reparameterization."""
    if not np.isfinite(scale) or not 0.0 < scale < 1.0:
        raise ValueError("scale must be finite and in (0, 1)")
    try:
        patch_blocks = (
            model.backbone.former.patch_embed3,
            model.backbone.former.patch_embed4,
        )
        decoder = model.backbone.dec4[0]
        deconvolution = decoder[0]
        batch_norm = decoder[1]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("unexpected CompletionFormer k18 module topology") from error
    if not isinstance(deconvolution, torch.nn.ConvTranspose2d):
        raise ValueError("dec4.0.0 is not ConvTranspose2d")
    if not isinstance(batch_norm, torch.nn.BatchNorm2d):
        raise ValueError("dec4.0.1 is not BatchNorm2d")
    with torch.no_grad():
        for block in patch_blocks:
            if (not isinstance(block.proj, torch.nn.Conv2d)
                    or not isinstance(block.norm, torch.nn.LayerNorm)
                    or block.proj.bias is None):
                raise ValueError("unexpected patch embedding topology")
            block.proj.weight.mul_(scale)
            block.proj.bias.mul_(scale)
            block.norm.eps *= scale * scale
        deconvolution.weight.mul_(scale)
        if deconvolution.bias is not None:
            deconvolution.bias.mul_(scale)
        batch_norm.running_mean.mul_(scale)
        batch_norm.running_var.mul_(scale * scale)
        batch_norm.eps *= scale * scale
    return 3


_DCN_SHAPES = []
_DCN_SYMBOLIC_INDEX = 0


def _pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("DCN spatial attribute must contain two values")
        return [int(value[0]), int(value[1])]
    return [int(value), int(value)]


def _shape(value, captured):
    sizes = value.type().sizes() if hasattr(value.type(), "sizes") else None
    if sizes is None or any(item is None for item in sizes):
        sizes = captured
    shape = [int(item) for item in sizes]
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("DCN symbolic requires a positive static shape")
    return shape


def _native_dcn_symbolic(
        graph, input_value, offset, mask, weight, bias, stride, padding,
        dilation, groups, deformable_groups, im2col_step):
    global _DCN_SYMBOLIC_INDEX
    if _DCN_SYMBOLIC_INDEX >= len(_DCN_SHAPES):
        raise ValueError("DCN shape capture was not populated")
    captured = _DCN_SHAPES[_DCN_SYMBOLIC_INDEX]
    _DCN_SYMBOLIC_INDEX += 1
    stride_pair = _pair(stride)
    padding_pair = _pair(padding)
    dilation_pair = _pair(dilation)
    node = graph.op(
        "com.opencorex::ModulatedDeformConv2D",
        input_value, offset, mask, weight, bias,
        strides_i=stride_pair,
        pads_i=[padding_pair[0], padding_pair[1],
                padding_pair[0], padding_pair[1]],
        dilations_i=dilation_pair,
        groups_i=int(groups),
        deformable_groups_i=int(deformable_groups),
        im2col_step_i=int(im2col_step),
        feature_shape_i=_shape(input_value, captured["feature"]),
        offset_shape_i=_shape(offset, captured["offset"]),
        mask_shape_i=_shape(mask, captured["mask"]),
        weight_shape_i=_shape(weight, captured["weight"]),
        bias_shape_i=_shape(bias, captured["bias"]),
        output_shape_i=list(captured["output"]),
    )
    return node.setType(input_value.type())


def install_native_dcn_symbolic() -> int:
    module = sys.modules.get("model.nlspn_module")
    if module is None:
        raise RuntimeError("CompletionFormer propagation module was not loaded")
    function = module.ModulatedDeformConvFunction
    original_forward = function.forward

    def recording_forward(ctx, input_value, offset, mask, weight, bias, *args):
        output = original_forward(
            ctx, input_value, offset, mask, weight, bias, *args)
        _DCN_SHAPES.append({
            "feature": tuple(input_value.shape),
            "offset": tuple(offset.shape),
            "mask": tuple(mask.shape),
            "weight": tuple(weight.shape),
            "bias": tuple(bias.shape),
            "output": tuple(output.shape),
        })
        return output

    function.forward = staticmethod(recording_forward)
    function.symbolic = staticmethod(_native_dcn_symbolic)
    return 1


class _Prediction(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, depth):
        output = self.model({"rgb": rgb, "dep": depth})
        return output["pred"]


def export(args: argparse.Namespace) -> dict:
    from scripts.benchmark_nyu_vanilla_gpu_worker import (
        _runtime_args,
        load_fixed_model_args,
    )
    from scripts.nyu_model_runtime import NYUModelRuntime
    from spn_quant.rknn_graph import (
        decompose_convtranspose_for_rknn,
        decompose_modulated_deform_conv_for_rknn,
    )
    import onnx
    import onnxruntime as ort

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = config["models"]["completionformer"]
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, args.device))
    model = runtime.build_model(runtime.device)
    model_args = load_fixed_model_args(
        args.fixed_inputs, "completionformer", runtime.device)
    if (len(model_args) != 1 or not isinstance(model_args[0], dict)
            or set(model_args[0]) != {"rgb", "dep"}):
        raise ValueError("unexpected fixed CompletionFormer input contract")
    rgb = model_args[0]["rgb"]
    depth = model_args[0]["dep"]
    with torch.no_grad():
        reference = model(*model_args)["pred"].detach().cpu().numpy()
    stabilize_completionformer_module(model, scale=args.scale)
    install_native_dcn_symbolic()
    wrapper = _Prediction(model).eval()
    args.native_output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (rgb, depth), str(args.native_output),
        input_names=["rgb", "dep"], output_names=["pred"],
        opset_version=17, do_constant_folding=True, dynamo=False)
    runtime.close()
    del wrapper, model, runtime, model_args, rgb, depth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    native = onnx.load(str(args.native_output), load_external_data=False)
    deform_count = sum(
        node.domain == "com.opencorex"
        and node.op_type == "ModulatedDeformConv2D"
        for node in native.graph.node)
    transposed = [
        node for node in native.graph.node if node.op_type == "ConvTranspose"]
    if deform_count != 26 or len(transposed) != 5:
        raise RuntimeError(
            "unexpected export topology: DCN=%d ConvTranspose=%d" %
            (deform_count, len(transposed)))
    shapes = ([1, 512, 7, 9], [1, 576, 14, 19], [1, 256, 28, 38],
              [1, 128, 57, 76], [1, 192, 114, 152])
    overrides = {
        node.input[0]: shape for node, shape in zip(transposed, shapes)}
    dcn_count = decompose_modulated_deform_conv_for_rknn(
        args.native_output, args.dcn_output)
    deconv_count = decompose_convtranspose_for_rknn(
        args.dcn_output, args.safe_output,
        input_shape_overrides=overrides)

    session = ort.InferenceSession(
        str(args.safe_output), providers=["CPUExecutionProvider"])
    fixed = torch.load(args.fixed_inputs, map_location="cpu", weights_only=True)
    fixed_args = fixed["completionformer"]["args"][0]
    actual = session.run(None, {
        "rgb": fixed_args["rgb"].numpy(),
        "dep": fixed_args["dep"].numpy(),
    })[0]
    absolute = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    result = {
        "scale": args.scale,
        "native_dcn_count": deform_count,
        "decomposed_dcn_count": dcn_count,
        "decomposed_convtranspose_count": deconv_count,
        "all_finite": bool(np.isfinite(actual).all()),
        "fp32_safe_onnx_vs_pytorch": {
            "maximum_absolute_error": float(absolute.max()),
            "mean_absolute_error": float(absolute.mean()),
            "p95_absolute_error": float(np.quantile(absolute, 0.95)),
            "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        },
    }
    args.validation_output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixed-inputs", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--dcn-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = export(_parser().parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
