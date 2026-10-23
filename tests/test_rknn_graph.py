from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch

from spn_quant.rknn_graph import (
    decompose_convtranspose_for_rknn,
    decompose_modulated_deform_conv_for_rknn,
    replace_sign_for_rknn,
    rewrite_convtranspose_output_padding_for_rknn,
    stabilize_completionformer_fp16_export,
)
from torchvision.ops import deform_conv2d


def test_stabilize_completionformer_fp16_export_is_algebraically_exact(
        tmp_path: Path):
    source = tmp_path / "completionformer.onnx"
    converted = tmp_path / "completionformer_stable.onnx"
    rng = np.random.default_rng(7)
    initializers = []
    nodes = []
    patch_outputs = []
    for stage in (3, 4):
        prefix = "model.backbone.former.patch_embed%d" % stage
        weight_name = prefix + ".proj.weight"
        bias_name = prefix + ".proj.bias"
        weight = rng.standard_normal((2, 2, 1, 1)).astype(np.float32)
        bias = rng.standard_normal((2,)).astype(np.float32)
        initializers.extend([
            onnx.numpy_helper.from_array(weight, weight_name),
            onnx.numpy_helper.from_array(bias, bias_name),
        ])
        conv_output = "patch%d_conv" % stage
        centered = "patch%d_centered" % stage
        variance = "patch%d_variance" % stage
        variance_eps = "patch%d_variance_eps" % stage
        denominator = "patch%d_denominator" % stage
        output = "patch%d_output" % stage
        nodes.extend([
            helper.make_node(
                "Conv", ["x", weight_name, bias_name], [conv_output],
                name="patch%d_conv" % stage),
            helper.make_node(
                "ReduceMean", [conv_output], ["patch%d_mean" % stage],
                name="patch%d_mean_node" % stage, axes=[1], keepdims=1),
            helper.make_node(
                "Sub", [conv_output, "patch%d_mean" % stage], [centered]),
            helper.make_node("Mul", [centered, centered], ["patch%d_sq" % stage]),
            helper.make_node(
                "ReduceMean", ["patch%d_sq" % stage], [variance],
                axes=[1], keepdims=1),
            helper.make_node(
                "Constant", [], ["patch%d_eps" % stage],
                name="Constant_%d" % (1424 if stage == 3 else 2604),
                value=helper.make_tensor(
                    "patch%d_eps_value" % stage, TensorProto.FLOAT, [], [1e-5])),
            helper.make_node("Add", [variance, "patch%d_eps" % stage], [variance_eps]),
            helper.make_node("Sqrt", [variance_eps], [denominator]),
            helper.make_node("Div", [centered, denominator], [output]),
        ])
        patch_outputs.append(output)

    decoder_prefix = "model.backbone.dec4.0"
    decoder_weight_name = decoder_prefix + ".0.weight"
    decoder_weight = rng.standard_normal((2, 2, 1, 1)).astype(np.float32)
    bn_names = {
        suffix: decoder_prefix + ".1." + suffix
        for suffix in ("weight", "bias", "running_mean", "running_var")
    }
    initializers.extend([
        onnx.numpy_helper.from_array(decoder_weight, decoder_weight_name),
        onnx.numpy_helper.from_array(
            np.array([1.1, 0.9], dtype=np.float32), bn_names["weight"]),
        onnx.numpy_helper.from_array(
            np.array([0.2, -0.3], dtype=np.float32), bn_names["bias"]),
        onnx.numpy_helper.from_array(
            np.array([0.4, -0.5], dtype=np.float32), bn_names["running_mean"]),
        onnx.numpy_helper.from_array(
            np.array([1.2, 0.8], dtype=np.float32), bn_names["running_var"]),
    ])
    nodes.extend([
        helper.make_node(
            "ConvTranspose", ["x", decoder_weight_name], ["decoder_conv"],
            name="decoder_conv"),
        helper.make_node(
            "BatchNormalization",
            ["decoder_conv", bn_names["weight"], bn_names["bias"],
             bn_names["running_mean"], bn_names["running_var"]],
            ["decoder_output"], name="decoder_bn", epsilon=1e-5),
    ])
    nodes.append(helper.make_node(
        "Concat", patch_outputs + ["decoder_output"], ["y"], axis=1))
    graph = helper.make_graph(
        nodes, "completionformer_stabilization",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 3, 4])],
        initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, source)

    assert stabilize_completionformer_fp16_export(source, converted) == 3

    inputs = {"x": rng.standard_normal((1, 2, 3, 4)).astype(np.float32)}
    original = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    stable = ort.InferenceSession(
        str(converted), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    np.testing.assert_allclose(stable, original, rtol=2e-6, atol=2e-6)


def test_replace_sign_uses_standard_ops_and_preserves_zero(tmp_path: Path):
    source = tmp_path / "sign.onnx"
    converted = tmp_path / "sign_rewritten.onnx"
    graph = helper.make_graph(
        [helper.make_node("Sign", ["x"], ["y"], name="sign")],
        "sign_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, source)

    count = replace_sign_for_rknn(source, converted)

    rewritten = onnx.load(converted)
    assert count == 1
    assert "Sign" not in [node.op_type for node in rewritten.graph.node]
    assert [node.op_type for node in rewritten.graph.node] == [
        "Constant", "Greater", "Less", "Cast", "Cast", "Sub"]
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": np.array([-2.0, 0.0, 3.0], dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.array([-1.0, 0.0, 1.0], dtype=np.float32))


def test_rewrite_convtranspose_output_padding_is_exact(tmp_path: Path):
    source = tmp_path / "deconv.onnx"
    converted = tmp_path / "deconv_rewritten.onnx"
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node(
            "ConvTranspose", ["x", "weight"], ["y"], name="deconv",
            kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1],
            output_padding=[1, 1],
        )],
        "deconv_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 5, 7])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 10, 14])],
        [onnx.numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, source)

    count = rewrite_convtranspose_output_padding_for_rknn(source, converted)

    rewritten = onnx.load(converted)
    node = rewritten.graph.node[0]
    attributes = {
        attr.name: list(helper.get_attribute_value(attr))
        for attr in node.attribute
        if attr.name in {"pads", "output_padding"}
    }
    assert count == 1
    assert attributes == {"output_padding": [0, 0], "pads": [1, 1, 0, 0]}
    inputs = {"x": rng.standard_normal((1, 4, 5, 7)).astype(np.float32)}
    original_output = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    rewritten_output = ort.InferenceSession(
        str(converted), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    np.testing.assert_array_equal(rewritten_output, original_output)


def test_decompose_convtranspose_is_exact_for_dense_and_depthwise(tmp_path: Path):
    rng = np.random.default_rng(23)
    cases = [
        {
            "name": "dense",
            "input_shape": [1, 4, 5, 7],
            "weight_shape": [4, 3, 3, 3],
            "output_shape": [1, 3, 10, 14],
            "group": 1,
            "pads": [1, 1, 1, 1],
            "output_padding": [1, 1],
        },
        {
            "name": "depthwise",
            "input_shape": [1, 4, 5, 7],
            "weight_shape": [4, 1, 2, 2],
            "output_shape": [1, 4, 10, 14],
            "group": 4,
            "pads": [0, 0, 0, 0],
            "output_padding": [0, 0],
        },
    ]
    for case in cases:
        source = tmp_path / (case["name"] + ".onnx")
        converted = tmp_path / (case["name"] + "_decomposed.onnx")
        weight = rng.standard_normal(case["weight_shape"]).astype(np.float32)
        bias = rng.standard_normal(case["output_shape"][1]).astype(np.float32)
        graph = helper.make_graph(
            [helper.make_node(
                "ConvTranspose", ["x", "weight", "bias"], ["y"],
                name="deconv", kernel_shape=case["weight_shape"][2:],
                strides=[2, 2], pads=case["pads"],
                output_padding=case["output_padding"], group=case["group"],
            )],
            case["name"] + "_graph",
            [helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, case["input_shape"])],
            [helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, case["output_shape"])],
            [onnx.numpy_helper.from_array(weight, "weight"),
             onnx.numpy_helper.from_array(bias, "bias")],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.save(model, source)

        count = decompose_convtranspose_for_rknn(source, converted)

        rewritten = onnx.load(converted)
        op_types = [node.op_type for node in rewritten.graph.node]
        assert count == 1
        assert "ConvTranspose" not in op_types
        assert op_types == ["Resize", "Mul", "Conv"]
        inputs = {
            "x": rng.standard_normal(case["input_shape"]).astype(np.float32)}
        original_output = ort.InferenceSession(
            str(source), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
        rewritten_output = ort.InferenceSession(
            str(converted), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
        np.testing.assert_allclose(
            rewritten_output, original_output, rtol=1e-6, atol=1e-6)


def test_decompose_convtranspose_resolves_identity_aliased_weight(tmp_path: Path):
    source = tmp_path / "identity_weight.onnx"
    converted = tmp_path / "identity_weight_decomposed.onnx"
    rng = np.random.default_rng(29)
    weight = rng.standard_normal((2, 3, 2, 2)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["weight"], ["weight_alias"]),
            helper.make_node(
                "ConvTranspose", ["x", "weight_alias"], ["y"],
                kernel_shape=[2, 2], strides=[2, 2]),
        ],
        "identity_weight_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 6, 8])],
        [onnx.numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, source)

    assert decompose_convtranspose_for_rknn(source, converted) == 1

    inputs = {"x": rng.standard_normal((1, 2, 3, 4)).astype(np.float32)}
    original_output = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    rewritten_output = ort.InferenceSession(
        str(converted), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    np.testing.assert_allclose(
        rewritten_output, original_output, rtol=1e-6, atol=1e-6)


def test_decompose_convtranspose_accepts_measured_static_shape_override(
        tmp_path: Path):
    source = tmp_path / "unknown_shape.onnx"
    converted = tmp_path / "unknown_shape_decomposed.onnx"
    rng = np.random.default_rng(30)
    weight = rng.standard_normal((2, 3, 2, 2)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node(
            "ConvTranspose", ["x", "weight"], ["y"], name="deconv",
            kernel_shape=[2, 2], strides=[2, 2])],
        "unknown_shape_graph",
        [helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, ["batch", 2, "height", "width"])],
        [helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, ["batch", 3, "out_height", "out_width"])],
        [onnx.numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, source)

    assert decompose_convtranspose_for_rknn(
        source, converted, input_shape_overrides={"x": [1, 2, 3, 4]}) == 1

    inputs = {"x": rng.standard_normal((1, 2, 3, 4)).astype(np.float32)}
    original_output = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    rewritten_output = ort.InferenceSession(
        str(converted), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    np.testing.assert_allclose(
        rewritten_output, original_output, rtol=1e-6, atol=1e-6)


def test_decompose_modulated_deform_conv_matches_torchvision(tmp_path: Path):
    source = tmp_path / "dcn.onnx"
    converted = tmp_path / "dcn_decomposed.onnx"
    rng = np.random.default_rng(31)
    input_value = rng.standard_normal((1, 1, 5, 7)).astype(np.float32)
    offset_value = (rng.standard_normal((1, 18, 5, 7)) * 0.2).astype(np.float32)
    mask_value = rng.random((1, 9, 5, 7)).astype(np.float32)
    weight = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
    bias = rng.standard_normal((1,)).astype(np.float32)
    node = helper.make_node(
        "ModulatedDeformConv2D", ["x", "offset", "mask", "weight", "bias"],
        ["y"], domain="com.opencorex", name="dcn", feature_shape=[1, 1, 5, 7],
        offset_shape=[1, 18, 5, 7], mask_shape=[1, 9, 5, 7],
        weight_shape=[1, 1, 3, 3], bias_shape=[1],
        output_shape=[1, 1, 5, 7], strides=[1, 1], pads=[1, 1, 1, 1],
        dilations=[1, 1], groups=1, deformable_groups=1, im2col_step=64)
    graph = helper.make_graph(
        [node], "dcn_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 7]),
         helper.make_tensor_value_info(
             "offset", TensorProto.FLOAT, [1, 18, 5, 7]),
         helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 9, 5, 7])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 5, 7])],
        [onnx.numpy_helper.from_array(weight, "weight"),
         onnx.numpy_helper.from_array(bias, "bias")])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14),
                              helper.make_opsetid("com.opencorex", 1)])
    model.ir_version = 8
    onnx.save(model, source)

    assert decompose_modulated_deform_conv_for_rknn(source, converted) == 1

    rewritten = onnx.load(converted)
    op_types = [node.op_type for node in rewritten.graph.node]
    assert op_types.count("Split") == 2
    assert "Slice" not in op_types
    assert all(len(node.input) == 2 for node in rewritten.graph.node
               if node.op_type == "Split")

    expected = deform_conv2d(
        torch.from_numpy(input_value), torch.from_numpy(offset_value),
        torch.from_numpy(weight), torch.from_numpy(bias), padding=(1, 1),
        mask=torch.from_numpy(mask_value)).numpy()
    actual = ort.InferenceSession(
        str(converted), providers=["CPUExecutionProvider"]).run(None, {
            "x": input_value, "offset": offset_value, "mask": mask_value})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
