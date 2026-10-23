"""Small, semantics-preserving ONNX rewrites for RKNN admission."""

from pathlib import Path
from typing import Dict, Set

import numpy as np
import onnx
from onnx import helper, numpy_helper


_COMPLETIONFORMER_PATCH_STAGES = (
    ("model.backbone.former.patch_embed3", "Constant_1424"),
    ("model.backbone.former.patch_embed4", "Constant_2604"),
)


def _scale_initializer(
        initializers: Dict[str, onnx.TensorProto], name: str,
        factor: float) -> None:
    initializer = initializers.get(name)
    if initializer is None:
        raise ValueError("missing required initializer: %s" % name)
    value = numpy_helper.to_array(initializer)
    scaled = (value * factor).astype(value.dtype)
    initializer.CopyFrom(numpy_helper.from_array(scaled, name))


def stabilize_completionformer_fp16_export(
        source: Path, output: Path, scale: float = 0.5) -> int:
    """Reparameterize three CompletionFormer blocks for finite FP16 execution.

    Scaling a projection before LayerNorm is exactly cancelled by scaling the
    LayerNorm epsilon by ``scale**2``.  Likewise, scaling a ConvTranspose and
    the following BatchNorm running mean, variance, and epsilon preserves the
    BatchNorm result.  The rewrite therefore leaves the real-valued function
    unchanged while keeping the three measured FP16 overflow sites in range.

    This is deliberately tied to the audited k18 vanilla export.  It fails
    closed if any required node, initializer, or topology is absent.
    """
    source = Path(source)
    output = Path(output)
    if not np.isfinite(scale) or scale <= 0.0 or scale >= 1.0:
        raise ValueError("CompletionFormer FP16 stabilization scale must be in (0, 1)")
    model = onnx.load(str(source))
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer}
    nodes_by_name = {node.name: node for node in model.graph.node}

    rewritten = 0
    for prefix, epsilon_node_name in _COMPLETIONFORMER_PATCH_STAGES:
        weight_name = prefix + ".proj.weight"
        bias_name = prefix + ".proj.bias"
        consumers = [
            node for node in model.graph.node
            if node.op_type == "Conv" and weight_name in node.input
            and bias_name in node.input
        ]
        if len(consumers) != 1:
            raise ValueError(
                "%s must feed exactly one Conv node" % weight_name)
        epsilon_node = nodes_by_name.get(epsilon_node_name)
        if epsilon_node is None or epsilon_node.op_type != "Constant":
            raise ValueError(
                "missing audited LayerNorm epsilon node: %s" % epsilon_node_name)
        value_attributes = [
            attribute for attribute in epsilon_node.attribute
            if attribute.name == "value"
        ]
        if len(value_attributes) != 1:
            raise ValueError(
                "%s must contain one tensor value" % epsilon_node_name)
        epsilon_tensor = value_attributes[0].t
        epsilon = numpy_helper.to_array(epsilon_tensor)
        if epsilon.size != 1 or not np.isfinite(epsilon).all():
            raise ValueError("LayerNorm epsilon must be one finite scalar")
        _scale_initializer(initializers, weight_name, scale)
        _scale_initializer(initializers, bias_name, scale)
        scaled_epsilon = (epsilon * scale * scale).astype(epsilon.dtype)
        epsilon_tensor.CopyFrom(numpy_helper.from_array(
            scaled_epsilon, epsilon_tensor.name))
        rewritten += 1

    decoder_prefix = "model.backbone.dec4.0"
    decoder_weight_name = decoder_prefix + ".0.weight"
    decoder_nodes = [
        node for node in model.graph.node
        if node.op_type == "ConvTranspose" and decoder_weight_name in node.input
    ]
    if (len(decoder_nodes) != 1 or len(decoder_nodes[0].input) != 2
            or len(decoder_nodes[0].output) != 1):
        raise ValueError(
            "%s must feed exactly one bias-free ConvTranspose node" %
            decoder_weight_name)
    batch_norm_nodes = [
        node for node in model.graph.node
        if node.op_type == "BatchNormalization"
        and node.input and node.input[0] == decoder_nodes[0].output[0]
    ]
    if len(batch_norm_nodes) != 1:
        raise ValueError("audited decoder ConvTranspose must feed one BatchNormalization")
    batch_norm = batch_norm_nodes[0]
    expected_batch_norm_inputs = [
        decoder_nodes[0].output[0],
        decoder_prefix + ".1.weight",
        decoder_prefix + ".1.bias",
        decoder_prefix + ".1.running_mean",
        decoder_prefix + ".1.running_var",
    ]
    if list(batch_norm.input) != expected_batch_norm_inputs:
        raise ValueError("audited decoder BatchNormalization inputs changed")
    epsilon_attributes = [
        attribute for attribute in batch_norm.attribute
        if attribute.name == "epsilon"
    ]
    if len(epsilon_attributes) != 1:
        raise ValueError("audited decoder BatchNormalization needs explicit epsilon")
    _scale_initializer(initializers, decoder_weight_name, scale)
    _scale_initializer(
        initializers, decoder_prefix + ".1.running_mean", scale)
    _scale_initializer(
        initializers, decoder_prefix + ".1.running_var", scale * scale)
    epsilon_attributes[0].f *= scale * scale
    rewritten += 1

    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return rewritten


def _tensor_types(model: onnx.ModelProto) -> Dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + \
        list(inferred.graph.output)
    return {
        value.name: int(value.type.tensor_type.elem_type)
        for value in values
        if value.type.HasField("tensor_type")
    }


def _tensor_shapes(model: onnx.ModelProto) -> Dict[str, list]:
    inferred = onnx.shape_inference.infer_shapes(model)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + \
        list(inferred.graph.output)
    return {
        value.name: [int(dimension.dim_value)
                     if dimension.HasField("dim_value") else 0
                     for dimension in value.type.tensor_type.shape.dim]
        for value in values
        if value.type.HasField("tensor_type")
    }


def replace_sign_for_rknn(source: Path, output: Path) -> int:
    """Replace ONNX Sign with Greater/Less/Cast/Sub without changing zeros."""
    source = Path(source)
    output = Path(output)
    model = onnx.load(str(source))
    tensor_types = _tensor_types(model)
    rewritten = []
    count = 0
    for index, node in enumerate(model.graph.node):
        if node.domain not in ("", "ai.onnx") or node.op_type != "Sign":
            rewritten.append(node)
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            raise ValueError("Sign node must have one input and one output")
        dtype = tensor_types.get(node.input[0])
        if dtype not in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT,
                         onnx.TensorProto.DOUBLE):
            raise ValueError("RKNN Sign rewrite requires a floating input")
        prefix = node.name or "Sign_%d" % index
        zero = prefix + "__zero"
        positive = prefix + "__positive"
        negative = prefix + "__negative"
        positive_value = prefix + "__positive_value"
        negative_value = prefix + "__negative_value"
        rewritten.extend([
            helper.make_node(
                "Constant", [], [zero], name=prefix + "__constant",
                value=helper.make_tensor(
                    prefix + "__zero_value", dtype, [1], [0.0])),
            helper.make_node(
                "Greater", [node.input[0], zero], [positive],
                name=prefix + "__greater"),
            helper.make_node(
                "Less", [node.input[0], zero], [negative],
                name=prefix + "__less"),
            helper.make_node(
                "Cast", [positive], [positive_value],
                name=prefix + "__cast_positive", to=dtype),
            helper.make_node(
                "Cast", [negative], [negative_value],
                name=prefix + "__cast_negative", to=dtype),
            helper.make_node(
                "Sub", [positive_value, negative_value], [node.output[0]],
                name=prefix + "__sub"),
        ])
        count += 1
    if count == 0:
        raise ValueError("model contains no standard-domain Sign node")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return count


def rewrite_convtranspose_output_padding_for_rknn(
        source: Path, output: Path) -> int:
    """Encode ConvTranspose output padding as an equivalent trailing pad change.

    RKNN 2.5 can mis-handle ``output_padding`` while optimizing wide
    ConvTranspose nodes.  For explicit padding, subtracting output padding
    from the trailing spatial pads produces exactly the same ONNX samples and
    shape, while leaving ``output_padding`` zero for RKNN.
    """
    source = Path(source)
    output = Path(output)
    model = onnx.load(str(source))
    count = 0
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "ConvTranspose":
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        output_padding_attribute = attributes.get("output_padding")
        if output_padding_attribute is None:
            continue
        output_padding = list(output_padding_attribute.ints)
        if not output_padding or not any(output_padding):
            continue
        if "output_shape" in attributes:
            raise ValueError(
                "cannot rewrite ConvTranspose with both output_shape and "
                "output_padding")
        auto_pad = attributes.get("auto_pad")
        if auto_pad is not None and helper.get_attribute_value(auto_pad) not in (
                b"", b"NOTSET"):
            raise ValueError(
                "ConvTranspose output_padding rewrite requires explicit pads")
        rank = len(output_padding)
        pads_attribute = attributes.get("pads")
        pads = ([0] * (2 * rank) if pads_attribute is None
                else list(pads_attribute.ints))
        if len(pads) != 2 * rank:
            raise ValueError("ConvTranspose pads rank does not match output_padding")
        rewritten_trailing_pads = [
            pads[rank + axis] - output_padding[axis]
            for axis in range(rank)
        ]
        if any(value < 0 for value in rewritten_trailing_pads):
            raise ValueError(
                "cannot encode ConvTranspose output_padding with nonnegative pads")
        pads[rank:] = rewritten_trailing_pads
        output_padding_attribute.ints[:] = [0] * rank
        if pads_attribute is None:
            node.attribute.extend([helper.make_attribute("pads", pads)])
        else:
            pads_attribute.ints[:] = pads
        count += 1
    if count == 0:
        raise ValueError("model contains no rewritable ConvTranspose output_padding")
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return count


def _unique_tensor_name(base: str, occupied: Set[str]) -> str:
    candidate = base
    suffix = 0
    while candidate in occupied:
        suffix += 1
        candidate = "%s_%d" % (base, suffix)
    occupied.add(candidate)
    return candidate


def _resolve_constant_initializer(
        tensor_name: str, initializers: Dict[str, onnx.TensorProto],
        producers: Dict[str, onnx.NodeProto]) -> onnx.TensorProto:
    """Resolve an initializer through export-time Identity aliases."""
    visited = set()
    current = tensor_name
    while current not in visited:
        visited.add(current)
        initializer = initializers.get(current)
        if initializer is not None:
            return initializer
        producer = producers.get(current)
        if (producer is None or producer.domain not in ("", "ai.onnx")
                or producer.op_type != "Identity" or len(producer.input) != 1):
            break
        current = producer.input[0]
    raise ValueError("ConvTranspose rewrite requires constant weights")


def decompose_convtranspose_for_rknn(
        source: Path, output: Path,
        input_shape_overrides: Dict[str, list] | None = None) -> int:
    """Replace common stride-2 ConvTranspose with exact NPU-safe operations.

    The rewrite applies nearest-neighbour Resize and a fixed checkerboard mask
    to insert zeros between input samples, then applies a spatially flipped
    ordinary convolution.  It is exact for the fixed-shape 2-D operators used
    by the four SPN exports and avoids RK3588's ConvTranspose submission path.
    """
    source = Path(source)
    output = Path(output)
    input_shape_overrides = input_shape_overrides or {}
    for tensor_name, shape in input_shape_overrides.items():
        if len(shape) != 4 or any(int(value) <= 0 for value in shape):
            raise ValueError(
                "ConvTranspose input shape override for %s must be a "
                "positive static 4-D shape" % tensor_name)
    model = onnx.load(str(source))
    tensor_shapes = _tensor_shapes(model)
    initializers = {initializer.name: initializer
                    for initializer in model.graph.initializer}
    producers = {
        tensor_name: node
        for node in model.graph.node
        for tensor_name in node.output
    }
    occupied = set(initializers)
    for node in model.graph.node:
        occupied.update(node.input)
        occupied.update(node.output)
    rewritten = []
    count = 0
    for index, node in enumerate(model.graph.node):
        if node.domain not in ("", "ai.onnx") or node.op_type != "ConvTranspose":
            rewritten.append(node)
            continue
        if len(node.input) not in (2, 3) or len(node.output) != 1:
            raise ValueError("ConvTranspose must have weight and optional bias")
        weight_initializer = _resolve_constant_initializer(
            node.input[1], initializers, producers)
        weight = numpy_helper.to_array(weight_initializer)
        if weight.ndim != 4:
            raise ValueError("ConvTranspose rewrite supports only 2-D operators")
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("auto_pad", b"NOTSET") not in (b"", b"NOTSET"):
            raise ValueError("ConvTranspose rewrite requires explicit pads")
        if "output_shape" in attributes:
            raise ValueError("ConvTranspose output_shape is not supported")
        strides = list(attributes.get("strides", [1, 1]))
        if len(strides) != 2 or strides[0] != strides[1] or strides[0] <= 1:
            raise ValueError("ConvTranspose rewrite requires equal spatial strides")
        block_size = int(strides[0])
        dilations = list(attributes.get("dilations", [1, 1]))
        pads = list(attributes.get("pads", [0, 0, 0, 0]))
        output_padding = list(attributes.get("output_padding", [0, 0]))
        if len(dilations) != 2 or len(pads) != 4 or len(output_padding) != 2:
            raise ValueError("ConvTranspose spatial attributes must be 2-D")
        group = int(attributes.get("group", 1))
        input_channels = int(weight.shape[0])
        if input_channels % group:
            raise ValueError("ConvTranspose input channels must divide group")
        output_channels_per_group = int(weight.shape[1])
        output_channels = output_channels_per_group * group
        input_channels_per_group = input_channels // group
        kernel_shape = list(attributes.get("kernel_shape", weight.shape[2:]))
        if kernel_shape != list(weight.shape[2:]):
            raise ValueError("ConvTranspose kernel_shape does not match weights")

        final_pads = [
            dilations[0] * (kernel_shape[0] - 1) - pads[0],
            dilations[1] * (kernel_shape[1] - 1) - pads[1],
            dilations[0] * (kernel_shape[0] - 1) - pads[2]
            + output_padding[0] - (block_size - 1),
            dilations[1] * (kernel_shape[1] - 1) - pads[3]
            + output_padding[1] - (block_size - 1),
        ]
        if any(value < 0 for value in final_pads):
            raise ValueError("ConvTranspose decomposition needs nonnegative Conv pads")
        input_shape = input_shape_overrides.get(
            node.input[0], tensor_shapes.get(node.input[0]))
        if (input_shape is None or len(input_shape) != 4
                or any(value <= 0 for value in input_shape[2:])):
            raise ValueError("ConvTranspose decomposition requires static H and W")
        if int(input_shape[1]) != input_channels:
            raise ValueError(
                "ConvTranspose input shape channel count does not match weights")

        grouped_weight = weight.reshape(
            group, input_channels_per_group, output_channels_per_group,
            kernel_shape[0], kernel_shape[1])
        final_weight = grouped_weight.transpose(0, 2, 1, 3, 4)
        final_weight = final_weight.reshape(
            output_channels, input_channels_per_group,
            kernel_shape[0], kernel_shape[1])[:, :, ::-1, ::-1]
        prefix = node.name or "ConvTranspose_%d" % index
        final_weight_name = _unique_tensor_name(
            prefix + "__conv_weight", occupied)
        model.graph.initializer.append(numpy_helper.from_array(
            np.ascontiguousarray(final_weight), final_weight_name))
        resized = _unique_tensor_name(prefix + "__resized", occupied)
        upsampled = _unique_tensor_name(prefix + "__zero_inserted", occupied)
        resize_scales_name = _unique_tensor_name(
            prefix + "__resize_scales", occupied)
        checkerboard_name = _unique_tensor_name(
            prefix + "__checkerboard", occupied)
        resize_scales = np.asarray(
            [1.0, 1.0, float(block_size), float(block_size)],
            dtype=np.float32)
        checkerboard = np.zeros(
            (1, 1, input_shape[2] * block_size,
             input_shape[3] * block_size), dtype=weight.dtype)
        checkerboard[:, :, ::block_size, ::block_size] = 1
        model.graph.initializer.append(numpy_helper.from_array(
            resize_scales, resize_scales_name))
        model.graph.initializer.append(numpy_helper.from_array(
            checkerboard, checkerboard_name))
        rewritten.extend([
            helper.make_node(
                "Resize", [node.input[0], "", resize_scales_name], [resized],
                name=prefix + "__resize", mode="nearest",
                coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
            helper.make_node(
                "Mul", [resized, checkerboard_name], [upsampled],
                name=prefix + "__zero_insert"),
            helper.make_node(
                "Conv", [upsampled, final_weight_name] + list(node.input[2:]),
                list(node.output), name=prefix + "__conv",
                kernel_shape=kernel_shape, dilations=dilations,
                pads=final_pads, strides=[1, 1], group=group),
        ])
        count += 1
    if count == 0:
        raise ValueError("model contains no decomposable ConvTranspose node")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return count


def decompose_modulated_deform_conv_for_rknn(
        source: Path, output: Path) -> int:
    """Lower the single-channel SPN DCNv2 nodes to standard GridSample ops.

    The propagation layers in the official NLSPN and CompletionFormer exports
    use only one input/output channel, one deformable group, and static spatial
    shapes.  Expanding each kernel point into a bilinear GridSample preserves
    DCNv2 semantics and lets RKNN make an explicit CPU/NPU placement decision
    instead of rejecting the custom ``com.opencorex`` operator.
    """
    source = Path(source)
    output = Path(output)
    model = onnx.load(str(source))
    initializers = {initializer.name: initializer
                    for initializer in model.graph.initializer}
    producers = {
        tensor_name: node
        for node in model.graph.node
        for tensor_name in node.output
    }
    occupied = set(initializers)
    for node in model.graph.node:
        occupied.update(node.input)
        occupied.update(node.output)

    def add_initializer(prefix: str, value: np.ndarray) -> str:
        name = _unique_tensor_name(prefix, occupied)
        model.graph.initializer.append(
            numpy_helper.from_array(np.ascontiguousarray(value), name))
        return name

    rewritten = []
    count = 0
    scale_cache = {}
    base_grid_cache = {}
    for index, node in enumerate(model.graph.node):
        if (node.domain != "com.opencorex"
                or node.op_type != "ModulatedDeformConv2D"):
            rewritten.append(node)
            continue
        if len(node.input) != 5 or len(node.output) != 1:
            raise ValueError("ModulatedDeformConv2D must have five inputs")
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        feature_shape = list(attributes.get("feature_shape", []))
        offset_shape = list(attributes.get("offset_shape", []))
        mask_shape = list(attributes.get("mask_shape", []))
        output_shape = list(attributes.get("output_shape", []))
        weight_shape = list(attributes.get("weight_shape", []))
        if (len(feature_shape) != 4 or len(offset_shape) != 4
                or len(mask_shape) != 4 or len(output_shape) != 4
                or len(weight_shape) != 4):
            raise ValueError("deformable convolution requires static 4-D shapes")
        if (feature_shape[0:2] != [1, 1]
                or output_shape[0:2] != [1, 1]
                or weight_shape[0:2] != [1, 1]
                or int(attributes.get("groups", 1)) != 1
                or int(attributes.get("deformable_groups", 1)) != 1):
            raise ValueError(
                "RKNN deformable-convolution lowering supports one-channel SPN nodes")
        kernel_h, kernel_w = weight_shape[2:]
        point_count = kernel_h * kernel_w
        if offset_shape[1] != 2 * point_count or mask_shape[1] != point_count:
            raise ValueError("offset or mask channel count does not match kernel")
        strides = list(attributes.get("strides", [1, 1]))
        dilations = list(attributes.get("dilations", [1, 1]))
        pads = list(attributes.get("pads", [0, 0, 0, 0]))
        if len(strides) != 2 or len(dilations) != 2 or len(pads) != 4:
            raise ValueError("deformable convolution spatial attributes must be 2-D")
        weight_initializer = _resolve_constant_initializer(
            node.input[3], initializers, producers)
        bias_initializer = _resolve_constant_initializer(
            node.input[4], initializers, producers)
        weight = numpy_helper.to_array(weight_initializer)
        bias = numpy_helper.to_array(bias_initializer)
        if list(weight.shape) != weight_shape or bias.size != 1:
            raise ValueError("deformable convolution constants do not match metadata")

        input_h, input_w = feature_shape[2:]
        output_h, output_w = output_shape[2:]
        prefix = node.name or "ModulatedDeformConv2D_%d" % index
        scale_key = (input_h, input_w, weight.dtype.str)
        scale_name = scale_cache.get(scale_key)
        if scale_name is None:
            scale_name = add_initializer(
                prefix + "__offset_scale",
                np.asarray([2.0 / input_w, 2.0 / input_h], dtype=weight.dtype)
                .reshape(1, 1, 1, 2))
            scale_cache[scale_key] = scale_name
        offset_channels = [
            _unique_tensor_name(
                "%s__offset_channel_%d" % (prefix, channel), occupied)
            for channel in range(2 * point_count)
        ]
        mask_channels = [
            _unique_tensor_name(
                "%s__mask_channel_%d" % (prefix, channel), occupied)
            for channel in range(point_count)
        ]
        offset_split_sizes = add_initializer(
            prefix + "__offset_split_sizes",
            np.ones(2 * point_count, dtype=np.int64))
        mask_split_sizes = add_initializer(
            prefix + "__mask_split_sizes",
            np.ones(point_count, dtype=np.int64))
        rewritten.extend([
            helper.make_node(
                "Split", [node.input[1], offset_split_sizes], offset_channels,
                name=prefix + "__split_offset", axis=1),
            helper.make_node(
                "Split", [node.input[2], mask_split_sizes], mask_channels,
                name=prefix + "__split_mask", axis=1),
        ])
        accumulated = None
        for point in range(point_count):
            kernel_y, kernel_x = divmod(point, kernel_w)
            y = (np.arange(output_h, dtype=np.float32) * strides[0]
                 - pads[0] + kernel_y * dilations[0])
            x = (np.arange(output_w, dtype=np.float32) * strides[1]
                 - pads[1] + kernel_x * dilations[1])
            grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
            base_grid = np.stack([
                (2.0 * grid_x + 1.0) / input_w - 1.0,
                (2.0 * grid_y + 1.0) / input_h - 1.0,
            ], axis=-1).astype(weight.dtype).reshape(
                1, output_h, output_w, 2)
            base_key = (
                input_h, input_w, output_h, output_w,
                strides[0], strides[1], pads[0], pads[1],
                dilations[0], dilations[1], kernel_y, kernel_x,
                weight.dtype.str,
            )
            base_name = base_grid_cache.get(base_key)
            if base_name is None:
                base_name = add_initializer(
                    "%s__base_grid_%d" % (prefix, point), base_grid)
                base_grid_cache[base_key] = base_name
            weight_name = add_initializer(
                "%s__weight_%d" % (prefix, point),
                np.asarray([weight[0, 0, kernel_y, kernel_x]],
                           dtype=weight.dtype))

            offset_y = offset_channels[2 * point]
            offset_x = offset_channels[2 * point + 1]
            mask = mask_channels[point]
            offset_xy = _unique_tensor_name(
                "%s__offset_xy_%d" % (prefix, point), occupied)
            offset_grid = _unique_tensor_name(
                "%s__offset_grid_%d" % (prefix, point), occupied)
            normalized_offset = _unique_tensor_name(
                "%s__normalized_offset_%d" % (prefix, point), occupied)
            grid = _unique_tensor_name(
                "%s__grid_%d" % (prefix, point), occupied)
            sampled = _unique_tensor_name(
                "%s__sampled_%d" % (prefix, point), occupied)
            masked = _unique_tensor_name(
                "%s__masked_%d" % (prefix, point), occupied)
            weighted = _unique_tensor_name(
                "%s__weighted_%d" % (prefix, point), occupied)
            rewritten.extend([
                helper.make_node(
                    "Concat", [offset_x, offset_y], [offset_xy],
                    name="%s__concat_offset_%d" % (prefix, point), axis=1),
                helper.make_node(
                    "Transpose", [offset_xy], [offset_grid],
                    name="%s__transpose_offset_%d" % (prefix, point),
                    perm=[0, 2, 3, 1]),
                helper.make_node(
                    "Mul", [offset_grid, scale_name], [normalized_offset],
                    name="%s__normalize_offset_%d" % (prefix, point)),
                helper.make_node(
                    "Add", [base_name, normalized_offset], [grid],
                    name="%s__make_grid_%d" % (prefix, point)),
                helper.make_node(
                    "GridSample", [node.input[0], grid], [sampled],
                    name="%s__grid_sample_%d" % (prefix, point),
                    align_corners=0, mode="bilinear", padding_mode="zeros"),
                helper.make_node(
                    "Mul", [sampled, mask], [masked],
                    name="%s__apply_mask_%d" % (prefix, point)),
                helper.make_node(
                    "Mul", [masked, weight_name], [weighted],
                    name="%s__apply_weight_%d" % (prefix, point)),
            ])
            if accumulated is None:
                accumulated = weighted
            else:
                combined = _unique_tensor_name(
                    "%s__sum_%d" % (prefix, point), occupied)
                rewritten.append(helper.make_node(
                    "Add", [accumulated, weighted], [combined],
                    name="%s__accumulate_%d" % (prefix, point)))
                accumulated = combined
        rewritten.append(helper.make_node(
            "Add", [accumulated, node.input[4]], list(node.output),
            name=prefix + "__add_bias"))
        count += 1
    if count == 0:
        raise ValueError("model contains no com.opencorex ModulatedDeformConv2D")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx") and opset.version < 16:
            opset.version = 16
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return count
