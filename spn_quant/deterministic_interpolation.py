"""Deterministic CompletionFormer bilinear input gradients for QAT."""

from __future__ import annotations

import math
from types import MethodType
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def _reverse_interpolation_map(
        input_size: int,
        output_size: int,
        align_corners: bool,
        device: torch.device,
        dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    contributions = [[] for _ in range(input_size)]
    for output_index in range(output_size):
        if align_corners:
            source = 0.0 if output_size == 1 else \
                output_index * (input_size - 1) / float(output_size - 1)
        else:
            source = (output_index + 0.5) * input_size / \
                float(output_size) - 0.5
            source = max(source, 0.0)
        lower = min(int(math.floor(source)), input_size - 1)
        upper = min(lower + 1, input_size - 1)
        if lower == upper:
            contributions[lower].append((output_index, 1.0))
        else:
            upper_weight = source - lower
            contributions[lower].append((output_index, 1.0 - upper_weight))
            contributions[upper].append((output_index, upper_weight))
    width = max(len(rows) for rows in contributions)
    index_rows = []
    weight_rows = []
    for rows in contributions:
        padding = width - len(rows)
        index_rows.append(
            [output_index for output_index, weight in rows] + [0] * padding)
        weight_rows.append(
            [weight for output_index, weight in rows] + [0.0] * padding)
    indices = torch.tensor(
        index_rows, device=device, dtype=torch.long)
    weights = torch.tensor(
        weight_rows, device=device, dtype=dtype)
    return indices, weights


def _transpose_interpolation_axis(
        gradient: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor) -> torch.Tensor:
    input_size = int(indices.shape[0])
    selected = torch.index_select(
        gradient, -1, indices.reshape(-1))
    selected = selected.reshape(
        gradient.shape[:-1] + (input_size, indices.shape[1]))
    weight_shape = (1,) * (selected.ndim - 2) + weights.shape
    return (selected * weights.reshape(weight_shape)).sum(dim=-1)


class _DeterministicBilinear2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, output_size, align_corners):
        ctx.input_size = tuple(int(size) for size in value.shape[-2:])
        ctx.output_size = tuple(int(size) for size in output_size)
        ctx.align_corners = bool(align_corners)
        height_indices, height_weights = _reverse_interpolation_map(
            ctx.input_size[0],
            ctx.output_size[0],
            ctx.align_corners,
            value.device,
            value.dtype,
        )
        width_indices, width_weights = _reverse_interpolation_map(
            ctx.input_size[1],
            ctx.output_size[1],
            ctx.align_corners,
            value.device,
            value.dtype,
        )
        ctx.save_for_backward(
            height_indices,
            height_weights,
            width_indices,
            width_weights,
        )
        return F.interpolate(
            value,
            size=ctx.output_size,
            mode="bilinear",
            align_corners=ctx.align_corners,
        )

    @staticmethod
    def backward(ctx, gradient):
        height_indices, height_weights, width_indices, width_weights = \
            ctx.saved_tensors
        height_gradient = _transpose_interpolation_axis(
            gradient.permute(0, 1, 3, 2),
            height_indices,
            height_weights,
        )
        input_gradient = _transpose_interpolation_axis(
            height_gradient.permute(0, 1, 3, 2),
            width_indices,
            width_weights,
        )
        return input_gradient, None, None


def deterministic_bilinear2d(
        value: torch.Tensor,
        output_size: Sequence[int],
        align_corners: bool) -> torch.Tensor:
    if value.ndim != 4 or not value.is_floating_point():
        raise ValueError("deterministic bilinear input must be floating NCHW")
    size = tuple(int(dimension) for dimension in output_size)
    if len(size) != 2 or any(dimension <= 0 for dimension in size):
        raise ValueError("deterministic bilinear output size is invalid")
    if not isinstance(align_corners, bool):
        raise TypeError("deterministic bilinear align_corners must be bool")
    return _DeterministicBilinear2d.apply(value, size, align_corners)


def deterministic_adaptive_max_pool2d_1(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or not value.is_floating_point():
        raise ValueError(
            "deterministic adaptive max-pool input must be floating NCHW")
    maximum, indices = torch.max(value.flatten(2), dim=2, keepdim=True)
    del indices
    return maximum.unsqueeze(-1)


def _decoder_forward(module, value, size):
    value = deterministic_bilinear2d(
        value, size, module.align_corners)
    value = module.conv(value)
    if module.bn is not None:
        value = module.bn(value)
    if module.relu is not None:
        value = module.relu(value)
    return value


def _adaptive_max_pool_forward(module, value):
    del module
    return deterministic_adaptive_max_pool2d_1(value)


def _concat(backbone, decoder, encoder, dim=1):
    decoder = deterministic_bilinear2d(
        decoder, encoder.shape[-2:], backbone.concat_align_corners)
    return torch.cat((decoder, encoder), dim=dim)


def _position_embedding(former, pos_embed, patch_embed, height, width):
    if height * width == former.patch_embed1.num_patches:
        return pos_embed
    value = pos_embed.reshape(
        1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2)
    value = deterministic_bilinear2d(value, (height, width), False)
    return value.reshape(1, -1, height * width).permute(0, 2, 1)


def install_completionformer_qat_interpolation(model) -> Dict[str, object]:
    if type(model).__name__ != "CompletionFormer":
        raise TypeError("deterministic interpolation requires CompletionFormer")
    backbone = model.backbone
    if type(backbone).__name__ != "Backbone" or \
            type(backbone.former).__name__ != "PVT":
        raise TypeError("CompletionFormer interpolation topology changed")
    if not isinstance(backbone.use_interpolate_decoder, bool):
        raise TypeError("CompletionFormer decoder topology flag changed")
    channel_attention_modules = tuple(
        name for name, module in model.named_modules()
        if type(module).__name__ == "ChannelAttention")
    expected_max_pool_modules = tuple(
        name + ".max_pool" for name in channel_attention_modules)
    if not channel_attention_modules or any(
            not name.startswith("backbone.")
            for name in channel_attention_modules):
        raise RuntimeError("CompletionFormer channel attention is missing")
    decoder_modules = []
    adaptive_max_pool_modules = []
    for name, module in model.named_modules():
        if type(module).__name__ == "AdaptiveMaxPool2d":
            if module.output_size not in (1, (1, 1)) or \
                    name not in expected_max_pool_modules:
                raise ValueError(
                    "CompletionFormer adaptive max-pool contract changed")
            module.forward = MethodType(_adaptive_max_pool_forward, module)
            adaptive_max_pool_modules.append(name)
        if type(module).__name__ != "InterpolateConvBNReLU":
            continue
        if module.mode != "bilinear" or \
                not isinstance(module.align_corners, bool):
            raise ValueError(
                "CompletionFormer decoder interpolation contract changed")
        module.forward = MethodType(_decoder_forward, module)
        decoder_modules.append(name)
    if backbone.use_interpolate_decoder and not decoder_modules:
        raise RuntimeError("CompletionFormer interpolation decoder is missing")
    if not backbone.use_interpolate_decoder and decoder_modules:
        raise RuntimeError(
            "CompletionFormer transpose-conv decoder has interpolation modules")
    if tuple(adaptive_max_pool_modules) != expected_max_pool_modules:
        raise RuntimeError("CompletionFormer channel max-pool coverage changed")
    backbone._concat = MethodType(_concat, backbone)
    backbone.former._get_pos_embed = MethodType(
        _position_embedding, backbone.former)
    return {
        "implementation":
            "official_forward_deterministic_input_gradient",
        "decoder_modules": decoder_modules,
        "adaptive_max_pool_modules": adaptive_max_pool_modules,
        "concat_method": "backbone._concat",
        "position_method": "backbone.former._get_pos_embed",
    }
