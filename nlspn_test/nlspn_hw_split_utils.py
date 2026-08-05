from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class InputChunk:
    start: int
    end: int
    conv: nn.Conv2d


def split_conv2d_input_channels(conv: nn.Conv2d, chunk_size: int = 64) -> List[InputChunk]:
    """Split Conv2d along input channels for exact Host-sum deployment.

    Each returned Conv2d has no bias. The original bias must be applied after
    summing all partial outputs, usually through a post-sum affine block.
    """

    if conv.groups != 1:
        raise ValueError("input-channel split only supports groups=1")
    if conv.in_channels % chunk_size != 0:
        raise ValueError(f"in_channels={conv.in_channels} is not divisible by chunk_size={chunk_size}")

    chunks: List[InputChunk] = []
    for start in range(0, conv.in_channels, chunk_size):
        end = start + chunk_size
        part = nn.Conv2d(
            chunk_size,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            groups=1,
            bias=False,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            part.weight.copy_(conv.weight[:, start:end, :, :])
        chunks.append(InputChunk(start=start, end=end, conv=part))
    return chunks


def split_conv2d_input_output_channels(
    conv: nn.Conv2d,
    input_chunk_size: int = 64,
    output_chunk_size: int = 64,
) -> List[Tuple[int, int, List[InputChunk]]]:
    """Split Conv2d into output-channel blocks, each with input-channel chunks."""

    if conv.out_channels % output_chunk_size != 0:
        raise ValueError(
            f"out_channels={conv.out_channels} is not divisible by output_chunk_size={output_chunk_size}"
        )
    blocks = []
    for out_start in range(0, conv.out_channels, output_chunk_size):
        out_end = out_start + output_chunk_size
        sub = nn.Conv2d(
            conv.in_channels,
            output_chunk_size,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            groups=1,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            sub.weight.copy_(conv.weight[out_start:out_end])
            if conv.bias is not None:
                sub.bias.copy_(conv.bias[out_start:out_end])
        blocks.append((out_start, out_end, split_conv2d_input_channels(sub, input_chunk_size)))
    return blocks


def make_affine1x1(
    scale: torch.Tensor,
    bias: torch.Tensor,
    relu: bool = False,
    pad_to_channels: Optional[int] = None,
) -> nn.Sequential:
    """Create a diagonal 1x1 affine Conv, optionally followed by ReLU.

    `pad_to_channels` supports the board rule where 1-channel post-sum heads
    are padded to 8 channels before the RHB affine/ReLU block.
    """

    if scale.ndim != 1 or bias.ndim != 1:
        raise ValueError("scale and bias must be 1D tensors")
    if scale.numel() != bias.numel():
        raise ValueError("scale and bias must have the same length")

    channels = int(scale.numel())
    out_channels = int(pad_to_channels or channels)
    if out_channels < channels:
        raise ValueError("pad_to_channels must be >= number of channels")

    conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        for c in range(channels):
            conv.weight[c, c, 0, 0] = scale[c]
            conv.bias[c] = bias[c]
        for c in range(channels, out_channels):
            conv.weight[c, c, 0, 0] = 1.0
    layers: List[nn.Module] = [conv]
    if relu:
        layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)


def affine_from_conv_bias(conv: nn.Conv2d, relu: bool, pad_to_channels: Optional[int] = None) -> nn.Sequential:
    channels = conv.out_channels
    scale = torch.ones(channels, dtype=conv.weight.dtype, device=conv.weight.device)
    if conv.bias is None:
        bias = torch.zeros(channels, dtype=conv.weight.dtype, device=conv.weight.device)
    else:
        bias = conv.bias.detach().clone()
    return make_affine1x1(scale, bias, relu=relu, pad_to_channels=pad_to_channels)


def affine_from_batchnorm(
    bn: nn.BatchNorm2d,
    conv_bias: Optional[torch.Tensor] = None,
    relu: bool = False,
    pad_to_channels: Optional[int] = None,
) -> nn.Sequential:
    """Fold BatchNorm2d into a post-sum 1x1 affine block."""

    if bn.training:
        raise ValueError("BatchNorm must be in eval mode to fold running stats")
    var = bn.running_var
    mean = bn.running_mean
    gamma = bn.weight if bn.affine else torch.ones_like(var)
    beta = bn.bias if bn.affine else torch.zeros_like(var)
    conv_b = torch.zeros_like(mean) if conv_bias is None else conv_bias
    scale = gamma / torch.sqrt(var + bn.eps)
    bias = beta + (conv_b - mean) * scale
    return make_affine1x1(scale.detach(), bias.detach(), relu=relu, pad_to_channels=pad_to_channels)


def sum_input_channel_partials(x: torch.Tensor, chunks: Sequence[InputChunk]) -> torch.Tensor:
    outputs = []
    for chunk in chunks:
        outputs.append(chunk.conv(x[:, chunk.start:chunk.end]))
    return torch.stack(outputs, dim=0).sum(dim=0)


def pad_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.shape[1] > channels:
        raise ValueError("cannot pad to fewer channels")
    if x.shape[1] == channels:
        return x
    pad = x.new_zeros((x.shape[0], channels - x.shape[1], x.shape[2], x.shape[3]))
    return torch.cat([x, pad], dim=1)
