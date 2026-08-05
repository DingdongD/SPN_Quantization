from __future__ import annotations

import torch.nn as nn
import torch

from .dyspn_hw_aligned import DySPNHWAlignedModel, load_dyspn_hw_checkpoint


def get_nested_attr(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


class SingleInputPart(nn.Module):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        self.part = get_nested_attr(self.net, path)
        self.eval()

    def forward(self, x):
        return self.part(x)


def folded_conv_params(part: nn.Module) -> tuple[nn.Conv2d, torch.Tensor, torch.Tensor]:
    if isinstance(part, nn.Sequential):
        conv = part[0]
        if len(part) >= 2 and isinstance(part[1], nn.BatchNorm2d):
            bn = part[1]
            scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            bias = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.running_mean.detach())
            weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
            folded_bias = (bias - bn.running_mean.detach()) * scale + bn.bias.detach()
            return conv, weight, folded_bias
        return conv, conv.weight.detach(), conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)
    if isinstance(part, nn.Conv2d):
        return part, part.weight.detach(), part.bias.detach() if part.bias is not None else torch.zeros(part.out_channels)
    raise TypeError(f"unsupported part for folded conv: {type(part)!r}")


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    bias = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.running_mean.detach())
    weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    folded_bias = (bias - bn.running_mean.detach()) * scale + bn.bias.detach()
    return weight, folded_bias


class FoldedConvBN(nn.Module):
    def __init__(self, conv_path: str, bn_path: str, relu: bool = False) -> None:
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        conv = get_nested_attr(self.net, conv_path)
        bn = get_nested_attr(self.net, bn_path)
        weight, bias = fold_conv_bn(conv, bn)
        self.conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
            self.conv.bias.copy_(bias)
        self.relu = nn.ReLU(inplace=False) if relu else nn.Identity()
        self.eval()

    def forward(self, x):
        return self.relu(self.conv(x))


class FoldedInputChunkConv(nn.Module):
    def __init__(self, path: str, in_start: int, in_end: int, include_bias: bool) -> None:
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        part = get_nested_attr(self.net, path)
        conv, weight, bias = folded_conv_params(part)
        self.in_start = in_start
        self.in_end = in_end
        self.conv = nn.Conv2d(
            in_end - in_start,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight[:, in_start:in_end])
            self.conv.bias.copy_(bias if include_bias else torch.zeros_like(bias))
        self.eval()

    def forward(self, x):
        return self.conv(x)


class FoldedInputOutputChunkConv(nn.Module):
    def __init__(
        self,
        path: str,
        in_start: int,
        in_end: int,
        out_start: int,
        out_end: int,
        include_bias: bool,
    ) -> None:
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        part = get_nested_attr(self.net, path)
        conv, weight, bias = folded_conv_params(part)
        self.conv = nn.Conv2d(
            in_end - in_start,
            out_end - out_start,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight[out_start:out_end, in_start:in_end])
            self.conv.bias.copy_(bias[out_start:out_end] if include_bias else torch.zeros(out_end - out_start))
        self.eval()

    def forward(self, x):
        return self.conv(x)


class FoldedConvBNInputOutputChunk(nn.Module):
    def __init__(
        self,
        conv_path: str,
        bn_path: str,
        in_start: int,
        in_end: int,
        out_start: int,
        out_end: int,
        include_bias: bool,
        relu: bool = False,
    ) -> None:
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        conv = get_nested_attr(self.net, conv_path)
        bn = get_nested_attr(self.net, bn_path)
        weight, bias = fold_conv_bn(conv, bn)
        self.conv = nn.Conv2d(
            in_end - in_start,
            out_end - out_start,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight[out_start:out_end, in_start:in_end])
            self.conv.bias.copy_(bias[out_start:out_end] if include_bias else torch.zeros(out_end - out_start))
        self.relu = nn.ReLU(inplace=False) if relu else nn.Identity()
        self.eval()

    def forward(self, x):
        return self.relu(self.conv(x))
