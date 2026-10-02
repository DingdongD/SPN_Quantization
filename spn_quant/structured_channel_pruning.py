"""Interface-preserving structured channel pruning helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


def _top_channels(score: torch.Tensor, keep_channels: int) -> torch.Tensor:
    width = int(score.numel())
    if keep_channels < 1 or keep_channels >= width:
        raise ValueError(
            "keep_channels must be between 1 and %d, got %d" %
            (width - 1, keep_channels))
    selected = torch.argsort(score, descending=True)[:keep_channels]
    return torch.sort(selected).values


def _copy_parameter(target: torch.Tensor, source: torch.Tensor) -> None:
    with torch.no_grad():
        target.copy_(source)


def _new_conv2d(source: nn.Conv2d, out_channels: int,
                in_channels: int | None = None) -> nn.Conv2d:
    if in_channels is None:
        in_channels = source.in_channels
    target = nn.Conv2d(
        in_channels, out_channels, source.kernel_size,
        stride=source.stride, padding=source.padding,
        dilation=source.dilation, groups=source.groups,
        bias=source.bias is not None, padding_mode=source.padding_mode)
    target = target.to(device=source.weight.device, dtype=source.weight.dtype)
    target.train(source.training)
    target.weight.requires_grad_(source.weight.requires_grad)
    if target.bias is not None:
        target.bias.requires_grad_(source.bias.requires_grad)
    return target


def _new_deconv2d(source: nn.ConvTranspose2d,
                  in_channels: int) -> nn.ConvTranspose2d:
    target = nn.ConvTranspose2d(
        in_channels, source.out_channels, source.kernel_size,
        stride=source.stride, padding=source.padding,
        output_padding=source.output_padding, groups=source.groups,
        bias=source.bias is not None, dilation=source.dilation,
        padding_mode=source.padding_mode)
    target = target.to(device=source.weight.device, dtype=source.weight.dtype)
    target.train(source.training)
    target.weight.requires_grad_(source.weight.requires_grad)
    if target.bias is not None:
        target.bias.requires_grad_(source.bias.requires_grad)
    return target


def _new_batch_norm(source: nn.BatchNorm2d,
                    num_features: int) -> nn.BatchNorm2d:
    target = nn.BatchNorm2d(
        num_features, eps=source.eps, momentum=source.momentum,
        affine=source.affine, track_running_stats=source.track_running_stats)
    reference = source.weight if source.affine else source.running_mean
    target = target.to(device=reference.device, dtype=reference.dtype)
    target.train(source.training)
    return target


def prune_conv_bn_deconv_bridge(
        encoder_tail: nn.Sequential,
        decoder_head: nn.Sequential,
        keep_channels: int) -> torch.Tensor:
    """Prune a Conv-BN bridge while preserving its external tensor shapes."""
    conv = encoder_tail[0]
    batch_norm = encoder_tail[1]
    deconv = decoder_head[0]
    if not isinstance(conv, nn.Conv2d) or \
            not isinstance(batch_norm, nn.BatchNorm2d) or \
            not isinstance(deconv, nn.ConvTranspose2d):
        raise TypeError("expected Conv2d-BatchNorm2d and ConvTranspose2d")
    if conv.groups != 1 or deconv.groups != 1:
        raise ValueError("grouped bridge convolutions are not supported")
    if conv.out_channels != deconv.in_channels:
        raise ValueError("bridge channel dimensions do not match")

    incoming = conv.weight.detach().float().flatten(1).norm(dim=1)
    outgoing = deconv.weight.detach().float().flatten(1).norm(dim=1)
    score = incoming * outgoing
    if batch_norm.affine:
        score = score * batch_norm.weight.detach().float().abs()
    selected = _top_channels(score, keep_channels)

    new_conv = _new_conv2d(conv, keep_channels)
    _copy_parameter(new_conv.weight, conv.weight[selected])
    if new_conv.bias is not None:
        _copy_parameter(new_conv.bias, conv.bias[selected])

    new_batch_norm = _new_batch_norm(batch_norm, keep_channels)
    if batch_norm.affine:
        _copy_parameter(new_batch_norm.weight, batch_norm.weight[selected])
        _copy_parameter(new_batch_norm.bias, batch_norm.bias[selected])
        new_batch_norm.weight.requires_grad_(batch_norm.weight.requires_grad)
        new_batch_norm.bias.requires_grad_(batch_norm.bias.requires_grad)
    if batch_norm.track_running_stats:
        _copy_parameter(
            new_batch_norm.running_mean, batch_norm.running_mean[selected])
        _copy_parameter(
            new_batch_norm.running_var, batch_norm.running_var[selected])
        _copy_parameter(
            new_batch_norm.num_batches_tracked,
            batch_norm.num_batches_tracked)

    new_deconv = _new_deconv2d(deconv, keep_channels)
    _copy_parameter(new_deconv.weight, deconv.weight[selected])
    if new_deconv.bias is not None:
        _copy_parameter(new_deconv.bias, deconv.bias)

    encoder_tail[0] = new_conv
    encoder_tail[1] = new_batch_norm
    decoder_head[0] = new_deconv
    return selected.detach().cpu()


def prune_mlp_hidden(mlp: nn.Module,
                     keep_channels: int) -> torch.Tensor:
    """Prune an MLP hidden dimension without changing its public width."""
    fc1 = mlp.fc1
    fc2 = mlp.fc2
    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        raise TypeError("expected an MLP with Linear fc1 and fc2")
    if fc1.out_features != fc2.in_features:
        raise ValueError("MLP hidden dimensions do not match")

    incoming = fc1.weight.detach().float().norm(dim=1)
    outgoing = fc2.weight.detach().float().norm(dim=0)
    selected = _top_channels(incoming * outgoing, keep_channels)

    new_fc1 = nn.Linear(
        fc1.in_features, keep_channels, bias=fc1.bias is not None)
    new_fc2 = nn.Linear(
        keep_channels, fc2.out_features, bias=fc2.bias is not None)
    new_fc1 = new_fc1.to(device=fc1.weight.device, dtype=fc1.weight.dtype)
    new_fc2 = new_fc2.to(device=fc2.weight.device, dtype=fc2.weight.dtype)
    _copy_parameter(new_fc1.weight, fc1.weight[selected])
    _copy_parameter(new_fc2.weight, fc2.weight[:, selected])
    if new_fc1.bias is not None:
        _copy_parameter(new_fc1.bias, fc1.bias[selected])
    if new_fc2.bias is not None:
        _copy_parameter(new_fc2.bias, fc2.bias)
    new_fc1.train(fc1.training)
    new_fc2.train(fc2.training)
    new_fc1.weight.requires_grad_(fc1.weight.requires_grad)
    new_fc2.weight.requires_grad_(fc2.weight.requires_grad)
    if new_fc1.bias is not None:
        new_fc1.bias.requires_grad_(fc1.bias.requires_grad)
    if new_fc2.bias is not None:
        new_fc2.bias.requires_grad_(fc2.bias.requires_grad)

    mlp.fc1 = new_fc1
    mlp.fc2 = new_fc2
    return selected.detach().cpu()


def prune_basic_block_hidden(block: nn.Module,
                             keep_channels: int) -> torch.Tensor:
    """Prune a two-convolution residual branch without changing its I/O."""
    conv1 = block.conv1
    batch_norm = block.bn1
    conv2 = block.conv2
    if not isinstance(conv1, nn.Conv2d) or \
            not isinstance(batch_norm, nn.BatchNorm2d) or \
            not isinstance(conv2, nn.Conv2d):
        raise TypeError("expected Conv2d-BatchNorm2d-Conv2d basic block")
    if conv1.groups != 1 or conv2.groups != 1:
        raise ValueError("grouped basic-block convolutions are not supported")
    if conv1.out_channels != conv2.in_channels:
        raise ValueError("basic-block hidden dimensions do not match")

    incoming = conv1.weight.detach().float().flatten(1).norm(dim=1)
    outgoing = conv2.weight.detach().float().permute(1, 0, 2, 3) \
        .flatten(1).norm(dim=1)
    score = incoming * outgoing
    if batch_norm.affine:
        score = score * batch_norm.weight.detach().float().abs()
    selected = _top_channels(score, keep_channels)

    new_conv1 = _new_conv2d(conv1, keep_channels)
    _copy_parameter(new_conv1.weight, conv1.weight[selected])
    if new_conv1.bias is not None:
        _copy_parameter(new_conv1.bias, conv1.bias[selected])

    new_batch_norm = _new_batch_norm(batch_norm, keep_channels)
    if batch_norm.affine:
        _copy_parameter(new_batch_norm.weight, batch_norm.weight[selected])
        _copy_parameter(new_batch_norm.bias, batch_norm.bias[selected])
        new_batch_norm.weight.requires_grad_(batch_norm.weight.requires_grad)
        new_batch_norm.bias.requires_grad_(batch_norm.bias.requires_grad)
    if batch_norm.track_running_stats:
        _copy_parameter(
            new_batch_norm.running_mean, batch_norm.running_mean[selected])
        _copy_parameter(
            new_batch_norm.running_var, batch_norm.running_var[selected])
        _copy_parameter(
            new_batch_norm.num_batches_tracked,
            batch_norm.num_batches_tracked)

    new_conv2 = _new_conv2d(
        conv2, conv2.out_channels, in_channels=keep_channels)
    _copy_parameter(new_conv2.weight, conv2.weight[:, selected])
    if new_conv2.bias is not None:
        _copy_parameter(new_conv2.bias, conv2.bias)

    block.conv1 = new_conv1
    block.bn1 = new_batch_norm
    block.conv2 = new_conv2
    return selected.detach().cpu()
