"""PyTorch-1.10-compatible Torchvision RAFT-Small inference.

Architecture and tensor math are adapted from Torchvision 0.22.1
torchvision.models.optical_flow.raft and its _utils module. Torchvision is
BSD-3-Clause licensed. Only the official small inference configuration is
included here; training and weight-registry APIs are intentionally omitted.
"""

from __future__ import division

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_WEIGHT_PATH = Path(
    "/root/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth")
EXPECTED_WEIGHT_SHA256 = (
    "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, dilation=1, bias=None):
        if padding is None:
            if isinstance(kernel_size, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                padding = tuple(
                    (value - 1) // 2 * dilation for value in kernel_size)
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, bias=bias)
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super(Conv2dNormActivation, self).__init__(*layers)
        self.out_channels = out_channels


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, stride=1):
        super(BottleneckBlock, self).__init__()
        self.convnormrelu1 = Conv2dNormActivation(
            in_channels, out_channels // 4, norm_layer=norm_layer,
            kernel_size=1, bias=True)
        self.convnormrelu2 = Conv2dNormActivation(
            out_channels // 4, out_channels // 4, norm_layer=norm_layer,
            kernel_size=3, stride=stride, bias=True)
        self.convnormrelu3 = Conv2dNormActivation(
            out_channels // 4, out_channels, norm_layer=norm_layer,
            kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = Conv2dNormActivation(
                in_channels, out_channels, norm_layer=norm_layer,
                kernel_size=1, stride=stride, bias=True,
                activation_layer=None)

    def forward(self, value):
        result = self.convnormrelu1(value)
        result = self.convnormrelu2(result)
        result = self.convnormrelu3(result)
        return self.relu(self.downsample(value) + result)


class FeatureEncoder(nn.Module):
    def __init__(self, layers, norm_layer):
        super(FeatureEncoder, self).__init__()
        if len(layers) != 5:
            raise ValueError("feature encoder requires five layer widths")
        self.convnormrelu = Conv2dNormActivation(
            3, layers[0], norm_layer=norm_layer, kernel_size=7,
            stride=2, bias=True)
        self.layer1 = self._make_2_blocks(
            layers[0], layers[1], norm_layer, 1)
        self.layer2 = self._make_2_blocks(
            layers[1], layers[2], norm_layer, 2)
        self.layer3 = self._make_2_blocks(
            layers[2], layers[3], norm_layer, 2)
        self.conv = nn.Conv2d(layers[3], layers[4], kernel_size=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.output_dim = layers[-1]
        self.downsample_factor = 8

    @staticmethod
    def _make_2_blocks(in_channels, out_channels, norm_layer, first_stride):
        return nn.Sequential(
            BottleneckBlock(
                in_channels, out_channels, norm_layer=norm_layer,
                stride=first_stride),
            BottleneckBlock(
                out_channels, out_channels, norm_layer=norm_layer, stride=1),
        )

    def forward(self, value):
        value = self.convnormrelu(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        return self.conv(value)


class MotionEncoder(nn.Module):
    def __init__(self, in_channels_corr, corr_layers=(96,),
                 flow_layers=(64, 32), out_channels=82):
        super(MotionEncoder, self).__init__()
        self.convcorr1 = Conv2dNormActivation(
            in_channels_corr, corr_layers[0], norm_layer=None, kernel_size=1)
        self.convcorr2 = nn.Identity()
        self.convflow1 = Conv2dNormActivation(
            2, flow_layers[0], norm_layer=None, kernel_size=7)
        self.convflow2 = Conv2dNormActivation(
            flow_layers[0], flow_layers[1], norm_layer=None, kernel_size=3)
        self.conv = Conv2dNormActivation(
            corr_layers[-1] + flow_layers[-1], out_channels - 2,
            norm_layer=None, kernel_size=3)
        self.out_channels = out_channels

    def forward(self, flow, corr_features):
        corr = self.convcorr2(self.convcorr1(corr_features))
        encoded_flow = self.convflow2(self.convflow1(flow))
        merged = self.conv(torch.cat([corr, encoded_flow], dim=1))
        return torch.cat([merged, flow], dim=1)


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super(ConvGRU, self).__init__()
        channels = hidden_size + input_size
        self.convz = nn.Conv2d(
            channels, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(
            channels, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convq = nn.Conv2d(
            channels, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, hidden, value):
        combined = torch.cat([hidden, value], dim=1)
        update = torch.sigmoid(self.convz(combined))
        reset = torch.sigmoid(self.convr(combined))
        candidate = torch.tanh(
            self.convq(torch.cat([reset * hidden, value], dim=1)))
        return (1 - update) * hidden + update * candidate


class RecurrentBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=(3,),
                 padding=(1,)):
        super(RecurrentBlock, self).__init__()
        if len(kernel_size) != 1 or len(padding) != 1:
            raise ValueError("RAFT-Small uses one ConvGRU")
        self.convgru1 = ConvGRU(
            input_size, hidden_size, kernel_size[0], padding[0])
        self.convgru2 = _pass_through_h
        self.hidden_size = hidden_size

    def forward(self, hidden, value):
        hidden = self.convgru1(hidden, value)
        return self.convgru2(hidden, value)


def _pass_through_h(hidden, unused):
    del unused
    return hidden


class FlowHead(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value):
        return self.conv2(self.relu(self.conv1(value)))


class UpdateBlock(nn.Module):
    def __init__(self, motion_encoder, recurrent_block, flow_head):
        super(UpdateBlock, self).__init__()
        self.motion_encoder = motion_encoder
        self.recurrent_block = recurrent_block
        self.flow_head = flow_head
        self.hidden_state_size = recurrent_block.hidden_size

    def forward(self, hidden_state, context, corr_features, flow):
        motion = self.motion_encoder(flow, corr_features)
        value = torch.cat([context, motion], dim=1)
        hidden_state = self.recurrent_block(hidden_state, value)
        return hidden_state, self.flow_head(hidden_state)


def _meshgrid(row, column):
    try:
        return torch.meshgrid(row, column, indexing="ij")
    except TypeError as error:
        if "indexing" not in str(error):
            raise
        return torch.meshgrid(row, column)


def grid_sample(image, absolute_grid, mode="bilinear", align_corners=None):
    height, width = image.shape[-2:]
    x_grid, y_grid = absolute_grid.split([1, 1], dim=-1)
    x_grid = 2 * x_grid / (width - 1) - 1
    if height > 1:
        y_grid = 2 * y_grid / (height - 1) - 1
    normalized_grid = torch.cat([x_grid, y_grid], dim=-1)
    return F.grid_sample(
        image, normalized_grid, mode=mode, align_corners=align_corners)


def make_coords_grid(batch_size, height, width, device="cpu"):
    device = torch.device(device)
    rows, columns = _meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device))
    coords = torch.stack((columns, rows), dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)


def upsample_flow(flow, up_mask=None, factor=8):
    batch_size, channels, height, width = flow.shape
    new_height, new_width = height * factor, width * factor
    if up_mask is None:
        return factor * F.interpolate(
            flow, size=(new_height, new_width), mode="bilinear",
            align_corners=True)
    up_mask = up_mask.view(
        batch_size, 1, 9, factor, factor, height, width)
    up_mask = torch.softmax(up_mask, dim=2)
    unfolded = F.unfold(
        factor * flow, kernel_size=3, padding=1).view(
            batch_size, channels, 9, 1, 1, height, width)
    result = torch.sum(up_mask * unfolded, dim=2)
    return result.permute(0, 1, 4, 2, 5, 3).reshape(
        batch_size, channels, new_height, new_width)


class CorrBlock(nn.Module):
    def __init__(self, num_levels=4, radius=3):
        super(CorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = [torch.tensor(0)]
        self.out_channels = num_levels * (2 * radius + 1) ** 2

    def build_pyramid(self, fmap1, fmap2):
        if fmap1.shape != fmap2.shape:
            raise ValueError("RAFT feature map shapes differ")
        minimum = 2 * (2 ** (self.num_levels - 1))
        if any(size < minimum for size in fmap1.shape[-2:]):
            raise ValueError("RAFT feature maps are too small")
        corr = self._compute_corr_volume(fmap1, fmap2)
        batch_size, height, width, channels, _, _ = corr.shape
        corr = corr.reshape(
            batch_size * height * width, channels, height, width)
        self.corr_pyramid = [corr]
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr)

    def index_pyramid(self, centroids_coords):
        side = 2 * self.radius + 1
        di = torch.linspace(
            -self.radius, self.radius, side,
            device=centroids_coords.device,
            dtype=centroids_coords.dtype)
        dj = torch.linspace(
            -self.radius, self.radius, side,
            device=centroids_coords.device,
            dtype=centroids_coords.dtype)
        delta_i, delta_j = _meshgrid(di, dj)
        delta = torch.stack([delta_i, delta_j], dim=-1).view(
            1, side, side, 2)
        batch_size, _, height, width = centroids_coords.shape
        centroids = centroids_coords.permute(0, 2, 3, 1).reshape(
            batch_size * height * width, 1, 1, 2)
        indexed = []
        for corr in self.corr_pyramid:
            sampled = grid_sample(
                corr, centroids + delta, mode="bilinear",
                align_corners=True).view(batch_size, height, width, -1)
            indexed.append(sampled)
            centroids = centroids / 2
        result = torch.cat(indexed, dim=-1).permute(
            0, 3, 1, 2).contiguous()
        expected = (batch_size, self.out_channels, height, width)
        if result.shape != expected:
            raise ValueError("RAFT correlation output shape is invalid")
        return result

    @staticmethod
    def _compute_corr_volume(fmap1, fmap2):
        batch_size, channels, height, width = fmap1.shape
        fmap1 = fmap1.view(batch_size, channels, height * width)
        fmap2 = fmap2.view(batch_size, channels, height * width)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch_size, height, width, 1, height, width)
        scale = torch.sqrt(torch.tensor(
            channels, dtype=corr.dtype, device=corr.device))
        return corr / scale


class RAFT(nn.Module):
    def __init__(self, feature_encoder, context_encoder, corr_block,
                 update_block, mask_predictor=None):
        super(RAFT, self).__init__()
        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder
        self.corr_block = corr_block
        self.update_block = update_block
        self.mask_predictor = mask_predictor

    def forward(self, image1, image2, num_flow_updates=12):
        batch_size, _, height, width = image1.shape
        if image2.shape[-2:] != (height, width):
            raise ValueError("RAFT input image shapes differ")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("RAFT image dimensions must be divisible by 8")
        feature_maps = self.feature_encoder(
            torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.chunk(feature_maps, chunks=2, dim=0)
        if fmap1.shape[-2:] != (height // 8, width // 8):
            raise ValueError("RAFT feature encoder downsample is invalid")
        self.corr_block.build_pyramid(fmap1, fmap2)
        context_out = self.context_encoder(image1)
        hidden_size = self.update_block.hidden_state_size
        context_channels = context_out.shape[1] - hidden_size
        if context_channels <= 0:
            raise ValueError("RAFT context encoder output is invalid")
        hidden_state, context = torch.split(
            context_out, [hidden_size, context_channels], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = F.relu(context)
        coords0 = make_coords_grid(
            batch_size, height // 8, width // 8, fmap1.device)
        coords1 = make_coords_grid(
            batch_size, height // 8, width // 8, fmap1.device)
        predictions = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach()
            corr_features = self.corr_block.index_pyramid(coords1)
            flow = coords1 - coords0
            hidden_state, delta_flow = self.update_block(
                hidden_state, context, corr_features, flow)
            coords1 = coords1 + delta_flow
            up_mask = (
                None if self.mask_predictor is None
                else self.mask_predictor(hidden_state))
            predictions.append(upsample_flow(coords1 - coords0, up_mask))
        return predictions


def raft_small():
    feature_encoder = FeatureEncoder(
        layers=(32, 32, 64, 96, 128), norm_layer=nn.InstanceNorm2d)
    context_encoder = FeatureEncoder(
        layers=(32, 32, 64, 96, 160), norm_layer=None)
    corr_block = CorrBlock(num_levels=4, radius=3)
    motion_encoder = MotionEncoder(
        in_channels_corr=corr_block.out_channels,
        corr_layers=(96,), flow_layers=(64, 32), out_channels=82)
    recurrent_block = RecurrentBlock(
        input_size=motion_encoder.out_channels + (160 - 96),
        hidden_size=96, kernel_size=(3,), padding=(1,))
    flow_head = FlowHead(in_channels=96, hidden_size=128)
    update_block = UpdateBlock(
        motion_encoder, recurrent_block, flow_head)
    return RAFT(
        feature_encoder, context_encoder, corr_block, update_block,
        mask_predictor=None)


def load_official_weights(model, path=DEFAULT_WEIGHT_PATH):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("official RAFT-Small weight digest mismatch")
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def prepare_backward_pair(current_rgb, previous_rgb):
    if current_rgb.shape != previous_rgb.shape:
        raise ValueError("current and previous RGB shapes differ")
    if current_rgb.ndim != 4 or current_rgb.shape[1:] != (3, 228, 304):
        raise ValueError("RAFT RGB must have shape [batch, 3, 228, 304]")
    current = F.pad(current_rgb, (0, 0, 2, 2), mode="replicate")
    previous = F.pad(previous_rgb, (0, 0, 2, 2), mode="replicate")
    return (
        (current * 2.0 - 1.0).contiguous(),
        (previous * 2.0 - 1.0).contiguous(),
    )


def predict_backward_flow(model, current_rgb, previous_rgb):
    current, previous = prepare_backward_pair(current_rgb, previous_rgb)
    flow = model(current, previous, num_flow_updates=12)[-1][:, :, 2:-2]
    if flow.shape != (current.shape[0], 2, 228, 304):
        raise ValueError("RAFT-Small returned invalid backward-flow shape")
    if not torch.isfinite(flow).all():
        raise ValueError("RAFT-Small returned non-finite backward flow")
    return flow

