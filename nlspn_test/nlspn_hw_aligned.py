from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


TensorDict = Dict[str, torch.Tensor]


@dataclass
class NLSPNHWConfig:
    input_height: int = 128
    input_width: int = 128
    network: str = "resnet18"
    from_scratch: bool = True
    pretrained_resnet_path: Optional[str] = None
    conf_prop: bool = True
    preserve_input: bool = False
    prop_time: int = 6
    prop_kernel: int = 3
    max_depth: float = 10.0
    enable_pred_init_calibrator: bool = False
    pred_init_calibrator_weight: float = 1.0
    pred_init_calibrator_bias: float = 0.0
    allow_approx_fixed_neighbor: bool = False


def conv_bn_relu(
    ch_in: int,
    ch_out: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    bn: bool = True,
    relu: bool = True,
) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn)
    ]
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResizeConvBNReLU(nn.Module):
    """Hardware-aligned replacement for ConvTranspose2d blocks.

    The resize runs in Host glue for deployment. The Conv/BN/ReLU part is the
    RHB candidate. This is a trainable architecture replacement, not an exact
    ConvTranspose2d decomposition.
    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel: int = 3,
        scale_factor: int = 2,
        padding: int = 1,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = conv_bn_relu(ch_in, ch_out, kernel, 1, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class FixedNeighborPropagation(nn.Module):
    """Approximate fixed-neighbor replacement for NLSPN deformable sampling.

    Guidance predicts eight neighbor weights. We normalize by L1 magnitude and
    iterate a fixed 3x3 shift-sum. This is not strict/ref NLSPN because it
    removes learned deformable offsets. Use only for explicitly retrained
    approximate hardware-aligned experiments.
    """

    def __init__(self, num_neighbors: int = 8, prop_time: int = 6) -> None:
        super().__init__()
        if num_neighbors != 8:
            raise ValueError("FixedNeighborPropagation expects 8 neighbors")
        self.num_neighbors = num_neighbors
        self.prop_time = prop_time
        self.offsets: Tuple[Tuple[int, int], ...] = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )

    def _shift(self, x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
        y0 = 1 + dy
        x0 = 1 + dx
        return padded[:, :, y0:y0 + x.shape[-2], x0:x0 + x.shape[-1]]

    def forward(
        self,
        feat_init: torch.Tensor,
        guidance: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        feat_fix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        aff = torch.tanh(guidance)
        denom = aff.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
        aff = aff / denom
        if confidence is not None:
            aff = aff * confidence

        aff_ref = 1.0 - aff.sum(dim=1, keepdim=True)
        mask_fix = None
        if feat_fix is not None:
            mask_fix = (feat_fix > 0.0).to(feat_init.dtype)

        feat = feat_init
        intermediates: List[torch.Tensor] = []
        for _ in range(self.prop_time):
            if mask_fix is not None:
                feat = feat * (1.0 - mask_fix) + feat_fix * mask_fix
            accum = aff_ref * feat
            for idx, (dy, dx) in enumerate(self.offsets):
                accum = accum + aff[:, idx:idx + 1] * self._shift(feat, dy, dx)
            feat = accum
            intermediates.append(feat)
        return feat, intermediates, aff


def _build_resnet_layers(network: str, pretrained_path: Optional[str]) -> nn.Module:
    if network == "resnet18":
        net = resnet18(weights=None)
    elif network == "resnet34":
        net = resnet34(weights=None)
    else:
        raise ValueError(f"unsupported network: {network}")
    if pretrained_path:
        state = torch.load(pretrained_path, map_location="cpu")
        net.load_state_dict(state)
    return net


def load_nlspn_hw_checkpoint(model: nn.Module, ckpt_path: Optional[str] = None, strict: bool = True) -> None:
    path = ckpt_path or os.environ.get("NLSPN_HW_CKPT", "")
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if strict:
        model_keys = set(model.state_dict().keys())
        state = {key: value for key, value in state.items() if key in model_keys}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[NLSPN_HW_CKPT] missing={missing} unexpected={unexpected}")


class NLSPNHWAlignedModel(nn.Module):
    """NLSPN-style depth completion model aligned to current RHB rules."""

    def __init__(self, config: Optional[NLSPNHWConfig] = None) -> None:
        super().__init__()
        self.config = config or NLSPNHWConfig()
        self.num_neighbors = self.config.prop_kernel * self.config.prop_kernel - 1

        self.conv1_rgb = conv_bn_relu(3, 48, 3, 1, 1, bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, 3, 1, 1, bn=False)

        net = _build_resnet_layers(
            self.config.network,
            None if self.config.from_scratch else self.config.pretrained_resnet_path,
        )
        self.conv2 = net.layer1
        self.conv3 = net.layer2
        self.conv4 = net.layer3
        self.conv5 = net.layer4

        self.conv6 = conv_bn_relu(512, 512, 3, stride=2, padding=1)

        self.dec5 = ResizeConvBNReLU(512, 256)
        self.dec4 = ResizeConvBNReLU(256 + 512, 128)
        self.dec3 = ResizeConvBNReLU(128 + 256, 64)
        self.dec2 = ResizeConvBNReLU(64 + 128, 64)

        self.id_dec1 = conv_bn_relu(64 + 64, 64, 3, 1, 1)
        self.id_dec0 = conv_bn_relu(64 + 64, 1, 3, 1, 1, bn=False, relu=True)
        if self.config.enable_pred_init_calibrator:
            self.pred_init_calibrator = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
            with torch.no_grad():
                self.pred_init_calibrator.weight.fill_(float(self.config.pred_init_calibrator_weight))
                self.pred_init_calibrator.bias.fill_(float(self.config.pred_init_calibrator_bias))
        else:
            self.pred_init_calibrator = nn.Identity()

        self.gd_dec1 = conv_bn_relu(64 + 64, 64, 3, 1, 1)
        self.gd_dec0 = conv_bn_relu(64 + 64, self.num_neighbors, 3, 1, 1, bn=False, relu=False)

        if self.config.conf_prop:
            self.cf_dec1 = conv_bn_relu(64 + 64, 32, 3, 1, 1)
            self.cf_dec0_conv = nn.Conv2d(32 + 64, 1, kernel_size=3, stride=1, padding=1)
        else:
            self.cf_dec1 = None
            self.cf_dec0_conv = None

        self.prop_layer = FixedNeighborPropagation(self.num_neighbors, self.config.prop_time)

    def _concat(self, fd: torch.Tensor, fe: torch.Tensor, dim: int = 1) -> torch.Tensor:
        _, _, hd, wd = fd.shape
        _, _, he, we = fe.shape
        if hd > he:
            fd = fd[:, :, :he, :]
        if wd > we:
            fd = fd[:, :, :, :we]
        if fd.shape[-2:] != fe.shape[-2:]:
            fd = F.interpolate(fd, size=fe.shape[-2:], mode="nearest")
        return torch.cat((fd, fe), dim=dim)

    def forward_features(self, rgb: torch.Tensor, dep: torch.Tensor) -> TensorDict:
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)
        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        return {
            "fe1": fe1,
            "fe2": fe2,
            "fe3": fe3,
            "fe4": fe4,
            "fe5": fe5,
            "fe6": fe6,
            "fd2": fd2,
        }

    def forward_heads(self, feats: TensorDict) -> TensorDict:
        fe1 = feats["fe1"]
        fe2 = feats["fe2"]
        fd2 = feats["fd2"]

        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))
        pred_init = self.pred_init_calibrator(pred_init)

        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guidance = self.gd_dec0(self._concat(gd_fd1, fe1))

        confidence_logits = None
        confidence = None
        if self.config.conf_prop and self.cf_dec1 is not None and self.cf_dec0_conv is not None:
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence_logits = self.cf_dec0_conv(self._concat(cf_fd1, fe1))
            confidence = torch.sigmoid(confidence_logits)

        return {
            "pred_init": pred_init,
            "guidance": guidance,
            "confidence_logits": confidence_logits,
            "confidence": confidence,
        }

    def forward(self, sample: TensorDict) -> TensorDict:
        if not self.config.allow_approx_fixed_neighbor:
            raise RuntimeError(
                "NLSPNHWAlignedModel.forward() would run approximate fixed-neighbor propagation. "
                "For strict/ref NLSPN, call forward_features()/forward_heads() and run original "
                "NLSPN_ECCV20 propagation on Host. Set allow_approx_fixed_neighbor=True only for "
                "explicit retrain-required approximate experiments."
            )
        rgb = sample["rgb"]
        dep = sample["dep"]
        feats = self.forward_features(rgb, dep)
        heads = self.forward_heads(feats)
        pred, pred_inter, aff = self.prop_layer(
            heads["pred_init"],
            heads["guidance"],
            heads["confidence"],
            dep if self.config.preserve_input else None,
        )
        pred = pred.clamp_min(0.0).clamp_max(self.config.max_depth)
        return {
            "pred": pred,
            "pred_init": heads["pred_init"],
            "pred_inter": pred_inter,
            "guidance": heads["guidance"],
            "aff": aff,
            "confidence": heads["confidence"],
            "confidence_logits": heads["confidence_logits"],
        }


class Model(nn.Module):
    """cv_onnx-compatible wrapper returning Conv-head tensors only.

    The RHB deployment should validate Conv-heavy regions first. The Host
    runner applies sigmoid. Strict/ref final depth must use the original
    NLSPN_ECCV20 deformable propagation, not this file's approximate
    fixed-neighbor module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = NLSPNHWAlignedModel(NLSPNHWConfig())
        load_nlspn_hw_checkpoint(self.model, strict=True)

    def forward(self, rgb: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(rgb, dep)
        heads = self.model.forward_heads(feats)
        confidence_logits = heads["confidence_logits"]
        if confidence_logits is None:
            confidence_logits = torch.zeros_like(heads["pred_init"])
        return torch.cat((heads["pred_init"], heads["guidance"], confidence_logits), dim=1)


ifmap_sz = [(3, 128, 128), (1, 128, 128)]
op_version = 18
