from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nlspn_hw_aligned import NLSPNHWAlignedModel, NLSPNHWConfig, TensorDict


@dataclass
class NLSPNHWStrictRefConfig(NLSPNHWConfig):
    affinity: str = "TGASS"
    affinity_gamma: float = 0.5
    legacy: bool = False


class GridSampleNLSPNPropagation(nn.Module):
    """Differentiable strict/ref NLSPN propagation.

    This mirrors `NLSPN_ECCV20/src/model/nlspnmodel.py::NLSPN` for the
    ch_f=1, k_f=3 case, but replaces the custom DCNv2 op with PyTorch
    `grid_sample`. The state_dict intentionally keeps original-compatible keys
    such as `prop_layer.conv_offset_aff.*`.
    """

    def __init__(
        self,
        num_neighbors: int,
        prop_time: int,
        affinity: str = "TGASS",
        affinity_gamma: float = 0.5,
        conf_prop: bool = True,
        preserve_input: bool = True,
        legacy: bool = False,
    ) -> None:
        super().__init__()
        if num_neighbors != 8:
            raise ValueError("strict NLSPN currently expects k=3 and 8 non-center neighbors")
        if affinity not in {"AS", "ASS", "TC", "TGASS"}:
            raise ValueError(f"unsupported affinity: {affinity}")
        self.num = num_neighbors
        self.idx_ref = self.num // 2
        self.prop_time = int(prop_time)
        self.affinity = affinity
        self.conf_prop = bool(conf_prop)
        self.preserve_input = bool(preserve_input)
        self.legacy = bool(legacy)
        self.k_g = 3
        self.k_f = 3
        self.padding = 1
        self.conv_offset_aff = nn.Conv2d(num_neighbors, 3 * num_neighbors, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.zeros_(self.conv_offset_aff.weight)
        nn.init.zeros_(self.conv_offset_aff.bias)
        if affinity == "TGASS":
            self.aff_scale_const = nn.Parameter(affinity_gamma * num_neighbors * torch.ones(1))
        elif affinity == "TC":
            self.aff_scale_const = nn.Parameter(num_neighbors * torch.ones(1), requires_grad=False)
        else:
            self.aff_scale_const = nn.Parameter(torch.ones(1), requires_grad=False)

    @staticmethod
    def _base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        ys = torch.arange(height, device=device, dtype=dtype).view(1, height, 1).expand(batch, height, width)
        xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, width).expand(batch, height, width)
        return ys, xs

    @staticmethod
    def _sample(x: torch.Tensor, offset_y: torch.Tensor, offset_x: torch.Tensor, kh: int, kw: int, pad: int) -> torch.Tensor:
        batch, _, height, width = x.shape
        ys, xs = GridSampleNLSPNPropagation._base_grid(batch, height, width, x.device, x.dtype)
        sample_y = ys + kh - pad + offset_y
        sample_x = xs + kw - pad + offset_x
        norm_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
        norm_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
        grid = torch.stack((norm_x, norm_y), dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _get_offset_affinity(self, guidance: torch.Tensor, confidence: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = guidance.shape
        offset_aff = self.conv_offset_aff(guidance)
        o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1).view(batch, self.num, 2, height, width)
        parts = list(torch.chunk(offset, self.num, dim=1))
        parts.insert(self.idx_ref, torch.zeros((batch, 1, 2, height, width), dtype=offset.dtype, device=offset.device))
        offset = torch.cat(parts, dim=1).view(batch, -1, height, width)

        if self.affinity == "TC":
            aff = torch.tanh(aff) / self.aff_scale_const
        elif self.affinity == "TGASS":
            aff = torch.tanh(aff) / (self.aff_scale_const + 1.0e-8)

        if self.conf_prop and confidence is not None:
            conf_parts: List[torch.Tensor] = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)
            for idx_off in range(self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f
                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue
                off = offset_each[idx_off]
                if self.legacy:
                    off = off.clone()
                    off[:, 0] = off[:, 0] + hh - (self.k_f - 1) / 2
                    off[:, 1] = off[:, 1] + ww - (self.k_f - 1) / 2
                conf_parts.append(self._sample(confidence, off[:, 0], off[:, 1], 1, 1, 0))
            aff = aff * torch.cat(conf_parts, dim=1)

        aff_abs_sum = torch.sum(torch.abs(aff), dim=1, keepdim=True) + 1.0e-4
        if self.affinity in {"ASS", "TGASS"}:
            aff_abs_sum = torch.where(aff_abs_sum < 1.0, torch.ones_like(aff_abs_sum), aff_abs_sum)
        if self.affinity in {"AS", "ASS", "TGASS"}:
            aff = aff / aff_abs_sum
        aff_ref = 1.0 - torch.sum(aff, dim=1, keepdim=True)
        aff_parts = list(torch.chunk(aff, self.num, dim=1))
        aff_parts.insert(self.idx_ref, aff_ref)
        return offset, torch.cat(aff_parts, dim=1)

    def _propagate_once(self, feat: torch.Tensor, offset: torch.Tensor, aff: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros_like(feat)
        offset_each = torch.chunk(offset, self.num + 1, dim=1)
        aff_each = torch.chunk(aff, self.num + 1, dim=1)
        idx = 0
        for kh in range(self.k_f):
            for kw in range(self.k_f):
                off = offset_each[idx]
                acc = acc + self._sample(feat, off[:, 0], off[:, 1], kh, kw, self.padding) * aff_each[idx]
                idx += 1
        return acc

    def forward(
        self,
        feat_init: torch.Tensor,
        guidance: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        feat_fix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        offset, aff = self._get_offset_affinity(guidance, confidence)
        mask_fix = None
        if self.preserve_input and feat_fix is not None:
            mask_fix = (feat_fix > 0.0).to(feat_init.dtype)

        feat = feat_init
        intermediates: List[torch.Tensor] = []
        for _ in range(self.prop_time):
            if mask_fix is not None:
                feat = feat * (1.0 - mask_fix) + feat_fix * mask_fix
            feat = self._propagate_once(feat, offset, aff)
            intermediates.append(feat)
        return feat, intermediates, offset, aff, self.aff_scale_const


class NLSPNHWStrictRefModel(NLSPNHWAlignedModel):
    """RHB-friendly decoder/head model with strict/ref NLSPN propagation."""

    def __init__(self, config: Optional[NLSPNHWStrictRefConfig] = None) -> None:
        cfg = config or NLSPNHWStrictRefConfig()
        super().__init__(cfg)
        self.config = cfg
        self.prop_layer = GridSampleNLSPNPropagation(
            self.num_neighbors,
            cfg.prop_time,
            affinity=cfg.affinity,
            affinity_gamma=cfg.affinity_gamma,
            conf_prop=cfg.conf_prop,
            preserve_input=cfg.preserve_input,
            legacy=cfg.legacy,
        )

    def forward(self, sample: TensorDict) -> TensorDict:
        rgb = sample["rgb"]
        dep = sample["dep"]
        feats = self.forward_features(rgb, dep)
        heads = self.forward_heads(feats)
        pred, pred_inter, offset, aff, gamma = self.prop_layer(
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
            "offset": offset,
            "aff": aff,
            "gamma": gamma,
            "confidence": heads["confidence"],
            "confidence_logits": heads["confidence_logits"],
        }
