"""Model-specific propagation-aware quantization adapters."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from spn_quant.propagation.controller import (
    PropagationQuantConfig,
    PropagationQuantController,
)
from spn_quant.propagation.fixed_point import Q13_ONE


_NEIGHBOR_PADS = (
    (0, 2, 0, 2),
    (1, 1, 0, 2),
    (2, 0, 0, 2),
    (0, 2, 1, 1),
    (2, 0, 1, 1),
    (0, 2, 2, 0),
    (1, 1, 2, 0),
    (2, 0, 2, 0),
)


def _pad_cspn_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.shape[1] != 8:
        raise ValueError("CSPN guidance must have eight channels")
    channels = torch.chunk(tensor, 8, dim=1)
    return torch.cat([
        F.pad(channel, padding).unsqueeze(1)
        for channel, padding in zip(channels, _NEIGHBOR_PADS)
    ], dim=1)


def _pad_cspn_state(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError("CSPN state must have one channel")
    return torch.cat([
        F.pad(tensor, padding).unsqueeze(1)
        for padding in _NEIGHBOR_PADS
    ], dim=1)


def _crop_cspn(value: torch.Tensor) -> torch.Tensor:
    return value.squeeze(1)[:, :, 1:-1, 1:-1]


class CSPNPropagationAdapter(object):
    """Replace only the CSPN propagation loop while preserving its interface."""

    def __init__(self, module: Any) -> None:
        if not hasattr(module, "prop_time") or not hasattr(module, "norm_type"):
            raise TypeError("CSPN propagation module is missing required fields")
        self.module = module
        self.controller = PropagationQuantController()
        self.original_forward = module.forward
        self._last_states = []  # type: List[torch.Tensor]
        self._adapter_statistics = []  # type: List[Dict[str, float]]

        def forward(guidance: torch.Tensor, blur_depth: torch.Tensor,
                    sparse_depth: torch.Tensor = None) -> torch.Tensor:
            if self.controller.mode == "bypass":
                return self.original_forward(
                    guidance, blur_depth, sparse_depth)
            return self._forward(guidance, blur_depth, sparse_depth)

        self.patched_forward = forward
        module.forward = self.patched_forward

    def observe(self) -> None:
        self.controller.observe()
        self._last_states = []
        self._adapter_statistics = []

    def freeze(self) -> None:
        self.controller.freeze()

    def configure(self, config: PropagationQuantConfig) -> None:
        self.controller.configure(config)
        self._last_states = []
        self._adapter_statistics = []

    def disable(self) -> None:
        self.controller.disable()
        self._last_states = []

    def _float_coefficients(self, raw: torch.Tensor):
        denominator = raw.abs().sum(dim=1, keepdim=True)
        denominator = torch.clamp(
            denominator, min=torch.finfo(raw.dtype).eps)
        neighbor = raw / denominator
        center = 1.0 - neighbor.sum(dim=1, keepdim=True)
        return neighbor, center

    def _record_anchor(self, state: torch.Tensor, initial: torch.Tensor,
                       mask: torch.Tensor, iteration: int) -> None:
        if not bool(mask.any()):
            error = state.new_zeros(1)
        else:
            error = (state[mask] - initial[mask]).abs()
        self._adapter_statistics.append({
            "signal": "anchor",
            "iteration": int(iteration),
            "numel": int(mask.sum().item()),
            "anchor_mae": float(error.mean().item()),
            "anchor_max_error": float(error.max().item()),
        })

    def _forward(self, guidance: torch.Tensor, initial: torch.Tensor,
                 sparse_depth: torch.Tensor = None) -> torch.Tensor:
        if "8sum" not in self.module.norm_type:
            raise ValueError("unknown CSPN norm %s" % self.module.norm_type)
        self._last_states = []
        self._adapter_statistics = []
        raw = _pad_cspn_channels(guidance)
        if "abs" in self.module.norm_type:
            raw = raw.abs()

        if self.controller.mode == "observe":
            self.controller.observe_signal("affinity_raw", raw)
            neighbor, center = self._float_coefficients(raw)
        else:
            neighbor, center_codes, _ = self.controller.signed_affinity(
                raw, denominator_floor=False, eps=0.0)
            center = center_codes.to(initial.dtype) / float(Q13_ONE)

        state = initial
        mask = None if sparse_depth is None else sparse_depth != 0
        for iteration in range(1, int(self.module.prop_time) + 1):
            padded = _pad_cspn_state(state)
            neighbor_sum = _crop_cspn((neighbor * padded).sum(
                dim=1, keepdim=True))
            center_value = _crop_cspn(center)
            state = neighbor_sum + center_value * initial
            if self.controller.mode == "observe":
                self.controller.observe_signal("state", state)
            else:
                state = self.controller.quantize_state(state, iteration)
            if mask is not None:
                state = torch.where(mask, initial, state)
                if self.controller.mode == "quantize":
                    self._record_anchor(state, initial, mask, iteration)
            self._last_states.append(state.detach().cpu().clone())
        return state

    def last_states(self) -> List[torch.Tensor]:
        return list(self._last_states)

    def statistics(self) -> List[Dict[str, float]]:
        return self.controller.statistics() + [
            dict(row) for row in self._adapter_statistics]

    def close(self) -> None:
        self.disable()
        self.module.forward = self.original_forward
