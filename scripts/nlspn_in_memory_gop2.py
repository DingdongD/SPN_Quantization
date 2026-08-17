"""Pure-memory online GOP2 engine for frozen NLSPN."""

from __future__ import division

from dataclasses import dataclass
import time

import numpy as np
import torch

from scripts import nlspn_temporal_residual as residual
from scripts import raft_small_compat


@dataclass
class OnlineState:
    previous_rgb: torch.Tensor
    previous_depth: torch.Tensor
    previous_guidance: torch.Tensor
    previous_confidence: torch.Tensor
    local_index: int


@dataclass
class FrameResult:
    prediction: torch.Tensor
    latency_ms: float
    kind: str


def frame_kind(local_index):
    if isinstance(local_index, bool) or int(local_index) != local_index:
        raise ValueError("local frame index must be an integer")
    local_index = int(local_index)
    if local_index < 0:
        raise ValueError("local frame index must be non-negative")
    return "I" if local_index % 2 == 0 else "P"


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_cpu_tensor(value, name):
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        raise TypeError("%s must be a NumPy array or Torch tensor" % name)
    if tensor.device.type != "cpu":
        raise ValueError("%s must reside in CPU memory" % name)
    return tensor


def _validate_inputs(rgb, sparse):
    rgb = _as_cpu_tensor(rgb, "RGB")
    sparse = _as_cpu_tensor(sparse, "sparse depth")
    if tuple(rgb.shape) != (3, residual.HEIGHT, residual.WIDTH):
        raise ValueError("RGB must have shape [3, 228, 304]")
    if tuple(sparse.shape) != (residual.HEIGHT, residual.WIDTH):
        raise ValueError("sparse depth must have shape [228, 304]")
    if not torch.isfinite(rgb).all() or not torch.isfinite(sparse).all():
        raise ValueError("online input contains non-finite values")
    if torch.any(sparse < 0):
        raise ValueError("sparse depth cannot be negative")
    if int(torch.count_nonzero(sparse > 0).item()) != residual.SPARSE_COUNT:
        raise ValueError("sparse depth must contain exactly 500 points")
    return rgb, sparse


class InMemoryGOP2Engine(object):
    def __init__(self, nlspn, raft, device):
        self.nlspn = nlspn
        self.raft = raft
        self.device = torch.device(device)
        self.state = None

    def reset(self):
        self.state = None

    def infer_i(self, rgb_cpu, sparse_cpu, local_index):
        if frame_kind(local_index) != "I":
            raise ValueError("infer_i requires an I-frame local index")
        rgb_cpu, sparse_cpu = _validate_inputs(rgb_cpu, sparse_cpu)
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            current_rgb = rgb_cpu[None].to(
                device=self.device, dtype=torch.float32)
            current_sparse = sparse_cpu[None, None].to(
                device=self.device, dtype=torch.float32)
            output = self.nlspn({
                "rgb": current_rgb,
                "dep": current_sparse,
            })
            required = ("pred", "guidance", "confidence")
            if not isinstance(output, dict) or not all(
                    key in output for key in required):
                raise ValueError("NLSPN output is missing online state fields")
            prediction = output["pred"]
            if tuple(prediction.shape) != (
                    1, 1, residual.HEIGHT, residual.WIDTH):
                raise ValueError("NLSPN prediction shape is invalid")
            if not all(torch.isfinite(output[key]).all() for key in required):
                raise ValueError("NLSPN output contains non-finite values")
            self.state = OnlineState(
                previous_rgb=current_rgb.detach(),
                previous_depth=prediction.detach(),
                previous_guidance=output["guidance"].detach(),
                previous_confidence=output["confidence"].detach(),
                local_index=int(local_index),
            )
            prediction_cpu = prediction.detach().to("cpu")[0, 0]
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(prediction_cpu, latency_ms, "I")

    def infer_p(self, rgb_cpu, sparse_cpu, local_index):
        if frame_kind(local_index) != "P":
            raise ValueError("infer_p requires a P-frame local index")
        if self.state is None:
            raise RuntimeError("P frame requires live temporal state")
        if self.state.local_index != int(local_index) - 1:
            raise RuntimeError("P frame temporal state is stale or misordered")
        rgb_cpu, sparse_cpu = _validate_inputs(rgb_cpu, sparse_cpu)
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            current_rgb = rgb_cpu[None].to(
                device=self.device, dtype=torch.float32)
            current_sparse = sparse_cpu[None, None].to(
                device=self.device, dtype=torch.float32)
            flow = raft_small_compat.predict_backward_flow(
                self.raft, current_rgb, self.state.previous_rgb)
            base, _ = residual.backward_warp(
                self.state.previous_depth, flow)
            guidance, _ = residual.backward_warp(
                self.state.previous_guidance, flow)
            confidence, _ = residual.backward_warp(
                self.state.previous_confidence, flow)
            seed = torch.zeros_like(base)
            sparse_mask = current_sparse > 0.0
            seed[sparse_mask] = (
                current_sparse[sparse_mask] - base[sparse_mask])
            dense = self.nlspn.prop_layer(
                seed, guidance, confidence, None, current_rgb)[0]
            prediction = torch.clamp(
                base + dense, min=0.0, max=residual.MAX_DEPTH)
            values = (flow, base, guidance, confidence, dense, prediction)
            if not all(torch.isfinite(value).all() for value in values):
                raise ValueError("P-frame inference contains non-finite values")
            self.state = OnlineState(
                previous_rgb=current_rgb.detach(),
                previous_depth=prediction.detach(),
                previous_guidance=guidance.detach(),
                previous_confidence=confidence.detach(),
                local_index=int(local_index),
            )
            prediction_cpu = prediction.detach().to("cpu")[0, 0]
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(prediction_cpu, latency_ms, "P")

