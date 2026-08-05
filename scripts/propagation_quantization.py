#!/usr/bin/env python3
"""Runtime adapters for quantizing depth state after each SPN iteration."""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.rtn_quantization import MinMaxObserver, QuantizationStats


class StateQuantizerController(object):
    def __init__(self):
        self.mode = "bypass"
        self.observer = MinMaxObserver()
        self.frozen = False
        self.quantizer = None
        self.stats = {}
        self._last_states = []

    def begin_forward(self):
        self._last_states = []

    def process(self, tensor, iteration):
        if self.mode == "observe":
            self.observer.update(tensor)
            return tensor
        if self.mode == "quantize":
            stats = self.stats.setdefault(int(iteration), QuantizationStats())
            tensor = self.quantizer(tensor, stats)
            self._last_states.append(tensor.detach().cpu().clone())
            return tensor
        if self.mode == "capture":
            self._last_states.append(tensor.detach().cpu().clone())
        return tensor

    def observe(self):
        self.mode = "observe"
        self.observer = MinMaxObserver()
        self.frozen = False
        self.quantizer = None
        self.stats = {}

    def freeze(self):
        if not self.observer.observed:
            raise RuntimeError("no propagation states were observed")
        self.frozen = True
        self.mode = "bypass"

    def configure(self, bits):
        if not self.frozen:
            raise RuntimeError("propagation-state calibration must be frozen")
        self.quantizer = self.observer.quantizer(bits)
        self.stats = {}
        self.mode = "quantize"

    def capture(self):
        self.mode = "capture"

    def disable(self):
        self.mode = "bypass"
        self.quantizer = None
        self._last_states = []

    def last_states(self):
        return list(self._last_states)

    def statistics(self):
        rows = []
        for iteration, stats in sorted(self.stats.items()):
            rows.append({
                "iteration": iteration,
                "numel": stats.numel,
                "mse": stats.mse,
                "sqnr_db": stats.sqnr_db,
                "cosine": stats.cosine,
                "saturation_rate": stats.saturation_rate,
                "sign_flip_rate": stats.sign_flip_rate,
            })
        return rows


class _AdapterBase(object):
    def __init__(self):
        self.controller = StateQuantizerController()

    def observe(self):
        self.controller.observe()

    def freeze(self):
        self.controller.freeze()

    def configure(self, bits):
        self.controller.configure(bits)

    def capture(self):
        self.controller.capture()

    def disable(self):
        self.controller.disable()

    def last_states(self):
        return self.controller.last_states()

    def statistics(self):
        return self.controller.statistics()


class NLSPNStateAdapter(_AdapterBase):
    """Instrument modules exposing `_propagate_once`, used by NLSPN variants."""

    def __init__(self, module):
        super(NLSPNStateAdapter, self).__init__()
        self.module = module
        self.original_propagate_once = module._propagate_once

        def propagate_once(feat, offset, affinity):
            result = self.original_propagate_once(feat, offset, affinity)
            iteration = getattr(self, "_iteration", 0) + 1
            self._iteration = iteration
            return self.controller.process(result, iteration)

        self.patched_propagate_once = propagate_once
        module._propagate_once = self.patched_propagate_once
        self.pre_handle = module.register_forward_pre_hook(self._begin_forward)

    def _begin_forward(self, module, inputs):
        del module, inputs
        self._iteration = 0
        self.controller.begin_forward()

    def close(self):
        self.disable()
        self.pre_handle.remove()
        self.module._propagate_once = self.original_propagate_once


class CSPNStateAdapter(_AdapterBase):
    def __init__(self, module):
        super(CSPNStateAdapter, self).__init__()
        self.module = module
        self.original_forward = module.forward

        def forward(guidance, blur_depth, sparse_depth=None):
            if self.controller.mode == "bypass":
                return self.original_forward(guidance, blur_depth, sparse_depth)
            return self._forward_with_state_qdq(guidance, blur_depth, sparse_depth)

        self.patched_forward = forward
        module.forward = self.patched_forward

    def _forward_with_state_qdq(self, guidance, blur_depth, sparse_depth=None):
        module = self.module
        self.controller.begin_forward()
        module.sum_conv = nn.Conv3d(8, 1, kernel_size=1, stride=1,
                                    padding=0, bias=False).to(guidance.device)
        weight = torch.ones(1, 8, 1, 1, 1, device=guidance.device,
                            dtype=guidance.dtype)
        module.sum_conv.weight = nn.Parameter(weight, requires_grad=False)
        gate_wb, gate_sum = module.affinity_normalization(guidance)
        raw_depth_input = blur_depth
        result_depth = blur_depth
        sparse_mask = sparse_depth.sign() if sparse_depth is not None else None

        for iteration in range(1, module.prop_time + 1):
            padded = module.pad_blur_depth(result_depth)
            neighbor_sum = module.sum_conv(gate_wb * padded).squeeze(1)
            result_depth = neighbor_sum[:, :, 1:-1, 1:-1]
            if "8sum" not in module.norm_type:
                raise ValueError("unknown norm %s" % module.norm_type)
            result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            if sparse_depth is not None:
                result_depth = (1.0 - sparse_mask) * result_depth \
                    + sparse_mask * raw_depth_input
            result_depth = self.controller.process(result_depth, iteration)
        return result_depth

    def close(self):
        self.disable()
        self.module.forward = self.original_forward


class DySPNStateAdapter(_AdapterBase):
    def __init__(self, module):
        super(DySPNStateAdapter, self).__init__()
        self.module = module
        self.original_forward = module.forward

        def forward(input_depth, guide, sparse_depth, confidence):
            if self.controller.mode == "bypass":
                return self.original_forward(input_depth, guide, sparse_depth, confidence)
            return self._forward_with_state_qdq(
                input_depth, guide, sparse_depth, confidence)

        self.patched_forward = forward
        module.forward = self.patched_forward

    def _forward_with_state_qdq(self, input_depth, guide, sparse_depth, confidence):
        module = self.module
        self.controller.begin_forward()
        batch, _, height, width = input_depth.shape
        offset_affinity = module.conv_offset_aff(guide)
        offset, affinity = torch.split(
            offset_affinity, [2 * module.ch, module.ch], dim=1)
        confidence = torch.sigmoid(confidence) * sparse_depth.sign()
        affinity = affinity.view(
            batch, module.iteration, module.num, height, width)
        offsets = torch.unbind(
            module.get_refgrid(batch, height, width, offset).float(), dim=1)
        affinities = torch.chunk(
            torch.softmax(affinity, dim=2), module.iteration, dim=1)

        state = input_depth.float()
        initial = state
        intermediate = []
        for iteration in range(module.iteration):
            output = torch.zeros_like(state)
            for neighbor in range(module.num):
                sampled = F.grid_sample(
                    state,
                    offsets[iteration][:, neighbor, :, :, :],
                    align_corners=False,
                    padding_mode="zeros",
                    mode="bilinear",
                )
                output = output + sampled * affinities[iteration][:, :, neighbor, :, :]
            state = (1.0 - confidence) * output + confidence * sparse_depth
            state = self.controller.process(state, iteration + 1)
            intermediate.append(state)
        return {
            "pred": state,
            "pred_init": initial,
            "list_feat": intermediate,
            "offset": offsets,
            "aff": affinities,
        }

    def close(self):
        self.disable()
        self.module.forward = self.original_forward


def install_state_adapter(model_name, model):
    if model_name == "cspn":
        return CSPNStateAdapter(model.post_process_layer)
    if model_name == "dyspn":
        name = "dyspn_%d_%d" % (model.iteration, model.num_sample)
        return DySPNStateAdapter(getattr(model, name))
    if model_name in ("nlspn", "completionformer"):
        return NLSPNStateAdapter(model.prop_layer)
    raise ValueError("unknown model: %s" % model_name)

