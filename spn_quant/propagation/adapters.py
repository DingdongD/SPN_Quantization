"""Model-specific propagation-aware quantization adapters."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from spn_quant.propagation.controller import (
    PropagationQuantConfig,
    PropagationQuantController,
)
from spn_quant.propagation.fixed_point import (
    Q13_ONE,
    q13_multiply_accumulate_int32,
    requantize_q13_accumulator,
)


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


def cspn_float_affinity(guidance: torch.Tensor,
                        norm_type: str) -> torch.Tensor:
    if "8sum" not in norm_type:
        raise ValueError("unknown CSPN norm %s" % norm_type)
    raw = _pad_cspn_channels(guidance)
    if "abs" in norm_type:
        raw = raw.abs()
    return raw / raw.abs().sum(dim=1, keepdim=True)


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
        self.statistics_enabled = True
        self._training_state_capture = False

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
        self._training_state_capture = False
        self._last_states = []
        self._adapter_statistics = []

    def freeze(self) -> None:
        self.controller.freeze()
        self._training_state_capture = False

    def configure(self, config: PropagationQuantConfig) -> None:
        self.controller.configure(config)
        self._training_state_capture = False
        self._last_states = []
        self._adapter_statistics = []

    def capture(self) -> None:
        self.controller.capture()
        self._training_state_capture = False
        self._last_states = []
        self._adapter_statistics = []

    def capture_training_states(self) -> None:
        self.controller.capture()
        self._training_state_capture = True
        self._last_states = []
        self._adapter_statistics = []

    def disable(self) -> None:
        self.controller.disable()
        self._training_state_capture = False
        self._last_states = []

    def _float_coefficients(self, raw: torch.Tensor):
        neighbor = raw / raw.abs().sum(dim=1, keepdim=True)
        center = 1.0 - neighbor.sum(dim=1, keepdim=True)
        return neighbor, center

    def _record_anchor(self, state: torch.Tensor, initial: torch.Tensor,
                       mask: torch.Tensor, iteration: int) -> None:
        if not self.statistics_enabled:
            return
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
        self.controller.begin_forward()
        raw = _pad_cspn_channels(guidance)
        if "abs" in self.module.norm_type:
            raw = raw.abs()

        if self.controller.mode != "quantize":
            if self.controller.mode == "observe":
                self.controller.observe_signal("affinity_raw", raw)
                self.controller.observe_signal("state", initial)
            neighbor, center = self._float_coefficients(raw)
        else:
            neighbor, center_codes, neighbor_codes = \
                self.controller.signed_affinity(
                raw, denominator_floor=False, eps=0.0)
            center = center_codes.to(initial.dtype) / float(Q13_ONE)

        state = initial
        mask = None if sparse_depth is None else sparse_depth != 0
        if self.controller.mode == "quantize":
            state, state_codes, state_scale = \
                self.controller.quantize_state_with_codes(initial, 0)
            initial_codes = state_codes
        for iteration in range(1, int(self.module.prop_time) + 1):
            if self.controller.mode == "quantize":
                padded_codes = _pad_cspn_state(state_codes).to(torch.int32)
                neighbor_accumulator = _crop_cspn(
                    q13_multiply_accumulate_int32(
                        neighbor_codes, padded_codes, dim=1))
                center_accumulator = _crop_cspn(
                    center_codes.to(torch.int32)) * initial_codes
                accumulator = neighbor_accumulator + center_accumulator
                if accumulator.dtype != torch.int32:
                    raise RuntimeError("CSPN propagation accumulator is not INT32")
                state_codes = requantize_q13_accumulator(
                    accumulator, self.controller.config.state_bits)
                reference = accumulator.to(initial.dtype) * \
                    (state_scale / float(Q13_ONE))
                state = self.controller.state_from_codes(
                    reference, state_codes, state_scale, iteration)
                if self.statistics_enabled:
                    self._adapter_statistics.append({
                        "signal": "state_accumulator",
                        "iteration": int(iteration),
                        "numel": int(accumulator.numel()),
                        "accumulator_dtype": "int32",
                        "accumulator_absmax": float(
                            accumulator.abs().max().item()),
                    })
            else:
                padded = _pad_cspn_state(state)
                neighbor_sum = _crop_cspn((neighbor * padded).sum(
                    dim=1, keepdim=True))
                center_value = _crop_cspn(center)
                state = neighbor_sum + center_value * initial
            if self.controller.mode == "observe":
                self.controller.observe_signal("state", state)
            if mask is not None:
                if self.controller.mode == "quantize":
                    state = torch.where(mask, initial, state)
                    state_codes = torch.where(mask, initial_codes, state_codes)
                    self._record_anchor(state, initial, mask, iteration)
                else:
                    mask_value = mask.to(state.dtype)
                    state = (1.0 - mask_value) * state + \
                        mask_value * initial
            if self._training_state_capture:
                self._last_states.append(state.detach().clone())
            elif self.statistics_enabled:
                self._last_states.append(state.detach().cpu().clone())
        return state

    def last_states(self) -> List[torch.Tensor]:
        return list(self._last_states)

    def statistics(self) -> List[Dict[str, float]]:
        return self.controller.statistics() + [
            dict(row) for row in self._adapter_statistics]

    def set_runtime_statistics(self, enabled: bool) -> None:
        self.statistics_enabled = bool(enabled)
        self.controller.set_runtime_statistics(enabled)

    def close(self) -> None:
        self.disable()
        self.module.forward = self.original_forward


class NLSPNPropagationAdapter(object):
    """Shared propagation-domain adapter for NLSPN and CompletionFormer."""

    def __init__(self, module: Any, model_name: str) -> None:
        if model_name not in ("nlspn", "completionformer"):
            raise ValueError("unsupported NLSPN-family model: %s" % model_name)
        for field in ("conv_offset_aff", "num", "idx_ref", "affinity",
                      "prop_time", "args", "_propagate_once"):
            if not hasattr(module, field):
                raise TypeError("NLSPN propagation module is missing %s" % field)
        self.module = module
        self.model_name = model_name
        self.controller = PropagationQuantController()
        self.original_forward = module.forward
        self._last_states = []  # type: List[torch.Tensor]
        self._adapter_statistics = []  # type: List[Dict[str, float]]
        self._coefficient_codes = None
        self._confidence_codes = None

        def forward(feat_init: torch.Tensor, guidance: torch.Tensor,
                    confidence: torch.Tensor = None,
                    feat_fix: torch.Tensor = None,
                    rgb: torch.Tensor = None):
            if self.controller.mode == "bypass":
                return self.original_forward(
                    feat_init, guidance, confidence, feat_fix, rgb)
            return self._forward(
                feat_init, guidance, confidence, feat_fix, rgb)

        self.patched_forward = forward
        module.forward = self.patched_forward

    def observe(self) -> None:
        self.controller.observe()
        self._reset_forward_records()

    def freeze(self) -> None:
        self.controller.freeze()

    def configure(self, config: PropagationQuantConfig) -> None:
        self.controller.configure(config)
        self._reset_forward_records()

    def capture(self) -> None:
        self.controller.capture()
        self._reset_forward_records()

    def disable(self) -> None:
        self.controller.disable()
        self._reset_forward_records()

    def _reset_forward_records(self) -> None:
        self._last_states = []
        self._adapter_statistics = []
        self._coefficient_codes = None
        self._confidence_codes = None

    def _transform_affinity(self, affinity: torch.Tensor) -> torch.Tensor:
        if self.module.affinity in ("AS", "ASS"):
            return affinity
        divisor = 100.0 if self.model_name == "completionformer" else 1.0
        scale = self.module.aff_scale_const
        if self.module.affinity == "TC":
            return torch.tanh(affinity / divisor) / scale
        if self.module.affinity == "TGASS":
            return torch.tanh(affinity / divisor) / (scale + 1e-8)
        raise ValueError("unknown affinity mode %s" % self.module.affinity)

    def _insert_center_offset(self, raw: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = raw.shape
        offset = raw.view(batch, int(self.module.num), 2, height, width)
        parts = list(torch.chunk(offset, int(self.module.num), dim=1))
        parts.insert(int(self.module.idx_ref), torch.zeros(
            batch, 1, 2, height, width, device=raw.device,
            dtype=raw.dtype))
        return torch.cat(parts, dim=1).view(batch, -1, height, width)

    def _sample_confidence(self, confidence: torch.Tensor,
                           offset: torch.Tensor) -> torch.Tensor:
        custom = getattr(
            self.module, "_sample_confidence_for_affinity", None)
        if callable(custom):
            return custom(confidence, offset)

        function = self.original_forward.__globals__.get(
            "ModulatedDeformConvFunction")
        if function is None:
            get_affinity = getattr(self.module, "_get_offset_affinity", None)
            function = getattr(get_affinity, "__globals__", {}).get(
                "ModulatedDeformConvFunction")
        if function is None:
            raise RuntimeError("NLSPN confidence sampler is unavailable")

        batch, _, height, width = confidence.shape
        offset_each = torch.chunk(offset, int(self.module.num) + 1, dim=1)
        modulation = torch.ones(
            batch, 1, height, width, device=offset.device,
            dtype=offset.dtype).detach()
        sampled = []
        for index, current in enumerate(offset_each):
            ww = index % int(self.module.k_f)
            hh = index // int(self.module.k_f)
            center = (int(self.module.k_f) - 1) // 2
            if ww == center and hh == center:
                continue
            current = current.detach()
            if bool(self.module.args.legacy):
                current[:, 0] = current[:, 0] + hh - center
                current[:, 1] = current[:, 1] + ww - center
            sampled.append(function.apply(
                confidence, current, modulation, self.module.w_conf,
                self.module.b, self.module.stride, 0, self.module.dilation,
                self.module.groups, self.module.deformable_groups,
                self.module.im2col_step))
        return torch.cat(sampled, dim=1)

    def _insert_center_affinity(self, neighbor: torch.Tensor,
                                center: torch.Tensor) -> torch.Tensor:
        parts = list(torch.chunk(neighbor, int(self.module.num), dim=1))
        parts.insert(int(self.module.idx_ref), center)
        return torch.cat(parts, dim=1)

    def _coefficient_values(self, raw_affinity: torch.Tensor):
        if self.module.affinity == "TC":
            neighbor, center_codes, neighbor_codes = \
                self.controller.direct_signed_affinity(raw_affinity)
        else:
            floor = self.module.affinity in ("ASS", "TGASS")
            neighbor, center_codes, neighbor_codes = \
                self.controller.signed_affinity(
                raw_affinity, denominator_floor=floor, eps=1e-4)
        center = center_codes.to(raw_affinity.dtype) / float(Q13_ONE)
        affinity = self._insert_center_affinity(neighbor, center)
        codes = self._insert_center_affinity(neighbor_codes, center_codes)
        return affinity, codes

    def _float_coefficients(self, raw_affinity: torch.Tensor):
        if self.module.affinity in ("AS", "ASS", "TGASS"):
            denominator = raw_affinity.abs().sum(dim=1, keepdim=True) + 1e-4
            if self.module.affinity in ("ASS", "TGASS"):
                denominator = torch.maximum(
                    denominator, torch.ones_like(denominator))
            raw_affinity = raw_affinity / denominator
        center = 1.0 - raw_affinity.sum(dim=1, keepdim=True)
        return self._insert_center_affinity(raw_affinity, center)

    def _record_anchor_injection(self, state: torch.Tensor,
                                 fixed: torch.Tensor, mask: torch.Tensor,
                                 iteration: int) -> None:
        error = (state[mask] - fixed[mask]).abs()
        self._adapter_statistics.append({
            "signal": "anchor_injection",
            "iteration": int(iteration),
            "numel": int(mask.sum().item()),
            "anchor_mae": float(error.mean().item()) if error.numel() else 0.0,
            "anchor_max_error": float(error.max().item()) if error.numel() else 0.0,
        })

    def _forward(self, initial: torch.Tensor, guidance: torch.Tensor,
                 confidence: torch.Tensor = None,
                 fixed: torch.Tensor = None, rgb: torch.Tensor = None):
        del rgb
        self._reset_forward_records()
        self.controller.begin_forward()
        projection = self.module.conv_offset_aff(guidance)
        o1, o2, raw_affinity = torch.chunk(projection, 3, dim=1)
        raw_offset = torch.cat((o1, o2), dim=1)

        if self.controller.mode != "quantize":
            if self.controller.mode == "observe":
                self.controller.observe_signal("offset", raw_offset)
            quantized_offset = raw_offset
        else:
            quantized_offset = self.controller.quantize_offset(raw_offset)
        offset = self._insert_center_offset(quantized_offset)
        raw_affinity = self._transform_affinity(raw_affinity)

        if bool(self.module.args.conf_prop):
            if confidence is None:
                raise ValueError("confidence is required by this NLSPN model")
            if self.controller.mode == "quantize":
                confidence, self._confidence_codes = \
                    self.controller.quantize_confidence(confidence)
            sampled = self._sample_confidence(confidence, offset)
            raw_affinity = raw_affinity * sampled.contiguous()

        if self.controller.mode != "quantize":
            if self.controller.mode == "observe":
                self.controller.observe_signal("affinity_raw", raw_affinity)
            affinity = self._float_coefficients(raw_affinity)
        else:
            affinity, self._coefficient_codes = self._coefficient_values(
                raw_affinity)

        preserve = bool(self.module.args.preserve_input)
        if preserve:
            if fixed is None or fixed.shape != initial.shape:
                raise ValueError("preserve_input requires matching fixed depth")
            mask = fixed > 0
        else:
            mask = None

        state = initial
        intermediate = []
        for iteration in range(1, int(self.module.prop_time) + 1):
            if mask is not None:
                state = torch.where(mask, fixed, state)
                if self.controller.mode == "quantize":
                    self._record_anchor_injection(
                        state, fixed, mask, iteration)
            state = self.module._propagate_once(state, offset, affinity)
            if self.controller.mode == "observe":
                self.controller.observe_signal("state", state)
            elif self.controller.mode == "quantize":
                state = self.controller.quantize_state(state, iteration)
            intermediate.append(state)
            self._last_states.append(state.detach().cpu().clone())
        return state, intermediate, offset, affinity, \
            self.module.aff_scale_const.data

    def last_states(self) -> List[torch.Tensor]:
        return list(self._last_states)

    def last_coefficient_codes(self) -> torch.Tensor:
        if self._coefficient_codes is None:
            raise RuntimeError("no quantized affinity was produced")
        return self._coefficient_codes

    def last_confidence_codes(self) -> torch.Tensor:
        if self._confidence_codes is None:
            raise RuntimeError("no quantized confidence was produced")
        return self._confidence_codes

    def statistics(self) -> List[Dict[str, float]]:
        return self.controller.statistics() + [
            dict(row) for row in self._adapter_statistics]

    def close(self) -> None:
        self.disable()
        self.module.forward = self.original_forward


class DySPNPropagationAdapter(object):
    """Propagation-domain quantization for the official dynamic SPN module."""

    def __init__(self, module: Any) -> None:
        for field in ("conv_offset_aff", "iteration", "num", "ch",
                      "get_refgrid"):
            if not hasattr(module, field):
                raise TypeError("DySPN propagation module is missing %s" % field)
        self.module = module
        self.controller = PropagationQuantController()
        self.original_forward = module.forward
        self._last_states = []  # type: List[torch.Tensor]
        self._coefficient_codes = None
        self._confidence_codes = None
        self._adapter_statistics = []  # type: List[Dict[str, float]]
        self.statistics_enabled = True

        def forward(initial: torch.Tensor, guidance: torch.Tensor,
                    sparse_depth: torch.Tensor,
                    confidence_logits: torch.Tensor):
            if self.controller.mode == "bypass":
                return self.original_forward(
                    initial, guidance, sparse_depth, confidence_logits)
            return self._forward(
                initial, guidance, sparse_depth, confidence_logits)

        self.patched_forward = forward
        module.forward = self.patched_forward

    def observe(self) -> None:
        self.controller.observe()
        self._reset_forward_records()

    def freeze(self) -> None:
        self.controller.freeze()

    def configure(self, config: PropagationQuantConfig) -> None:
        self.controller.configure(config)
        self._reset_forward_records()

    def capture(self) -> None:
        self.controller.capture()
        self._reset_forward_records()

    def disable(self) -> None:
        self.controller.disable()
        self._reset_forward_records()
        self.controller.begin_forward()

    def _reset_forward_records(self) -> None:
        self._last_states = []
        self._coefficient_codes = None
        self._confidence_codes = None
        self._adapter_statistics = []

    def _record_anchor_injection(
            self, state: torch.Tensor, propagated: torch.Tensor,
            sparse_depth: torch.Tensor, confidence: torch.Tensor,
            mask: torch.Tensor, iteration: int) -> None:
        if not self.statistics_enabled:
            return
        expected = (1.0 - confidence) * propagated + \
            confidence * sparse_depth
        error = (state[mask] - expected[mask]).abs()
        self._adapter_statistics.append({
            "signal": "anchor_injection",
            "iteration": int(iteration),
            "numel": int(mask.sum().item()),
            "anchor_mae": float(error.mean().item())
            if error.numel() else 0.0,
            "anchor_max_error": float(error.max().item())
            if error.numel() else 0.0,
        })

    def _forward(self, initial: torch.Tensor, guidance: torch.Tensor,
                 sparse_depth: torch.Tensor,
                 confidence_logits: torch.Tensor):
        self._reset_forward_records()
        self.controller.begin_forward()
        batch, _, height, width = initial.shape
        projection = self.module.conv_offset_aff(guidance)
        raw_offset, logits = torch.split(
            projection, [2 * int(self.module.ch), int(self.module.ch)], dim=1)
        logits = logits.view(
            batch, int(self.module.iteration), int(self.module.num),
            height, width)

        confidence = torch.sigmoid(confidence_logits)
        if self.controller.mode != "quantize":
            if self.controller.mode == "observe":
                self.controller.observe_signal("offset", raw_offset)
                self.controller.observe_signal("affinity_raw", logits)
            quantized_offset = raw_offset
            affinity = torch.softmax(logits, dim=2)
        else:
            quantized_offset = self.controller.quantize_offset(raw_offset)
            affinity, self._coefficient_codes = \
                self.controller.softmax_affinity(logits, dim=2)
            confidence, self._confidence_codes = \
                self.controller.quantize_confidence(confidence)

        sparse_mask = sparse_depth.sign()
        anchor_mask = sparse_depth != 0
        confidence = confidence * sparse_mask
        offset_grid = self.module.get_refgrid(
            batch, height, width, quantized_offset).float()
        offsets = torch.unbind(offset_grid, dim=1)

        state = initial.float()
        intermediate = []
        affinities = torch.chunk(affinity, int(self.module.iteration), dim=1)
        for iteration in range(int(self.module.iteration)):
            propagated = torch.zeros_like(state)
            for neighbor in range(int(self.module.num)):
                sampled = F.grid_sample(
                    state,
                    offsets[iteration][:, neighbor],
                    align_corners=False,
                    padding_mode="zeros",
                    mode="bilinear",
                )
                propagated = propagated + sampled * \
                    affinities[iteration][:, :, neighbor]
            state = (1.0 - confidence) * propagated + \
                confidence * sparse_depth
            if self.controller.mode == "quantize":
                self._record_anchor_injection(
                    state, propagated, sparse_depth, confidence,
                    anchor_mask, iteration + 1)
            if self.controller.mode == "observe":
                self.controller.observe_signal("state", state)
            elif self.controller.mode == "quantize":
                state = self.controller.quantize_state(state, iteration + 1)
            intermediate.append(state)
            self._last_states.append(state.detach().cpu().clone())

        return {
            "pred": state,
            "pred_init": initial,
            "list_feat": intermediate,
            "offset": offsets,
            "aff": affinities,
        }

    def last_states(self) -> List[torch.Tensor]:
        return list(self._last_states)

    def last_coefficient_codes(self) -> torch.Tensor:
        if self._coefficient_codes is None:
            raise RuntimeError("no quantized affinity was produced")
        return self._coefficient_codes

    def last_confidence_codes(self) -> torch.Tensor:
        if self._confidence_codes is None:
            raise RuntimeError("no quantized confidence was produced")
        return self._confidence_codes

    def statistics(self) -> List[Dict[str, float]]:
        return self.controller.statistics() + [
            dict(row) for row in self._adapter_statistics]

    def set_runtime_statistics(self, enabled: bool) -> None:
        self.statistics_enabled = bool(enabled)
        self.controller.set_runtime_statistics(enabled)

    def close(self) -> None:
        self.disable()
        self.module.forward = self.original_forward


def propagation_projection_outputs(model_name: str, model: Any):
    if model_name == "cspn":
        names = {"gud_up_proj_layer6.conv1"}
    elif model_name == "dyspn":
        names = {"dyspn_%d_%d.conv_offset_aff" % (
            int(model.iteration), int(model.num_sample))}
    elif model_name in ("nlspn", "completionformer"):
        names = {"prop_layer.conv_offset_aff"}
    else:
        raise ValueError("unknown propagation model: %s" % model_name)
    available = set(dict(model.named_modules()))
    missing = names - available
    if missing:
        raise RuntimeError("propagation projection is missing: %s" %
                           sorted(missing))
    return names


def install_propagation_adapter(model_name: str, model: Any):
    if model_name == "cspn":
        return CSPNPropagationAdapter(model.post_process_layer)
    if model_name == "dyspn":
        module_name = "dyspn_%d_%d" % (
            int(model.iteration), int(model.num_sample))
        return DySPNPropagationAdapter(getattr(model, module_name))
    if model_name in ("nlspn", "completionformer"):
        return NLSPNPropagationAdapter(model.prop_layer, model_name)
    raise ValueError("unknown propagation model: %s" % model_name)
