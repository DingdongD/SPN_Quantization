"""Joint integer Attention and concat adapter for CompletionFormer."""

from __future__ import annotations

from collections import deque
import types
from typing import Any, Deque, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.completionformer_attention import IntegerAttentionController
from spn_quant.completionformer_concat import SplitConcatConvController


class CompletionFormerJointAdapter(object):
    def __init__(self, model: nn.Module, expected_attention_modules: int,
                 expected_concat_modules: int, weight_bits: int,
                 qkv_bits: int, probability_bits: int, concat_bits: int,
                 output_bits: int, clip_factors: Sequence[float],
                 search_rounds: int, cache_sample_limit: int,
                 cache_byte_limit: int) -> None:
        if model.training:
            raise RuntimeError("CompletionFormer joint adapter requires eval mode")
        self.model = model
        self.weight_bits = int(weight_bits)
        self.qkv_bits = int(qkv_bits)
        self.probability_bits = int(probability_bits)
        self.concat_bits = int(concat_bits)
        self.output_bits = int(output_bits)
        self.clip_factors = tuple(float(value) for value in clip_factors)
        self.search_rounds = int(search_rounds)
        self.cache_sample_limit = int(cache_sample_limit)
        self.cache_byte_limit = int(cache_byte_limit)
        self.phase = "disabled"
        self._closed = False
        self._attention_enabled = False
        self._concat_enabled = False
        self._current_calls = {}  # type: Dict[str, int]
        self._target_forwards = 0
        self._reconstruction_forwards = 0
        self._qdrop_range_forwards = 0
        self._qdrop_sites = {}
        self._qdrop_quantizers = {}
        self._qdrop_attention = {}
        self._qdrop_concat = {}
        self._qdrop_execution_enabled = False

        self._attention_modules = dict(
            (name, module) for name, module in model.named_modules()
            if name.startswith("backbone.former") and
            module.__class__.__name__ == "Attention")
        self._concat_modules = dict(
            (name, module) for name, module in model.named_modules()
            if name.startswith("backbone.former") and
            name.endswith(".concat_conv") and isinstance(module, nn.Conv2d))
        if len(self._attention_modules) != int(expected_attention_modules):
            raise RuntimeError(
                "expected %d Attention modules but found %d" %
                (int(expected_attention_modules),
                 len(self._attention_modules)))
        if len(self._concat_modules) != int(expected_concat_modules):
            raise RuntimeError(
                "expected %d concat_conv modules but found %d" %
                (int(expected_concat_modules), len(self._concat_modules)))

        self._validate_attention_modules()
        self._validate_concat_modules()
        self.attention_controllers = self._build_attention_controllers()
        self.concat_controllers = self._build_concat_controllers()
        self._attention_targets = dict(
            (name, deque()) for name in self._attention_modules
        )  # type: Dict[str, Deque[Tuple[int, torch.Tensor]]]
        self._concat_targets = dict(
            (name, deque()) for name in self._concat_modules
        )  # type: Dict[str, Deque[Tuple[int, torch.Tensor]]]
        self._original_attention_forwards = dict(
            (name, module.forward)
            for name, module in self._attention_modules.items())
        self._original_concat_forwards = dict(
            (name, module.forward)
            for name, module in self._concat_modules.items())
        self._install()

    def _validate_attention_modules(self) -> None:
        for name, module in self._attention_modules.items():
            if not isinstance(module.q, nn.Linear) or \
                    not isinstance(module.kv, nn.Linear) or \
                    not isinstance(module.proj, nn.Linear):
                raise TypeError("Attention projections must be Linear: %s" % name)
            if int(module.num_heads) <= 0 or \
                    module.q.out_features % int(module.num_heads) != 0:
                raise ValueError("Attention head shape is invalid: %s" % name)
            if module.kv.out_features != 2 * module.q.out_features:
                raise ValueError("Attention KV projection is invalid: %s" % name)
            if int(module.sr_ratio) > 1:
                if not isinstance(module.sr, nn.Conv2d) or \
                        not isinstance(module.norm, nn.LayerNorm):
                    raise TypeError(
                        "reduced Attention requires Conv2d and LayerNorm: %s" %
                        name)

    def _validate_concat_modules(self) -> None:
        for name, module in self._concat_modules.items():
            if module.in_channels % 2 != 0:
                raise ValueError("concat Conv channels must split evenly: %s" % name)

    def _build_attention_controllers(
            self) -> Dict[str, IntegerAttentionController]:
        controllers = {}
        for name, module in self._attention_modules.items():
            controllers[name] = IntegerAttentionController(
                name=name,
                num_heads=int(module.num_heads),
                head_dim=module.q.out_features // int(module.num_heads),
                head_scale=float(module.scale),
                qkv_bits=self.qkv_bits,
                probability_bits=self.probability_bits,
                clip_factors=self.clip_factors,
                search_rounds=self.search_rounds,
                cache_sample_limit=self.cache_sample_limit,
                cache_byte_limit=self.cache_byte_limit)
        return controllers

    def _build_concat_controllers(
            self) -> Dict[str, SplitConcatConvController]:
        controllers = {}
        for name, module in self._concat_modules.items():
            controllers[name] = SplitConcatConvController(
                name=name,
                module=module,
                branch_channels=module.in_channels // 2,
                weight_bits=self.weight_bits,
                activation_bits=self.concat_bits,
                output_bits=self.output_bits,
                clip_factors=self.clip_factors,
                search_rounds=self.search_rounds,
                cache_sample_limit=self.cache_sample_limit,
                cache_byte_limit=self.cache_byte_limit)
        return controllers

    def _install(self) -> None:
        for name, module in self._attention_modules.items():
            original = self._original_attention_forwards[name]
            module.forward = types.MethodType(
                self._make_attention_forward(name, original), module)
        for name, module in self._concat_modules.items():
            original = self._original_concat_forwards[name]
            module.forward = types.MethodType(
                self._make_concat_forward(name, original), module)
        self._pre_handle = self.model.register_forward_pre_hook(
            self._begin_forward)
        self._post_handle = self.model.register_forward_hook(
            self._complete_forward)

    def _begin_forward(self, module: nn.Module,
                       inputs: Tuple[Any, ...]) -> None:
        del module, inputs
        if self.phase not in (
                "capture_targets", "reconstruction", "qdrop_ranges"):
            return
        if self.phase == "reconstruction" and \
                self._reconstruction_forwards >= self._target_forwards:
            raise RuntimeError("reconstruction exceeds captured target forwards")
        self._current_calls = dict(
            (name, 0) for name in self._all_runtime_names())

    def _complete_forward(self, module: nn.Module, inputs: Tuple[Any, ...],
                          output: Any) -> None:
        del module, inputs, output
        if self.phase not in (
                "capture_targets", "reconstruction", "qdrop_ranges"):
            return
        invalid = sorted(
            name for name, count in self._current_calls.items() if count != 1)
        if invalid:
            raise RuntimeError(
                "joint calibration requires one call per module: %s" % invalid)
        if self.phase == "capture_targets":
            self._target_forwards += 1
        elif self.phase == "qdrop_ranges":
            self._qdrop_range_forwards += 1
        else:
            self._reconstruction_forwards += 1

    def _all_runtime_names(self) -> List[str]:
        return list(self._attention_modules) + list(self._concat_modules)

    def _record_call(self, name: str) -> None:
        self._current_calls[name] += 1
        if self._current_calls[name] > 1:
            raise RuntimeError("joint calibration module called more than once: %s" %
                               name)

    def _forward_index(self) -> int:
        if self.phase == "capture_targets":
            return self._target_forwards
        if self.phase == "qdrop_ranges":
            return self._qdrop_range_forwards
        return self._reconstruction_forwards

    @staticmethod
    def _cpu_target(tensor: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("joint calibration target must be finite")
        return tensor.detach().to(
            device="cpu", dtype=torch.float32).contiguous()

    def _append_target(self, queue: Deque[Tuple[int, torch.Tensor]],
                       tensor: torch.Tensor) -> None:
        queue.append((self._forward_index(), self._cpu_target(tensor)))

    def _consume_target(self, queue: Deque[Tuple[int, torch.Tensor]],
                        reference: torch.Tensor) -> torch.Tensor:
        if not queue:
            raise RuntimeError("joint reconstruction target queue is empty")
        sample_index, target = queue.popleft()
        if sample_index != self._forward_index():
            raise RuntimeError("joint reconstruction target order changed")
        return target.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _attention_qkv(module: nn.Module, x: torch.Tensor,
                       height: int, width: int
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, channels = x.shape
        heads = int(module.num_heads)
        head_dim = channels // heads
        if channels % heads != 0 or module.q.out_features != channels:
            raise ValueError("Attention input channels do not match projections")
        q = module.q(x).reshape(
            batch, tokens, heads, head_dim).permute(0, 2, 1, 3)
        if int(module.sr_ratio) > 1:
            if tokens != int(height) * int(width):
                raise ValueError("Attention token count does not match height and width")
            reduced = x.permute(0, 2, 1).reshape(
                batch, channels, int(height), int(width))
            reduced = module.sr(reduced).reshape(
                batch, channels, -1).permute(0, 2, 1)
            reduced = module.norm(reduced)
        else:
            reduced = x
        reduced_tokens = reduced.shape[1]
        kv = module.kv(reduced).reshape(
            batch, reduced_tokens, 2, heads, head_dim).permute(
                2, 0, 3, 1, 4)
        return q, kv[0], kv[1]

    @staticmethod
    def _float_context(module: nn.Module, q: torch.Tensor,
                       k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * float(module.scale)
        probability = torch.softmax(scores, dim=-1)
        return torch.matmul(module.attn_drop(probability), v)

    @staticmethod
    def _attention_projection(module: nn.Module,
                              context: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, head_dim = context.shape
        output = context.transpose(1, 2).reshape(
            batch, tokens, heads * head_dim)
        return module.proj_drop(module.proj(output))

    def _qdrop_attention_context(
            self, name: str, module: nn.Module, x: torch.Tensor,
            height: int, width: int) -> torch.Tensor:
        q, k, v = self._attention_qkv(module, x, height, width)
        quantizers = self._qdrop_attention[name]
        q = quantizers["q"].quantize_with_codes(q)[0]
        k = quantizers["k"].quantize_with_codes(k)[0]
        v = quantizers["v"].quantize_with_codes(v)[0]
        score = torch.matmul(q, k.transpose(-2, -1)) * float(module.scale)
        probability = torch.softmax(score.to(torch.float16), dim=-1)
        context = torch.matmul(
            module.attn_drop(probability), v.to(torch.float16))
        return context.to(dtype=x.dtype)

    def _make_attention_forward(self, name: str, original: Any):
        def forward(module: nn.Module, x: torch.Tensor,
                    height: int, width: int) -> torch.Tensor:
            if self.phase == "qdrop_ranges":
                self._record_call(name)
                q, k, v = self._attention_qkv(module, x, height, width)
                context = self._float_context(module, q, k, v)
                self.attention_controllers[name].observe_range(
                    q, k, v, context)
                return self._attention_projection(module, context)
            if self._qdrop_execution_enabled and \
                    name in self._qdrop_attention and \
                    self.phase != "capture_targets":
                if self.phase == "reconstruction":
                    self._record_call(name)
                context = self._qdrop_attention_context(
                    name, module, x, height, width)
                if self.phase == "reconstruction":
                    self._consume_target(
                        self._attention_targets[name], context)
                return self._attention_projection(module, context)
            if self.phase == "disabled" or \
                    (self.phase == "quantize" and
                     not self._attention_enabled):
                return original(x, height, width)
            if self.phase in ("capture_targets", "reconstruction"):
                self._record_call(name)
            q, k, v = self._attention_qkv(module, x, height, width)
            context = self._float_context(module, q, k, v)
            if self.phase == "capture_targets":
                self._append_target(self._attention_targets[name], context)
            elif self.phase == "reconstruction":
                target = self._consume_target(
                    self._attention_targets[name], context)
                self.attention_controllers[name].observe(q, k, v, target)
            elif self.phase == "quantize":
                context = self.attention_controllers[name].quantize(q, k, v)
            else:
                raise RuntimeError("unknown CompletionFormer joint phase")
            return self._attention_projection(module, context)
        return forward

    def _make_concat_forward(self, name: str, original: Any):
        def forward(module: nn.Module, merged: torch.Tensor) -> torch.Tensor:
            del module
            if self.phase == "qdrop_ranges":
                self._record_call(name)
                output = original(merged)
                self.concat_controllers[name].observe_range(merged, output)
                return output
            if self._qdrop_execution_enabled and \
                    name in self._qdrop_concat and \
                    self.phase != "capture_targets":
                if self.phase == "reconstruction":
                    self._record_call(name)
                channels = merged.shape[1] // 2
                transformer = merged[:, :channels]
                cnn = merged[:, channels:]
                quantizers = self._qdrop_concat[name]
                transformer = quantizers[
                    "transformer"].quantize_with_codes(transformer)[0]
                cnn = quantizers["cnn"].quantize_with_codes(cnn)[0]
                output = original(torch.cat((transformer, cnn), dim=1))
                if self.phase == "reconstruction":
                    self._consume_target(
                        self._concat_targets[name], output)
                return output
            if self.phase == "disabled" or \
                    (self.phase == "quantize" and not self._concat_enabled):
                return original(merged)
            if self.phase in ("capture_targets", "reconstruction"):
                self._record_call(name)
            if self.phase == "capture_targets":
                output = original(merged)
                self._append_target(self._concat_targets[name], output)
                return output
            if self.phase == "reconstruction":
                output = original(merged)
                target = self._consume_target(self._concat_targets[name], output)
                self.concat_controllers[name].observe(merged, target)
                return output
            if self.phase == "quantize":
                return self.concat_controllers[name].quantize(merged)
            raise RuntimeError("unknown CompletionFormer joint phase")
        return forward

    def attention_names(self) -> List[str]:
        return sorted(self._attention_modules)

    def concat_names(self) -> List[str]:
        return sorted(self._concat_modules)

    @staticmethod
    def _qdrop_parts(site: Any) -> Tuple[str, str, str]:
        parts = site.site.split("::")
        if len(parts) != 3:
            raise ValueError("invalid CompletionFormer QDrop site: %s" %
                             site.site)
        return parts[0], parts[1], parts[2]

    def bind_qdrop_sites(self, sites: Sequence[Any],
                         quantizers: Dict[str, Any]) -> None:
        sites = tuple(sites)
        names = set(site.site for site in sites)
        if names != set(quantizers):
            raise KeyError("CompletionFormer QDrop site/quantizer mismatch")
        next_sites = dict(self._qdrop_sites)
        next_quantizers = dict(self._qdrop_quantizers)
        next_attention = dict(
            (name, dict(roles))
            for name, roles in self._qdrop_attention.items())
        next_concat = dict(
            (name, dict(roles))
            for name, roles in self._qdrop_concat.items())
        for site in sites:
            if site.owner_kind == "attention_probability" or \
                    site.role == "attention_probability":
                raise ValueError(
                    "attention probability cannot be a QDrop site")
            if site.site in next_sites:
                raise RuntimeError(
                    "CompletionFormer QDrop site is already bound: %s" %
                    site.site)
            family, module_name, role = self._qdrop_parts(site)
            quantizer = quantizers[site.site]
            if quantizer.site != site.site:
                raise ValueError("CompletionFormer QDrop quantizer site mismatch")
            if family == "attention" and \
                    site.owner_kind == "attention_qkv":
                if module_name not in self._attention_modules:
                    raise KeyError("unknown QDrop Attention module: %s" %
                                   module_name)
                if role not in ("q", "k", "v"):
                    raise ValueError("unknown QDrop Attention role: %s" % role)
                if module_name not in next_attention:
                    next_attention[module_name] = {}
                if role in next_attention[module_name]:
                    raise RuntimeError("duplicate QDrop Attention role")
                next_attention[module_name][role] = quantizer
            elif family == "concat" and \
                    site.owner_kind == "concat_input":
                if module_name not in self._concat_modules:
                    raise KeyError("unknown QDrop concat module: %s" %
                                   module_name)
                if role not in ("transformer_input", "cnn_input"):
                    raise ValueError("unknown QDrop concat role: %s" % role)
                branch = role[:-len("_input")]
                if module_name not in next_concat:
                    next_concat[module_name] = {}
                if branch in next_concat[module_name]:
                    raise RuntimeError("duplicate QDrop concat role")
                next_concat[module_name][branch] = quantizer
            else:
                raise ValueError(
                    "unsupported CompletionFormer QDrop site: %s" %
                    site.site)
            next_sites[site.site] = site
            next_quantizers[site.site] = quantizer
        incomplete_attention = sorted(
            name for name, roles in next_attention.items()
            if set(roles) != {"q", "k", "v"})
        if incomplete_attention:
            raise RuntimeError(
                "incomplete QDrop Attention ownership: %s" %
                incomplete_attention)
        incomplete_concat = sorted(
            name for name, roles in next_concat.items()
            if set(roles) != {"transformer", "cnn"})
        if incomplete_concat:
            raise RuntimeError(
                "incomplete QDrop concat ownership: %s" % incomplete_concat)
        self._qdrop_sites = next_sites
        self._qdrop_quantizers = next_quantizers
        self._qdrop_attention = next_attention
        self._qdrop_concat = next_concat
        self._qdrop_execution_enabled = True

    def enable_qdrop_execution(self) -> None:
        if not self._qdrop_sites:
            raise RuntimeError("CompletionFormer QDrop sites are not bound")
        self._qdrop_execution_enabled = True

    def disable_qdrop_execution(self) -> None:
        self._qdrop_execution_enabled = False

    def qdrop_parameters(self, target: str) -> Tuple[nn.Parameter, ...]:
        parameters = []
        for site_name, site in self._qdrop_sites.items():
            if site.owner_name == target:
                parameters.extend(
                    self._qdrop_quantizers[site_name].parameters())
        if not parameters:
            raise RuntimeError(
                "CompletionFormer QDrop target has no parameters: %s" %
                target)
        return tuple(parameters)

    def qdrop_initialization_tensor(self, site: Any) -> torch.Tensor:
        family, module_name, role = self._qdrop_parts(site)
        if family == "attention" and site.owner_kind == "attention_qkv":
            if module_name not in self.attention_controllers:
                raise KeyError("unknown QDrop Attention module: %s" % module_name)
            maximum = self.attention_controllers[module_name].maxima[role]
            if maximum is None:
                raise RuntimeError(
                    "QDrop Attention site was not calibrated: %s" % site.site)
            maximum = maximum.detach().float().reshape(-1)
            device = self._attention_modules[module_name].q.weight.device
            return torch.cat((-maximum, maximum)).to(device=device)
        if family == "concat" and site.owner_kind == "concat_input":
            if module_name not in self.concat_controllers:
                raise KeyError("unknown QDrop concat module: %s" % module_name)
            controller = self.concat_controllers[module_name]
            branch = role[:-len("_input")]
            if branch == "transformer":
                maximum = float(controller.transformer_maximum)
            elif branch == "cnn":
                maximum = float(controller.cnn_maximum)
            else:
                raise ValueError("unknown QDrop concat role: %s" % role)
            if maximum <= 0.0:
                raise RuntimeError(
                    "QDrop concat site was not calibrated: %s" % site.site)
            device = self._concat_modules[module_name].weight.device
            return torch.tensor((-maximum, maximum), device=device)
        raise ValueError("unsupported CompletionFormer QDrop site: %s" %
                         site.site)

    def freeze_qdrop_sites(self, target: str) -> None:
        names = [
            site_name for site_name, site in self._qdrop_sites.items()
            if site.owner_name == target]
        if not names:
            raise RuntimeError(
                "CompletionFormer QDrop target is not bound: %s" % target)
        for site_name in names:
            self._qdrop_quantizers[site_name].freeze()

    def unbind_qdrop_sites(self) -> None:
        self._qdrop_sites = {}
        self._qdrop_quantizers = {}
        self._qdrop_attention = {}
        self._qdrop_concat = {}
        self._qdrop_execution_enabled = False

    def externally_owned_inputs(self) -> List[str]:
        return self.concat_names()

    def externally_owned_outputs(self) -> List[str]:
        names = list(self._concat_modules)
        for name in self._attention_modules:
            names.extend((name + ".q", name + ".kv"))
        return sorted(names)

    def attention_owned_outputs(self) -> List[str]:
        names = []
        for name in self._attention_modules:
            names.extend((name + ".q", name + ".kv"))
        return sorted(names)

    def concat_owned_inputs(self) -> List[str]:
        return self.concat_names()

    def concat_owned_outputs(self) -> List[str]:
        return self.concat_names()

    def capture_targets(self) -> None:
        if self.phase != "disabled":
            raise RuntimeError("target capture requires disabled joint adapter")
        for queue in self._attention_targets.values():
            queue.clear()
        for queue in self._concat_targets.values():
            queue.clear()
        self._target_forwards = 0
        self._reconstruction_forwards = 0
        self.phase = "capture_targets"

    def observe_qdrop_ranges(self) -> None:
        if self.phase != "disabled":
            raise RuntimeError(
                "QDrop range calibration requires disabled joint adapter")
        self._qdrop_range_forwards = 0
        self.phase = "qdrop_ranges"

    def freeze_qdrop_ranges(self) -> None:
        if self.phase != "qdrop_ranges" or self._qdrop_range_forwards == 0:
            raise RuntimeError("QDrop range calibration has no forwards")
        missing_attention = sorted(
            name for name, controller in self.attention_controllers.items()
            if controller.observations != self._qdrop_range_forwards)
        missing_concat = sorted(
            name for name, controller in self.concat_controllers.items()
            if controller.observations != self._qdrop_range_forwards)
        if missing_attention or missing_concat:
            raise RuntimeError(
                "QDrop range calibration call count mismatch: attention=%s "
                "concat=%s" % (missing_attention, missing_concat))
        self.phase = "disabled"

    def observe_reconstruction(self) -> None:
        if self.phase != "capture_targets" or self._target_forwards == 0:
            raise RuntimeError("reconstruction requires captured FP targets")
        expected = self._target_forwards
        for name, queue in self._attention_targets.items():
            if len(queue) != expected:
                raise RuntimeError("Attention target count is invalid: %s" % name)
        for name, queue in self._concat_targets.items():
            if len(queue) != expected:
                raise RuntimeError("concat target count is invalid: %s" % name)
        self._reconstruction_forwards = 0
        self.phase = "reconstruction"

    def freeze(self) -> None:
        if self.phase != "reconstruction":
            raise RuntimeError("joint freeze requires reconstruction phase")
        if self._reconstruction_forwards != self._target_forwards:
            raise RuntimeError("joint reconstruction targets were not fully consumed")
        if any(self._attention_targets[name]
               for name in self._attention_targets) or \
                any(self._concat_targets[name] for name in self._concat_targets):
            raise RuntimeError("joint reconstruction targets were not fully consumed")
        active_qdrop = sorted(
            site_name for site_name, quantizer in self._qdrop_quantizers.items()
            if quantizer.phase != "frozen")
        if active_qdrop:
            raise RuntimeError(
                "CompletionFormer QDrop sites are not frozen: %s" %
                active_qdrop)
        for name, controller in self.attention_controllers.items():
            if name not in self._qdrop_attention:
                controller.freeze()
        for name, controller in self.concat_controllers.items():
            if name not in self._qdrop_concat:
                controller.freeze()
        self.phase = "disabled"

    def configure(self, attention_enabled: bool, concat_enabled: bool,
                  qkv_bits: int, concat_bits: int,
                  output_bits: int) -> None:
        if self.phase not in ("disabled", "quantize"):
            raise RuntimeError("joint adapter must be frozen before configure")
        if self._qdrop_sites:
            raise RuntimeError(
                "standard joint quantization conflicts with QDrop bindings")
        for controller in self.attention_controllers.values():
            controller.disable()
        for controller in self.concat_controllers.values():
            controller.disable()
        qkv_bits = int(qkv_bits)
        concat_bits = int(concat_bits)
        output_bits = int(output_bits)
        if qkv_bits != self.qkv_bits:
            for controller in self.attention_controllers.values():
                controller.reconfigure(qkv_bits=qkv_bits)
            self.qkv_bits = qkv_bits
        if concat_bits != self.concat_bits or output_bits != self.output_bits:
            for controller in self.concat_controllers.values():
                controller.reconfigure(
                    activation_bits=concat_bits, output_bits=output_bits)
            self.concat_bits = concat_bits
            self.output_bits = output_bits
        for controller in self.attention_controllers.values():
            controller.reset_statistics()
        for controller in self.concat_controllers.values():
            controller.reset_statistics()
        self._attention_enabled = bool(attention_enabled)
        self._concat_enabled = bool(concat_enabled)
        if self._attention_enabled:
            for controller in self.attention_controllers.values():
                controller.enable()
        if self._concat_enabled:
            for controller in self.concat_controllers.values():
                controller.enable()
        self.phase = "quantize"

    def disable(self) -> None:
        if all(controller.phase != "observe"
               for controller in self.attention_controllers.values()):
            for controller in self.attention_controllers.values():
                controller.disable()
            for controller in self.concat_controllers.values():
                controller.disable()
        self._attention_enabled = False
        self._concat_enabled = False
        self.phase = "disabled"

    @staticmethod
    def _tag_rows(rows: List[Dict[str, object]], config_name: str,
                  family: str) -> List[Dict[str, object]]:
        tagged = []
        for source in rows:
            row = dict(source)
            row["config"] = str(config_name)
            row["family"] = family
            tagged.append(row)
        return tagged

    def attention_manifest_rows(self, config_name: str
                                ) -> List[Dict[str, object]]:
        rows = []
        for controller in self.attention_controllers.values():
            rows.extend(controller.manifest())
        return self._tag_rows(rows, config_name, "attention")

    def concat_manifest_rows(self, config_name: str
                             ) -> List[Dict[str, object]]:
        rows = [controller.manifest()
                for controller in self.concat_controllers.values()]
        return self._tag_rows(rows, config_name, "concat")

    def attention_metric_rows(self, config_name: str
                              ) -> List[Dict[str, object]]:
        rows = []
        for controller in self.attention_controllers.values():
            rows.extend(controller.statistics())
        return self._tag_rows(rows, config_name, "attention")

    def concat_metric_rows(self, config_name: str
                           ) -> List[Dict[str, object]]:
        rows = []
        for controller in self.concat_controllers.values():
            rows.extend(controller.statistics())
        return self._tag_rows(rows, config_name, "concat")

    def search_rows(self, config_name: str) -> List[Dict[str, object]]:
        rows = []
        for controller in self.attention_controllers.values():
            rows.extend(self._tag_rows(
                controller.search_rows(), config_name, "attention"))
        for controller in self.concat_controllers.values():
            rows.extend(self._tag_rows(
                controller.search_rows(), config_name, "concat"))
        return rows

    def calibration_metadata(self) -> Dict[str, object]:
        return {
            "target_forwards": self._target_forwards,
            "reconstruction_forwards": self._reconstruction_forwards,
            "qdrop_range_forwards": self._qdrop_range_forwards,
            "attention_modules": len(self._attention_modules),
            "concat_modules": len(self._concat_modules),
            "attention_updates": dict(
                (name, controller.observations)
                for name, controller in self.attention_controllers.items()),
            "attention_cached_samples": dict(
                (name, controller.cached_samples)
                for name, controller in self.attention_controllers.items()),
            "concat_updates": dict(
                (name, controller.observations)
                for name, controller in self.concat_controllers.items()),
            "concat_cached_samples": dict(
                (name, controller.cached_samples)
                for name, controller in self.concat_controllers.items()),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.phase = "disabled"
        self.unbind_qdrop_sites()
        self._pre_handle.remove()
        self._post_handle.remove()
        for name, module in self._attention_modules.items():
            module.forward = self._original_attention_forwards[name]
        for name, module in self._concat_modules.items():
            module.forward = self._original_concat_forwards[name]
        self._closed = True
