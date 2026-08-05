#!/usr/bin/env python3
"""QDQ primitives for a standard integer depth-completion backend."""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from scripts.outlier_mitigation_quantization import (
    apply_input_scale_to_weight,
    clipped_symmetric_weight_qdq,
    smoothquant_scale,
)
from scripts.rtn_quantization import QuantizationStats
from scripts.lognp_quantization import (
    ChannelLogNPObserver,
    LogNPActivationQuantizer,
    LogNPQuantizationStats,
    fit_bias_correction,
    fit_weight_correction,
)


class HardwareMinMaxObserver(object):
    def __init__(self):
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.samples = 0

    @property
    def observed(self):
        return self.samples > 0

    def update(self, tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        detached = tensor.detach()
        self.minimum = min(self.minimum, float(detached.min().item()))
        self.maximum = max(self.maximum, float(detached.max().item()))
        self.samples += 1

    def quantizer(self, bits, unsigned=False):
        if not self.observed:
            raise RuntimeError("cannot create a quantizer without observations")
        if unsigned:
            return UnsignedActivationQuantizer(bits, self.maximum)
        return SymmetricActivationQuantizer(
            bits, max(abs(self.minimum), abs(self.maximum)))


def _module_at(model, name):
    module = model
    if not name:
        return module
    for part in name.split("."):
        module = module._modules[part]
    return module


def _replace_module(model, name, replacement):
    parts = name.split(".")
    parent = _module_at(model, ".".join(parts[:-1]))
    parent._modules[parts[-1]] = replacement


def discover_conv_bn_pairs(model, example_args):
    if model.training:
        raise ValueError("Conv-BN discovery requires eval mode")
    names = dict((module, name) for name, module in model.named_modules())
    producers = {}
    discovered = {}
    handles = []

    def conv_hook(module, inputs, output):
        del inputs
        if torch.is_tensor(output):
            producers[id(output)] = (output, names[module])

    def bn_pre_hook(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        producer = producers.get(id(inputs[0]))
        if producer is None or producer[0] is not inputs[0]:
            return None
        conv_name = producer[1]
        bn_name = names[module]
        previous = discovered.get(bn_name)
        if previous is not None and previous != conv_name:
            raise RuntimeError("BatchNorm %s has ambiguous Conv producers" % bn_name)
        discovered[bn_name] = conv_name
        return None

    for module in names:
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_pre_hook(bn_pre_hook))
    try:
        with torch.no_grad():
            model(*example_args)
    finally:
        for handle in handles:
            handle.remove()
    return sorted([(conv_name, bn_name)
                   for bn_name, conv_name in discovered.items()])


def fold_conv_bn_pairs(model, pairs):
    if model.training:
        raise ValueError("Conv-BN folding requires eval mode")
    manifest = []
    for conv_name, bn_name in pairs:
        conv = _module_at(model, conv_name)
        bn = _module_at(model, bn_name)
        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            raise TypeError("invalid Conv-BN pair: %s -> %s" % (conv_name, bn_name))
        fused = fuse_conv_bn_eval(conv, bn)
        _replace_module(model, conv_name, fused)
        _replace_module(model, bn_name, nn.Identity())
        manifest.append({"conv": conv_name, "bn": bn_name})
    return manifest


def _detached_output(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return dict((key, _detached_output(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(_detached_output(item) for item in value)
    return value


def _maximum_output_error(reference, candidate):
    if torch.is_tensor(reference):
        return float(torch.max(torch.abs(reference - candidate)).item())
    if isinstance(reference, dict):
        common = set(reference) & set(candidate)
        return max([_maximum_output_error(reference[key], candidate[key])
                    for key in common] or [0.0])
    if isinstance(reference, (list, tuple)):
        return max([_maximum_output_error(left, right)
                    for left, right in zip(reference, candidate)] or [0.0])
    return 0.0


def discover_relu_input_producers(model, example_args):
    names = dict((module, name) for name, module in model.named_modules())
    producers = {}
    relu_calls = {}
    call_counts = {}
    handles = []

    def reset_calls(module, inputs):
        del module, inputs
        call_counts.clear()

    def conv_hook(module, inputs, output):
        del inputs
        if torch.is_tensor(output):
            producers[id(output)] = (output, names[module])

    def relu_pre_hook(module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return None
        name = names[module]
        index = call_counts.get(name, 0)
        call_counts[name] = index + 1
        producer = producers.get(id(inputs[0]))
        if producer is not None and producer[0] is inputs[0]:
            producer = producer[1]
        else:
            producer = None
        relu_calls["%s#%d" % (name, index)] = producer
        return None

    handles.append(model.register_forward_pre_hook(reset_calls))
    for module in names:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, (nn.ReLU, nn.ReLU6)):
            handles.append(module.register_forward_pre_hook(relu_pre_hook))
    try:
        with torch.no_grad():
            model(*example_args)
    finally:
        for handle in handles:
            handle.remove()
    return relu_calls


def prepare_hardware_model(model, example_args, excluded_pairs=(),
                           fold=True):
    if model.training:
        raise ValueError("hardware preparation requires eval mode")
    with torch.no_grad():
        reference = _detached_output(model(*example_args))
    # Sensitive propagation models can amplify a numerically exact local
    # Conv-BN fold into a large end-to-end change.  In reference-QDQ mode the
    # original graph is the contract, so avoid extra discovery forwards too.
    pairs = discover_conv_bn_pairs(model, example_args) if fold else []
    excluded_pairs = set(tuple(pair) for pair in excluded_pairs)
    foldable_pairs = [pair for pair in pairs
                      if tuple(pair) not in excluded_pairs]
    folded_pairs = fold_conv_bn_pairs(model, foldable_pairs) if fold else []
    if fold:
        with torch.no_grad():
            folded = _detached_output(model(*example_args))
    else:
        # No graph rewrite occurred, so do not compare two potentially
        # nondeterministic DCN/PVT forwards just to validate an identity.
        folded = reference
    relu_inputs = (discover_relu_input_producers(model, example_args)
                   if fold else {})
    return {
        "folded_pairs": folded_pairs,
        "unfolded_fanout_pairs": [
            {"conv": pair[0], "bn": pair[1]}
            for pair in pairs if tuple(pair) in excluded_pairs
        ],
        "unfolded_conv_bn_pairs": [
            {"conv": pair[0], "bn": pair[1]}
            for pair in pairs if not fold and tuple(pair) not in excluded_pairs
        ],
        "max_abs_error": _maximum_output_error(reference, folded),
        "fused_relu_producers": relu_inputs,
    }


class HardwareAlignedInstrumentor(object):
    """Hardware-contract QDQ for a model whose Conv-BN pairs are folded."""

    def __init__(self, model, group_fn, fused_relu_producers=None):
        self.model = model
        self.mode = "bypass"
        self.frozen = False
        self.w_bits = None
        self.a_bits = None
        self.enabled_groups = set()
        self.modules = {}
        self.groups = {}
        self.original_weights = {}
        self.original_biases = {}
        self.observers = {}
        self.relu_observers = {}
        self.lognp_observers = {}
        self.lognp_relu_observers = {}
        self.quantizers = {}
        self.relu_quantizers = {}
        self.lognp_quantizers = {}
        self.lognp_relu_quantizers = {}
        self.weight_scales = {}
        self.smooth_scales = {}
        self.stats = {}
        self.relu_stats = {}
        self.lognp_stats = {}
        self.lognp_relu_stats = {}
        self.compensation_modules = set()
        self.compensation_sample_limit = 0
        self.compensation_cache = {}
        self.compensation_rows = []
        self.activation_mode = "uniform"
        self.calibration_activation_mode = "uniform"
        self.handles = []
        self.relu_call_counts = {}
        self.relu_names = {}
        fused_relu_producers = fused_relu_producers or {}
        self.relu_producers = dict(fused_relu_producers)
        self.fused_relu_producers = set(
            name for name in fused_relu_producers.values() if name is not None)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                group = group_fn(name, module)
                if group is None:
                    continue
                self.modules[name] = module
                self.groups[name] = str(group)
                self.original_weights[name] = module.weight.detach().cpu().clone()
                self.original_biases[name] = None if module.bias is None else \
                    module.bias.detach().cpu().clone()
                self.observers[(name, "input")] = HardwareMinMaxObserver()
                self.lognp_observers[(name, "input")] = ChannelLogNPObserver()
                if name not in self.fused_relu_producers:
                    self.observers[(name, "output")] = HardwareMinMaxObserver()
                    self.lognp_observers[(name, "output")] = \
                        ChannelLogNPObserver()
                self.handles.append(
                    module.register_forward_pre_hook(self._make_pre_hook(name)))
                self.handles.append(
                    module.register_forward_hook(self._make_post_hook(name)))
            elif isinstance(module, (nn.ReLU, nn.ReLU6)):
                self.relu_names[module] = name
                self.handles.append(module.register_forward_hook(self._relu_hook))
        self.handles.append(model.register_forward_pre_hook(self._reset_relu_calls))

    def enable_compensation_capture(self, modules=None, sample_limit=8192):
        """Capture bounded original-domain rows during the next observe pass."""
        sample_limit = int(sample_limit)
        if sample_limit <= 0:
            raise ValueError("compensation sample_limit must be positive")
        requested = set(self.modules if modules is None else modules)
        unknown = requested - set(self.modules)
        if unknown:
            raise ValueError("unknown compensation modules: %s" % sorted(unknown))
        self.compensation_modules = requested
        self.compensation_sample_limit = sample_limit

    def _select_rows(self, inputs, target):
        limit = self.compensation_sample_limit
        if inputs.shape[0] > limit:
            indices = torch.linspace(
                0, inputs.shape[0] - 1, steps=limit, dtype=torch.long)
            return inputs.index_select(0, indices), target.index_select(0, indices)
        return inputs, target

    def _capture_module_rows(self, name, module, inputs, target):
        if name not in self.compensation_modules:
            return
        if not torch.is_tensor(inputs) or not torch.is_tensor(target):
            return
        inputs = inputs.detach().float()
        target = target.detach().float()
        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise ValueError("compensation capture requires groups=1")
            rows = F.unfold(
                inputs, module.kernel_size, module.dilation,
                module.padding, module.stride).transpose(1, 2).reshape(
                    -1, module.in_channels * module.kernel_size[0] *
                    module.kernel_size[1])
            targets = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
        elif isinstance(module, nn.Linear):
            rows = inputs.reshape(-1, module.in_features)
            targets = target.reshape(-1, module.out_features)
        else:
            return
        rows, targets = self._select_rows(rows.cpu(), targets.cpu())
        previous = self.compensation_cache.get(name)
        if previous is not None:
            rows = torch.cat((previous[0], rows))
            targets = torch.cat((previous[1], targets))
            rows, targets = self._select_rows(rows, targets)
        self.compensation_cache[name] = (rows, targets)

    def _reset_relu_calls(self, module, inputs):
        del module, inputs
        self.relu_call_counts = {}

    def _relu_hook(self, module, inputs, output):
        del inputs
        name = self.relu_names[module]
        index = self.relu_call_counts.get(name, 0)
        self.relu_call_counts[name] = index + 1
        key = "%s#%d" % (name, index)
        if self.mode == "observe":
            observer = self.relu_observers.setdefault(key, HardwareMinMaxObserver())
            observer.update(output)
            if self.calibration_activation_mode == "lognp":
                lognp_observer = self.lognp_relu_observers.setdefault(
                    key, ChannelLogNPObserver())
                lognp_observer.update(output)
            return None
        if self.mode != "quantize" or not self.enabled_groups:
            return None
        quantizer = self.relu_quantizers.get(key)
        if self.activation_mode == "lognp":
            quantizer = self.lognp_relu_quantizers.get(key)
            if quantizer is None:
                return None
            quantized, codes = quantizer.quantize_with_codes(output)
            self.lognp_relu_stats[key].update(
                output, quantized, codes=codes,
                qmin=quantizer.qmin, qmax=quantizer.qmax)
            return quantized
        if quantizer is None:
            return None
        quantized, _ = quantizer.quantize_with_codes(output)
        self.relu_stats[key].update(output, quantized)
        return quantized

    def _make_pre_hook(self, name):
        def hook(module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            tensor = inputs[0]
            if self.mode == "observe":
                self.observers[(name, "input")].update(tensor)
                if self.calibration_activation_mode == "lognp":
                    self.lognp_observers[(name, "input")].update(tensor)
                return None
            if self.mode != "quantize" or self.groups[name] not in self.enabled_groups:
                return None
            scale = self.smooth_scales.get(name)
            scale_shape = None
            quantizer_input = tensor
            if scale is not None:
                scale = scale.to(device=tensor.device, dtype=tensor.dtype)
                if isinstance(module, nn.Conv2d):
                    scale_shape = scale.reshape(1, -1, 1, 1)
                else:
                    scale_shape = scale.reshape(
                        *([1] * (tensor.ndim - 1)), scale.numel())
                quantizer_input = tensor / scale_shape
            if self.activation_mode == "lognp":
                quantizer = self.lognp_quantizers[(name, "input")]
                quantized, codes = quantizer.quantize_with_codes(tensor)
                self.lognp_stats[(name, "input")].update(
                    tensor, quantized, codes=codes,
                    qmin=quantizer.qmin, qmax=quantizer.qmax)
                return (quantized,) + tuple(inputs[1:])
            quantizer = self.quantizers[(name, "input")]
            quantized, _ = quantizer.quantize_with_codes(quantizer_input)
            comparable = quantized if scale_shape is None else quantized * scale_shape
            self.stats[(name, "input")].update(tensor, comparable)
            return (quantized,) + tuple(inputs[1:])
        return hook

    def _make_post_hook(self, name):
        def hook(module, inputs, output):
            key = (name, "output")
            if key not in self.observers or not torch.is_tensor(output):
                return None
            if self.mode == "observe":
                self.observers[key].update(output)
                if self.calibration_activation_mode == "lognp":
                    self.lognp_observers[key].update(output)
                if self.calibration_activation_mode == "lognp" and \
                        inputs and torch.is_tensor(inputs[0]):
                    self._capture_module_rows(name, module, inputs[0], output)
                return None
            if self.mode != "quantize" or self.groups[name] not in self.enabled_groups:
                return None
            if self.activation_mode == "lognp":
                quantizer = self.lognp_quantizers[key]
                quantized, codes = quantizer.quantize_with_codes(output)
                self.lognp_stats[key].update(
                    output, quantized, codes=codes,
                    qmin=quantizer.qmin, qmax=quantizer.qmax)
                return quantized
            quantizer = self.quantizers[key]
            quantized, _ = quantizer.quantize_with_codes(output)
            self.stats[key].update(output, quantized)
            return quantized
        return hook

    def _restore_parameters(self):
        with torch.no_grad():
            for name, module in self.modules.items():
                module.weight.copy_(self.original_weights[name].to(
                    device=module.weight.device, dtype=module.weight.dtype))
                original_bias = self.original_biases[name]
                if original_bias is not None:
                    module.bias.copy_(original_bias.to(
                        device=module.bias.device, dtype=module.bias.dtype))

    def observe(self, activation_mode="uniform"):
        if activation_mode not in ("uniform", "lognp"):
            raise ValueError("unknown activation mode: %s" % activation_mode)
        self._restore_parameters()
        self.mode = "observe"
        self.calibration_activation_mode = activation_mode
        self.frozen = False
        self.quantizers = {}
        self.relu_quantizers = {}
        self.lognp_quantizers = {}
        self.lognp_relu_quantizers = {}
        self.stats = {}
        self.relu_stats = {}
        self.lognp_stats = {}
        self.lognp_relu_stats = {}
        self.compensation_cache = {}
        self.compensation_rows = []
        for key in self.lognp_observers:
            self.lognp_observers[key] = ChannelLogNPObserver()
        self.lognp_relu_observers = {}

    def freeze(self):
        if not any(observer.observed for observer in self.observers.values()):
            raise RuntimeError("no hardware activation tensors were observed")
        self.frozen = True
        self.mode = "bypass"

    def configure(self, w_bits, a_bits, enabled_groups,
                  activation_overrides=None, smooth_channel_maxima=None,
                  smooth_alpha=None, weight_clip_ratio=1.0,
                  activation_bit_overrides=None, activation_mode="uniform",
                  alpha_factor=1.0, max_z=24.0,
                  lognp_per_channel=True):
        if not self.frozen:
            raise RuntimeError("calibration must be frozen before quantization")
        if activation_mode not in ("uniform", "lognp"):
            raise ValueError("unknown activation mode: %s" % activation_mode)
        self._restore_parameters()
        self.activation_mode = activation_mode
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)
        self.enabled_groups = set(enabled_groups)
        self.quantizers = {}
        self.relu_quantizers = {}
        self.lognp_quantizers = {}
        self.lognp_relu_quantizers = {}
        self.stats = {}
        self.relu_stats = {}
        self.lognp_stats = {}
        self.lognp_relu_stats = {}
        self.smooth_scales = {}
        activation_overrides = activation_overrides or {}
        activation_bit_overrides = activation_bit_overrides or {}
        smooth_channel_maxima = smooth_channel_maxima or {}
        if smooth_channel_maxima and smooth_alpha is None:
            raise ValueError("SmoothQuant maxima require an alpha")
        with torch.no_grad():
            for name, module in self.modules.items():
                if self.groups[name] not in self.enabled_groups:
                    continue
                original_weight = self.original_weights[name]
                quantization_weight = original_weight
                if name in smooth_channel_maxima:
                    scale = smoothquant_scale(
                        original_weight, smooth_channel_maxima[name], smooth_alpha)
                    self.smooth_scales[name] = scale.detach().cpu()
                    quantization_weight = apply_input_scale_to_weight(
                        original_weight, scale)
                if float(weight_clip_ratio) < 1.0:
                    quantized_weight, weight_scale = clipped_symmetric_weight_qdq(
                        quantization_weight, self.w_bits, weight_clip_ratio)
                else:
                    quantized_weight, weight_scale = symmetric_weight_qdq(
                        quantization_weight, self.w_bits)
                self.weight_scales[name] = weight_scale
                weight_stats = QuantizationStats()
                weight_stats.update(quantization_weight, quantized_weight)
                self.stats[(name, "weight")] = weight_stats
                module.weight.copy_(quantized_weight.to(
                    device=module.weight.device, dtype=module.weight.dtype))
                for kind in ("input", "output"):
                    key = (name, kind)
                    if self.activation_mode == "lognp":
                        lognp_observer = self.lognp_observers.get(key)
                        if lognp_observer is None or not lognp_observer.observed:
                            continue
                        bits = int(activation_bit_overrides.get(
                            key, activation_bit_overrides.get(
                                name, self.a_bits)))
                        minimum = lognp_observer.minimum
                        unsigned = kind == "input" and minimum >= 0.0
                        lognp_observer.freeze(
                            bits, unsigned, alpha_factor=alpha_factor,
                            max_z=max_z, per_channel=lognp_per_channel)
                        self.lognp_quantizers[key] = lognp_observer.quantizer()
                        self.lognp_stats[key] = LogNPQuantizationStats()
                        continue
                    observer = self.observers.get(key)
                    if observer is None or not observer.observed:
                        continue
                    bits = int(activation_bit_overrides.get(
                        key, activation_bit_overrides.get(name, self.a_bits)))
                    unsigned = kind == "input" and observer.minimum >= 0.0
                    maximum = activation_overrides.get(key)
                    if kind == "input" and name in self.smooth_scales:
                        activation_max = torch.as_tensor(
                            smooth_channel_maxima[name], dtype=torch.float32)
                        maximum = float((activation_max /
                                         self.smooth_scales[name]).max().item())
                    if maximum is None:
                        quantizer = observer.quantizer(
                            bits, unsigned=unsigned)
                    elif unsigned:
                        quantizer = UnsignedActivationQuantizer(
                            bits, maximum)
                    else:
                        quantizer = SymmetricActivationQuantizer(
                            bits, maximum)
                    self.quantizers[key] = quantizer
                    self.stats[key] = QuantizationStats()
                original_bias = self.original_biases[name]
                if self.activation_mode == "lognp":
                    continue
                if original_bias is not None:
                    input_scale = self.quantizers[(name, "input")].scale
                    quantized_bias, _, bias_scale = int32_bias_qdq(
                        original_bias, input_scale, weight_scale)
                    module.bias.copy_(quantized_bias.to(
                        device=module.bias.device, dtype=module.bias.dtype))
                    bias_stats = QuantizationStats()
                    bias_stats.update(original_bias, quantized_bias)
                    bias_stats.bias_scale = bias_scale
                    self.stats[(name, "bias")] = bias_stats
            for key, observer in self.relu_observers.items():
                if observer.observed:
                    producer = self.relu_producers.get(key)
                    bits = int(activation_bit_overrides.get(
                        key, activation_bit_overrides.get(
                            producer, self.a_bits)))
                    if self.activation_mode == "lognp":
                        lognp_observer = self.lognp_relu_observers[key]
                        lognp_observer.freeze(
                            bits, unsigned=True, alpha_factor=alpha_factor,
                            max_z=max_z, per_channel=lognp_per_channel)
                        self.lognp_relu_quantizers[key] = \
                            lognp_observer.quantizer()
                        self.lognp_relu_stats[key] = LogNPQuantizationStats()
                        continue
                    self.relu_quantizers[key] = observer.quantizer(
                        bits, unsigned=True)
                    self.relu_stats[key] = QuantizationStats()
        self.mode = "quantize"

    def disable(self):
        self._restore_parameters()
        self.mode = "bypass"
        self.activation_mode = "uniform"
        self.enabled_groups = set()

    def module_groups(self):
        return dict(self.groups)

    def manifest(self):
        rows = []
        if self.activation_mode == "lognp":
            for (name, kind), quantizer in sorted(
                    self.lognp_quantizers.items()):
                row = dict(quantizer.manifest())
                row.update({"module": name, "kind": kind})
                rows.append(row)
            for key, quantizer in sorted(self.lognp_relu_quantizers.items()):
                row = dict(quantizer.manifest())
                row.update({"module": key, "kind": "relu_output"})
                rows.append(row)
            return rows
        for (name, kind), quantizer in sorted(self.quantizers.items()):
            rows.append({
                "module": name, "kind": kind,
                "bits": quantizer.bits,
                "unsigned": isinstance(quantizer, UnsignedActivationQuantizer),
                "qmin": quantizer.qmin, "qmax": quantizer.qmax,
                "scale": quantizer.scale,
            })
        for key, quantizer in sorted(self.relu_quantizers.items()):
            rows.append({
                "module": key, "kind": "relu_output", "unsigned": True,
                "bits": quantizer.bits,
                "qmin": quantizer.qmin, "qmax": quantizer.qmax,
                "scale": quantizer.scale,
            })
        return rows

    def metadata(self):
        if self.activation_mode == "lognp":
            return {
                "activation_mode": "lognp",
                "bias_contract": "reference_float_reconstruction",
                "quantization_execution": "float_qdq_reference",
            }
        return {
            "activation_mode": "uniform",
            "bias_contract": "int32 scale=sx*sw[o]",
            "quantization_execution": "hardware_aligned_qdq",
        }

    def apply_compensation(self, method="bias", ridge=1e-4):
        """Apply calibration-only bias or weight compensation to LogNP QDQ."""
        if self.activation_mode != "lognp" or self.mode != "quantize":
            raise RuntimeError("LogNP quantization must be active")
        if method not in ("bias", "weight"):
            raise ValueError("unknown compensation method: %s" % method)
        rows = []
        with torch.no_grad():
            for name in sorted(self.compensation_modules):
                cached = self.compensation_cache.get(name)
                quantizer = self.lognp_quantizers.get((name, "input"))
                if cached is None or quantizer is None:
                    continue
                module = self.modules[name]
                inputs, target = cached
                if isinstance(module, nn.Conv2d):
                    kernel_area = int(module.kernel_size[0] *
                                      module.kernel_size[1])
                    patch_quantizer = LogNPActivationQuantizer(
                        quantizer.bits,
                        quantizer.alpha.repeat_interleave(kernel_area),
                        quantizer.scale.repeat_interleave(kernel_area),
                        quantizer.unsigned, quantizer.max_z)
                    reconstructed = patch_quantizer(inputs)
                else:
                    reconstructed = quantizer(inputs)
                reconstructed = reconstructed.reshape(
                    -1, module.weight.reshape(module.weight.shape[0], -1).shape[1])
                target = target.reshape(-1, module.weight.shape[0])
                weight = module.weight.detach().float().cpu().reshape(
                    module.weight.shape[0], -1)
                bias = (module.bias.detach().float().cpu()
                        if module.bias is not None else
                        torch.zeros(module.weight.shape[0]))
                before = torch.matmul(reconstructed, weight.t()) + bias
                before_mse = float(torch.mean((before - target) ** 2).item())
                applied = False
                correction_norm = 0.0
                if method == "bias" and module.bias is not None:
                    corrected_bias = fit_bias_correction(target, before, bias)
                    module.bias.copy_(corrected_bias.to(
                        device=module.bias.device, dtype=module.bias.dtype))
                    correction_norm = float(torch.norm(corrected_bias - bias).item())
                    applied = True
                    after_mse = float(torch.mean(
                        (before - bias + corrected_bias - target) ** 2).item())
                elif method == "weight":
                    target_without_bias = target - bias
                    fitted = fit_weight_correction(
                        reconstructed, target_without_bias, ridge=ridge)
                    quantized, _ = symmetric_weight_qdq(fitted, self.w_bits)
                    after = torch.matmul(reconstructed, quantized.t()) + bias
                    after_mse = float(torch.mean((after - target) ** 2).item())
                    if torch.isfinite(quantized).all() and \
                            after_mse <= before_mse * 1.01:
                        module.weight.copy_(quantized.reshape_as(module.weight).to(
                            device=module.weight.device,
                            dtype=module.weight.dtype))
                        correction_norm = float(torch.norm(quantized - weight).item())
                        applied = True
                    else:
                        after_mse = before_mse
                else:
                    after_mse = before_mse
                rows.append({
                    "module": name, "method": method,
                    "rows": int(inputs.shape[0]),
                    "before_mse": before_mse,
                    "after_mse": after_mse,
                    "correction_norm": correction_norm,
                    "applied": int(applied),
                })
        self.compensation_rows = rows
        return list(rows)

    def statistics(self):
        rows = []
        if self.activation_mode == "lognp":
            for (name, kind), stats in sorted(self.lognp_stats.items()):
                rows.append({
                    "module": name, "group": self.groups[name],
                    "kind": kind, "numel": stats.numel,
                    "mse": stats.mse, "sqnr_db": stats.sqnr_db,
                    "transformed_sqnr_db": stats.transformed_sqnr_db,
                    "p50": stats.p50, "p75": stats.p75,
                    "p99": stats.p99, "p99_9": stats.p99_9,
                    "saturation_rate": stats.saturation_rate,
                    "zero_code_rate": stats.zero_code_rate,
                    "nonfinite_rate": stats.nonfinite_rate,
                    "sign_flip_rate": stats.sign_flip_rate,
                })
            for key, stats in sorted(self.lognp_relu_stats.items()):
                rows.append({
                    "module": key, "group": "relu",
                    "kind": "relu_output", "numel": stats.numel,
                    "mse": stats.mse, "sqnr_db": stats.sqnr_db,
                    "transformed_sqnr_db": stats.transformed_sqnr_db,
                    "p50": stats.p50, "p75": stats.p75,
                    "p99": stats.p99, "p99_9": stats.p99_9,
                    "saturation_rate": stats.saturation_rate,
                    "zero_code_rate": stats.zero_code_rate,
                    "nonfinite_rate": stats.nonfinite_rate,
                    "sign_flip_rate": stats.sign_flip_rate,
                })
            return rows
        for (name, kind), stats in sorted(self.stats.items()):
            row = {
                "module": name, "group": self.groups[name], "kind": kind,
                "numel": stats.numel, "mse": stats.mse,
                "sqnr_db": stats.sqnr_db, "cosine": stats.cosine,
                "saturation_rate": stats.saturation_rate,
                "sign_flip_rate": stats.sign_flip_rate,
            }
            if kind == "bias":
                row["scale_min"] = float(stats.bias_scale.min().item())
                row["scale_max"] = float(stats.bias_scale.max().item())
            rows.append(row)
        return rows

    def close(self):
        self.disable()
        for handle in self.handles:
            handle.remove()
        self.handles = []


def symmetric_weight_qdq(weight, bits):
    if bits < 2:
        raise ValueError("weight bits must be at least 2")
    if weight.ndim < 2:
        raise ValueError("weight must have an output-channel dimension")
    qmax = 2 ** (bits - 1) - 1
    flat = weight.reshape(weight.shape[0], -1)
    maximum = flat.abs().max(dim=1)[0]
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
    scale = (safe_maximum / float(qmax)).reshape(shape)
    codes = torch.round(weight / scale).clamp(-qmax, qmax)
    return codes * scale, scale


class SymmetricActivationQuantizer(object):
    def __init__(self, bits, maximum):
        if bits < 2:
            raise ValueError("activation bits must be at least 2")
        self.bits = int(bits)
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -self.qmax
        self.maximum = abs(float(maximum))
        self.scale = self.maximum / float(self.qmax) if self.maximum > 0 else 1.0
        self.zero_point = 0

    def quantize_with_codes(self, tensor):
        codes = torch.round(tensor / self.scale).clamp(self.qmin, self.qmax)
        return codes * self.scale, codes.to(torch.int32)

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


class UnsignedActivationQuantizer(object):
    def __init__(self, bits, maximum):
        if bits < 1:
            raise ValueError("activation bits must be positive")
        self.bits = int(bits)
        self.qmin = 0
        self.qmax = 2 ** bits - 1
        self.maximum = max(float(maximum), 0.0)
        self.scale = self.maximum / float(self.qmax) if self.maximum > 0 else 1.0
        self.zero_point = 0

    def quantize_with_codes(self, tensor):
        codes = torch.round(tensor / self.scale).clamp(self.qmin, self.qmax)
        return codes * self.scale, codes.to(torch.int32)

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


def int32_bias_qdq(bias, input_scale, weight_scale):
    if bias.ndim != 1:
        raise ValueError("bias must have one value per output channel")
    flat_weight_scale = weight_scale.reshape(weight_scale.shape[0], -1)[:, 0]
    scale = flat_weight_scale * torch.as_tensor(
        input_scale, device=bias.device, dtype=bias.dtype)
    scale = scale.to(device=bias.device, dtype=bias.dtype)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    limits = torch.iinfo(torch.int32)
    codes = torch.round(bias / safe_scale).clamp(limits.min, limits.max).to(torch.int32)
    return codes.to(bias.dtype) * safe_scale, codes, safe_scale
