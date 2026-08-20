#!/usr/bin/env python3
"""QDQ primitives for a standard integer depth-completion backend."""

from __future__ import division

import copy

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from scripts.rtn_quantization import QuantizationStats
from spn_quant.specs import QuantSpec


HARDWARE_INTEGER_BITS = (2, 4, 6, 8)


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
        if not bool(torch.isfinite(detached).all().item()):
            raise ValueError("activation calibration tensor must be finite")
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

class ChannelMinMaxObserver(object):
    """Per-channel min/max observer for Conv activation boundaries."""

    def __init__(self, channel_dim=1):
        self.channel_dim = int(channel_dim)
        self.minimum = None
        self.maximum = None
        self.square_sum = None
        self.scalar_count = 0
        self.samples = 0

    @property
    def observed(self):
        return self.samples > 0 and self.minimum is not None

    def update(self, tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        if tensor.ndim <= self.channel_dim:
            raise ValueError("channel observer requires a channel dimension")
        values = tensor.detach().movedim(self.channel_dim, 0).reshape(
            tensor.shape[self.channel_dim], -1)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("activation calibration tensor must be finite")
        minimum = values.min(dim=1).values.cpu()
        maximum = values.max(dim=1).values.cpu()
        square_sum = values.to(torch.float64).square().sum(dim=1).cpu()
        if self.minimum is None:
            self.minimum = minimum
            self.maximum = maximum
            self.square_sum = square_sum
        else:
            self.minimum = torch.minimum(self.minimum, minimum)
            self.maximum = torch.maximum(self.maximum, maximum)
            self.square_sum += square_sum
        self.scalar_count += int(values.shape[1])
        self.samples += 1

    def channel_rms(self):
        if not self.observed or self.square_sum is None or \
                self.scalar_count <= 0:
            raise RuntimeError("channel RMS requires observations")
        return torch.sqrt(
            self.square_sum / float(self.scalar_count)).to(torch.float32)

    def quantizer(self, bits, unsigned=False):
        if not self.observed:
            raise RuntimeError("cannot create a quantizer without observations")
        return ChannelActivationQuantizer(
            bits, self.minimum, self.maximum, self.channel_dim, unsigned)

    def quantizer_for(self, spec, unsigned=None, maximum=None):
        if not self.observed:
            raise RuntimeError("cannot create a quantizer without observations")
        if not isinstance(spec, QuantSpec):
            raise TypeError("activation spec must be QuantSpec")
        if spec.observer != "minmax" or spec.transform != "none":
            raise ValueError("activation QDQ requires untransformed MinMax")
        declared_unsigned = not spec.signed
        if unsigned is not None and bool(unsigned) != declared_unsigned:
            raise ValueError("activation spec unsigned contract does not match site")
        if declared_unsigned:
            if spec.scheme != "affine" or not spec.preserve_zero:
                raise ValueError("unsigned activation must preserve zero")
        elif spec.scheme != "symmetric":
            raise ValueError("signed activation must use symmetric QDQ")

        if spec.dynamic:
            if maximum is not None:
                raise ValueError(
                    "dynamic activation cannot use a static maximum")
            if spec.granularity == "tensor":
                return DynamicTensorActivationQuantizer(
                    spec.bits, declared_unsigned)
            if spec.granularity == "channel":
                raise ValueError(
                    "dynamic channel activation is outside this contract")
            if spec.axis != self.channel_dim:
                raise ValueError(
                    "activation spec axis does not match channel dimension")
            channels = int(self.minimum.numel())
            group_size = int(spec.group_size)
            if channels % group_size != 0:
                raise ValueError("group size must divide activation channels")
            return DynamicGroupedActivationQuantizer(
                spec.bits, self.channel_dim, group_size, channels,
                declared_unsigned)

        if spec.granularity == "tensor":
            if maximum is None:
                extent = self.maximum if declared_unsigned else torch.maximum(
                    self.minimum.abs(), self.maximum.abs())
                maximum = float(extent.max().item())
            return UnsignedActivationQuantizer(spec.bits, maximum) \
                if declared_unsigned else \
                SymmetricActivationQuantizer(spec.bits, maximum)
        if spec.axis != self.channel_dim:
            raise ValueError("activation spec axis does not match channel dimension")
        if spec.granularity == "channel":
            if maximum is not None:
                maximum = torch.as_tensor(maximum, dtype=torch.float32)
                if maximum.numel() != self.minimum.numel():
                    raise ValueError("channel maximum count does not match channels")
                minimum = torch.zeros_like(maximum) \
                    if declared_unsigned else -maximum
            else:
                minimum = self.minimum
                maximum = self.maximum
            return ChannelActivationQuantizer(
                spec.bits, minimum, maximum, self.channel_dim,
                declared_unsigned)

        channels = int(self.minimum.numel())
        group_size = int(spec.group_size)
        if channels % group_size != 0:
            raise ValueError("group size must divide activation channels")
        group_count = channels // group_size
        if maximum is None:
            minimum = self.minimum.reshape(group_count, group_size).amin(dim=1)
            maximum = self.maximum.reshape(group_count, group_size).amax(dim=1)
        else:
            maximum = torch.as_tensor(maximum, dtype=torch.float32)
            if maximum.numel() != group_count:
                raise ValueError("group maximum count does not match groups")
            minimum = torch.zeros_like(maximum) \
                if declared_unsigned else -maximum
        return GroupedActivationQuantizer(
            spec.bits, minimum, maximum, self.channel_dim,
            group_size, channels, declared_unsigned)

def activation_maximum_for_spec(spec, channel_maximum):
    if not isinstance(spec, QuantSpec):
        raise TypeError("activation spec must be QuantSpec")
    maximum = torch.as_tensor(channel_maximum, dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(maximum).all().item()) or \
            bool((maximum < 0.0).any().item()):
        raise ValueError("activation channel maxima must be finite and nonnegative")
    if spec.granularity == "tensor":
        return float(maximum.max().item())
    if spec.granularity == "channel":
        return maximum
    group_size = int(spec.group_size)
    if maximum.numel() % group_size != 0:
        raise ValueError("group size must divide activation channels")
    return maximum.reshape(-1, group_size).amax(dim=1)


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
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
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
        if not isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d)) or \
                not isinstance(bn, nn.BatchNorm2d):
            raise TypeError("invalid Conv-BN pair: %s -> %s" % (conv_name, bn_name))
        fused = _fuse_conv_transpose_bn_eval(conv, bn) \
            if isinstance(conv, nn.ConvTranspose2d) else \
            fuse_conv_bn_eval(conv, bn)
        _replace_module(model, conv_name, fused)
        _replace_module(model, bn_name, nn.Identity())
        manifest.append({"conv": conv_name, "bn": bn_name})
    return manifest


def _fuse_conv_transpose_bn_eval(conv, bn):
    if conv.training or bn.training:
        raise ValueError("Conv-BN folding requires eval mode")
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm running statistics are required")
    if not isinstance(conv, nn.ConvTranspose2d):
        raise TypeError("expected ConvTranspose2d")
    if conv.groups != 1:
        raise ValueError("ConvTranspose2d BN folding requires groups=1")

    running_mean = bn.running_mean
    running_var = bn.running_var
    bn_weight = torch.ones_like(running_mean) \
        if bn.weight is None else bn.weight
    bn_bias = torch.zeros_like(running_mean) \
        if bn.bias is None else bn.bias
    conv_bias = torch.zeros_like(running_mean) \
        if conv.bias is None else conv.bias
    coefficient = bn_weight * torch.rsqrt(running_var + bn.eps)
    shape = [1, coefficient.numel()] + [1] * (conv.weight.ndim - 2)
    weight = conv.weight * coefficient.reshape(shape)
    bias = (conv_bias - running_mean) * coefficient + bn_bias

    fused = copy.deepcopy(conv)
    fused.weight = nn.Parameter(weight, requires_grad=conv.weight.requires_grad)
    fused.bias = nn.Parameter(bias, requires_grad=conv.weight.requires_grad)
    return fused


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


def _primary_output_error(reference, candidate):
    """Compare the model's primary depth prediction when outputs are structured.

    Auxiliary propagation tensors such as offsets may accumulate small floating
    point differences after Conv-BN folding.  The folded graph is accepted only
    on the depth prediction, while the full auxiliary error remains reported.
    """
    if isinstance(reference, dict) and isinstance(candidate, dict):
        for key in ("pred", "prediction", "depth", "output"):
            if key in reference and key in candidate:
                return _maximum_output_error(reference[key], candidate[key])
    return _maximum_output_error(reference, candidate)


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
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
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
        "primary_max_abs_error": _primary_output_error(reference, folded),
        "fused_relu_producers": relu_inputs,
    }


class HardwareAlignedInstrumentor(object):
    """Hardware-contract QDQ for a model whose Conv-BN pairs are folded."""

    def __init__(self, model, group_fn, fused_relu_producers=None,
                 fuse_layernorm=True, externally_owned_outputs=None,
                 per_channel_activation_inputs=None,
                 externally_owned_inputs=None):
        self.model = model
        self.mode = "bypass"
        self.frozen = False
        self.w_bits = None
        self.weight_bits = {}
        self.a_bits = None
        self.enabled_groups = set()
        self.modules = {}
        self.groups = {}
        self.original_weights = {}
        self.original_biases = {}
        self.observers = {}
        self.channel_observers = {}
        self.relu_observers = {}
        self.relu_channel_observers = {}
        self.quantizers = {}
        self.relu_quantizers = {}
        self._skipped_output_modules = set()
        self._externally_owned_outputs = set(externally_owned_outputs or ())
        self._externally_owned_inputs = set(externally_owned_inputs or ())
        self._active_externally_owned_outputs = \
            set(self._externally_owned_outputs)
        self._active_externally_owned_inputs = \
            set(self._externally_owned_inputs)
        self.external_output_ownership = True
        self._per_channel_activation_modules = set() \
            if per_channel_activation_inputs is None else \
            set(per_channel_activation_inputs)
        self._layernorm_fusion_pairs = []
        self._layernorm_output_modules = set()
        self.weight_scales = {}
        self.stats = {}
        self.relu_stats = {}
        self.quantize_bias = True
        self.activation_recorder = None
        self.runtime_statistics_enabled = True
        self.calibration_recorder = None
        self.calibration_owners = set()
        self.activation_call_counts = {}
        self.handles = []
        self.relu_call_counts = {}
        self.relu_names = {}
        self.relu_module_groups = {}
        fused_relu_producers = fused_relu_producers or {}
        self.relu_producers = dict(fused_relu_producers)
        self.fused_relu_producers = set(
            name for name in fused_relu_producers.values() if name is not None)

        named_modules = dict(model.named_modules())
        quantized_types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        quantized_names = set(
            name for name, module in named_modules.items()
            if isinstance(module, quantized_types))
        unknown_per_channel_inputs = self._per_channel_activation_modules - \
            quantized_names
        if unknown_per_channel_inputs:
            raise ValueError("unknown per-channel activation inputs: %s" %
                             sorted(unknown_per_channel_inputs))
        unknown_owned_outputs = self._externally_owned_outputs - \
            set(named_modules)
        if unknown_owned_outputs:
            raise ValueError("unknown externally owned outputs: %s" %
                             sorted(unknown_owned_outputs))
        unknown_owned_inputs = self._externally_owned_inputs - quantized_names
        if unknown_owned_inputs:
            raise ValueError("unknown externally owned inputs: %s" %
                             sorted(unknown_owned_inputs))
        conflicting_inputs = self._externally_owned_inputs & \
            self._per_channel_activation_modules
        if conflicting_inputs:
            raise ValueError(
                "externally owned inputs cannot use generic per-channel QDQ: %s" %
                sorted(conflicting_inputs))
        fused_conv_to_norm = {}
        if fuse_layernorm:
            for parent_name, parent in named_modules.items():
                norm = getattr(parent, "norm", None)
                if not isinstance(norm, nn.LayerNorm):
                    continue
                norm_name = "%s.norm" % parent_name if parent_name else "norm"
                for attr in ("proj", "sr"):
                    conv = getattr(parent, attr, None)
                    if not isinstance(conv, nn.Conv2d):
                        continue
                    conv_name = "%s.%s" % (parent_name, attr) \
                        if parent_name else attr
                    if named_modules.get(conv_name) is not conv:
                        continue
                    fused_conv_to_norm[conv_name] = norm_name
                    self._layernorm_fusion_pairs.append({
                        "conv": conv_name, "layernorm": norm_name,
                    })
                    self._layernorm_output_modules.add(norm_name)
        for name, module in model.named_modules():
            if isinstance(module, quantized_types):
                if isinstance(module, nn.ConvTranspose2d) and module.groups != 1:
                    raise ValueError(
                        "ConvTranspose2d quantization requires groups=1: %s" %
                        name)
                group = group_fn(name, module)
                if group is None:
                    if name in self._per_channel_activation_modules:
                        raise ValueError(
                            "per-channel activation input has no group: %s" %
                            name)
                    continue
                self.modules[name] = module
                self.groups[name] = str(group)
                self.original_weights[name] = module.weight.detach().cpu().clone()
                self.original_biases[name] = None if module.bias is None else \
                    module.bias.detach().cpu().clone()
                per_channel_input = name in \
                    self._per_channel_activation_modules
                observer_type = ChannelMinMaxObserver if per_channel_input \
                    else HardwareMinMaxObserver
                self.observers[(name, "input")] = observer_type()
                if per_channel_input:
                    self.channel_observers[(name, "input")] = \
                        self.observers[(name, "input")]
                else:
                    self.channel_observers[(name, "input")] = \
                        ChannelMinMaxObserver(
                            self._activation_channel_dim(module))
                skip_output = name in fused_conv_to_norm
                if skip_output:
                    self._skipped_output_modules.add(name)
                if name not in self.fused_relu_producers and not skip_output:
                    self.observers[(name, "output")] = \
                        HardwareMinMaxObserver()
                    self.channel_observers[(name, "output")] = \
                        ChannelMinMaxObserver(
                            self._activation_channel_dim(module))
                self.handles.append(
                    module.register_forward_pre_hook(self._make_pre_hook(name)))
                self.handles.append(
                    module.register_forward_hook(self._make_post_hook(name)))
            elif isinstance(module, nn.LayerNorm) and \
                    name in self._layernorm_output_modules:
                conv_name = next(
                    row["conv"] for row in self._layernorm_fusion_pairs
                    if row["layernorm"] == name)
                self.groups[name] = self.groups[conv_name]
                self.observers[(name, "output")] = \
                    ChannelMinMaxObserver(channel_dim=-1)
                self.channel_observers[(name, "output")] = \
                    self.observers[(name, "output")]
                self.handles.append(
                    module.register_forward_hook(self._make_post_hook(name)))
            elif isinstance(module, (nn.ReLU, nn.ReLU6)):
                group = group_fn(name, module)
                if group is None:
                    continue
                self.relu_names[module] = name
                self.relu_module_groups[name] = str(group)
                self.handles.append(module.register_forward_hook(self._relu_hook))
        self.handles.append(model.register_forward_pre_hook(self._reset_relu_calls))

    def _reset_relu_calls(self, module, inputs):
        del module, inputs
        self.relu_call_counts = {}
        self.activation_call_counts = {}

    def set_activation_recorder(self, recorder):
        if recorder is None or not callable(recorder.record):
            raise TypeError("activation recorder must define record")
        self.activation_recorder = recorder

    def clear_activation_recorder(self):
        self.activation_recorder = None

    def set_calibration_recorder(self, recorder, owners):
        if recorder is None or not callable(recorder.record_reference):
            raise TypeError(
                "calibration recorder must define record_reference")
        owners = set(tuple(owner) for owner in owners)
        available = set(
            key if isinstance(key, tuple) else (key, "relu_output")
            for key in self.activation_site_keys(self._known_groups()))
        if owners != available:
            raise ValueError(
                "calibration owners must match activation sites: missing=%s extra=%s" %
                (sorted(available - owners), sorted(owners - available)))
        self.calibration_recorder = recorder
        self.calibration_owners = owners

    def clear_calibration_recorder(self):
        self.calibration_recorder = None
        self.calibration_owners = set()

    def _record_calibration(self, name, kind, group, tensor, channel_dim):
        if self.calibration_recorder is None or \
                (name, kind) not in self.calibration_owners:
            return
        self.calibration_recorder.record_reference(
            name, kind, group, tensor, channel_dim)

    def _update_uniform_observers(self, key, tensor):
        observer = self.observers[key]
        observer.update(tensor)
        channel_observer = self.channel_observers[key]
        if channel_observer is not observer:
            channel_observer.update(tensor)

    @staticmethod
    def _activation_channel_dim(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            return 1
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            return -1
        raise TypeError("unsupported activation module: %s" %
                        type(module).__name__)

    def _relu_channel_dim(self, key, output):
        owner = self._relu_owner(key)[0]
        if owner in self.modules:
            return self._activation_channel_dim(self.modules[owner])
        if output.ndim == 4:
            return 1
        return -1

    def _record_activation(self, name, kind, group, reference,
                           quantized, codes, quantizer, channel_dim):
        if self.activation_recorder is None:
            return
        key = (name, kind)
        call_index = self.activation_call_counts[key] \
            if key in self.activation_call_counts else 0
        self.activation_call_counts[key] = call_index + 1
        self.activation_recorder.record(
            name, kind, call_index, group, reference,
            quantized, codes, quantizer, channel_dim)

    @staticmethod
    def _override_value(overrides, key, module_name, default_value):
        if key in overrides:
            return overrides[key]
        if module_name in overrides:
            return overrides[module_name]
        return default_value

    @staticmethod
    def _activation_quantizer(observer, bits, unsigned,
                              maximum=None, spec=None,
                              channel_observer=None):
        if spec is not None:
            if channel_observer is None:
                raise ValueError("QuantSpec activation requires channel ranges")
            if int(bits) != int(spec.bits):
                raise ValueError("activation bits do not match QuantSpec")
            return channel_observer.quantizer_for(
                spec, unsigned=unsigned, maximum=maximum)
        if maximum is None:
            return observer.quantizer(bits, unsigned=unsigned)
        if unsigned:
            return UnsignedActivationQuantizer(bits, maximum)
        return SymmetricActivationQuantizer(bits, maximum)

    def _relu_hook(self, module, inputs, output):
        del inputs
        name = self.relu_names[module]
        index = self.relu_call_counts.get(name, 0)
        self.relu_call_counts[name] = index + 1
        key = "%s#%d" % (name, index)
        if self.mode == "observe":
            observer = self.relu_observers.setdefault(key, HardwareMinMaxObserver())
            observer.update(output)
            channel_observer = self.relu_channel_observers.setdefault(
                key, ChannelMinMaxObserver(
                    self._relu_channel_dim(key, output)))
            channel_observer.update(output)
            self._record_calibration(
                key, "relu_output", self._relu_owner(key)[1], output,
                self._relu_channel_dim(key, output))
            return None
        if self.mode != "quantize" or not self.enabled_groups:
            return None
        quantizer = self.relu_quantizers[key] \
            if key in self.relu_quantizers else None
        if quantizer is None:
            return None
        quantized, codes = quantizer.quantize_with_codes(output)
        if self.runtime_statistics_enabled:
            update_activation_stats(
                self.relu_stats[key], quantizer,
                output, quantized, codes, output)
        self._record_activation(
            key, "relu_output", self._relu_owner(key)[1],
            output, quantized, codes, quantizer,
            self._relu_channel_dim(key, output))
        return quantized

    def _relu_owner(self, key):
        producer = self.relu_producers[key] \
            if key in self.relu_producers else None
        if producer is not None:
            return producer, self.groups[producer]
        module_name = key.rpartition("#")[0]
        return module_name, self.relu_module_groups[module_name]

    def _channel_observer_for_site(self, key):
        if isinstance(key, str):
            return self.relu_channel_observers[key]
        return self.channel_observers[key]

    def _make_pre_hook(self, name):
        def hook(module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            tensor = inputs[0]
            key = (name, "input")
            if key not in self.observers:
                return None
            if self.mode == "observe":
                self._update_uniform_observers(key, tensor)
                self._record_calibration(
                    name, "input", self.groups[name], tensor,
                    self._activation_channel_dim(module))
                return None
            if self.mode != "quantize" or self.groups[name] not in self.enabled_groups:
                return None
            quantizer = self.quantizers[(name, "input")] \
                if (name, "input") in self.quantizers else None
            if quantizer is None:
                return None
            quantized, codes = quantizer.quantize_with_codes(tensor)
            if self.runtime_statistics_enabled:
                update_activation_stats(
                    self.stats[(name, "input")], quantizer,
                    tensor, quantized, codes, tensor)
            self._record_activation(
                name, "input", self.groups[name], tensor,
                quantized, codes, quantizer,
                self._activation_channel_dim(module))
            return (quantized,) + tuple(inputs[1:])
        return hook

    def _make_post_hook(self, name):
        def hook(module, inputs, output):
            key = (name, "output")
            if key not in self.observers or not torch.is_tensor(output):
                return None
            if self.mode == "observe":
                self._update_uniform_observers(key, output)
                self._record_calibration(
                    name, "output", self.groups[name], output,
                    self._activation_channel_dim(module))
                return None
            if self.mode != "quantize" or self.groups[name] not in self.enabled_groups:
                return None
            quantizer = self.quantizers[key] \
                if key in self.quantizers else None
            if quantizer is None:
                return None
            quantized, codes = quantizer.quantize_with_codes(output)
            if self.runtime_statistics_enabled:
                update_activation_stats(
                    self.stats[key], quantizer,
                    output, quantized, codes, output)
            self._record_activation(
                name, "output", self.groups[name], output,
                quantized, codes, quantizer,
                self._activation_channel_dim(module))
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

    def refresh_parameter_sources(self):
        if self.mode != "bypass":
            raise RuntimeError(
                "parameter sources require bypass instrumentor mode")
        for name, module in self.modules.items():
            has_bias = module.bias is not None
            if has_bias != (self.original_biases[name] is not None):
                raise RuntimeError(
                    "module bias structure changed: %s" % name)
            self.original_weights[name] = \
                module.weight.detach().cpu().clone()
            self.original_biases[name] = None if module.bias is None else \
                module.bias.detach().cpu().clone()

    def observe(self):
        self._restore_parameters()
        self.mode = "observe"
        self.frozen = False
        self.quantizers = {}
        self.relu_quantizers = {}
        self.stats = {}
        self.relu_stats = {}

    def freeze(self):
        if not any(observer.observed for observer in self.observers.values()):
            raise RuntimeError("no hardware activation tensors were observed")
        self.frozen = True
        self.mode = "bypass"

    def _known_groups(self):
        return set(self.groups.values()) | set(self.relu_module_groups.values())

    def _validate_component_groups(self, groups, component):
        unknown = set(groups) - self._known_groups()
        if unknown:
            raise ValueError("unknown %s groups: %s" %
                             (component, sorted(unknown)))

    def activation_site_keys(self, groups):
        groups = set(groups)
        self._validate_component_groups(groups, "activation")
        keys = []
        for key, observer in self.channel_observers.items():
            name, kind = key
            if self.groups[name] not in groups or not observer.observed:
                continue
            if kind == "input" and name in self._active_externally_owned_inputs:
                continue
            if kind == "output" and self.external_output_ownership and \
                    name in self._active_externally_owned_outputs:
                continue
            keys.append(key)
        for key, observer in self.relu_channel_observers.items():
            if observer.observed and self._relu_owner(key)[1] in groups:
                keys.append(key)
        return tuple(sorted(keys, key=str))

    def tensor_activation_specs(self, bits, groups):
        specs = {}
        for key in self.activation_site_keys(groups):
            if isinstance(key, str):
                specs[key] = QuantSpec.unsigned_tensor(int(bits))
                continue
            name, kind = key
            observer = self.observers[(name, kind)]
            minimum = observer.minimum
            nonnegative = bool(torch.all(minimum >= 0.0).item()) \
                if torch.is_tensor(minimum) else minimum >= 0.0
            specs[key] = QuantSpec.unsigned_tensor(int(bits)) \
                if kind == "input" and nonnegative else \
                QuantSpec.signed_tensor(int(bits))
        return specs

    def configure_components(self, w_bits, a_bits, weight_groups,
                             activation_groups, activation_specs,
                             quantize_bias):
        return self.configure_components_with_ranges(
            w_bits, a_bits, weight_groups, activation_groups,
            activation_specs, quantize_bias, activation_maxima={})

    def configure_components_with_ranges(
            self, w_bits, a_bits, weight_groups, activation_groups,
            activation_specs, quantize_bias, activation_maxima,
            weight_bit_overrides=None, weight_modules=None):
        weight_groups = set(weight_groups)
        activation_groups = set(activation_groups)
        self._validate_component_groups(weight_groups, "weight")
        self._validate_component_groups(activation_groups, "activation")
        if weight_modules is None:
            active_weight_modules = {
                name for name in self.modules
                if self.groups[name] in weight_groups}
        else:
            declared_weight_modules = tuple(
                str(name) for name in weight_modules)
            if len(declared_weight_modules) != len(set(declared_weight_modules)):
                raise ValueError("selected weight modules contain duplicates")
            active_weight_modules = set(declared_weight_modules)
            unknown_weight_modules = active_weight_modules - set(self.modules)
            if unknown_weight_modules:
                raise ValueError("selected weight modules are unknown: %s" %
                                 sorted(unknown_weight_modules))
            wrong_group_modules = {
                name for name in active_weight_modules
                if self.groups[name] not in weight_groups}
            if wrong_group_modules:
                raise ValueError(
                    "selected weight modules are outside enabled groups: %s" %
                    sorted(wrong_group_modules))
        if weight_bit_overrides is not None and \
                not set(weight_bit_overrides) <= active_weight_modules:
            raise ValueError(
                "weight bit overrides include unselected modules: %s" %
                sorted(set(weight_bit_overrides) - active_weight_modules))
        expected_specs = set(self.activation_site_keys(activation_groups))
        provided_specs = set(activation_specs)
        if provided_specs != expected_specs:
            missing = expected_specs - provided_specs
            extra = provided_specs - expected_specs
            raise ValueError(
                "activation spec coverage mismatch: missing=%s extra=%s" %
                (sorted(missing, key=str), sorted(extra, key=str)))
        if quantize_bias and weight_groups != activation_groups:
            raise ValueError(
                "component-isolated quantization requires FP32 bias")
        unknown_maxima = set(activation_maxima) - provided_specs
        if unknown_maxima:
            raise ValueError("activation maxima lack declared specs: %s" %
                             sorted(unknown_maxima, key=str))
        enabled_groups = weight_groups | activation_groups
        activation_bit_overrides = dict(
            (key, activation_specs[key].bits) for key in activation_specs)
        self.configure(
            w_bits, a_bits, enabled_groups,
            weight_bit_overrides=weight_bit_overrides,
            activation_specs=activation_specs,
            activation_bit_overrides=activation_bit_overrides,
            activation_overrides=activation_maxima,
            quantize_bias=quantize_bias)

        with torch.no_grad():
            for name, module in self.modules.items():
                if name in active_weight_modules:
                    continue
                module.weight.copy_(self.original_weights[name].to(
                    device=module.weight.device, dtype=module.weight.dtype))
                original_bias = self.original_biases[name]
                if original_bias is not None:
                    module.bias.copy_(original_bias.to(
                        device=module.bias.device, dtype=module.bias.dtype))

        self.quantizers = dict(
            (key, quantizer) for key, quantizer in self.quantizers.items()
            if self.groups[key[0]] in activation_groups)
        self.relu_quantizers = dict(
            (key, quantizer) for key, quantizer in self.relu_quantizers.items()
            if self._relu_owner(key)[1] in activation_groups)
        retained_stats = {}
        for key, stats in self.stats.items():
            name, kind = key
            group = self.groups[name]
            if kind in ("weight", "bias") and \
                    name in active_weight_modules:
                retained_stats[key] = stats
            elif kind not in ("weight", "bias") and \
                    group in activation_groups:
                retained_stats[key] = stats
        self.stats = retained_stats
        self.relu_stats = dict(
            (key, stats) for key, stats in self.relu_stats.items()
            if self._relu_owner(key)[1] in activation_groups)
        self.weight_scales = dict(
            (name, scale) for name, scale in self.weight_scales.items()
            if name in active_weight_modules)
        self.weight_bits = dict(
            (name, bits) for name, bits in self.weight_bits.items()
            if name in active_weight_modules)
        self.enabled_groups = activation_groups

    def configure(self, w_bits, a_bits, enabled_groups,
                  activation_overrides=None,
                  weight_bit_overrides=None,
                  activation_bit_overrides=None,
                  external_output_ownership=True, quantize_bias=True,
                  weight_source_overrides=None, activation_specs=None,
                  ):
        if not self.frozen:
            raise RuntimeError("calibration must be frozen before quantization")
        self._restore_parameters()
        self.quantize_bias = bool(quantize_bias)
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)
        self.enabled_groups = set(enabled_groups)
        self.external_output_ownership = bool(external_output_ownership)
        self.quantizers = {}
        self.relu_quantizers = {}
        self.stats = {}
        self.relu_stats = {}
        self.weight_bits = {}
        if activation_overrides is None:
            activation_overrides = {}
        if weight_bit_overrides is None:
            weight_bit_overrides = {}
        if weight_source_overrides is None:
            weight_source_overrides = {}
        unknown_weight_overrides = set(weight_bit_overrides) - \
            set(self.modules)
        if unknown_weight_overrides:
            raise ValueError("unknown weight bit overrides: %s" %
                             sorted(unknown_weight_overrides))
        unknown_weight_sources = set(weight_source_overrides) - \
            set(self.modules)
        if unknown_weight_sources:
            raise ValueError("unknown weight source overrides: %s" %
                             sorted(unknown_weight_sources))
        if activation_bit_overrides is None:
            activation_bit_overrides = {}
        if activation_specs is None:
            activation_specs = {}
        available_activation_specs = set(self.channel_observers) | \
            set(self.relu_channel_observers)
        unknown_activation_specs = set(activation_specs) - \
            available_activation_specs
        if unknown_activation_specs:
            raise ValueError("unknown activation specs: %s" %
                             sorted(unknown_activation_specs, key=str))
        for key in activation_specs:
            if not isinstance(activation_specs[key], QuantSpec):
                raise TypeError("activation spec must be QuantSpec: %s" %
                                (key,))
        with torch.no_grad():
            for name, module in self.modules.items():
                if self.groups[name] not in self.enabled_groups:
                    continue
                fully_owned = name in self._active_externally_owned_inputs and \
                    self.external_output_ownership and \
                    name in self._active_externally_owned_outputs
                if fully_owned:
                    continue
                original_weight = weight_source_overrides[name] \
                    if name in weight_source_overrides else \
                    self.original_weights[name]
                if original_weight.shape != module.weight.shape:
                    raise ValueError(
                        "weight source shape mismatch: %s" % name)
                weight_bits = int(weight_bit_overrides[name]) \
                    if name in weight_bit_overrides else self.w_bits
                if weight_bits not in HARDWARE_INTEGER_BITS:
                    raise ValueError(
                        "weight bits must be one of %s: %s=%d" %
                        (HARDWARE_INTEGER_BITS, name, weight_bits))
                self.weight_bits[name] = weight_bits
                quantized_weight, weight_scale = symmetric_weight_qdq(
                    original_weight, weight_bits,
                    channel_dim=self._weight_output_channel_dim(module))
                self.weight_scales[name] = weight_scale
                weight_stats = QuantizationStats()
                weight_stats.update(original_weight, quantized_weight)
                self.stats[(name, "weight")] = weight_stats
                module.weight.copy_(quantized_weight.to(
                    device=module.weight.device, dtype=module.weight.dtype))
                for kind in ("input", "output"):
                    key = (name, kind)
                    if kind == "output" and self.external_output_ownership and \
                            name in self._active_externally_owned_outputs:
                        continue
                    if kind == "input" and \
                            name in self._active_externally_owned_inputs:
                        continue
                    observer = self.observers[key] \
                        if key in self.observers else None
                    if observer is None or not observer.observed:
                        continue
                    bits = int(self._override_value(
                        activation_bit_overrides, key, name, self.a_bits))
                    minimum = observer.minimum
                    if torch.is_tensor(minimum):
                        nonnegative = bool(torch.all(minimum >= 0.0).item())
                    else:
                        nonnegative = minimum >= 0.0
                    unsigned = kind == "input" and nonnegative
                    maximum = activation_overrides[key] \
                        if key in activation_overrides else None
                    quantizer = self._activation_quantizer(
                        observer, bits, unsigned, maximum,
                        activation_specs[key]
                        if key in activation_specs else None,
                        self.channel_observers[key])
                    self.quantizers[key] = quantizer
                    self.stats[key] = QuantizationStats()
                original_bias = self.original_biases[name]
                if original_bias is not None and self.quantize_bias and \
                        name in self._active_externally_owned_inputs:
                    raise RuntimeError(
                        "externally owned input bias must be quantized externally: %s" %
                        name)
                if original_bias is not None and self.quantize_bias:
                    input_scale = self.quantizers[(name, "input")].scale
                    quantized_bias, _, bias_scale = int32_bias_qdq(
                        original_bias, input_scale, weight_scale)
                    module.bias.copy_(quantized_bias.to(
                        device=module.bias.device, dtype=module.bias.dtype))
                    bias_stats = QuantizationStats()
                    bias_stats.update(original_bias, quantized_bias)
                    bias_stats.bias_scale = bias_scale
                    self.stats[(name, "bias")] = bias_stats
            for name in sorted(self._layernorm_output_modules):
                if self.groups[name] not in self.enabled_groups:
                    continue
                key = (name, "output")
                observer = self.observers[key]
                if not observer.observed:
                    continue
                bits = int(self._override_value(
                    activation_bit_overrides, key, name, self.a_bits))
                maximum = activation_overrides[key] \
                    if key in activation_overrides else None
                extent = torch.maximum(
                    observer.minimum.abs(), observer.maximum.abs())
                uniform_maximum = float(extent.max().item()) \
                    if maximum is None else maximum
                quantizer = self._activation_quantizer(
                    observer, bits, False, uniform_maximum,
                    activation_specs[key]
                    if key in activation_specs else None,
                    self.channel_observers[key])
                self.quantizers[key] = quantizer
                self.stats[key] = QuantizationStats()
            for key, observer in self.relu_observers.items():
                if observer.observed:
                    owner, group = self._relu_owner(key)
                    if group not in self.enabled_groups:
                        continue
                    bits = int(self._override_value(
                        activation_bit_overrides, key, owner, self.a_bits))
                    maximum = activation_overrides[key] \
                        if key in activation_overrides else None
                    self.relu_quantizers[key] = self._activation_quantizer(
                        observer, bits, True, maximum,
                        activation_specs[key]
                        if key in activation_specs else None,
                        self.relu_channel_observers[key])
                    self.relu_stats[key] = QuantizationStats()
        self.mode = "quantize"

    def disable(self):
        self._restore_parameters()
        self.mode = "bypass"
        self.quantize_bias = True
        self.enabled_groups = set()

    def set_runtime_statistics(self, enabled):
        self.runtime_statistics_enabled = bool(enabled)

    def module_groups(self):
        return dict(self.groups)

    def skipped_output_modules(self):
        return sorted(self._skipped_output_modules)

    def externally_owned_outputs(self):
        return sorted(self._externally_owned_outputs)

    def externally_owned_inputs(self):
        return sorted(self._externally_owned_inputs)

    def set_external_ownership(self, inputs, outputs):
        active_inputs = set(inputs)
        active_outputs = set(outputs)
        unknown_inputs = active_inputs - self._externally_owned_inputs
        if unknown_inputs:
            raise ValueError("undeclared externally owned inputs: %s" %
                             sorted(unknown_inputs))
        unknown_outputs = active_outputs - self._externally_owned_outputs
        if unknown_outputs:
            raise ValueError("undeclared externally owned outputs: %s" %
                             sorted(unknown_outputs))
        self._active_externally_owned_inputs = active_inputs
        self._active_externally_owned_outputs = active_outputs

    def active_externally_owned_inputs(self):
        return sorted(self._active_externally_owned_inputs)

    def active_externally_owned_outputs(self):
        return sorted(self._active_externally_owned_outputs)

    def layernorm_fusions(self):
        return sorted(self._layernorm_fusion_pairs,
                      key=lambda row: (row["conv"], row["layernorm"]))

    def per_channel_activation_modules(self):
        return sorted(self._per_channel_activation_modules)

    def weight_bits_by_module(self):
        return dict(self.weight_bits)

    @staticmethod
    def _weight_output_channel_dim(module):
        if isinstance(module, nn.ConvTranspose2d):
            return 1
        return 0

    def manifest(self):
        rows = []
        for (name, kind), quantizer in sorted(self.quantizers.items()):
            row = {
                "module": name, "kind": kind,
                "bits": quantizer.bits,
                "format": quantizer.format,
                "unsigned": bool(quantizer.unsigned),
                "qmin": quantizer.qmin, "qmax": quantizer.qmax,
                "scale": quantizer.scale,
            }
            rows.append(row)
        for key, quantizer in sorted(self.relu_quantizers.items()):
            row = {
                "module": key, "kind": "relu_output",
                "format": quantizer.format,
                "unsigned": bool(quantizer.unsigned),
                "bits": quantizer.bits,
                "qmin": quantizer.qmin, "qmax": quantizer.qmax,
                "scale": quantizer.scale,
            }
            rows.append(row)
        return rows

    def metadata(self):
        layernorm_metadata = {
            "conv_layernorm_fusion_boundaries": self.layernorm_fusions(),
            "conv_layernorm_kernel_fused": False,
            "externally_owned_inputs": self.externally_owned_inputs(),
            "externally_owned_outputs": self.externally_owned_outputs(),
            "active_externally_owned_inputs":
                self.active_externally_owned_inputs(),
            "active_externally_owned_outputs":
                self.active_externally_owned_outputs(),
            "external_output_ownership_active":
                self.external_output_ownership,
            "conv_layernorm_contract": (
                "Aq input -> Wq Conv -> high-precision accumulator "
                "LayerNorm -> Aq output"),
        }
        bias_contract = "int32 scale=sx*sw[o]" \
            if self.quantize_bias else "fp32_isolation"
        return dict(layernorm_metadata, **{
            "activation_mode": "uniform",
            "bias_contract": bias_contract,
            "quantization_execution": "hardware_aligned_qdq",
        })


    def dynamic_activation_rows(self):
        rows = []
        dynamic_types = (
            DynamicTensorActivationQuantizer,
            DynamicGroupedActivationQuantizer,
        )
        for key, quantizer in sorted(self.quantizers.items(), key=str):
            if not isinstance(quantizer, dynamic_types):
                continue
            name, kind = key
            rows.append({
                "module": name,
                "kind": kind,
                "group": self.groups[name],
                "granularity": quantizer.granularity,
                "group_size": "" if quantizer.group_size is None else
                quantizer.group_size,
                "invocations": quantizer.invocations,
                "runtime_scale_count": quantizer.runtime_scale_count,
                "reduction_elements": quantizer.reduction_elements,
            })
        for key, quantizer in sorted(self.relu_quantizers.items()):
            if not isinstance(quantizer, dynamic_types):
                continue
            owner, group = self._relu_owner(key)
            rows.append({
                "module": owner,
                "kind": "relu_output",
                "group": group,
                "granularity": quantizer.granularity,
                "group_size": "" if quantizer.group_size is None else
                quantizer.group_size,
                "invocations": quantizer.invocations,
                "runtime_scale_count": quantizer.runtime_scale_count,
                "reduction_elements": quantizer.reduction_elements,
            })
        return rows

    def statistics(self):
        rows = []

        for (name, kind), stats in sorted(self.stats.items()):
            row = {
                "module": name, "group": self.groups[name], "kind": kind,
                "numel": stats.numel, "mse": stats.mse,
                "sqnr_db": stats.sqnr_db, "cosine": stats.cosine,
                "signal_sq": stats.signal_sq,
                "error_sq": stats.error_sq,
                "saturation_rate": stats.saturation_rate,
                "zero_code_rate": stats.zero_code_rate,
                "nonfinite_rate": stats.nonfinite_rate,
                "sign_flip_rate": stats.sign_flip_rate,
            }
            if kind == "bias":
                row["scale_min"] = float(stats.bias_scale.min().item())
                row["scale_max"] = float(stats.bias_scale.max().item())
            rows.append(row)
        for key, stats in sorted(self.relu_stats.items()):
            rows.append({
                "module": key, "group": self._relu_owner(key)[1],
                "kind": "relu_output", "numel": stats.numel,
                "mse": stats.mse, "sqnr_db": stats.sqnr_db,
                "cosine": stats.cosine, "signal_sq": stats.signal_sq,
                "error_sq": stats.error_sq,
                "saturation_rate": stats.saturation_rate,
                "zero_code_rate": stats.zero_code_rate,
                "nonfinite_rate": stats.nonfinite_rate,
                "sign_flip_rate": stats.sign_flip_rate,
            })
        return rows

    def close(self):
        self.disable()
        self.clear_calibration_recorder()
        for handle in self.handles:
            handle.remove()
        self.handles = []


def symmetric_weight_qdq(weight, bits, channel_dim=0):
    if bits < 2:
        raise ValueError("weight bits must be at least 2")
    if weight.ndim < 2:
        raise ValueError("weight must have an output-channel dimension")
    channel_dim = int(channel_dim)
    if channel_dim < 0 or channel_dim >= weight.ndim:
        raise ValueError("weight channel dimension is out of range")
    qmax = 2 ** (bits - 1) - 1
    flat = weight.movedim(channel_dim, 0).reshape(
        weight.shape[channel_dim], -1)
    maximum = flat.abs().max(dim=1)[0]
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    shape = [1] * weight.ndim
    shape[channel_dim] = weight.shape[channel_dim]
    scale = (safe_maximum / float(qmax)).reshape(shape)
    codes = torch.round(weight / scale).clamp(-qmax, qmax)
    return codes * scale, scale


class SymmetricActivationQuantizer(object):
    format = "uniform"
    unsigned = False
    granularity = "tensor"
    group_size = None
    scale_count = 1

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

    def scale_for(self, tensor):
        del tensor
        return self.scale

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


class UnsignedActivationQuantizer(object):
    format = "uniform"
    unsigned = True
    granularity = "tensor"
    group_size = None
    scale_count = 1

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

    def scale_for(self, tensor):
        del tensor
        return self.scale

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


class DynamicTensorActivationQuantizer(object):
    """Uniform QDQ with one runtime scale per inference sample."""

    format = "uniform"
    granularity = "tensor"
    group_size = None
    scale_count = 1
    dynamic = True

    def __init__(self, bits, unsigned=False):
        if bits < 2:
            raise ValueError("activation bits must be at least 2")
        self.bits = int(bits)
        self.unsigned = bool(unsigned)
        self.qmin = 0 if self.unsigned else -(2 ** (bits - 1) - 1)
        self.qmax = 2 ** bits - 1 if self.unsigned else \
            2 ** (bits - 1) - 1
        self.zero_point = 0
        self.invocations = 0
        self.runtime_scale_count = 0
        self.reduction_elements = 0

    def _scale(self, tensor):
        if tensor.ndim < 2:
            raise ValueError("dynamic activation requires a batch dimension")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("dynamic activation tensor must be finite")
        dimensions = tuple(range(1, tensor.ndim))
        extent = tensor.amax(dim=dimensions, keepdim=True) \
            if self.unsigned else \
            tensor.abs().amax(dim=dimensions, keepdim=True)
        return torch.where(
            extent > 0, extent / float(self.qmax), torch.ones_like(extent))

    def quantize_with_codes(self, tensor):
        scale = self._scale(tensor)
        codes = torch.round(tensor / scale).clamp(self.qmin, self.qmax)
        self.invocations += 1
        self.runtime_scale_count += int(tensor.shape[0])
        self.reduction_elements += int(tensor.numel())
        return codes * scale, codes.to(torch.int32)

    def scale_for(self, tensor):
        return self._scale(tensor)

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


class DynamicGroupedActivationQuantizer(object):
    """Uniform QDQ with one runtime scale per sample and channel group."""

    format = "uniform"
    granularity = "group"
    dynamic = True

    def __init__(self, bits, channel_dim, group_size, channels,
                 unsigned=False):
        if bits < 2:
            raise ValueError("activation bits must be at least 2")
        self.bits = int(bits)
        self.channel_dim = int(channel_dim)
        self.group_size = int(group_size)
        self.channels = int(channels)
        if self.channels % self.group_size != 0:
            raise ValueError("group size must divide activation channels")
        self.groups = self.channels // self.group_size
        self.scale_count = self.groups
        self.unsigned = bool(unsigned)
        self.qmin = 0 if self.unsigned else -(2 ** (bits - 1) - 1)
        self.qmax = 2 ** bits - 1 if self.unsigned else \
            2 ** (bits - 1) - 1
        self.zero_point = 0
        self.invocations = 0
        self.runtime_scale_count = 0
        self.reduction_elements = 0

    def _scale(self, tensor):
        if tensor.ndim < 2:
            raise ValueError("dynamic activation requires a batch dimension")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("dynamic activation tensor must be finite")
        axis = self.channel_dim \
            if self.channel_dim >= 0 else tensor.ndim + self.channel_dim
        if axis <= 0 or axis >= tensor.ndim:
            raise ValueError("dynamic activation channel dimension is invalid")
        if int(tensor.shape[axis]) != self.channels:
            raise ValueError("grouped activation channel count changed")
        moved = tensor.movedim(axis, 1)
        grouped = moved.reshape(
            int(tensor.shape[0]), self.groups, self.group_size, -1)
        extent = grouped.amax(dim=(2, 3)) if self.unsigned else \
            grouped.abs().amax(dim=(2, 3))
        scale = torch.where(
            extent > 0, extent / float(self.qmax), torch.ones_like(extent))
        expanded = scale.repeat_interleave(self.group_size, dim=1)
        shape = [int(tensor.shape[0]), self.channels] + \
            [1] * (tensor.ndim - 2)
        return expanded.reshape(shape).movedim(1, axis)

    def quantize_with_codes(self, tensor):
        scale = self._scale(tensor)
        codes = torch.round(tensor / scale).clamp(self.qmin, self.qmax)
        self.invocations += 1
        self.runtime_scale_count += int(tensor.shape[0]) * self.groups
        self.reduction_elements += int(tensor.numel())
        return codes * scale, codes.to(torch.int32)

    def scale_for(self, tensor):
        return self._scale(tensor)

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


class ChannelActivationQuantizer(object):
    """Per-channel symmetric/unsigned fake quantizer for Conv activations."""

    format = "uniform"
    granularity = "channel"
    group_size = None

    def __init__(self, bits, minimum, maximum, channel_dim=1, unsigned=False):
        if bits < 2:
            raise ValueError("activation bits must be at least 2")
        self.bits = int(bits)
        self.channel_dim = int(channel_dim)
        self.unsigned = bool(unsigned)
        self.qmin = 0 if self.unsigned else -(2 ** (bits - 1) - 1)
        self.qmax = 2 ** bits - 1 if self.unsigned else 2 ** (bits - 1) - 1
        minimum = minimum.detach().float()
        maximum = maximum.detach().float()
        extent = maximum if self.unsigned else torch.maximum(
            minimum.abs(), maximum.abs())
        self.scale = torch.where(extent > 0, extent / float(self.qmax),
                                 torch.ones_like(extent))
        self.scale_count = int(self.scale.numel())
        self.zero_point = 0

    def _scale_shape(self, tensor):
        shape = [1] * tensor.ndim
        shape[self.channel_dim] = self.scale.numel()
        return self.scale.to(device=tensor.device, dtype=tensor.dtype).reshape(shape)

    def quantize_with_codes(self, tensor):
        scale = self._scale_shape(tensor)
        codes = torch.round(tensor / scale).clamp(self.qmin, self.qmax)
        return codes * scale, codes.to(torch.int32)

    def scale_for(self, tensor):
        return self._scale_shape(tensor)

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


class GroupedActivationQuantizer(object):
    """Uniform QDQ with one scale per contiguous channel group."""

    format = "uniform"
    granularity = "group"

    def __init__(self, bits, minimum, maximum, channel_dim,
                 group_size, channels, unsigned=False):
        if bits < 2:
            raise ValueError("activation bits must be at least 2")
        self.bits = int(bits)
        self.channel_dim = int(channel_dim)
        self.group_size = int(group_size)
        self.channels = int(channels)
        if self.channels % self.group_size != 0:
            raise ValueError("group size must divide activation channels")
        self.unsigned = bool(unsigned)
        self.qmin = 0 if self.unsigned else -(2 ** (bits - 1) - 1)
        self.qmax = 2 ** bits - 1 if self.unsigned else 2 ** (bits - 1) - 1
        minimum = minimum.detach().float().reshape(-1)
        maximum = maximum.detach().float().reshape(-1)
        expected = self.channels // self.group_size
        if minimum.numel() != expected or maximum.numel() != expected:
            raise ValueError("group range count does not match channels")
        extent = maximum if self.unsigned else torch.maximum(
            minimum.abs(), maximum.abs())
        self.scale = torch.where(
            extent > 0, extent / float(self.qmax), torch.ones_like(extent))
        self.scale_count = int(self.scale.numel())
        self.zero_point = 0

    def _scale_shape(self, tensor):
        axis = self.channel_dim \
            if self.channel_dim >= 0 else tensor.ndim + self.channel_dim
        if int(tensor.shape[axis]) != self.channels:
            raise ValueError("grouped activation channel count changed")
        expanded = self.scale.repeat_interleave(self.group_size)
        shape = [1] * tensor.ndim
        shape[axis] = self.channels
        return expanded.to(
            device=tensor.device, dtype=tensor.dtype).reshape(shape)

    def quantize_with_codes(self, tensor):
        scale = self._scale_shape(tensor)
        codes = torch.round(tensor / scale).clamp(self.qmin, self.qmax)
        return codes * scale, codes.to(torch.int32)

    def scale_for(self, tensor):
        return self._scale_shape(tensor)

    def __call__(self, tensor):
        return self.quantize_with_codes(tensor)[0]


def update_activation_stats(stats, quantizer, reference, quantized, codes,
                            coding_reference):
    if codes is None:
        return
    if quantizer.format != "uniform":
        raise ValueError("unknown activation format: %s" % quantizer.format)
    scale = quantizer.scale_for(coding_reference)
    normalized = coding_reference / scale
    saturated = int(((normalized < quantizer.qmin) |
                     (normalized > quantizer.qmax)).sum().item())
    zero_codes = int((codes == 0).sum().item())
    nonfinite = int((~torch.isfinite(quantized)).sum().item())
    stats.update(
        reference, quantized, saturated=saturated,
        zero_codes=zero_codes, nonfinite=nonfinite)


def int32_bias_qdq(bias, input_scale, weight_scale):
    if bias.ndim != 1:
        raise ValueError("bias must have one value per output channel")
    flat_weight_scale = weight_scale.reshape(-1)
    if flat_weight_scale.numel() != bias.numel():
        raise ValueError("weight scale must have one value per output channel")
    scale = flat_weight_scale * torch.as_tensor(
        input_scale, device=bias.device, dtype=bias.dtype)
    scale = scale.to(device=bias.device, dtype=bias.dtype)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    limits = torch.iinfo(torch.int32)
    codes = torch.round(bias / safe_scale).clamp(limits.min, limits.max).to(torch.int32)
    return codes.to(bias.dtype) * safe_scale, codes, safe_scale
