#!/usr/bin/env python3
"""Graph-semantic Add/Concat QDQ adapters.

Merge boundaries are explicit requantization sites.  The default ``shared``
policy preserves the former standard-backend contract. ``independent`` keeps
branch scales separate before wide-domain Add/Concat, while ``grouped`` uses
channel-group scales on the merged tensor (Concat) or common channel groups
across branches (Add).
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from spn_quant.merge import MergeSiteController, TensorMinMaxObserver
from spn_quant.runtime import EdgeQDQRuntime
from spn_quant.completionformer_concat import (
    SplitConcatConvController,
    split_concat_branches,
)


class SharedMergeQuantizer(object):
    """Backward-compatible shared-scale branch quantizer."""

    def __init__(self, unsigned: bool) -> None:
        self.unsigned = bool(unsigned)
        self.observer = TensorMinMaxObserver()
        self.bits = None
        self.quantizer = None

    @property
    def scale(self) -> float:
        if self.quantizer is None:
            raise RuntimeError("merge quantizer is not frozen")
        return self.quantizer.scale

    def observe(self, branches: Sequence[torch.Tensor]) -> None:
        for branch in branches:
            self.observer.update(branch)

    def freeze(self, bits: int) -> None:
        self.bits = int(bits)
        self.quantizer = self.observer.quantizer(bits, unsigned=self.unsigned)

    def quantize(self, branches: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        if self.quantizer is None:
            raise RuntimeError("merge quantizer is not frozen")
        return tuple(self.quantizer(branch) for branch in branches)

    def qparams(self) -> Dict[str, Any]:
        if self.quantizer is None:
            raise RuntimeError("merge quantizer is not frozen")
        return self.quantizer.qparams()


class _CallIndexedMergeAdapter(object):
    def __init__(self, model: Any, method_name: str, operation: str,
                 policy: str = "shared", axis: int = 1,
                 group_size: Optional[int] = None,
                 expected_calls: Optional[int] = None,
                 runtime: Optional[EdgeQDQRuntime] = None,
                 manage_runtime: bool = True) -> None:
        self.model = model
        self.method_name = str(method_name)
        self.operation = str(operation)
        self.policy = str(policy)
        self.axis = int(axis)
        self.group_size = group_size
        self.expected_calls = expected_calls
        self.runtime = runtime or EdgeQDQRuntime()
        self.manage_runtime = bool(manage_runtime)
        self.mode = "bypass"
        self.bits = None
        self.call_counts = {}  # type: Dict[str, int]
        self.controllers = {}  # type: Dict[str, MergeSiteController]
        self.originals = {}  # type: Dict[str, Tuple[Any, Any]]
        self.handle = model.register_forward_pre_hook(self._reset_calls)

        for name, module in model.named_modules():
            method = getattr(module, self.method_name, None)
            if method is None or not callable(method):
                continue
            self.originals[name] = (module, method)
            setattr(module, self.method_name, types.MethodType(
                self._make_wrapper(name, method), module))

    def _reset_calls(self, module: Any, inputs: Any) -> None:
        del module, inputs
        self.call_counts = {}
        if self.manage_runtime:
            self.runtime.begin_forward()

    def _key(self, name: str, index: int) -> str:
        prefix = "%s.%s" % (name, self.method_name) if name else self.method_name
        return "%s#%d" % (prefix, index)

    @staticmethod
    def _extract_branches(args: Sequence[Any], kwargs: Dict[str, Any]
                          ) -> Tuple[List[torch.Tensor], List[Tuple[str, Any]]]:
        branches = []
        locations = []
        for index, value in enumerate(args):
            if torch.is_tensor(value):
                branches.append(value)
                locations.append(("arg", index))
        for key in sorted(kwargs):
            value = kwargs[key]
            if torch.is_tensor(value):
                branches.append(value)
                locations.append(("kwarg", key))
        return branches, locations

    @staticmethod
    def _replace_branches(args: Sequence[Any], kwargs: Dict[str, Any],
                          locations: Sequence[Tuple[str, Any]],
                          branches: Sequence[torch.Tensor]
                          ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        updated_args = list(args)
        updated_kwargs = dict(kwargs)
        for location, value in zip(locations, branches):
            if location[0] == "arg":
                updated_args[location[1]] = value
            else:
                updated_kwargs[location[1]] = value
        return tuple(updated_args), updated_kwargs

    def _controller(self, key: str) -> MergeSiteController:
        controller = self.controllers.get(key)
        if controller is None:
            controller = MergeSiteController(
                key, operation=self.operation, policy=self.policy,
                axis=self.axis, group_size=self.group_size,
                runtime=self.runtime)
            self.controllers[key] = controller
        return controller

    def _make_wrapper(self, name: str, original: Any):
        def wrapper(module: Any, *args: Any, **kwargs: Any) -> Any:
            del module
            index = self.call_counts.get(name, 0)
            self.call_counts[name] = index + 1
            key = self._key(name, index)
            branches, locations = self._extract_branches(args, kwargs)
            if len(branches) < 2:
                raise RuntimeError("%s did not receive at least two tensor branches" % key)
            controller = self._controller(key)
            if self.mode == "observe":
                output = original(*args, **kwargs)
                controller.observe(branches, merged=output if torch.is_tensor(output) else None)
                return output
            if self.mode == "quantize":
                if self.operation == "concat" and self.policy == "grouped":
                    output = original(*args, **kwargs)
                    if not torch.is_tensor(output):
                        raise TypeError("grouped concat must return a tensor")
                    return controller.quantize_output(output)
                quantized = controller.quantize_branches(branches)
                updated_args, updated_kwargs = self._replace_branches(
                    args, kwargs, locations, quantized)
                output = original(*updated_args, **updated_kwargs)
                if not torch.is_tensor(output):
                    raise TypeError("quantized merge must return a tensor")
                if self.operation == "add":
                    return controller.quantize_output(output)
                return self.runtime.mark_quantized(
                    "%s:output" % key, output)
            return original(*args, **kwargs)
        return wrapper

    def observe(self) -> None:
        self.mode = "observe"

    def freeze(self, bits: int, policy: Optional[str] = None,
               group_size: Optional[int] = None) -> None:
        if policy is not None and policy != self.policy:
            raise ValueError("merge policy is fixed when the adapter is installed")
        if group_size is not None and group_size != self.group_size:
            raise ValueError("merge group_size is fixed when the adapter is installed")
        if self.expected_calls is not None and len(self.controllers) != int(self.expected_calls):
            raise RuntimeError(
                "expected %d %s calls but observed %d" % (
                    int(self.expected_calls), self.operation, len(self.controllers)))
        if self.originals and not self.controllers:
            raise RuntimeError("merge methods were installed but no calls were observed")
        self.bits = int(bits)
        for controller in self.controllers.values():
            controller.freeze(bits)
        self.mode = "bypass"

    def quantize(self) -> None:
        if not self.originals:
            self.mode = "bypass"
            return
        if not self.controllers:
            raise RuntimeError("merge calibration must be frozen first")
        self.mode = "quantize"

    def disable(self) -> None:
        self.mode = "bypass"

    def manifest(self) -> List[Dict[str, Any]]:
        return [controller.qparams()
                for key, controller in sorted(self.controllers.items())]

    def edge_statistics(self) -> List[Dict[str, int]]:
        return self.runtime.statistics()

    def close(self) -> None:
        self.disable()
        self.handle.remove()
        for name, (module, original) in self.originals.items():
            del name
            setattr(module, self.method_name, original)
        self.originals = {}


class CallIndexedConcatAdapter(_CallIndexedMergeAdapter):
    def __init__(self, model: Any, policy: str = "shared", axis: int = 1,
                 group_size: Optional[int] = None,
                 expected_calls: Optional[int] = None,
                 runtime: Optional[EdgeQDQRuntime] = None,
                 manage_runtime: bool = True) -> None:
        super(CallIndexedConcatAdapter, self).__init__(
            model, method_name="_concat", operation="concat", policy=policy,
            axis=axis, group_size=group_size,
            expected_calls=expected_calls, runtime=runtime,
            manage_runtime=manage_runtime)


class CallIndexedConcatConvAdapter(object):
    """Own integer Conv execution after independently calibrated concat branches."""

    def __init__(self, model: Any, consumer_modules: Sequence[str],
                 weight_bits: int, activation_bits: int,
                 output_bits: int, cache_sample_limit: int,
                 cache_byte_limit: int,
                 call_consumer_modules: Optional[Sequence[Optional[str]]] = None,
                 method_name: str = "_concat") -> None:
        self.model = model
        self.method_name = str(method_name)
        self.consumer_modules = tuple(str(name) for name in consumer_modules)
        if not self.consumer_modules:
            raise ValueError("scale-aware concat requires consumer modules")
        if len(set(self.consumer_modules)) != len(self.consumer_modules):
            raise ValueError("concat consumer modules must be unique")
        self.call_consumer_modules = tuple(
            self.consumer_modules if call_consumer_modules is None
            else call_consumer_modules)
        if not self.call_consumer_modules:
            raise ValueError("concat call consumer mapping must be nonempty")
        declared_consumers = tuple(
            name for name in self.call_consumer_modules if name is not None)
        if set(declared_consumers) != set(self.consumer_modules) or \
                len(declared_consumers) != len(self.consumer_modules):
            raise ValueError("concat call consumer mapping differs from owners")
        named_modules = dict(model.named_modules())
        unknown = set(self.consumer_modules) - set(named_modules)
        if unknown:
            raise ValueError("unknown concat consumer modules: %s" %
                             sorted(unknown))
        for name in self.consumer_modules:
            if not isinstance(named_modules[name], torch.nn.Conv2d):
                raise TypeError("concat consumer must be Conv2d: %s" % name)
        self.controllers = {}
        for name in self.consumer_modules:
            module = named_modules[name]
            self.controllers[name] = SplitConcatConvController(
                name=name, module=module,
                branch_channels=module.in_channels // 2,
                weight_bits=int(weight_bits),
                activation_bits=int(activation_bits),
                output_bits=int(output_bits),
                clip_factors=(1.0,), search_rounds=1,
                cache_sample_limit=int(cache_sample_limit),
                cache_byte_limit=int(cache_byte_limit))
        self.concat_originals = {}
        self.consumer_originals = {}
        self.concat_call_counts = {}
        self.concat_call_index = 0
        self.pending = {}
        self.call_to_consumer = {}
        self.observed_calls = set()
        self.mode = "bypass"
        self.fp_format_quantizers = {}
        self.handle = model.register_forward_pre_hook(self._reset)
        for name, module in model.named_modules():
            method = getattr(module, self.method_name, None)
            if method is None or not callable(method):
                continue
            self.concat_originals[name] = (module, method)
            setattr(module, self.method_name, types.MethodType(
                self._make_concat_wrapper(name, method), module))
        for name in self.consumer_modules:
            module = named_modules[name]
            original = module.forward
            self.consumer_originals[name] = (module, original)
            module.forward = types.MethodType(
                self._make_consumer_wrapper(name, original), module)

    def _reset(self, module: Any, inputs: Any) -> None:
        del module, inputs
        self.concat_call_counts = {}
        self.concat_call_index = 0
        self.pending = {}

    def _make_concat_key(self, name: str, index: int) -> str:
        prefix = "%s.%s" % (name, self.method_name) \
            if name else self.method_name
        return "%s#%d" % (prefix, index)

    def _make_concat_wrapper(self, name: str, original: Any):
        def wrapper(module: Any, *args: Any, **kwargs: Any) -> Any:
            del module
            index = self.concat_call_counts[name] \
                if name in self.concat_call_counts else 0
            self.concat_call_counts[name] = index + 1
            global_index = self.concat_call_index
            self.concat_call_index += 1
            if global_index >= len(self.call_consumer_modules):
                raise RuntimeError(
                    "concat call count exceeds declared consumer modules")
            key = self._make_concat_key(name, index)
            output = original(*args, **kwargs)
            if not torch.is_tensor(output):
                raise TypeError("concat output must be a tensor: %s" % key)
            consumer = self.call_consumer_modules[global_index]
            if consumer is not None:
                self.call_to_consumer[key] = consumer
                self.pending[id(output)] = key
            return output
        return wrapper

    def _make_consumer_wrapper(self, name: str, original: Any):
        def wrapper(module: Any, *args: Any, **kwargs: Any) -> Any:
            del module
            if not args or not torch.is_tensor(args[0]):
                return original(*args, **kwargs)
            merged = args[0]
            tensor_id = id(merged)
            if tensor_id not in self.pending:
                return original(*args, **kwargs)
            key = self.pending[tensor_id]
            expected_consumer = self.call_to_consumer[key]
            if expected_consumer != name:
                raise RuntimeError(
                    "concat output %s reached consumer %s, expected %s" %
                    (key, name, expected_consumer))
            controller = self.controllers[name]
            self.observed_calls.add(key)
            if self.mode == "observe":
                output = original(*args, **kwargs)
                controller.observe(merged, output)
                return output
            if self.mode == "quantize":
                return controller.quantize(merged)
            if self.mode == "fp_format":
                quantizers = self.fp_format_quantizers[name]
                transformer, cnn = split_concat_branches(
                    merged, controller.branch_channels)
                transformer = quantizers["transformer"].quantize_with_codes(
                    transformer)[0]
                cnn = quantizers["cnn"].quantize_with_codes(cnn)[0]
                output = original(torch.cat((transformer, cnn), dim=1),
                                  *args[1:], **kwargs)
                return quantizers["output"].quantize_with_codes(output)[0]
            return original(*args, **kwargs)
        return wrapper

    def observe(self) -> None:
        self.mode = "observe"

    def freeze(self) -> None:
        if self.mode != "observe":
            raise RuntimeError("concat Conv adapter must be observing")
        if self.concat_call_index != len(self.call_consumer_modules):
            raise RuntimeError(
                "observed %d concat calls but expected %d" %
                (self.concat_call_index, len(self.call_consumer_modules)))
        missing = set(self.call_to_consumer) - self.observed_calls
        if missing:
            raise RuntimeError("concat calls lack Conv consumers: %s" %
                               sorted(missing))
        for controller in self.controllers.values():
            controller.freeze()
        self.mode = "bypass"

    def configure(self, weight_bits: Mapping[str, int] | int,
                  activation_bits: Mapping[str, int] | int,
                  output_bits: Mapping[str, int] | int) -> None:
        if self.mode == "observe":
            raise RuntimeError("concat Conv adapter must be frozen first")
        for name in self.consumer_modules:
            weight = weight_bits[name] if isinstance(weight_bits, Mapping) \
                else weight_bits
            activation = activation_bits[name] \
                if isinstance(activation_bits, Mapping) else activation_bits
            output = output_bits[name] if isinstance(output_bits, Mapping) \
                else output_bits
            self.controllers[name].reconfigure_precision(
                weight, activation, output)
            self.controllers[name].enable()
        self.mode = "quantize"

    def configure_floating_point(self, quantizers) -> None:
        if self.mode == "observe":
            raise RuntimeError("concat Conv adapter must be frozen first")
        quantizers = dict(quantizers)
        if set(quantizers) != set(self.consumer_modules):
            raise ValueError("FP format concat consumer coverage differs")
        for name in self.consumer_modules:
            if set(quantizers[name]) != {"transformer", "cnn", "output"}:
                raise ValueError("FP format concat role coverage differs: %s" %
                                 name)
        self.fp_format_quantizers = quantizers
        self.mode = "fp_format"

    def disable(self) -> None:
        for controller in self.controllers.values():
            if controller.phase != "observe":
                controller.disable()
        self.mode = "bypass"
        self.fp_format_quantizers = {}

    def externally_owned_inputs(self) -> Tuple[str, ...]:
        return self.consumer_modules

    def externally_owned_outputs(self) -> Tuple[str, ...]:
        return self.consumer_modules

    def manifest(self) -> List[Dict[str, Any]]:
        rows = []
        for name in self.consumer_modules:
            row = dict(self.controllers[name].manifest())
            row["consumer_module"] = name
            row["branch_calibration"] = "independent"
            row["branch_scales"] = (
                row["transformer_scale"], row["cnn_scale"])
            row["accumulation"] = "branch_partial_int32_requantize_add"
            rows.append(row)
        return rows

    def statistics(self) -> List[Dict[str, Any]]:
        return [row for name in self.consumer_modules
                for row in self.controllers[name].statistics()]

    def close(self) -> None:
        self.disable()
        self.handle.remove()
        for name, (module, original) in self.concat_originals.items():
            del name
            setattr(module, self.method_name, original)
        for name, (module, original) in self.consumer_originals.items():
            del name
            module.forward = original
        self.concat_originals = {}
        self.consumer_originals = {}


class CallIndexedAddAdapter(_CallIndexedMergeAdapter):
    def __init__(self, model: Any, policy: str = "shared", axis: int = 1,
                 group_size: Optional[int] = None,
                 expected_calls: Optional[int] = None,
                 runtime: Optional[EdgeQDQRuntime] = None,
                 manage_runtime: bool = True) -> None:
        super(CallIndexedAddAdapter, self).__init__(
            model, method_name="_add", operation="add", policy=policy,
            axis=axis, group_size=group_size,
            expected_calls=expected_calls, runtime=runtime,
            manage_runtime=manage_runtime)
