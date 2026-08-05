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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from spn_quant.merge import MergeSiteController, TensorMinMaxObserver
from spn_quant.runtime import EdgeQDQRuntime


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
