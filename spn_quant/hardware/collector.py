"""Runtime hooks that collect executed SPN layer workloads."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .workload import LayerWorkload


MODEL_NAMES = frozenset(("cspn", "dyspn", "nlspn", "completionformer"))


def _tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        preferred = ("pred", "prediction", "pred_init")
        seen = set()
        for key in preferred:
            if key in value:
                seen.add(key)
                yield from _tensors(value[key])
        for key, item in value.items():
            if key not in seen:
                yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _first_tensor(value: Any) -> torch.Tensor:
    try:
        return next(iter(_tensors(value)))
    except StopIteration:
        raise ValueError("hook value contains no tensor")


def _is_attention(module: nn.Module) -> bool:
    return (
        hasattr(module, "num_heads") and
        hasattr(module, "sr_ratio") and
        isinstance(getattr(module, "q", None), nn.Linear) and
        isinstance(getattr(module, "kv", None), nn.Linear) and
        isinstance(getattr(module, "proj", None), nn.Linear)
    )


class WorkloadCollector(object):
    """Collect dense and derived operations from exactly one model execution."""

    def __init__(self, model: nn.Module, model_name: str,
                 propagation_iterations: Optional[int] = None,
                 neighborhood_size: Optional[int] = None) -> None:
        if model_name not in MODEL_NAMES:
            raise ValueError("unsupported SPN model: %s" % model_name)
        if (propagation_iterations is None) != (neighborhood_size is None):
            raise ValueError(
                "propagation iterations and neighborhood must be provided together")
        if propagation_iterations is not None and (
                int(propagation_iterations) <= 0 or
                int(neighborhood_size) <= 0):
            raise ValueError("propagation dimensions must be positive")
        self.model = model
        self.model_name = model_name
        self.propagation_iterations = None if propagation_iterations is None \
            else int(propagation_iterations)
        self.neighborhood_size = None if neighborhood_size is None \
            else int(neighborhood_size)
        self._rows: List[LayerWorkload] = []
        self._handles: List[Any] = []
        self._active = False

    def __enter__(self) -> "WorkloadCollector":
        if self._active:
            raise RuntimeError("workload collector is already active")
        self._active = True
        for name, module in self.model.named_modules():
            if not name:
                continue
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                self._handles.append(module.register_forward_hook(
                    self._dense_hook(name)))
        for name, module in self.model.named_modules():
            if name and _is_attention(module):
                self._handles.append(module.register_forward_hook(
                    self._attention_hook(name)))
        if self.propagation_iterations is not None:
            self._handles.append(self.model.register_forward_hook(
                self._propagation_hook()))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._active = False

    def _next_index(self) -> int:
        return len(self._rows)

    def _dense_hook(self, name: str):
        def record(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            input_tensor = _first_tensor(inputs)
            output_tensor = _first_tensor(output)
            metadata = (("module_class", type(module).__name__),)
            if isinstance(module, nn.Conv2d):
                row = LayerWorkload.conv2d(
                    name=name, input_shape=tuple(input_tensor.shape),
                    output_shape=tuple(output_tensor.shape),
                    kernel=module.kernel_size, stride=module.stride,
                    padding=module.padding, dilation=module.dilation,
                    groups=module.groups, execution_index=self._next_index(),
                    metadata=metadata)
            elif isinstance(module, nn.ConvTranspose2d):
                row = LayerWorkload.conv_transpose2d(
                    name=name, input_shape=tuple(input_tensor.shape),
                    output_shape=tuple(output_tensor.shape),
                    kernel=module.kernel_size, stride=module.stride,
                    padding=module.padding, dilation=module.dilation,
                    groups=module.groups, execution_index=self._next_index(),
                    metadata=metadata)
            elif isinstance(module, nn.Linear):
                row = LayerWorkload.linear(
                    name=name, input_shape=tuple(input_tensor.shape),
                    output_shape=tuple(output_tensor.shape),
                    execution_index=self._next_index(), metadata=metadata)
            else:  # pragma: no cover - registration restricts the types
                raise TypeError("unsupported dense module")
            self._rows.append(row)
        return record

    def _recent_layer(self, name: str) -> LayerWorkload:
        for row in reversed(self._rows):
            if row.name == name:
                return row
        raise RuntimeError("attention child was not executed: %s" % name)

    def _attention_hook(self, name: str):
        def record(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            input_tensor = _first_tensor(inputs)
            output_tensor = _first_tensor(output)
            if input_tensor.dim() != 3:
                raise RuntimeError(
                    "attention input must be [batch,tokens,channels]: %s" % name)
            batch, query_tokens, channels = tuple(
                int(value) for value in input_tensor.shape)
            heads = int(module.num_heads)
            if channels % heads:
                raise RuntimeError("attention channels are not head aligned: %s" % name)
            kv_layer = self._recent_layer(name + ".kv")
            if len(kv_layer.input_shape) != 3:
                raise RuntimeError("attention KV input must be rank three: %s" % name)
            kv_tokens = int(kv_layer.input_shape[-2])
            head_dim = channels // heads
            repetitions = batch * heads
            metadata = (
                ("num_heads", heads),
                ("sr_ratio", int(module.sr_ratio)),
                ("parent_module_class", type(module).__name__),
            )
            self._rows.append(LayerWorkload.attention(
                name=name + "::qk", kind="attention_qk",
                m=query_tokens, n=kv_tokens, k=head_dim,
                repetitions=repetitions,
                input_shape=tuple(input_tensor.shape),
                output_shape=(batch, heads, query_tokens, kv_tokens),
                execution_index=self._next_index(), metadata=metadata))
            self._rows.append(LayerWorkload.attention(
                name=name + "::av", kind="attention_av",
                m=query_tokens, n=head_dim, k=kv_tokens,
                repetitions=repetitions,
                input_shape=(batch, heads, query_tokens, kv_tokens),
                output_shape=tuple(output_tensor.shape),
                execution_index=self._next_index(), metadata=metadata))
        return record

    def _propagation_hook(self):
        def record(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            del module, inputs
            prediction = _first_tensor(output)
            if prediction.dim() != 4:
                raise RuntimeError(
                    "SPN prediction must be [batch,channel,height,width]")
            batch, channels, height, width = tuple(
                int(value) for value in prediction.shape)
            macs = (batch * channels * height * width *
                    int(self.propagation_iterations) *
                    int(self.neighborhood_size))
            self._rows.append(LayerWorkload.irregular(
                name="%s::propagation" % self.model_name,
                kind="propagation", input_shape=tuple(prediction.shape),
                output_shape=tuple(prediction.shape), macs=macs,
                execution_index=self._next_index(),
                iterations=int(self.propagation_iterations),
                neighborhood_size=int(self.neighborhood_size),
                metadata=(("estimate", "neighbor_weighted_updates"),)))
        return record

    def layers(self) -> Tuple[LayerWorkload, ...]:
        return tuple(self._rows)
