"""Environment-neutral SPN workload schema and GEMM lowering rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import json
from operator import mul
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


FORMAT_VERSION = 1
MAPPABLE_KINDS = frozenset((
    "conv2d", "conv_transpose2d", "linear",
    "attention_qk", "attention_av",
))
IRREGULAR_KINDS = frozenset(("propagation", "deformable_sampling"))
SUPPORTED_KINDS = MAPPABLE_KINDS | IRREGULAR_KINDS


def _positive_tuple(values: Sequence[int], label: str) -> Tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError("%s dimensions must be positive" % label)
    return result


def _pair(values: Sequence[int], label: str) -> Tuple[int, int]:
    result = _positive_tuple(values, label)
    if len(result) != 2:
        raise ValueError("%s must contain two dimensions" % label)
    return result[0], result[1]


def _product(values: Iterable[int]) -> int:
    return int(reduce(mul, values, 1))


@dataclass(frozen=True)
class GemmWorkload:
    """One repeatable matrix multiplication, excluding bias operations."""

    name: str
    kind: str
    m: int
    n: int
    k: int
    repetitions: int = 1
    execution_index: int = 0

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("GEMM name must be non-empty")
        if self.kind not in MAPPABLE_KINDS:
            raise ValueError("unsupported GEMM kind: %s" % self.kind)
        for label in ("m", "n", "k", "repetitions"):
            value = int(getattr(self, label))
            if value <= 0:
                raise ValueError("GEMM %s must be positive" % label)
            object.__setattr__(self, label, value)
        object.__setattr__(self, "execution_index", int(self.execution_index))

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k * self.repetitions


@dataclass(frozen=True)
class LayerWorkload:
    """One executed neural-network or SPN-specific operation."""

    name: str
    kind: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    macs: int
    array_mappable: bool
    execution_index: int = 0
    kernel: Tuple[int, int] = (1, 1)
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    m: int = 0
    n: int = 0
    k: int = 0
    repetitions: int = 1
    iterations: int = 1
    neighborhood_size: int = 1
    metadata: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("layer name must be non-empty")
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError("unsupported layer kind: %s" % self.kind)
        object.__setattr__(self, "input_shape", _positive_tuple(
            self.input_shape, "input"))
        object.__setattr__(self, "output_shape", _positive_tuple(
            self.output_shape, "output"))
        object.__setattr__(self, "kernel", _pair(self.kernel, "kernel"))
        object.__setattr__(self, "stride", _pair(self.stride, "stride"))
        padding = tuple(int(value) for value in self.padding)
        if len(padding) != 2 or any(value < 0 for value in padding):
            raise ValueError("padding must contain two non-negative dimensions")
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "dilation", _pair(
            self.dilation, "dilation"))
        for label in ("macs", "groups", "repetitions", "iterations",
                      "neighborhood_size"):
            value = int(getattr(self, label))
            if value <= 0:
                raise ValueError("layer %s must be positive" % label)
            object.__setattr__(self, label, value)
        object.__setattr__(self, "execution_index", int(self.execution_index))
        object.__setattr__(self, "array_mappable", bool(self.array_mappable))
        if self.array_mappable != (self.kind in MAPPABLE_KINDS):
            raise ValueError("array mapping flag disagrees with layer kind")
        if self.kind in ("attention_qk", "attention_av"):
            if min(int(self.m), int(self.n), int(self.k)) <= 0:
                raise ValueError("attention GEMM dimensions must be positive")
        object.__setattr__(self, "metadata", tuple(
            (str(key), value) for key, value in self.metadata))

    @classmethod
    def conv2d(cls, name: str, input_shape: Sequence[int],
               output_shape: Sequence[int], kernel: Sequence[int],
               stride: Sequence[int] = (1, 1), padding: Sequence[int] = (0, 0),
               dilation: Sequence[int] = (1, 1), groups: int = 1,
               execution_index: int = 0,
               metadata: Sequence[Tuple[str, Any]] = ()) -> "LayerWorkload":
        input_dims = _positive_tuple(input_shape, "input")
        output_dims = _positive_tuple(output_shape, "output")
        if len(input_dims) != 4 or len(output_dims) != 4:
            raise ValueError("Conv2d requires NCHW input and output")
        kernel_pair = _pair(kernel, "kernel")
        group_count = int(groups)
        if input_dims[1] % group_count or output_dims[1] % group_count:
            raise ValueError("Conv2d channels must be divisible by groups")
        macs = (output_dims[0] * output_dims[2] * output_dims[3] *
                output_dims[1] * (input_dims[1] // group_count) *
                kernel_pair[0] * kernel_pair[1])
        return cls(
            name=str(name), kind="conv2d", input_shape=input_dims,
            output_shape=output_dims, macs=macs, array_mappable=True,
            execution_index=execution_index, kernel=kernel_pair,
            stride=_pair(stride, "stride"), padding=tuple(padding),
            dilation=_pair(dilation, "dilation"), groups=group_count,
            metadata=tuple(metadata))

    @classmethod
    def conv_transpose2d(
            cls, name: str, input_shape: Sequence[int],
            output_shape: Sequence[int], kernel: Sequence[int],
            stride: Sequence[int] = (1, 1), padding: Sequence[int] = (0, 0),
            dilation: Sequence[int] = (1, 1), groups: int = 1,
            execution_index: int = 0,
            metadata: Sequence[Tuple[str, Any]] = ()) -> "LayerWorkload":
        input_dims = _positive_tuple(input_shape, "input")
        output_dims = _positive_tuple(output_shape, "output")
        if len(input_dims) != 4 or len(output_dims) != 4:
            raise ValueError("ConvTranspose2d requires NCHW input and output")
        kernel_pair = _pair(kernel, "kernel")
        group_count = int(groups)
        if input_dims[1] % group_count or output_dims[1] % group_count:
            raise ValueError(
                "ConvTranspose2d channels must be divisible by groups")
        macs = (input_dims[0] * input_dims[2] * input_dims[3] *
                input_dims[1] * (output_dims[1] // group_count) *
                kernel_pair[0] * kernel_pair[1])
        return cls(
            name=str(name), kind="conv_transpose2d", input_shape=input_dims,
            output_shape=output_dims, macs=macs, array_mappable=True,
            execution_index=execution_index, kernel=kernel_pair,
            stride=_pair(stride, "stride"), padding=tuple(padding),
            dilation=_pair(dilation, "dilation"), groups=group_count,
            metadata=tuple(metadata))

    @classmethod
    def linear(cls, name: str, input_shape: Sequence[int],
               output_shape: Sequence[int], execution_index: int = 0,
               metadata: Sequence[Tuple[str, Any]] = ()) -> "LayerWorkload":
        input_dims = _positive_tuple(input_shape, "input")
        output_dims = _positive_tuple(output_shape, "output")
        if len(input_dims) < 2 or len(input_dims) != len(output_dims):
            raise ValueError("Linear input and output ranks must match")
        if input_dims[:-1] != output_dims[:-1]:
            raise ValueError("Linear leading dimensions must match")
        macs = _product(input_dims[:-1]) * input_dims[-1] * output_dims[-1]
        return cls(
            name=str(name), kind="linear", input_shape=input_dims,
            output_shape=output_dims, macs=macs, array_mappable=True,
            execution_index=execution_index, metadata=tuple(metadata))

    @classmethod
    def attention(cls, name: str, kind: str, m: int, n: int, k: int,
                  repetitions: int, input_shape: Sequence[int],
                  output_shape: Sequence[int], execution_index: int = 0,
                  metadata: Sequence[Tuple[str, Any]] = ()) -> "LayerWorkload":
        if kind not in ("attention_qk", "attention_av"):
            raise ValueError("unsupported attention kind: %s" % kind)
        macs = int(m) * int(n) * int(k) * int(repetitions)
        return cls(
            name=str(name), kind=kind,
            input_shape=_positive_tuple(input_shape, "input"),
            output_shape=_positive_tuple(output_shape, "output"), macs=macs,
            array_mappable=True, execution_index=execution_index,
            m=int(m), n=int(n), k=int(k), repetitions=int(repetitions),
            metadata=tuple(metadata))

    @classmethod
    def irregular(cls, name: str, kind: str,
                  input_shape: Sequence[int], output_shape: Sequence[int],
                  macs: int, execution_index: int = 0, iterations: int = 1,
                  neighborhood_size: int = 1,
                  metadata: Sequence[Tuple[str, Any]] = ()) -> "LayerWorkload":
        if kind not in IRREGULAR_KINDS:
            raise ValueError("unsupported irregular kind: %s" % kind)
        return cls(
            name=str(name), kind=kind,
            input_shape=_positive_tuple(input_shape, "input"),
            output_shape=_positive_tuple(output_shape, "output"),
            macs=int(macs), array_mappable=False,
            execution_index=execution_index, iterations=iterations,
            neighborhood_size=neighborhood_size, metadata=tuple(metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "macs": self.macs,
            "array_mappable": self.array_mappable,
            "execution_index": self.execution_index,
            "kernel": list(self.kernel),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "groups": self.groups,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "repetitions": self.repetitions,
            "iterations": self.iterations,
            "neighborhood_size": self.neighborhood_size,
            "metadata": [[key, value] for key, value in self.metadata],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LayerWorkload":
        return cls(
            name=payload["name"], kind=payload["kind"],
            input_shape=tuple(payload["input_shape"]),
            output_shape=tuple(payload["output_shape"]),
            macs=payload["macs"], array_mappable=payload["array_mappable"],
            execution_index=payload.get("execution_index", 0),
            kernel=tuple(payload.get("kernel", (1, 1))),
            stride=tuple(payload.get("stride", (1, 1))),
            padding=tuple(payload.get("padding", (0, 0))),
            dilation=tuple(payload.get("dilation", (1, 1))),
            groups=payload.get("groups", 1), m=payload.get("m", 0),
            n=payload.get("n", 0), k=payload.get("k", 0),
            repetitions=payload.get("repetitions", 1),
            iterations=payload.get("iterations", 1),
            neighborhood_size=payload.get("neighborhood_size", 1),
            metadata=tuple(tuple(row) for row in payload.get("metadata", ())))


def conv2d_to_gemms(layer: LayerWorkload) -> Tuple[GemmWorkload, ...]:
    if layer.kind != "conv2d":
        raise ValueError("expected Conv2d workload")
    batch, input_channels, _, _ = layer.input_shape
    _, output_channels, output_h, output_w = layer.output_shape
    return (GemmWorkload(
        name=layer.name, kind=layer.kind,
        m=batch * output_h * output_w,
        n=output_channels // layer.groups,
        k=(input_channels // layer.groups) * layer.kernel[0] * layer.kernel[1],
        repetitions=layer.groups, execution_index=layer.execution_index),)


def conv_transpose2d_to_gemms(
        layer: LayerWorkload) -> Tuple[GemmWorkload, ...]:
    if layer.kind != "conv_transpose2d":
        raise ValueError("expected ConvTranspose2d workload")
    batch, input_channels, input_h, input_w = layer.input_shape
    _, output_channels, _, _ = layer.output_shape
    return (GemmWorkload(
        name=layer.name, kind=layer.kind, m=batch * input_h * input_w,
        n=(output_channels // layer.groups) * layer.kernel[0] * layer.kernel[1],
        k=input_channels // layer.groups, repetitions=layer.groups,
        execution_index=layer.execution_index),)


def linear_to_gemms(layer: LayerWorkload) -> Tuple[GemmWorkload, ...]:
    if layer.kind != "linear":
        raise ValueError("expected Linear workload")
    return (GemmWorkload(
        name=layer.name, kind=layer.kind, m=_product(layer.input_shape[:-1]),
        n=layer.output_shape[-1], k=layer.input_shape[-1],
        execution_index=layer.execution_index),)


def lower_layer_to_gemms(layer: LayerWorkload) -> Tuple[GemmWorkload, ...]:
    if layer.kind == "conv2d":
        return conv2d_to_gemms(layer)
    if layer.kind == "conv_transpose2d":
        return conv_transpose2d_to_gemms(layer)
    if layer.kind == "linear":
        return linear_to_gemms(layer)
    if layer.kind in ("attention_qk", "attention_av"):
        return (GemmWorkload(
            name=layer.name, kind=layer.kind, m=layer.m, n=layer.n, k=layer.k,
            repetitions=layer.repetitions,
            execution_index=layer.execution_index),)
    return ()


@dataclass(frozen=True)
class WorkloadProfile:
    """Complete, serializable workload for one official model inference."""

    model_name: str
    source_config: str
    checkpoint: str
    sample_index: int
    input_shapes: Tuple[Tuple[str, Tuple[int, ...]], ...]
    layers: Tuple[LayerWorkload, ...]
    metadata: Tuple[Tuple[str, Any], ...] = ()
    format_version: int = FORMAT_VERSION

    def __post_init__(self) -> None:
        if int(self.format_version) != FORMAT_VERSION:
            raise ValueError("unsupported workload format version")
        if not self.model_name or not self.source_config or not self.checkpoint:
            raise ValueError("profile identities must be non-empty")
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "input_shapes", tuple(
            (str(name), _positive_tuple(shape, "model input"))
            for name, shape in self.input_shapes))
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers:
            raise ValueError("workload profile requires executed layers")
        object.__setattr__(self, "metadata", tuple(
            (str(key), value) for key, value in self.metadata))

    @property
    def mapped_macs(self) -> int:
        return sum(layer.macs for layer in self.layers
                   if layer.array_mappable)

    @property
    def irregular_macs(self) -> int:
        return sum(layer.macs for layer in self.layers
                   if not layer.array_mappable)

    @property
    def mappable_gemm_count(self) -> int:
        return sum(len(lower_layer_to_gemms(layer)) for layer in self.layers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_name": self.model_name,
            "source_config": self.source_config,
            "checkpoint": self.checkpoint,
            "sample_index": self.sample_index,
            "input_shapes": [
                [name, list(shape)] for name, shape in self.input_shapes],
            "layers": [layer.to_dict() for layer in self.layers],
            "metadata": [[key, value] for key, value in self.metadata],
            "totals": {
                "mapped_macs": self.mapped_macs,
                "irregular_macs": self.irregular_macs,
                "mappable_gemm_count": self.mappable_gemm_count,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkloadProfile":
        return cls(
            model_name=str(payload["model_name"]),
            source_config=str(payload["source_config"]),
            checkpoint=str(payload["checkpoint"]),
            sample_index=int(payload["sample_index"]),
            input_shapes=tuple(
                (str(row[0]), tuple(row[1]))
                for row in payload["input_shapes"]),
            layers=tuple(LayerWorkload.from_dict(row)
                         for row in payload["layers"]),
            metadata=tuple(tuple(row) for row in payload.get("metadata", ())),
            format_version=int(payload["format_version"]),
        )

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return path

    @classmethod
    def read_json(cls, path: Path) -> "WorkloadProfile":
        return cls.from_dict(json.loads(
            Path(path).read_text(encoding="utf-8")))
