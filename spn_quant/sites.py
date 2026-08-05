"""Semantic tensor-site registry and execution-based edge discovery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.specs import DEFAULT_W4A4_ACTIVATION_SPEC, QuantSpec


SpecFactory = Callable[[str, str, Tuple[str, ...]], QuantSpec]
ModuleFilter = Callable[[str, nn.Module], bool]
GroupFunction = Callable[[str, nn.Module], Optional[str]]


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            for tensor in _iter_tensors(item):
                yield tensor
    elif isinstance(value, (list, tuple)):
        for item in value:
            for tensor in _iter_tensors(item):
                yield tensor


def _base_module_name(call_name: str) -> Optional[str]:
    if call_name.startswith("model_input#") or call_name.startswith("external#"):
        return None
    return call_name.rsplit("#", 1)[0]


@dataclass(frozen=True)
class QuantSite:
    """One logical quantized tensor produced once and consumed one or more times."""

    name: str
    role: str
    producer: str
    consumers: Tuple[str, ...]
    spec: QuantSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("site name cannot be empty")
        if not self.role:
            raise ValueError("site role cannot be empty")
        if not self.producer:
            raise ValueError("site producer cannot be empty")
        normalized = tuple(sorted(set(str(item) for item in self.consumers)))
        object.__setattr__(self, "consumers", normalized)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def with_spec(self, spec: QuantSpec) -> "QuantSite":
        return replace(self, spec=spec)

    def manifest(self) -> Dict[str, Any]:
        row = {
            "site": self.name,
            "role": self.role,
            "producer": self.producer,
            "consumers": ";".join(self.consumers),
            "fanout": len(self.consumers),
        }
        row.update(self.spec.manifest())
        for key, value in sorted(self.metadata.items()):
            row["meta_%s" % key] = value
        return row


class QuantSiteRegistry(object):
    """Ordered registry that fails closed on duplicate or unknown sites."""

    def __init__(self, sites: Iterable[QuantSite] = ()) -> None:
        self._sites = {}  # type: Dict[str, QuantSite]
        self._frozen = False
        self.extend(sites)

    def register(self, site: QuantSite) -> QuantSite:
        if self._frozen:
            raise RuntimeError("quant site registry is frozen")
        if site.name in self._sites:
            raise ValueError("duplicate quant site: %s" % site.name)
        self._sites[site.name] = site
        return site

    def extend(self, sites: Iterable[QuantSite]) -> None:
        for site in sites:
            self.register(site)

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def require(self, name: str) -> QuantSite:
        try:
            return self._sites[name]
        except KeyError:
            raise KeyError("unknown quant site: %s" % name)

    def set_spec(self, name: str, spec: QuantSpec) -> QuantSite:
        if self._frozen:
            raise RuntimeError("quant site registry is frozen")
        site = self.require(name).with_spec(spec)
        self._sites[name] = site
        return site

    def by_role(self, role: str) -> List[QuantSite]:
        return [site for site in self if site.role == role]

    def manifest(self) -> List[Dict[str, Any]]:
        return [site.manifest() for site in self]

    def __iter__(self) -> Iterator[QuantSite]:
        for name in sorted(self._sites):
            yield self._sites[name]

    def __len__(self) -> int:
        return len(self._sites)

    def __contains__(self, name: object) -> bool:
        return name in self._sites


@dataclass(frozen=True)
class TracedTensorEdge:
    producer: str
    consumers: Tuple[str, ...]


def trace_module_tensor_edges(model: nn.Module, example_args: Sequence[Any],
                              module_filter: Optional[ModuleFilter] = None
                              ) -> List[TracedTensorEdge]:
    """Discover producer/fan-out relations from one deterministic eval forward.

    Call indices are part of producer/consumer names so recurrent or reused
    modules receive separate semantic sites.
    """

    if model.training:
        raise ValueError("semantic edge tracing requires eval mode")
    module_filter = module_filter or (
        lambda name, module: isinstance(module, (nn.Conv2d, nn.Linear)))
    names = dict((module, name) for name, module in model.named_modules())
    targets = dict((module, name) for module, name in names.items()
                   if name and module_filter(name, module))
    producers = {}
    root_inputs = {}
    consumers = defaultdict(set)
    produced = set()
    call_counts = defaultdict(int)
    active_calls = defaultdict(list)
    handles = []

    def root_pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        del module
        producers.clear()
        root_inputs.clear()
        consumers.clear()
        produced.clear()
        call_counts.clear()
        active_calls.clear()
        for index, tensor in enumerate(_iter_tensors(inputs)):
            root_inputs[id(tensor)] = (tensor, "model_input#%d" % index)

    def module_pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        name = targets[module]
        index = call_counts[name]
        call_counts[name] += 1
        call_name = "%s#%d" % (name, index)
        active_calls[module].append(call_name)
        for tensor in _iter_tensors(inputs):
            producer = producers.get(id(tensor))
            if producer is not None and producer[0] is tensor:
                source = producer[1]
            else:
                root = root_inputs.get(id(tensor))
                if root is not None and root[0] is tensor:
                    source = root[1]
                else:
                    source = "external#%d" % len(root_inputs)
                    root_inputs[id(tensor)] = (tensor, source)
            consumers[source].add(call_name)

    def module_post_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        del inputs
        call_name = active_calls[module].pop()
        produced.add(call_name)
        for tensor in _iter_tensors(output):
            producers[id(tensor)] = (tensor, call_name)

    handles.append(model.register_forward_pre_hook(root_pre_hook))
    for module in targets:
        handles.append(module.register_forward_pre_hook(module_pre_hook))
        handles.append(module.register_forward_hook(module_post_hook))
    try:
        with torch.no_grad():
            model(*tuple(example_args))
    finally:
        for handle in handles:
            handle.remove()

    sources = sorted(set(consumers) | produced)
    return [TracedTensorEdge(source, tuple(sorted(consumers.get(source, ()))))
            for source in sources]


def build_module_site_registry(model: nn.Module, example_args: Sequence[Any],
                               group_fn: GroupFunction,
                               default_spec: QuantSpec = DEFAULT_W4A4_ACTIVATION_SPEC,
                               spec_factory: Optional[SpecFactory] = None,
                               module_filter: Optional[ModuleFilter] = None
                               ) -> QuantSiteRegistry:
    """Build semantic output-edge sites for Conv/Linear modules and model inputs."""

    modules = dict(model.named_modules())
    edges = trace_module_tensor_edges(model, example_args, module_filter=module_filter)
    registry = QuantSiteRegistry()
    for edge in edges:
        base_name = _base_module_name(edge.producer)
        if base_name is None:
            role = "model_input"
        else:
            group = group_fn(base_name, modules[base_name])
            role = "%s_activation" % (group or "unclassified")
        spec = (spec_factory(role, edge.producer, edge.consumers)
                if spec_factory is not None else default_spec)
        registry.register(QuantSite(
            name="activation::%s" % edge.producer,
            role=role,
            producer=edge.producer,
            consumers=edge.consumers,
            spec=spec,
            metadata={"source": "execution_trace"},
        ))
    registry.freeze()
    return registry
