"""Per-forward edge-QDQ runtime.

Module pre/post hooks can encounter the same logical tensor several times. The
runtime guarantees that a quantized tensor is reused across consumers unless a
caller explicitly requests a requantization boundary (for example, Add/Concat
scale alignment).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch


QuantizeWithCodes = Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]


@dataclass
class _CachedResult:
    source: torch.Tensor
    quantized: torch.Tensor
    codes: Any


class EdgeQDQRuntime(object):
    def __init__(self) -> None:
        self.forward_index = 0
        self._cache = {}
        self._quantized = {}
        self._counts = defaultdict(lambda: {
            "applied": 0,
            "same_site_reuse": 0,
            "upstream_reuse": 0,
            "forced_requant": 0,
            "marked_output": 0,
        })

    def begin_forward(self) -> None:
        self.forward_index += 1
        self._cache = {}
        self._quantized = {}

    def _is_quantized_object(self, tensor: torch.Tensor) -> bool:
        candidate = self._quantized.get(id(tensor))
        return candidate is tensor

    def mark_quantized(self, site: str, tensor: torch.Tensor) -> torch.Tensor:
        """Mark a merge-produced tensor as already represented by QDQ inputs."""
        if not torch.is_tensor(tensor):
            raise TypeError("edge QDQ expects a tensor")
        self._quantized[id(tensor)] = tensor
        self._counts[str(site)]["marked_output"] += 1
        return tensor

    def process_with_codes(self, site: str, tensor: torch.Tensor,
                           quantize: QuantizeWithCodes,
                           force: bool = False) -> Tuple[torch.Tensor, Any]:
        if not torch.is_tensor(tensor):
            raise TypeError("edge QDQ expects a tensor")
        site = str(site)
        stats = self._counts[site]
        key = (site, id(tensor))
        cached = self._cache.get(key)
        if not force and cached is not None and cached.source is tensor:
            stats["same_site_reuse"] += 1
            return cached.quantized, cached.codes
        if not force and self._is_quantized_object(tensor):
            stats["upstream_reuse"] += 1
            return tensor, None
        quantized, codes = quantize(tensor)
        if not torch.is_tensor(quantized):
            raise TypeError("edge quantizer must return a tensor")
        result = _CachedResult(tensor, quantized, codes)
        self._cache[key] = result
        self._quantized[id(quantized)] = quantized
        stats["applied"] += 1
        if force:
            stats["forced_requant"] += 1
        return quantized, codes

    def process(self, site: str, tensor: torch.Tensor,
                quantize: Callable[[torch.Tensor], torch.Tensor],
                force: bool = False) -> torch.Tensor:
        quantize_with_codes = getattr(quantize, "quantize_with_codes", None)
        if callable(quantize_with_codes):
            output, _ = self.process_with_codes(
                site, tensor, quantize_with_codes, force=force)
        else:
            output, _ = self.process_with_codes(
                site, tensor, lambda value: (quantize(value), None), force=force)
        return output

    def statistics(self) -> List[Dict[str, int]]:
        rows = []
        for site in sorted(self._counts):
            row = {"site": site}
            row.update(self._counts[site])
            rows.append(row)
        return rows


class EdgeAwareQuantizerProxy(object):
    """Wrap a QDQ object exposing ``quantize_with_codes``."""

    def __init__(self, quantizer: Any, runtime: EdgeQDQRuntime, site: str,
                 force: bool = False) -> None:
        self._quantizer = quantizer
        self._runtime = runtime
        self._site = str(site)
        self._force = bool(force)

    def quantize_with_codes(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return self._runtime.process_with_codes(
            self._site, tensor, self._quantizer.quantize_with_codes,
            force=self._force)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize_with_codes(tensor)[0]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._quantizer, name)


class EdgeAwareInstrumentorAdapter(object):
    """Drop-in wrapper for the existing HardwareAlignedInstrumentor.

    The adapter deliberately wraps only uniform quantizers. LogNP remains an
    explicit legacy/ablation backend because code-dependent transformed-domain
    statistics cannot be reused safely after an upstream edge is shared.
    """

    def __init__(self, instrumentor: Any,
                 runtime: EdgeQDQRuntime = None) -> None:
        self.instrumentor = instrumentor
        self.runtime = runtime or EdgeQDQRuntime()
        self._root_handle = instrumentor.model.register_forward_pre_hook(
            self._begin_forward)

    def _begin_forward(self, module: Any, inputs: Any) -> None:
        del module, inputs
        self.runtime.begin_forward()

    def _wrap_quantizer_dict(self, quantizers: Dict[Any, Any], prefix: str) -> None:
        for key, quantizer in list(quantizers.items()):
            if isinstance(quantizer, EdgeAwareQuantizerProxy):
                continue
            if isinstance(key, tuple):
                suffix = ":".join(str(item) for item in key)
            else:
                suffix = str(key)
            quantizers[key] = EdgeAwareQuantizerProxy(
                quantizer, self.runtime, "%s::%s" % (prefix, suffix))

    def configure(self, *args: Any, **kwargs: Any) -> Any:
        result = self.instrumentor.configure(*args, **kwargs)
        activation_mode = getattr(self.instrumentor, "activation_mode", "uniform")
        if activation_mode != "uniform":
            return result
        self._wrap_quantizer_dict(self.instrumentor.quantizers, "activation")
        self._wrap_quantizer_dict(self.instrumentor.relu_quantizers, "relu")
        return result

    def metadata(self) -> Dict[str, Any]:
        row = dict(self.instrumentor.metadata())
        row.update({
            "qdq_semantics": "logical_edge_once",
            "fanout_contract": "reuse_quantized_tensor",
            "explicit_requantization": "merge_boundaries_only",
        })
        return row

    def edge_statistics(self) -> List[Dict[str, int]]:
        return self.runtime.statistics()

    def close(self) -> None:
        self._root_handle.remove()
        self.instrumentor.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.instrumentor, name)
