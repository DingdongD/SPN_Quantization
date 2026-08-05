"""Compatibility exports for installing edge-once QDQ around legacy runners."""

from spn_quant.runtime import (
    EdgeAwareInstrumentorAdapter,
    EdgeAwareQuantizerProxy,
    EdgeQDQRuntime,
)

__all__ = [
    "EdgeAwareInstrumentorAdapter",
    "EdgeAwareQuantizerProxy",
    "EdgeQDQRuntime",
]
