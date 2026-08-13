"""Quantization-aware training for strict SPN deployment graphs."""

from spn_quant.qat.ste import hard_forward_proxy, round_ste


__all__ = (
    "hard_forward_proxy",
    "round_ste",
)
