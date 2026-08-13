"""Declarative quantization specifications.

The existing experiment scripts historically pass bit-width and calibration
flags independently. QuantSpec makes the complete tensor contract explicit so
model adapters, manifests, and future graph rewrites share one representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional


_VALID_SCHEMES = frozenset(("symmetric", "affine"))
_VALID_GRANULARITIES = frozenset(("tensor", "channel", "group"))
_VALID_OBSERVERS = frozenset(("minmax", "percentile", "mse", "zero_aware"))
_VALID_TRANSFORMS = frozenset(("none", "lognp", "smooth"))


@dataclass(frozen=True)
class QuantSpec:
    """Complete storage/QDQ contract for one logical tensor edge.

    ``transform`` defaults to ``none``. LogNP remains representable for
    controlled ablations, but is never selected implicitly by W4A4 policies.
    """

    bits: int
    scheme: str = "symmetric"
    granularity: str = "tensor"
    axis: Optional[int] = None
    group_size: Optional[int] = None
    observer: str = "minmax"
    transform: str = "none"
    signed: bool = True
    preserve_zero: bool = False
    dynamic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.bits, int) or isinstance(self.bits, bool):
            raise TypeError("bits must be an integer")
        if self.bits < 1 or self.bits > 32:
            raise ValueError("bits must be in [1, 32]")
        if self.scheme not in _VALID_SCHEMES:
            raise ValueError("unknown quantization scheme: %s" % self.scheme)
        if self.granularity not in _VALID_GRANULARITIES:
            raise ValueError("unknown quantization granularity: %s" % self.granularity)
        if self.observer not in _VALID_OBSERVERS:
            raise ValueError("unknown observer: %s" % self.observer)
        if self.transform not in _VALID_TRANSFORMS:
            raise ValueError("unknown transform: %s" % self.transform)
        if self.scheme == "symmetric" and not self.signed:
            raise ValueError("unsigned quantization must use the affine scheme")
        if self.granularity == "tensor":
            if self.axis is not None or self.group_size is not None:
                raise ValueError("tensor granularity cannot define axis/group_size")
        elif self.granularity == "channel":
            if self.axis is None:
                raise ValueError("channel granularity requires axis")
            if self.group_size is not None:
                raise ValueError("channel granularity cannot define group_size")
        elif self.granularity == "group":
            if self.axis is None:
                raise ValueError("group granularity requires axis")
            if self.group_size is None or int(self.group_size) <= 0:
                raise ValueError("group granularity requires a positive group_size")
        if self.preserve_zero and self.scheme != "affine":
            raise ValueError("preserve_zero requires an affine quantizer")

    @classmethod
    def signed_tensor(cls, bits: int, observer: str = "minmax") -> "QuantSpec":
        return cls(bits=bits, scheme="symmetric", granularity="tensor",
                   observer=observer, signed=True)

    @classmethod
    def unsigned_tensor(cls, bits: int, observer: str = "minmax",
                        preserve_zero: bool = True) -> "QuantSpec":
        return cls(bits=bits, scheme="affine", granularity="tensor",
                   observer=observer, signed=False,
                   preserve_zero=preserve_zero)

    @classmethod
    def signed_group(cls, bits: int, axis: int, group_size: int,
                     observer: str = "minmax") -> "QuantSpec":
        return cls(
            bits=bits, scheme="symmetric", granularity="group",
            axis=axis, group_size=group_size, observer=observer, signed=True)

    @classmethod
    def unsigned_group(cls, bits: int, axis: int, group_size: int,
                       observer: str = "minmax",
                       preserve_zero: bool = True) -> "QuantSpec":
        return cls(
            bits=bits, scheme="affine", granularity="group",
            axis=axis, group_size=group_size, observer=observer,
            signed=False, preserve_zero=preserve_zero)

    def with_bits(self, bits: int) -> "QuantSpec":
        return replace(self, bits=int(bits))

    def with_transform(self, transform: str) -> "QuantSpec":
        return replace(self, transform=str(transform))

    def with_dynamic(self, dynamic: bool = True) -> "QuantSpec":
        return replace(self, dynamic=bool(dynamic))

    def manifest(self) -> Dict[str, Any]:
        row = asdict(self)
        row["axis"] = "" if self.axis is None else int(self.axis)
        row["group_size"] = "" if self.group_size is None else int(self.group_size)
        return row


DEFAULT_W4A4_ACTIVATION_SPEC = QuantSpec.signed_tensor(4)
