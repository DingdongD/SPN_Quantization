"""Validated, serializable CSPN encoder search specifications."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator, Mapping, Tuple


_BASE_WIDTHS = (64, 128, 256, 512)
_WIDTH_MULTIPLIERS = (0.5, 0.75, 1.0)
_BASE_DECODER_WIDTHS = (512, 256, 128, 64, 64)


@dataclass(frozen=True)
class DecoderSpec:
    widths: Tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "widths", tuple(int(value) for value in self.widths))
        if len(self.widths) != 5:
            raise ValueError("decoder widths must contain five stages")
        if any(width <= 0 or width % 8 for width in self.widths):
            raise ValueError("decoder widths must be positive multiples of 8")
        if any(left < right for left, right in zip(self.widths, self.widths[1:])):
            raise ValueError("decoder widths must be non-increasing")

    @classmethod
    def default(cls) -> "DecoderSpec":
        return cls(_BASE_DECODER_WIDTHS)

    @classmethod
    def scaled(cls, multiplier: float) -> "DecoderSpec":
        if multiplier <= 0.0 or multiplier > 1.0:
            raise ValueError("decoder multiplier must be in (0, 1]")
        return cls(tuple(
            int(round(width * multiplier / 8.0)) * 8
            for width in _BASE_DECODER_WIDTHS
        ))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecoderSpec":
        return cls(tuple(value["widths"]))

    def to_dict(self) -> dict[str, Any]:
        return {"widths": list(self.widths), "slug": self.slug}

    @property
    def slug(self) -> str:
        return "dw" + "-".join(str(value) for value in self.widths)


@dataclass(frozen=True)
class EncoderSpec:
    stem_width: int
    widths: Tuple[int, int, int, int]
    depths: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "widths", tuple(int(value) for value in self.widths))
        object.__setattr__(self, "depths", tuple(int(value) for value in self.depths))
        object.__setattr__(self, "stem_width", int(self.stem_width))
        if self.stem_width <= 0 or self.stem_width % 16:
            raise ValueError("stem width must be a positive multiple of 16")
        if len(self.widths) != 4:
            raise ValueError("widths must contain four stages")
        if any(width <= 0 or width % 16 for width in self.widths):
            raise ValueError("stage widths must be positive multiples of 16")
        if any(left > right for left, right in zip(self.widths, self.widths[1:])):
            raise ValueError("stage widths must be non-decreasing")
        if len(self.depths) != 4:
            raise ValueError("depths must contain four stages")
        if self.depths[0] not in (1, 2):
            raise ValueError("stage 1 depth must be 1 or 2")
        if any(depth not in (0, 1, 2) for depth in self.depths[1:]):
            raise ValueError("stages 2 through 4 depths must be 0, 1, or 2")

    @classmethod
    def r18(cls) -> "EncoderSpec":
        return cls(64, _BASE_WIDTHS, (2, 2, 2, 2))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EncoderSpec":
        return cls(
            stem_width=int(value["stem_width"]),
            widths=tuple(value["widths"]),
            depths=tuple(value["depths"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem_width": self.stem_width,
            "widths": list(self.widths),
            "depths": list(self.depths),
            "slug": self.slug,
        }

    @property
    def slug(self) -> str:
        widths = "-".join(str(value) for value in self.widths)
        depths = "-".join(str(value) for value in self.depths)
        return f"s{self.stem_width}-w{widths}-d{depths}"


def _scaled_width(base: int, multiplier: float) -> int:
    return int(round(base * multiplier / 16.0)) * 16


def enumerate_encoder_specs() -> Iterator[EncoderSpec]:
    """Yield every legal encoder candidate in deterministic order."""
    seen = set()
    for stem_width in (32, 48, 64):
        for multipliers in product(_WIDTH_MULTIPLIERS, repeat=4):
            widths = tuple(
                _scaled_width(base, multiplier)
                for base, multiplier in zip(_BASE_WIDTHS, multipliers)
            )
            for stage1_depth in (1, 2):
                for later_depths in product((0, 1, 2), repeat=3):
                    spec = EncoderSpec(
                        stem_width=stem_width,
                        widths=widths,
                        depths=(stage1_depth,) + later_depths,
                    )
                    if spec in seen:
                        continue
                    seen.add(spec)
                    yield spec
