from __future__ import annotations

import re

from ._dyspn_hw_parts import FoldedConvBNInputOutputChunk


def _spec_from_name(name: str):
    base = name.rsplit(".", 1)[-1]
    pattern = (
        r"dyspn_hw_part_conv5_"
        r"b(?P<block>[01])_conv(?P<conv>[12])_"
        r"oc(?P<os>\d+)_(?P<oe>\d+)_"
        r"ic(?P<is>\d+)_(?P<ie>\d+)"
    )
    match = re.fullmatch(pattern, base)
    if not match:
        raise ValueError(f"cannot parse conv5 chunk module name: {name}")
    block = int(match.group("block"))
    conv = int(match.group("conv"))
    in_start = int(match.group("is"))
    in_end = int(match.group("ie"))
    out_start = int(match.group("os"))
    out_end = int(match.group("oe"))
    conv_path = f"conv5.{block}.conv{conv}"
    bn_path = f"conv5.{block}.bn{conv}"
    if block == 0 and conv == 1:
        spatial = 16
    else:
        spatial = 8
    return conv_path, bn_path, in_start, in_end, out_start, out_end, spatial


_CONV_PATH, _BN_PATH, _IN_START, _IN_END, _OUT_START, _OUT_END, _SPATIAL = _spec_from_name(__name__)

ifmap_sz = [(_IN_END - _IN_START, _SPATIAL, _SPATIAL)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedConvBNInputOutputChunk):
    def __init__(self):
        super().__init__(
            _CONV_PATH,
            _BN_PATH,
            _IN_START,
            _IN_END,
            _OUT_START,
            _OUT_END,
            include_bias=(_IN_START == 0),
            relu=False,
        )
