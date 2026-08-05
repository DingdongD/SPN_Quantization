from __future__ import annotations

import re

from ._dyspn_hw_parts import FoldedInputOutputChunkConv


def _spec_from_name(name: str):
    base = name.rsplit(".", 1)[-1]
    pattern = r"dyspn_hw_part_dec5_conv_oc(?P<os>\d+)_(?P<oe>\d+)_ic(?P<is>\d+)_(?P<ie>\d+)"
    match = re.fullmatch(pattern, base)
    if not match:
        raise ValueError(f"cannot parse dec5 chunk module name: {name}")
    out_start = int(match.group("os"))
    out_end = int(match.group("oe"))
    in_start = int(match.group("is"))
    in_end = int(match.group("ie"))
    return in_start, in_end, out_start, out_end


_IN_START, _IN_END, _OUT_START, _OUT_END = _spec_from_name(__name__)

ifmap_sz = [(_IN_END - _IN_START, 16, 16)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedInputOutputChunkConv):
    def __init__(self):
        super().__init__("dec5.conv", _IN_START, _IN_END, _OUT_START, _OUT_END, True)
