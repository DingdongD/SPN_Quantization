from ._dyspn_hw_parts import FoldedInputChunkConv

ifmap_sz = [(64, 128, 128)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedInputChunkConv):
    def __init__(self):
        super().__init__("gd_dec1", 0, 64, True)
