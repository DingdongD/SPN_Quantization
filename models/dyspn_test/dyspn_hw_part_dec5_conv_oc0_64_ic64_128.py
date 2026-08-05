from ._dyspn_hw_parts import FoldedInputOutputChunkConv

ifmap_sz = [(64, 16, 16)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedInputOutputChunkConv):
    def __init__(self):
        super().__init__("dec5.conv", 64, 128, 0, 64, False)
