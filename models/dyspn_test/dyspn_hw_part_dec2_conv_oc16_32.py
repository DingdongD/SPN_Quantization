from ._dyspn_hw_parts import FoldedInputOutputChunkConv

ifmap_sz = [(128, 128, 128)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedInputOutputChunkConv):
    def __init__(self):
        super().__init__("dec2.conv", 0, 128, 16, 32, True)
