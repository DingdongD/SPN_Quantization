from ._dyspn_hw_parts import FoldedConvBNInputOutputChunk

ifmap_sz = [(128, 16, 16)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedConvBNInputOutputChunk):
    def __init__(self):
        super().__init__("conv5.0.conv1", "conv5.0.bn1", 128, 256, 0, 128, False, relu=False)
