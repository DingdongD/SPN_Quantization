from ._dyspn_hw_parts import FoldedConvBN

ifmap_sz = [(64, 64, 64)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedConvBN):
    def __init__(self):
        super().__init__("conv2.0.conv2", "conv2.0.bn2", relu=False)
