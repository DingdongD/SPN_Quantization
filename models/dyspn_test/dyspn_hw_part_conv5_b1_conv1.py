from ._dyspn_hw_parts import FoldedConvBN

ifmap_sz = [(512, 8, 8)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedConvBN):
    def __init__(self):
        super().__init__("conv5.1.conv1", "conv5.1.bn1", relu=True)
