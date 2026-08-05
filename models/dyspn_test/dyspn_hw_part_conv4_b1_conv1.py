from ._dyspn_hw_parts import FoldedConvBN

ifmap_sz = [(256, 16, 16)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedConvBN):
    def __init__(self):
        super().__init__("conv4.1.conv1", "conv4.1.bn1", relu=True)
