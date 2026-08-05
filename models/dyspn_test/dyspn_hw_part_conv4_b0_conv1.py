from ._dyspn_hw_parts import FoldedConvBN

ifmap_sz = [(128, 32, 32)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(FoldedConvBN):
    def __init__(self):
        super().__init__("conv4.0.conv1", "conv4.0.bn1", relu=True)
