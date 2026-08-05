from ._dyspn_hw_parts import SingleInputPart

ifmap_sz = [(64, 64, 64)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(SingleInputPart):
    def __init__(self):
        super().__init__("conv3")
