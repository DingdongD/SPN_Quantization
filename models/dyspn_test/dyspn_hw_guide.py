from .dyspn_hw_aligned import DySPNHWAlignedModel, load_dyspn_hw_checkpoint

import torch.nn as nn


ifmap_sz = [(3, 128, 128), (1, 128, 128)]
input_layouts = ["BCHW", "BCHW"]
op_version = 18
batch_size = 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        self.net.eval()

    def forward(self, rgb, dep):
        return self.net.encode_decode(rgb, dep)["guide"]

