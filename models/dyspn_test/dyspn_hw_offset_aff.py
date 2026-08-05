from .dyspn_hw_aligned import DySPNHWAlignedModel, load_dyspn_hw_checkpoint

import torch.nn as nn


ifmap_sz = [(30, 128, 128)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        self.conv = self.net.propagation.conv_offset_aff
        self.eval()

    def forward(self, guide):
        return self.conv(guide)

