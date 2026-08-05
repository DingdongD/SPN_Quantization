from .dyspn_hw_aligned import DySPNHWAlignedModel, load_dyspn_hw_checkpoint

import torch
import torch.nn as nn


ifmap_sz = [(30, 128, 128)]
input_layouts = ["BCHW"]
op_version = 18
batch_size = 1


class Model(nn.Module):
    """Compile smoke probe for DySPN offset/aff Conv.

    The strict untrained DySPN initialization keeps this Conv at all zeros,
    which makes the legacy quantizer divide by a zero activation range. If a
    real `DYSPN_HW_CKPT` is supplied, use `dyspn_hw_offset_aff` instead. This
    probe only validates the operator shape for the no-checkpoint bootstrap.
    """

    def __init__(self):
        super().__init__()
        self.net = DySPNHWAlignedModel()
        load_dyspn_hw_checkpoint(self.net, strict=False)
        self.conv = self.net.propagation.conv_offset_aff
        if float(self.conv.weight.detach().abs().sum()) == 0.0 and float(self.conv.bias.detach().abs().sum()) == 0.0:
            with torch.no_grad():
                self.conv.weight.normal_(mean=0.0, std=0.01)
                self.conv.bias.zero_()
        self.eval()

    def forward(self, guide):
        return self.conv(guide)

