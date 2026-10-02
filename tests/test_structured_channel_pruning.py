import torch
import torch.nn as nn

from spn_quant.structured_channel_pruning import (
    prune_basic_block_hidden,
    prune_conv_bn_deconv_bridge,
    prune_mlp_hidden,
)


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, value):
        return self.fc2(self.act(self.fc1(value)))


class TinyBasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self, value):
        residual = value
        value = self.relu(self.bn1(self.conv1(value)))
        return self.relu(self.bn2(self.conv2(value)) + residual)


def test_prune_conv_bn_deconv_bridge_preserves_external_shapes():
    encoder_tail = nn.Sequential(
        nn.Conv2d(4, 8, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    )
    decoder_head = nn.Sequential(
        nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1,
                           output_padding=1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(),
    )

    kept = prune_conv_bn_deconv_bridge(encoder_tail, decoder_head, 5)
    output = decoder_head(encoder_tail(torch.randn(2, 4, 16, 16)))

    assert kept.shape == (5,)
    assert encoder_tail[0].weight.shape == (5, 4, 3, 3)
    assert encoder_tail[1].num_features == 5
    assert decoder_head[0].weight.shape == (5, 3, 3, 3)
    assert output.shape == (2, 3, 16, 16)


def test_prune_mlp_hidden_preserves_input_and_output_width():
    mlp = TinyMlp()

    kept = prune_mlp_hidden(mlp, 6)
    output = mlp(torch.randn(2, 7, 4))

    assert kept.shape == (6,)
    assert mlp.fc1.weight.shape == (6, 4)
    assert mlp.fc2.weight.shape == (4, 6)
    assert output.shape == (2, 7, 4)


def test_pruning_uses_joint_incoming_and_outgoing_channel_importance():
    mlp = TinyMlp()
    with torch.no_grad():
        mlp.fc1.weight.zero_()
        mlp.fc2.weight.zero_()
        mlp.fc1.weight[2].fill_(3.0)
        mlp.fc2.weight[:, 2].fill_(4.0)
        mlp.fc1.weight[6].fill_(2.0)
        mlp.fc2.weight[:, 6].fill_(5.0)

    kept = prune_mlp_hidden(mlp, 2)

    assert kept.tolist() == [2, 6]
    assert torch.all(mlp.fc1.weight[0] == 3.0)
    assert torch.all(mlp.fc1.weight[1] == 2.0)


def test_prune_basic_block_hidden_preserves_residual_interface():
    block = TinyBasicBlock()

    kept = prune_basic_block_hidden(block, 5)
    output = block(torch.randn(2, 4, 12, 16))

    assert kept.shape == (5,)
    assert block.conv1.weight.shape == (5, 4, 3, 3)
    assert block.bn1.num_features == 5
    assert block.conv2.weight.shape == (4, 5, 3, 3)
    assert output.shape == (2, 4, 12, 16)


def test_basic_block_pruning_uses_both_convolutions_for_importance():
    block = TinyBasicBlock()
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
        block.conv1.weight[1].fill_(4.0)
        block.conv2.weight[:, 1].fill_(3.0)
        block.conv1.weight[7].fill_(2.0)
        block.conv2.weight[:, 7].fill_(5.0)

    kept = prune_basic_block_hidden(block, 2)

    assert kept.tolist() == [1, 7]
    assert torch.all(block.conv1.weight[0] == 4.0)
    assert torch.all(block.conv1.weight[1] == 2.0)
