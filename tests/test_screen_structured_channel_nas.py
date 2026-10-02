from types import SimpleNamespace

import torch.nn as nn

from scripts.screen_structured_channel_nas import (
    candidate_ids,
    apply_structured_candidate,
    evaluation_indices,
)


def _bridge():
    return SimpleNamespace(base=SimpleNamespace(
        conv6=nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU()),
        dec5=nn.Sequential(
            nn.ConvTranspose2d(8, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3), nn.ReLU())))


class TinyMlp(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)


class TinyBlock(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.mlp = TinyMlp(width, hidden)


class TinyResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)


class TinyFormerBlock(TinyBlock):
    def __init__(self, width, hidden):
        super().__init__(width, hidden)
        self.resblock = TinyResidual(width)


def test_dyspn_candidate_prunes_bridge_without_changing_decoder_output():
    model = _bridge()

    report = apply_structured_candidate(
        model, "dyspn", "bridge_75pct", bridge_width=6)

    assert report["bridge_width"] == 6
    assert model.base.conv6[0].out_channels == 6
    assert model.base.dec5[0].in_channels == 6
    assert model.base.dec5[0].out_channels == 3


def test_completionformer_candidate_prunes_every_active_mlp():
    model = SimpleNamespace(backbone=SimpleNamespace(former=SimpleNamespace(
        block1=nn.ModuleList([TinyBlock(4, 8)]),
        block2=nn.ModuleList([TinyBlock(6, 12)]),
        block3=nn.ModuleList([]),
        block4=nn.ModuleList([TinyBlock(8, 16)]))))

    report = apply_structured_candidate(
        model, "completionformer", "mlp_75pct", mlp_ratio=0.75)

    assert report["pruned_mlp_count"] == 3
    assert model.backbone.former.block1[0].mlp.fc1.out_features == 6
    assert model.backbone.former.block2[0].mlp.fc1.out_features == 9
    assert model.backbone.former.block4[0].mlp.fc1.out_features == 12


def test_candidate_ids_include_baseline_and_three_pruning_levels():
    assert candidate_ids("dyspn")[:4] == (
        "baseline", "bridge_87p5pct", "bridge_75pct", "bridge_62p5pct")
    assert candidate_ids("dyspn")[4:7] == (
        "bridge62_stage5hidden_87p5pct",
        "bridge62_stage5hidden_75pct",
        "bridge62_stage5hidden_62p5pct",
    )
    assert candidate_ids("dyspn")[7:] == (
        "bridge62_s5hidden62_s4hidden_75pct",
        "bridge62_s5hidden62_s4hidden_50pct",
    )
    assert candidate_ids("completionformer")[:4] == (
        "baseline", "mlp_87p5pct", "mlp_75pct", "mlp_62p5pct")
    assert candidate_ids("completionformer")[4:7] == (
        "mlp62_stage4hidden_87p5pct",
        "mlp62_stage4hidden_75pct",
        "mlp62_stage4hidden_62p5pct",
    )
    assert candidate_ids("completionformer")[7:] == (
        "mlp62_s4hidden62_s3hidden_80pct",
        "mlp62_s4hidden62_s3hidden_60pct",
    )


def test_candidate_ids_can_select_only_requested_candidates():
    assert candidate_ids("dyspn", (
        "baseline", "bridge62_s5hidden62_s4hidden_50pct")) == (
            "baseline", "bridge62_s5hidden62_s4hidden_50pct")


def test_zero_sample_count_selects_the_full_dataset():
    assert evaluation_indices((8, 3, 1), 10, 0) == tuple(range(10))
    assert evaluation_indices((8, 3, 1), 10, 2) == (8, 3)


def test_dyspn_compound_candidate_prunes_bridge_and_stage5_hidden():
    model = _bridge()
    model.base.conv5 = nn.Sequential(
        TinyResidual(8), TinyResidual(8))

    report = apply_structured_candidate(
        model, "dyspn", "bridge62_stage5hidden_75pct",
        bridge_width=6)

    assert report["bridge_width"] == 6
    assert report["pruned_block_count"] == 2
    assert model.base.conv5[0].conv1.out_channels == 6
    assert model.base.conv5[0].conv2.in_channels == 6


def test_completionformer_compound_candidate_prunes_mlp_and_stage4_cnn():
    model = SimpleNamespace(backbone=SimpleNamespace(former=SimpleNamespace(
        block1=nn.ModuleList([TinyFormerBlock(4, 8)]),
        block2=nn.ModuleList([TinyFormerBlock(6, 12)]),
        block3=nn.ModuleList([]),
        block4=nn.ModuleList([TinyFormerBlock(8, 16)]))))

    report = apply_structured_candidate(
        model, "completionformer", "mlp62_stage4hidden_75pct")

    assert report["pruned_mlp_count"] == 3
    assert report["pruned_block_count"] == 1
    assert model.backbone.former.block1[0].mlp.fc1.out_features == 5
    assert model.backbone.former.block4[0].resblock.conv1.out_channels == 6


def test_dyspn_deep_compound_candidate_prunes_stage4_and_stage5():
    model = _bridge()
    model.base.conv4 = nn.Sequential(TinyResidual(4))
    model.base.conv5 = nn.Sequential(TinyResidual(8))

    report = apply_structured_candidate(
        model, "dyspn", "bridge62_s5hidden62_s4hidden_50pct",
        bridge_width=6)

    assert report["stage5_blocks"][0]["hidden_width"] == 5
    assert report["stage4_blocks"][0]["hidden_width"] == 2
    assert model.base.conv4[0].conv2.in_channels == 2


def test_completionformer_deep_candidate_prunes_stage3_and_stage4_cnn():
    model = SimpleNamespace(backbone=SimpleNamespace(former=SimpleNamespace(
        block1=nn.ModuleList([TinyFormerBlock(4, 8)]),
        block2=nn.ModuleList([TinyFormerBlock(6, 12)]),
        block3=nn.ModuleList([TinyFormerBlock(10, 20)]),
        block4=nn.ModuleList([TinyFormerBlock(8, 16)]))))

    report = apply_structured_candidate(
        model, "completionformer", "mlp62_s4hidden62_s3hidden_60pct")

    assert report["stage4_blocks"][0]["hidden_width"] == 5
    assert report["stage3_blocks"][0]["hidden_width"] == 6
    assert model.backbone.former.block3[0].resblock.conv2.in_channels == 6
