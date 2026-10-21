import torch
import torch.nn as nn

from spn_quant.hardware.collector import WorkloadCollector


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 12, 3, padding=1, groups=2)
        self.fc = nn.Linear(12, 5)

    def forward(self, x):
        y = self.conv(x)
        return self.fc(y.mean((-2, -1)))


def test_collector_records_executed_shapes_once():
    model = Toy().eval()
    with WorkloadCollector(model, model_name="dyspn") as collector:
        model(torch.zeros(1, 8, 10, 14))
    rows = {row.name: row for row in collector.layers()}
    assert rows["conv"].input_shape == (1, 8, 10, 14)
    assert rows["conv"].output_shape == (1, 12, 10, 14)
    assert rows["conv"].groups == 2
    assert rows["fc"].input_shape == (1, 12)
    assert rows["fc"].output_shape == (1, 5)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 128
        self.num_heads = 4
        self.sr_ratio = 4
        self.q = nn.Linear(128, 128, bias=False)
        self.kv = nn.Linear(128, 256, bias=False)
        self.proj = nn.Linear(128, 128, bias=False)

    def forward(self, x, height, width):
        del height, width
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(
            batch, tokens, self.num_heads, channels // self.num_heads)
        reduced = x[:, ::4, :]
        kv = self.kv(reduced).reshape(
            batch, reduced.shape[1], 2, self.num_heads,
            channels // self.num_heads)
        del q, kv
        return self.proj(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()

    def forward(self, x):
        return self.attn(x, 7, 8)


def test_collector_derives_attention_qk_and_av_shapes():
    model = AttentionModel().eval()
    with WorkloadCollector(model, model_name="completionformer") as collector:
        model(torch.zeros(1, 56, 128))
    derived = {
        row.kind: row for row in collector.layers()
        if row.kind in {"attention_qk", "attention_av"}
    }
    assert (derived["attention_qk"].m, derived["attention_qk"].n,
            derived["attention_qk"].k) == (56, 14, 32)
    assert (derived["attention_av"].m, derived["attention_av"].n,
            derived["attention_av"].k) == (56, 32, 14)
    assert derived["attention_qk"].repetitions == 4
    assert derived["attention_av"].repetitions == 4


class PredictionModel(nn.Module):
    def forward(self, x):
        return {"pred": x[:, :1]}


def test_collector_records_propagation_as_irregular_workload():
    model = PredictionModel().eval()
    with WorkloadCollector(
            model, model_name="nlspn", propagation_iterations=18,
            neighborhood_size=8) as collector:
        model(torch.zeros(1, 2, 10, 14))
    propagation, = tuple(
        row for row in collector.layers() if row.kind == "propagation")
    assert propagation.iterations == 18
    assert propagation.neighborhood_size == 8
    assert propagation.array_mappable is False
    assert propagation.macs == 1 * 10 * 14 * 18 * 8


def test_collector_removes_hooks_after_context_exit():
    model = Toy().eval()
    collector = WorkloadCollector(model, model_name="cspn")
    with collector:
        model(torch.zeros(1, 8, 10, 14))
    count = len(collector.layers())
    model(torch.zeros(1, 8, 10, 14))
    assert len(collector.layers()) == count
