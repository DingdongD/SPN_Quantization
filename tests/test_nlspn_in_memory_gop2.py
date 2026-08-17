import numpy as np
import pytest
import torch

from scripts import nlspn_in_memory_gop2 as online


HEIGHT = 228
WIDTH = 304


def make_inputs(sparse_value=3.0):
    rgb = torch.zeros(3, HEIGHT, WIDTH)
    sparse = torch.zeros(HEIGHT, WIDTH)
    sparse.view(-1)[:500] = sparse_value
    return rgb, sparse


class FakePropLayer(torch.nn.Module):
    def forward(self, seed, guidance, confidence, fixed, rgb):
        assert fixed is None
        return seed, [seed], None, None, torch.tensor(1.0)


class FakeNLSPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prop_layer = FakePropLayer()
        self.calls = 0

    def forward(self, sample):
        self.calls += 1
        batch, _, height, width = sample["dep"].shape
        prediction = torch.full(
            (batch, 1, height, width), 2.0,
            dtype=sample["dep"].dtype, device=sample["dep"].device)
        return {
            "pred": prediction,
            "guidance": torch.zeros(
                batch, 8, height, width, device=sample["dep"].device),
            "confidence": torch.ones(
                batch, 1, height, width, device=sample["dep"].device),
        }


class FakeRAFT(torch.nn.Module):
    def forward(self, current, previous, num_flow_updates=12):
        assert current.shape == previous.shape == (1, 3, 232, 304)
        assert num_flow_updates == 12
        return [torch.zeros(
            current.shape[0], 2, current.shape[2], current.shape[3],
            device=current.device)]


def test_fixed_gop2_schedule():
    assert [online.frame_kind(index) for index in range(6)] == [
        "I", "P", "I", "P", "I", "P"]
    with pytest.raises(ValueError):
        online.frame_kind(-1)


def test_p_frame_requires_live_state():
    engine = online.InMemoryGOP2Engine(
        FakeNLSPN(), FakeRAFT(), torch.device("cpu"))
    rgb, sparse = make_inputs()
    with pytest.raises(RuntimeError, match="state"):
        engine.infer_p(rgb, sparse, 1)


def test_reset_drops_every_temporal_tensor():
    engine = online.InMemoryGOP2Engine(
        FakeNLSPN(), FakeRAFT(), torch.device("cpu"))
    rgb, sparse = make_inputs()
    engine.infer_i(rgb, sparse, 0)
    assert engine.state is not None
    engine.reset()
    assert engine.state is None


def test_i_frame_runs_full_model_and_returns_cpu_prediction():
    model = FakeNLSPN()
    engine = online.InMemoryGOP2Engine(
        model, FakeRAFT(), torch.device("cpu"))
    rgb, sparse = make_inputs()
    result = engine.infer_i(rgb, sparse, 0)
    assert result.kind == "I"
    assert result.prediction.device.type == "cpu"
    assert result.prediction.shape == (HEIGHT, WIDTH)
    assert result.latency_ms >= 0.0
    assert model.calls == 1
    assert engine.state.local_index == 0
    assert engine.state.previous_rgb.shape == (1, 3, HEIGHT, WIDTH)
    assert engine.state.previous_depth.shape == (1, 1, HEIGHT, WIDTH)
    assert engine.state.previous_guidance.shape == (1, 8, HEIGHT, WIDTH)
    assert engine.state.previous_confidence.shape == (1, 1, HEIGHT, WIDTH)


def test_p_frame_propagates_signed_sparse_seed_and_updates_state():
    engine = online.InMemoryGOP2Engine(
        FakeNLSPN(), FakeRAFT(), torch.device("cpu"))
    first_rgb, first_sparse = make_inputs()
    engine.infer_i(first_rgb, first_sparse, 0)

    current_rgb, current_sparse = make_inputs(sparse_value=3.0)
    current_rgb.fill_(0.25)
    result = engine.infer_p(current_rgb, current_sparse, 1)

    expected = torch.full((HEIGHT, WIDTH), 2.0)
    expected.view(-1)[:500] = 3.0
    torch.testing.assert_close(result.prediction, expected)
    assert result.kind == "P"
    assert result.latency_ms >= 0.0
    assert engine.state.local_index == 1
    torch.testing.assert_close(
        engine.state.previous_rgb[0], current_rgb)
    torch.testing.assert_close(
        engine.state.previous_depth[0, 0], expected)
    assert torch.count_nonzero(engine.state.previous_guidance) == 0
    assert torch.all(engine.state.previous_confidence == 1.0)


def test_engine_rejects_wrong_sparse_count():
    engine = online.InMemoryGOP2Engine(
        FakeNLSPN(), FakeRAFT(), torch.device("cpu"))
    rgb = torch.zeros(3, HEIGHT, WIDTH)
    sparse = torch.zeros(HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="500"):
        engine.infer_i(rgb, sparse, 0)

