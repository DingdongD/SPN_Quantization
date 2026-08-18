import numpy as np
import pytest
import torch

from scripts import nlspn_in_memory_gop2 as online
from scripts import nlspn_temporal_residual as residual


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


class DirectionRecordingRAFT(torch.nn.Module):
    def __init__(self, dx=0.0):
        super().__init__()
        self.dx = float(dx)
        self.calls = []

    def forward(self, current, previous, num_flow_updates=12):
        self.calls.append((
            current.detach().clone(), previous.detach().clone(),
            num_flow_updates))
        flow = torch.zeros(
            current.shape[0], 2, current.shape[2], current.shape[3],
            device=current.device)
        flow[:, 0].fill_(self.dx)
        return [flow]


class CapturingPropLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.guidance = None
        self.confidence = None

    def forward(self, seed, guidance, confidence, fixed, rgb):
        self.guidance = guidance.detach().clone()
        self.confidence = confidence.detach().clone()
        zero = torch.zeros_like(seed)
        return zero, [zero], None, None, torch.tensor(1.0)


class SpatialNLSPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prop_layer = CapturingPropLayer()

    def forward(self, sample):
        batch, _, height, width = sample["dep"].shape
        x = torch.arange(
            width, dtype=sample["dep"].dtype,
            device=sample["dep"].device).view(1, 1, 1, width)
        x = x.expand(batch, 1, height, width)
        return {
            "pred": x / 100.0,
            "guidance": x.expand(batch, 8, height, width),
            "confidence": x + 1.0,
        }


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


def test_full_reference_frame_does_not_create_temporal_state():
    model = FakeNLSPN()
    engine = online.InMemoryGOP2Engine(
        model, FakeRAFT(), torch.device("cpu"))
    rgb, sparse = make_inputs()
    result = engine.infer_full(rgb, sparse)
    assert result.kind == "FULL"
    assert result.prediction.device.type == "cpu"
    assert result.prediction.shape == (HEIGHT, WIDTH)
    assert model.calls == 1
    assert engine.state is None


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


def test_p_frame_calls_raft_current_to_previous_with_twelve_updates():
    raft = DirectionRecordingRAFT()
    engine = online.InMemoryGOP2Engine(
        FakeNLSPN(), raft, torch.device("cpu"))
    previous_rgb, previous_sparse = make_inputs()
    engine.infer_i(previous_rgb, previous_sparse, 0)
    current_rgb, current_sparse = make_inputs()
    current_rgb.fill_(0.25)

    engine.infer_p(current_rgb, current_sparse, 1)

    assert len(raft.calls) == 1
    current, previous, updates = raft.calls[0]
    assert updates == 12
    assert torch.all(current == -0.5)
    assert torch.all(previous == -1.0)


def test_p_frame_strictly_warps_depth_guidance_and_confidence():
    model = SpatialNLSPN()
    raft = DirectionRecordingRAFT(dx=1.0)
    engine = online.InMemoryGOP2Engine(model, raft, torch.device("cpu"))
    previous_rgb, previous_sparse = make_inputs()
    engine.infer_i(previous_rgb, previous_sparse, 0)
    initial = engine.state
    flow = torch.zeros(1, 2, HEIGHT, WIDTH)
    flow[:, 0].fill_(1.0)
    expected_depth, _ = residual.backward_warp(initial.previous_depth, flow)
    expected_guidance, _ = residual.backward_warp(
        initial.previous_guidance, flow)
    expected_confidence, _ = residual.backward_warp(
        initial.previous_confidence, flow)
    current_rgb, current_sparse = make_inputs()

    result = engine.infer_p(current_rgb, current_sparse, 1)

    torch.testing.assert_close(result.prediction, expected_depth[0, 0])
    torch.testing.assert_close(
        model.prop_layer.guidance, expected_guidance)
    torch.testing.assert_close(
        model.prop_layer.confidence, expected_confidence)


def test_engine_rejects_wrong_sparse_count():
    engine = online.InMemoryGOP2Engine(
        FakeNLSPN(), FakeRAFT(), torch.device("cpu"))
    rgb = torch.zeros(3, HEIGHT, WIDTH)
    sparse = torch.zeros(HEIGHT, WIDTH)
    with pytest.raises(ValueError, match="500"):
        engine.infer_i(rgb, sparse, 0)


def test_latency_summary_reports_required_distribution():
    result = online.latency_summary([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["total_ms"] == 10.0
    assert result["mean_ms"] == 2.5
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0
    assert result["fps"] == 400.0


@pytest.mark.parametrize("values", [[], [-1.0], [float("nan")]])
def test_latency_summary_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        online.latency_summary(values)


def make_quality_case():
    gt = np.full((4, 2, 2), 5.0, dtype=np.float32)
    full = np.full((4, 2, 2), 6.0, dtype=np.float32)
    gop2 = np.empty_like(full)
    gop2[0] = 6.02
    gop2[1:] = 5.99
    valid = np.ones_like(gt, dtype=bool)
    return gt, full, gop2, valid


def test_quality_rows_keep_local_failure_when_pooled_gate_passes():
    gt, full, gop2, valid = make_quality_case()
    frame_ids = np.asarray([1, 2, 3, 4], dtype=np.int32)
    clips = np.asarray(["a", "a", "b", "b"])
    rows = online.frame_quality_rows(
        frame_ids, clips, full, gop2, gt, valid)
    summary = online.pooled_quality_summary(full, gop2, gt, valid)
    clip_rows = online.clip_quality_rows(rows)
    assert summary["passes"] is True
    assert summary["quality_ratio"] < 1.01
    assert rows[0]["quality_ratio"] == pytest.approx(1.02)
    assert rows[0]["passes_1pct"] is False
    assert len(clip_rows) == 2
    assert clip_rows[0]["clip"] == "a"
    assert clip_rows[0]["quality_ratio"] > 1.0


def test_benchmark_summary_separates_i_and_p_latency():
    gt, full, gop2, valid = make_quality_case()
    quality = online.pooled_quality_summary(full, gop2, gt, valid)
    result = online.benchmark_summary(
        full_latencies=[10.0, 10.0, 10.0, 10.0],
        gop2_latencies=[10.0, 2.0, 10.0, 2.0],
        gop2_kinds=["I", "P", "I", "P"],
        quality=quality,
    )
    assert result["speedup"] == pytest.approx(40.0 / 24.0)
    assert result["latency"]["i"]["count"] == 2
    assert result["latency"]["p"]["mean_ms"] == 2.0
