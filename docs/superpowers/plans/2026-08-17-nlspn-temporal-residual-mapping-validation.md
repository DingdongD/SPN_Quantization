# NLSPN Temporal Residual Mapping Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible 256-frame pilot that determines whether the frozen NLSPN propagation operator can reconstruct causal motion-compensated depth residuals within 1% of full per-frame NLSPN RMSE.

**Architecture:** A base-environment orchestrator prepares eight canonical clips, runs Torchvision RAFT-Small for backward flow, and computes metrics and artifacts. A Python 3.7 worker runs frozen NLSPN baseline inference and the existing `prop_layer` inside the `completionformer-py37` environment, exchanging versioned compressed NumPy payloads with the orchestrator. No NLSPN weights, model structure, or propagation settings change.

**Tech Stack:** Python, PyTorch, Torchvision RAFT-Small, NumPy, SciPy, Matplotlib, OpenCV/Pillow, Pytest, the existing NLSPN custom deformable-convolution environment.

---

## File map

- Create `scripts/nlspn_temporal_residual.py`: shared constants, clip and pair contracts, backward warping, residual construction, metric primitives, and quality gate.
- Create `scripts/run_nlspn_temporal_residual_worker.py`: Python 3.7-compatible NLSPN baseline and propagation stages.
- Create `scripts/run_nlspn_temporal_residual_validation.py`: preprocessing, RAFT inference, worker orchestration, cache validation, CSV/JSON/plot rendering, and CLI.
- Create `tests/test_nlspn_temporal_residual.py`: pure contract, warp, residual, and metric tests.
- Create `tests/test_run_nlspn_temporal_residual_worker.py`: worker payload and propagation tests with small fake models.
- Create `tests/test_run_nlspn_temporal_residual_validation.py`: orchestration, RAFT adapter, metadata, and artifact tests with injected fakes.
- Create `docs/2026-08-17-nlspn-temporal-residual-pilot-results.md`: generated pilot result interpretation after the run.

Do not modify the external NLSPN or CompletionFormer repositories. Import the existing model adapter from `scripts/run_spn_sequence_worker.py` and the existing preprocessing functions from `scripts/export_cspn_sequence_predictions.py`.

### Task 1: Freeze the pilot clip and pair contracts

**Files:**
- Create: `scripts/nlspn_temporal_residual.py`
- Create: `tests/test_nlspn_temporal_residual.py`

- [ ] **Step 1: Write failing clip-contract tests**

```python
# tests/test_nlspn_temporal_residual.py
import numpy as np
import pytest

from scripts import nlspn_temporal_residual as residual


def test_pilot_clips_cover_256_frames_without_cross_clip_pairs():
    assert residual.PILOT_CLIPS == (
        (1, 32), (282, 313), (563, 594), (844, 875),
        (1126, 1157), (1407, 1438), (1688, 1719), (1969, 2000),
    )
    frames = residual.clip_frame_ids(residual.PILOT_CLIPS)
    pairs = residual.clip_pairs(residual.PILOT_CLIPS)
    assert len(frames) == 256
    assert len(set(frames)) == 256
    assert len(pairs) == 248
    assert (32, 282) not in pairs
    assert pairs[0] == (1, 2)
    assert pairs[-1] == (1999, 2000)


def test_validate_clip_payload_requires_fixed_shapes_and_500_points():
    payload = {
        "frame_ids": np.arange(1, 33, dtype=np.int32),
        "rgb": np.zeros((32, 3, 228, 304), dtype=np.float32),
        "sparse": np.zeros((32, 228, 304), dtype=np.float32),
        "gt": np.ones((32, 228, 304), dtype=np.float32),
        "valid": np.ones((32, 228, 304), dtype=bool),
    }
    locations = np.arange(500)
    payload["sparse"].reshape(32, -1)[:, locations] = 1.0
    residual.validate_clip_payload(payload)
    payload["sparse"][0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="500 sparse points"):
        residual.validate_clip_payload(payload)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
python -m pytest -q tests/test_nlspn_temporal_residual.py
```

Expected: collection fails because `scripts.nlspn_temporal_residual` does not exist.

- [ ] **Step 3: Add the pilot contracts and payload validation**

```python
# scripts/nlspn_temporal_residual.py
"""Shared contracts and math for NLSPN temporal-residual validation."""

import hashlib
from pathlib import Path

import numpy as np


HEIGHT = 228
WIDTH = 304
SPARSE_COUNT = 500
SPARSE_SEED = 2026
MAX_DEPTH = 10.0
PILOT_CLIPS = (
    (1, 32), (282, 313), (563, 594), (844, 875),
    (1126, 1157), (1407, 1438), (1688, 1719), (1969, 2000),
)


def clip_frame_ids(clips):
    return tuple(
        frame_id
        for start, end in clips
        for frame_id in range(int(start), int(end) + 1)
    )


def clip_pairs(clips):
    return tuple(
        (frame_id, frame_id + 1)
        for start, end in clips
        for frame_id in range(int(start), int(end))
    )


def validate_clip_payload(payload):
    frame_ids = np.asarray(payload["frame_ids"])
    frame_count = frame_ids.size
    expected = {
        "rgb": (frame_count, 3, HEIGHT, WIDTH),
        "sparse": (frame_count, HEIGHT, WIDTH),
        "gt": (frame_count, HEIGHT, WIDTH),
        "valid": (frame_count, HEIGHT, WIDTH),
    }
    for key, shape in expected.items():
        value = np.asarray(payload[key])
        if value.shape != shape:
            raise ValueError("%s must have shape %r" % (key, shape))
    sparse_count = np.count_nonzero(payload["sparse"], axis=(1, 2))
    if not np.all(sparse_count == SPARSE_COUNT):
        raise ValueError("every frame must contain exactly 500 sparse points")
    if not np.isfinite(payload["rgb"]).all():
        raise ValueError("RGB contains non-finite values")
    for key in ("sparse", "gt"):
        if not np.isfinite(np.asarray(payload[key])).all():
            raise ValueError("%s contains non-finite values" % key)
    return payload


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 4: Run the contract tests**

Run:

```bash
python -m pytest -q tests/test_nlspn_temporal_residual.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the contracts**

```bash
git add scripts/nlspn_temporal_residual.py tests/test_nlspn_temporal_residual.py
git commit -m "feat: add NLSPN temporal pilot contracts"
```

### Task 2: Implement tested motion-compensation and residual math

**Files:**
- Modify: `scripts/nlspn_temporal_residual.py`
- Modify: `tests/test_nlspn_temporal_residual.py`

- [ ] **Step 1: Add failing synthetic warp and residual tests**

```python
import torch


def test_backward_warp_uses_target_to_source_pixel_flow_and_border_fill():
    source = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    flow = torch.zeros((1, 2, 1, 4))
    flow[:, 0] = 1.0
    warped, in_bounds = residual.backward_warp(source, flow)
    assert torch.equal(warped, torch.tensor([[[[2.0, 3.0, 4.0, 4.0]]]]))
    assert torch.equal(
        in_bounds, torch.tensor([[[[True, True, True, False]]]]))


def test_sparse_residual_seed_keeps_signed_values_only_at_measurements():
    base = np.full((2, 2), 2.0, dtype=np.float32)
    sparse = np.array([[0.0, 1.5], [3.0, 0.0]], dtype=np.float32)
    seed, mask = residual.sparse_residual_seed(sparse, base)
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])
    np.testing.assert_allclose(seed, [[0.0, -0.5], [1.0, 0.0]])


def test_pooled_quality_ratio_uses_pixels_not_mean_of_frame_rmse():
    gt = np.zeros((2, 1, 2), dtype=np.float32)
    full = np.array([[[1.0, 1.0]], [[2.0, 2.0]]], dtype=np.float32)
    reconstructed = full * 1.01
    valid = np.ones_like(gt, dtype=bool)
    result = residual.pooled_quality(full, reconstructed, gt, valid)
    assert result["rmse_full"] == pytest.approx(np.sqrt(2.5))
    assert result["quality_ratio"] == pytest.approx(1.01)
    assert result["passes"] is True
```

- [ ] **Step 2: Run the focused tests and verify missing-symbol failures**

Run:

```bash
python -m pytest -q \
  tests/test_nlspn_temporal_residual.py::test_backward_warp_uses_target_to_source_pixel_flow_and_border_fill \
  tests/test_nlspn_temporal_residual.py::test_sparse_residual_seed_keeps_signed_values_only_at_measurements \
  tests/test_nlspn_temporal_residual.py::test_pooled_quality_ratio_uses_pixels_not_mean_of_frame_rmse
```

Expected: all three fail because the functions are absent.

- [ ] **Step 3: Implement pixel-space warping, residual seeding, and pooling**

```python
# Add to scripts/nlspn_temporal_residual.py
def backward_warp(source, flow):
    import torch
    import torch.nn.functional as torch_f

    if source.ndim != 4 or flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("source and flow must be BCHW tensors")
    batch, _, height, width = source.shape
    if flow.shape != (batch, 2, height, width):
        raise ValueError("flow shape does not match source")
    y_axis = torch.arange(
        height, device=source.device, dtype=source.dtype)
    x_axis = torch.arange(
        width, device=source.device, dtype=source.dtype)
    try:
        ys, xs = torch.meshgrid(y_axis, x_axis, indexing="ij")
    except TypeError as error:
        if "indexing" not in str(error):
            raise
        ys, xs = torch.meshgrid(y_axis, x_axis)
    sample_x = xs[None] + flow[:, 0]
    sample_y = ys[None] + flow[:, 1]
    in_bounds = (
        (sample_x >= 0) & (sample_x <= width - 1) &
        (sample_y >= 0) & (sample_y <= height - 1)
    )
    grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = torch_f.grid_sample(
        source, grid, mode="bilinear", padding_mode="border",
        align_corners=True,
    )
    return warped, in_bounds[:, None]


def sparse_residual_seed(sparse, base):
    sparse = np.asarray(sparse, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    if sparse.shape != base.shape:
        raise ValueError("sparse and base shapes differ")
    mask = sparse > 0.0
    seed = np.zeros_like(sparse)
    seed[mask] = sparse[mask] - base[mask]
    return seed, mask


def pooled_quality(full, reconstructed, gt, valid):
    full = np.asarray(full, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (full.shape == reconstructed.shape == gt.shape == valid.shape):
        raise ValueError("quality arrays must have identical shapes")
    if not np.any(valid):
        raise ValueError("quality mask is empty")
    full_error = full[valid] - gt[valid]
    reconstructed_error = reconstructed[valid] - gt[valid]
    rmse_full = float(np.sqrt(np.mean(full_error ** 2)))
    if rmse_full <= 0.0:
        raise ValueError("full baseline RMSE must be positive")
    rmse_reconstructed = float(
        np.sqrt(np.mean(reconstructed_error ** 2)))
    ratio = rmse_reconstructed / rmse_full
    return {
        "rmse_full": rmse_full,
        "rmse_reconstructed": rmse_reconstructed,
        "quality_ratio": ratio,
        "passes": bool(ratio <= 1.01),
    }
```

The narrow `TypeError` branch preserves compatibility with Torch 1.10 in the
Python 3.7 worker. Add a test that monkeypatches `torch.meshgrid` to reject the
`indexing` keyword and verifies that the second call returns the same warp.
Another test raises an unrelated `TypeError` and verifies it is not
suppressed.

- [ ] **Step 4: Run all pure tests**

Run:

```bash
python -m pytest -q tests/test_nlspn_temporal_residual.py
```

Expected: `5 passed`.

- [ ] **Step 5: Commit the residual math**

```bash
git add scripts/nlspn_temporal_residual.py tests/test_nlspn_temporal_residual.py
git commit -m "feat: add temporal residual warp and quality math"
```

### Task 3: Prepare deterministic canonical pilot clips

**Files:**
- Create: `scripts/run_nlspn_temporal_residual_validation.py`
- Create: `tests/test_run_nlspn_temporal_residual_validation.py`

- [ ] **Step 1: Write failing preprocessing and cache tests**

```python
# tests/test_run_nlspn_temporal_residual_validation.py
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_nlspn_temporal_residual_validation as runner


def test_prepare_clip_uses_common_fixed_sparse_mask(monkeypatch, tmp_path):
    frame_ids = (1, 2)
    rgb = np.zeros((3, 228, 304), dtype=np.float32)
    gt = np.ones((228, 304), dtype=np.float32)
    valid = np.ones((228, 304), dtype=bool)
    monkeypatch.setattr(
        runner, "load_preprocessed_frame",
        lambda data_root, scene, frame_id: (rgb, gt, valid),
    )
    output = tmp_path / "clip_0001_0002.npz"
    payload = runner.prepare_clip(
        Path("/data"), "scene", frame_ids, output, seed=2026)
    assert output.is_file()
    assert payload["rgb"].shape == (2, 3, 228, 304)
    masks = payload["sparse"] > 0
    np.testing.assert_array_equal(masks[0], masks[1])
    assert masks[0].sum() == 500
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run:

```bash
python -m pytest -q \
  tests/test_run_nlspn_temporal_residual_validation.py::test_prepare_clip_uses_common_fixed_sparse_mask
```

Expected: collection fails because the runner module does not exist.

- [ ] **Step 3: Implement frame loading and atomic clip preparation**

```python
# scripts/run_nlspn_temporal_residual_validation.py
#!/usr/bin/env python3
"""Orchestrate the frozen-NLSPN temporal residual pilot."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone

import numpy as np

from scripts import export_cspn_sequence_predictions as sequence
from scripts import nlspn_temporal_residual as residual


def load_preprocessed_frame(data_root, scene, frame_id):
    rgb_path = Path(data_root) / scene / "rgb" / ("%04d.jpg" % frame_id)
    depth_path = (
        Path(data_root) / scene / "depth" / ("Image%04d.exr" % frame_id))
    return sequence.preprocess_pair(
        sequence.load_rgb(rgb_path), sequence.read_exr_depth(depth_path))


def write_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_clip(data_root, scene, frame_ids, output, seed):
    frames = [
        load_preprocessed_frame(data_root, scene, frame_id)
        for frame_id in frame_ids
    ]
    rgb, gt, valid = (np.stack(values) for values in zip(*frames))
    sparse, sparse_mask = sequence.build_shared_sparse_depths(
        gt, valid, count=residual.SPARSE_COUNT, seed=seed)
    payload = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "rgb": rgb.astype(np.float32),
        "sparse": sparse.astype(np.float32),
        "gt": gt.astype(np.float32),
        "valid": valid.astype(bool),
        "sparse_mask": sparse_mask.astype(bool),
    }
    residual.validate_clip_payload(payload)
    write_npz_atomic(output, **payload)
    return payload
```

- [ ] **Step 4: Run the preprocessing test and existing sequence tests**

Run:

```bash
python -m pytest -q \
  tests/test_run_nlspn_temporal_residual_validation.py \
  tests/test_export_cspn_sequence_predictions.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit deterministic pilot preparation**

```bash
git add scripts/run_nlspn_temporal_residual_validation.py \
  tests/test_run_nlspn_temporal_residual_validation.py
git commit -m "feat: prepare canonical NLSPN temporal clips"
```

### Task 4: Export frozen NLSPN baselines and intermediate fields

**Files:**
- Create: `scripts/run_nlspn_temporal_residual_worker.py`
- Create: `tests/test_run_nlspn_temporal_residual_worker.py`

- [ ] **Step 1: Write failing baseline-output tests with a fake model**

```python
# tests/test_run_nlspn_temporal_residual_worker.py
import numpy as np
import torch

from scripts import run_nlspn_temporal_residual_worker as worker


class FakeNLSPN(torch.nn.Module):
    def forward(self, sample):
        depth = sample["dep"] + 1.0
        batch, _, height, width = depth.shape
        return {
            "pred": depth,
            "pred_init": depth - 0.1,
            "guidance": depth.repeat(1, 8, 1, 1),
            "confidence": torch.full_like(depth, 0.75),
            "offset": depth.repeat(1, 16, 1, 1),
            "aff": depth.repeat(1, 9, 1, 1),
        }


def test_predict_baseline_exports_required_fields():
    rgb = np.zeros((2, 3, 4, 5), dtype=np.float32)
    sparse = np.zeros((2, 4, 5), dtype=np.float32)
    result, seconds = worker.predict_baseline(
        FakeNLSPN(), rgb, sparse, torch.device("cpu"))
    assert seconds >= 0.0
    assert set(result) == {
        "pred", "pred_init", "guidance", "confidence", "offset", "aff"}
    assert result["pred"].shape == (2, 4, 5)
    assert result["guidance"].shape == (2, 8, 4, 5)
    assert all(np.isfinite(value).all() for value in result.values())
```

- [ ] **Step 2: Run the worker test and verify the missing module failure**

Run:

```bash
python -m pytest -q \
  tests/test_run_nlspn_temporal_residual_worker.py::test_predict_baseline_exports_required_fields
```

Expected: collection fails because the worker module does not exist.

- [ ] **Step 3: Implement the Python 3.7-compatible baseline stage**

```python
# scripts/run_nlspn_temporal_residual_worker.py
#!/usr/bin/env python3
"""Run frozen NLSPN stages in the custom deform-convolution environment."""

from __future__ import print_function

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts import nlspn_temporal_residual as residual
from scripts import run_spn_sequence_worker as spn_worker


BASELINE_FIELDS = (
    "pred", "pred_init", "guidance", "confidence", "offset", "aff")


def predict_baseline(model, rgb, sparse, device):
    collected = dict((key, []) for key in BASELINE_FIELDS)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(rgb.shape[0]):
            sample = {
                "rgb": torch.from_numpy(rgb[index:index + 1]).to(device),
                "dep": torch.from_numpy(
                    sparse[index:index + 1, None]).to(device),
            }
            output = model(sample)
            for key in BASELINE_FIELDS:
                value = output[key].detach().cpu().numpy()[0]
                if key in ("pred", "pred_init"):
                    value = value[0]
                collected[key].append(value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    result = dict(
        (key, np.stack(values).astype(np.float32))
        for key, values in collected.items())
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("baseline output contains non-finite values")
    return result, seconds
```

Add a `baseline` CLI stage that loads one canonical clip, validates it, builds
NLSPN through `spn_worker.build_model("nlspn", args, device)`, loads the
checkpoint with `spn_worker.load_checkpoint_strict`, writes the six arrays
plus frame IDs and timing atomically, and records the checkpoint SHA-256.
The stage must reject an output whose frame IDs or shapes do not match the
input payload.

- [ ] **Step 4: Run worker unit tests in both Python environments**

Run:

```bash
python -m pytest -q tests/test_run_nlspn_temporal_residual_worker.py
conda run -n completionformer-py37 env \
  PYTHONPATH=/workspace/SPN_Quantization/.worktrees/four-model-five-frame-comparison:/workspace/external_depth_completion_models/NLSPN_ECCV20/src:/workspace/external_depth_completion_models/NLSPN_ECCV20/src/model/deformconv \
  python -m pytest -q tests/test_run_nlspn_temporal_residual_worker.py
```

Expected: all worker unit tests pass in both environments.

- [ ] **Step 5: Commit the baseline worker**

```bash
git add scripts/run_nlspn_temporal_residual_worker.py \
  tests/test_run_nlspn_temporal_residual_worker.py
git commit -m "feat: export frozen NLSPN temporal baselines"
```

### Task 5: Add causal RAFT-Small backward flow

**Files:**
- Modify: `scripts/run_nlspn_temporal_residual_validation.py`
- Modify: `tests/test_run_nlspn_temporal_residual_validation.py`

- [ ] **Step 1: Write failing RAFT adapter tests with an injected fake**

```python
import torch


class FakeRaft(torch.nn.Module):
    def forward(self, image1, image2):
        batch, _, height, width = image1.shape
        flow = torch.zeros((batch, 2, height, width), device=image1.device)
        flow[:, 0] = 2.0
        return [flow]


def test_predict_backward_flow_is_current_to_previous_and_crops_padding():
    rgb = np.zeros((3, 3, 228, 304), dtype=np.float32)
    flow, seconds = runner.predict_backward_flow(
        FakeRaft(), lambda current, previous: (current, previous),
        rgb, torch.device("cpu"), batch_size=2)
    assert seconds >= 0.0
    assert flow.shape == (2, 2, 228, 304)
    np.testing.assert_allclose(flow[:, 0], 2.0)
```

- [ ] **Step 2: Run the focused test and verify the missing-function failure**

Run:

```bash
python -m pytest -q \
  tests/test_run_nlspn_temporal_residual_validation.py::test_predict_backward_flow_is_current_to_previous_and_crops_padding
```

Expected: fail because `predict_backward_flow` is absent.

- [ ] **Step 3: Implement official RAFT loading and batched causal flow**

```python
# Add to scripts/run_nlspn_temporal_residual_validation.py
def build_raft(device):
    from torchvision.models.optical_flow import (
        Raft_Small_Weights, raft_small)

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    return model, weights.transforms(), weights


def predict_backward_flow(model, transform, rgb, device, batch_size=4):
    import torch
    import torch.nn.functional as torch_f
    import time

    rgb_tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32))
    flows = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for first in range(0, rgb_tensor.shape[0] - 1, batch_size):
            last = min(first + batch_size, rgb_tensor.shape[0] - 1)
            current = rgb_tensor[first + 1:last + 1].to(device)
            previous = rgb_tensor[first:last].to(device)
            current = torch_f.pad(current, (0, 0, 2, 2), mode="replicate")
            previous = torch_f.pad(previous, (0, 0, 2, 2), mode="replicate")
            current, previous = transform(current, previous)
            prediction = model(current, previous)[-1]
            flows.append(prediction[:, :, 2:-2].cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    flow = torch.cat(flows).numpy().astype(np.float32)
    expected = (rgb_tensor.shape[0] - 1, 2, 228, 304)
    if flow.shape != expected or not np.isfinite(flow).all():
        raise ValueError("RAFT flow has invalid shape or values")
    return flow, seconds
```

Record the resolved Torchvision weight file path and SHA-256. If the official
weight cannot be resolved or loaded, propagate the exception and leave run
metadata incomplete. Do not instantiate an untrained model.

- [ ] **Step 4: Run RAFT adapter and warp tests**

Run:

```bash
python -m pytest -q \
  tests/test_run_nlspn_temporal_residual_validation.py \
  tests/test_nlspn_temporal_residual.py
```

Expected: all tests pass without downloading weights because the unit test
injects `FakeRaft`.

- [ ] **Step 5: Commit RAFT support**

```bash
git add scripts/run_nlspn_temporal_residual_validation.py \
  tests/test_run_nlspn_temporal_residual_validation.py
git commit -m "feat: add causal RAFT flow for temporal validation"
```

### Task 6: Reconstruct residuals with the existing NLSPN propagation layer

**Files:**
- Modify: `scripts/run_nlspn_temporal_residual_worker.py`
- Modify: `tests/test_run_nlspn_temporal_residual_worker.py`

- [ ] **Step 1: Write failing oracle and causal propagation tests**

```python
class FakePropLayer(torch.nn.Module):
    def forward(self, feat_init, guidance, confidence, feat_fix, rgb):
        result = feat_init + guidance[:, :1] * 0.0
        return result, [result], None, None, torch.tensor(1.0)


class FakePropagationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prop_layer = FakePropLayer()


def test_reconstruct_pair_uses_sparse_signed_residual_and_original_prop():
    model = FakePropagationModel()
    previous_depth = torch.full((1, 1, 2, 3), 2.0)
    current_sparse = torch.tensor([[[[0.0, 1.5, 0.0],
                                     [3.0, 0.0, 0.0]]]])
    flow = torch.zeros((1, 2, 2, 3))
    guidance = torch.zeros((1, 8, 2, 3))
    confidence = torch.ones((1, 1, 2, 3))
    reconstructed, dense_residual = worker.reconstruct_pair(
        model, previous_depth, current_sparse, flow,
        guidance, confidence, torch.zeros((1, 3, 2, 3)))
    np.testing.assert_allclose(
        dense_residual.numpy()[0, 0],
        [[0.0, -0.5, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        reconstructed.numpy()[0, 0],
        [[2.0, 1.5, 2.0], [3.0, 2.0, 2.0]])
```

- [ ] **Step 2: Run the focused test and verify the missing-function failure**

Run:

```bash
python -m pytest -q \
  tests/test_run_nlspn_temporal_residual_worker.py::test_reconstruct_pair_uses_sparse_signed_residual_and_original_prop
```

Expected: fail because `reconstruct_pair` is absent.

- [ ] **Step 3: Implement the exact frozen-propagation reconstruction**

```python
# Add to scripts/run_nlspn_temporal_residual_worker.py
def reconstruct_pair(model, previous_depth, current_sparse, backward_flow,
                     guidance, confidence, current_rgb):
    base, _ = residual.backward_warp(previous_depth, backward_flow)
    sparse_mask = current_sparse > 0.0
    seed = torch.zeros_like(base)
    seed[sparse_mask] = current_sparse[sparse_mask] - base[sparse_mask]
    dense_residual = model.prop_layer(
        seed, guidance, confidence, None, current_rgb)[0]
    reconstructed = torch.clamp(
        base + dense_residual, min=0.0, max=residual.MAX_DEPTH)
    return reconstructed, dense_residual
```

Implement the `propagate` CLI stage. For each adjacent pair, construct the
same `d_base` and sparse seed, then run:

- oracle mode with `guidance[t+1]` and `confidence[t+1]`;
- causal mode with `backward_warp(guidance[t], flow)` and
  `backward_warp(confidence[t], flow)`.

Save `base`, `oracle_residual`, `oracle_prediction`, `causal_residual`,
`causal_prediction`, and `in_bounds` for all pairs. Verify that the model
configuration remains ResNet-34, 18 iterations, `TGASS`,
`preserve_input=False`, and that every output is finite before writing.
Synchronize CUDA and record propagation-only timing separately for oracle and
causal modes.

- [ ] **Step 4: Run all worker tests in both environments**

Run:

```bash
python -m pytest -q tests/test_run_nlspn_temporal_residual_worker.py
conda run -n completionformer-py37 env \
  PYTHONPATH=/workspace/SPN_Quantization/.worktrees/four-model-five-frame-comparison:/workspace/external_depth_completion_models/NLSPN_ECCV20/src:/workspace/external_depth_completion_models/NLSPN_ECCV20/src/model/deformconv \
  python -m pytest -q tests/test_run_nlspn_temporal_residual_worker.py
```

Expected: all worker tests pass in both environments.

- [ ] **Step 5: Commit original-propagation reconstruction**

```bash
git add scripts/run_nlspn_temporal_residual_worker.py \
  tests/test_run_nlspn_temporal_residual_worker.py
git commit -m "feat: reconstruct temporal residuals with NLSPN propagation"
```

### Task 7: Compute mapping statistics and quality artifacts

**Files:**
- Modify: `scripts/nlspn_temporal_residual.py`
- Modify: `scripts/run_nlspn_temporal_residual_validation.py`
- Modify: `tests/test_nlspn_temporal_residual.py`
- Modify: `tests/test_run_nlspn_temporal_residual_validation.py`

- [ ] **Step 1: Write failing residual-statistics and completion tests**

```python
def test_residual_statistics_reports_thresholds_and_finite_correlations():
    values = np.array([-0.10, -0.02, 0.0, 0.01, 0.04], dtype=np.float32)
    stats = residual.residual_statistics(values)
    assert stats["count"] == 5
    assert stats["rmse"] == pytest.approx(
        np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    assert stats["fraction_below_1cm"] == pytest.approx(0.4)
    assert stats["fraction_below_5cm"] == pytest.approx(0.8)
    assert all(np.isfinite(value) for value in stats.values())


def test_finalize_metadata_requires_every_artifact(tmp_path):
    metadata = {"complete": False}
    required = (tmp_path / "summary.json", tmp_path / "pair_metrics.csv")
    required[0].write_text("{}")
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.finalize_metadata(tmp_path / "run_metadata.json", metadata,
                                 required)
    required[1].write_text("model,pair\n")
    runner.finalize_metadata(tmp_path / "run_metadata.json", metadata,
                             required)
    saved = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert saved["complete"] is True
```

- [ ] **Step 2: Run the focused tests and verify missing-function failures**

Run:

```bash
python -m pytest -q \
  tests/test_nlspn_temporal_residual.py::test_residual_statistics_reports_thresholds_and_finite_correlations \
  tests/test_run_nlspn_temporal_residual_validation.py::test_finalize_metadata_requires_every_artifact
```

Expected: both fail because the functions are absent.

- [ ] **Step 3: Implement statistics, CSV/JSON output, and plots**

```python
# Add to scripts/nlspn_temporal_residual.py
def residual_statistics(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("residual statistics require finite values")
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mae": float(np.mean(absolute)),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "residual_energy": float(np.sum(values ** 2)),
        "fraction_below_1cm": float(np.mean(absolute <= 0.01)),
        "fraction_below_2cm": float(np.mean(absolute <= 0.02)),
        "fraction_below_5cm": float(np.mean(absolute <= 0.05)),
        "fraction_below_10cm": float(np.mean(absolute <= 0.10)),
    }
```

In the runner, add `write_json_atomic`, `write_csv_atomic`, and
`finalize_metadata`. `finalize_metadata` must check every required file is a
non-empty regular file, set `complete=true`, add `artifact_count`, and replace
the metadata file atomically. Add Pearson correlation with NumPy and Spearman
correlation with `scipy.stats.spearmanr`; constant inputs produce an explicit
zero correlation and a `constant_input=true` field instead of NaN.

Generate:

- `frame_metrics.csv` with full, oracle, and causal GT metrics;
- `pair_metrics.csv` with raw/aligned statistics, input-residual
  relationships, flow magnitude, photometric error, and in-bounds coverage;
- `clip_summary.csv` with pooled full/oracle/causal RMSE and ratios;
- `summary.json` with global pooled ratios and the three interpretation
  branches from the specification;
- `residual_mapping_overview.png` using the first pair and the largest-causal-
  error pair from each clip;
- `quality_ratio_by_frame.png` with a horizontal 1.01 gate line.

- [ ] **Step 4: Run all metric and artifact tests**

Run:

```bash
python -m pytest -q \
  tests/test_nlspn_temporal_residual.py \
  tests/test_run_nlspn_temporal_residual_validation.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit metrics and artifacts**

```bash
git add scripts/nlspn_temporal_residual.py \
  scripts/run_nlspn_temporal_residual_validation.py \
  tests/test_nlspn_temporal_residual.py \
  tests/test_run_nlspn_temporal_residual_validation.py
git commit -m "feat: report NLSPN temporal residual quality"
```

### Task 8: Complete orchestration, cache integrity, and CLI

**Files:**
- Modify: `scripts/run_nlspn_temporal_residual_validation.py`
- Modify: `tests/test_run_nlspn_temporal_residual_validation.py`

- [ ] **Step 1: Write failing worker-command and incomplete-run tests**

```python
def test_worker_command_uses_completionformer_environment(tmp_path):
    command, env = runner.build_worker_command(
        "baseline", tmp_path / "clip.npz", tmp_path / "baseline.npz",
        checkpoint=Path("/weights/best.pt"),
        args_json=Path("/weights/args.json"), device="cuda:0")
    assert command[:4] == [
        "conda", "run", "-n", "completionformer-py37"]
    assert "--stage" in command and "baseline" in command
    assert "NLSPN_ECCV20/src/model/deformconv" in env["PYTHONPATH"]


def test_smoke_frames_accepts_exactly_two_increasing_ids():
    args = runner.make_parser().parse_args(
        ["--smoke-frames", "1", "2"])
    assert runner.resolve_clips(args) == ((1, 2),)
    for values in (("1",), ("2", "1"), ("1", "2", "3")):
        args = runner.make_parser().parse_args(
            ["--smoke-frames"] + list(values))
        with pytest.raises(ValueError, match="two increasing"):
            runner.resolve_clips(args)


def test_main_leaves_metadata_incomplete_when_worker_fails(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner, "run_pilot",
        lambda args, metadata: (_ for _ in ()).throw(
            RuntimeError("worker failed")))
    with pytest.raises(RuntimeError, match="worker failed"):
        runner.main(["--output-dir", str(tmp_path)])
    metadata = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["complete"] is False
```

- [ ] **Step 2: Run the focused tests and verify failures**

Run:

```bash
python -m pytest -q \
  tests/test_run_nlspn_temporal_residual_validation.py::test_worker_command_uses_completionformer_environment \
  tests/test_run_nlspn_temporal_residual_validation.py::test_main_leaves_metadata_incomplete_when_worker_fails
```

Expected: fail because command construction and `main` are incomplete.

- [ ] **Step 3: Implement the CLI and digest-checked stage orchestration**

The parser must expose these exact arguments and defaults:

```python
def make_parser():
    parser = argparse.ArgumentParser(
        description="Validate causal NLSPN temporal residual mapping")
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument(
        "--checkpoint",
        default=("/workspace/CSPN/cspn_pytorch/output/"
                 "nyu_converged_baselines/nlspn_iter18/best.pt"))
    parser.add_argument(
        "--args-json",
        default=("/workspace/CSPN/cspn_pytorch/output/"
                 "nyu_converged_baselines/nlspn_iter18/args.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raft-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        default=("/workspace/VoxelNet/nlspn_temporal_residual_validation/"
                 "BeachApartmentInterior_My_ir/pilot_256"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke-frames", type=int, nargs="+", default=None,
        help=argparse.SUPPRESS)
    return parser
```

Implement `resolve_clips(args)` to return `PILOT_CLIPS` when
`args.smoke_frames is None`; otherwise require exactly two strictly increasing
IDs within 1--2000 and return one inclusive two-frame clip.

Implement `build_worker_command` with `conda run -n completionformer-py37`,
the repository root and both NLSPN source paths in `PYTHONPATH`, and explicit
stage/input/output/checkpoint/args/device arguments. Implement cache reuse only
when frame IDs, input digest, checkpoint digest, stage name, and expected array
shapes all match. A cache mismatch reruns that stage; it is never accepted with
a warning.

`main` writes initial metadata before preprocessing, records every subprocess
command and log path, and leaves `complete=false` on all exceptions. Worker
stdout and stderr go to per-clip stage logs. A nonzero worker exit raises an
error naming the log. Call `finalize_metadata` only after all eight clips,
248 pairs, CSV rows, plots, summaries, and compressed arrays validate.

- [ ] **Step 4: Run orchestration tests and the complete fast suite**

Run:

```bash
python -m pytest -q \
  tests/test_nlspn_temporal_residual.py \
  tests/test_run_nlspn_temporal_residual_worker.py \
  tests/test_run_nlspn_temporal_residual_validation.py
```

Expected: all new tests pass without performing real GPU inference.

- [ ] **Step 5: Commit orchestration**

```bash
git add scripts/run_nlspn_temporal_residual_validation.py \
  tests/test_run_nlspn_temporal_residual_validation.py
git commit -m "feat: orchestrate NLSPN temporal residual pilot"
```

### Task 9: Verify the real two-frame path, run the pilot, and record the result

**Files:**
- Modify: `scripts/run_nlspn_temporal_residual_validation.py` only if the smoke test exposes a defect
- Modify: the matching test file before any defect fix
- Create: `docs/2026-08-17-nlspn-temporal-residual-pilot-results.md`

- [ ] **Step 1: Run the full repository test suite before GPU validation**

Run:

```bash
python -m pytest -q
git diff --check
```

Expected: every test passes and `git diff --check` prints nothing.

- [ ] **Step 2: Resolve and hash the official RAFT-Small weight**

Run:

```bash
python -c "from torchvision.models.optical_flow import Raft_Small_Weights, raft_small; raft_small(weights=Raft_Small_Weights.DEFAULT); print(Raft_Small_Weights.DEFAULT.url)"
```

Expected: the official `raft_small_C_T_V2-01064c6d.pth` weight loads. Record
its local path and SHA-256 in run metadata. If loading fails, stop and report
the explicit error; do not continue with random weights.

- [ ] **Step 3: Run a real two-frame smoke test**

Run:

```bash
python scripts/run_nlspn_temporal_residual_validation.py \
  --output-dir /workspace/VoxelNet/nlspn_temporal_residual_validation/smoke_0001_0002 \
  --device cuda:0 \
  --smoke-frames 1 2 \
  --force
```

Expected: complete metadata, one pair, finite full/oracle/causal predictions,
500 sparse points in each frame, and both required plots.

- [ ] **Step 4: Run the approved 256-frame pilot**

Run:

```bash
python scripts/run_nlspn_temporal_residual_validation.py \
  --output-dir /workspace/VoxelNet/nlspn_temporal_residual_validation/BeachApartmentInterior_My_ir/pilot_256 \
  --device cuda:0 \
  --raft-batch-size 4
```

Expected: `run_metadata.json` has `complete=true`, exactly 256 frames and 248
pairs are recorded, all required artifacts are non-empty, and `summary.json`
contains finite oracle and causal quality ratios plus an interpretation branch.

- [ ] **Step 5: Inspect visualizations and independently recompute the gate**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/workspace/VoxelNet/nlspn_temporal_residual_validation/BeachApartmentInterior_My_ir/pilot_256')
metadata = json.loads((root / 'run_metadata.json').read_text())
summary = json.loads((root / 'summary.json').read_text())
assert metadata['complete'] is True
assert metadata['frame_count'] == 256
assert metadata['pair_count'] == 248
assert summary['causal']['quality_ratio'] > 0
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

Open `residual_mapping_overview.png` and `quality_ratio_by_frame.png` with the
local image viewer. Confirm flow direction visually, ensure borders are not
silently masked from GT metrics, and compare the printed causal ratio to the
CSV-pooled squared errors with an independent short calculation.

- [ ] **Step 6: Write the result report without overstating the outcome**

Create `docs/2026-08-17-nlspn-temporal-residual-pilot-results.md` containing:

```markdown
# NLSPN Temporal Residual Pilot Results

## Configuration

- Dataset and exact frame ranges
- NLSPN and RAFT checkpoint digests
- Input geometry and sparse-point policy
- Hardware and synchronized runtime components

## Residual Mapping

- Raw versus aligned residual statistics
- Input/output residual correlations
- Representative visual observations

## Existing Propagation Reconstruction

- Full baseline, current-guidance oracle, and warped-history causal RMSE
- Oracle and causal quality ratios
- Per-clip and worst-frame behavior

## Decision

- The exact pass/fail branch from the approved design
- Whether a separate codec implementation design is justified
- No speedup claim unless the causal quality gate passes
```

Replace every bullet with measured values from the complete artifacts. Do not
describe a failed gate as promising or average frame RMSE ratios instead of
pooling pixel errors.

- [ ] **Step 7: Run final verification and commit the measured report**

Run:

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: all tests pass, no whitespace errors, and only the measured report
or a test-backed smoke defect fix remains uncommitted.

Commit:

```bash
git add docs/2026-08-17-nlspn-temporal-residual-pilot-results.md
git add scripts/run_nlspn_temporal_residual_validation.py \
  scripts/run_nlspn_temporal_residual_worker.py \
  scripts/nlspn_temporal_residual.py \
  tests/test_run_nlspn_temporal_residual_validation.py \
  tests/test_run_nlspn_temporal_residual_worker.py \
  tests/test_nlspn_temporal_residual.py
git commit -m "docs: report NLSPN temporal residual pilot"
```

The generated pilot arrays, CSV files, JSON files, and PNG files remain under
`/workspace/VoxelNet/nlspn_temporal_residual_validation/` and are not added to
Git.
