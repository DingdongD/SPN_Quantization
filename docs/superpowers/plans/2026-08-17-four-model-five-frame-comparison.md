# Four-Model Five-Frame Depth-Completion Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run CSPN, DySPN, NLSPN, and CompletionFormer on the same five canonical `304 x 228` RGB/sparse-depth inputs and export validated per-frame, temporal, metric, and visual comparisons.

**Architecture:** A Python-3.7-compatible IO contract module validates the existing CSPN NPZ files and all worker results. A model worker runs one external model inside its native Conda/PyTorch environment, while a neutral orchestrator launches workers, reuses the existing CSPN predictions, computes common metrics, and renders all comparison artifacts.

**Tech Stack:** Python, NumPy, PyTorch, Conda subprocesses, CUDA, Matplotlib, CSV/JSON, pytest.

---

## File map

- Create `scripts/spn_sequence_io.py`: canonical input/result schemas, content digests, atomic NPZ writing, and common frame/temporal metrics.
- Create `scripts/run_spn_sequence_worker.py`: strict DySPN/NLSPN/CompletionFormer construction, checkpoint loading, and five-frame inference.
- Create `scripts/compare_spn_sequence_models.py`: worker command orchestration, result caching/validation, CSPN reuse, CSV/metadata writing, and plots.
- Create `tests/test_spn_sequence_io.py`: input/result contract and metric tests.
- Create `tests/test_run_spn_sequence_worker.py`: model-configuration and inference-adapter tests without loading real networks.
- Create `tests/test_compare_spn_sequence_models.py`: subprocess-command, aggregation, and artifact-manifest tests.
- Generate artifacts only under `/workspace/VoxelNet/spn_model_comparison/`; do not commit generated inference output.

### Task 1: Canonical input and result contracts

**Files:**
- Create: `scripts/spn_sequence_io.py`
- Create: `tests/test_spn_sequence_io.py`

- [ ] **Step 1: Write failing canonical-input tests**

Create `tests/test_spn_sequence_io.py` with helpers that write five synthetic
`frame_0001.npz` files and assertions for the exact contract:

```python
from pathlib import Path
import tempfile

import numpy as np
import pytest

from scripts import spn_sequence_io as sequence_io


def write_canonical(root, sparse_masks=None):
    root = Path(root)
    root.mkdir(parents=True)
    base_mask = np.zeros((228, 304), dtype=bool)
    base_mask.flat[:500] = True
    for index, frame_id in enumerate(range(1, 6)):
        mask = base_mask if sparse_masks is None else sparse_masks[index]
        gt = np.full((228, 304), 1.0 + index, dtype=np.float32)
        sparse = np.where(mask, gt, 0.0).astype(np.float32)
        np.savez_compressed(
            root / ("frame_%04d.npz" % frame_id),
            frame_id=np.asarray(frame_id),
            rgb=np.full((3, 228, 304), index / 5.0, dtype=np.float32),
            sparse=sparse,
            gt=gt,
            pred_raw=gt + 0.2,
            pred_clamped=gt + 0.2,
            valid=np.ones((228, 304), dtype=bool),
            abs_err=np.full((228, 304), 0.2, dtype=np.float32),
        )


def test_load_canonical_frames_stacks_exact_five_frame_contract(tmp_path):
    write_canonical(tmp_path)
    data = sequence_io.load_canonical_frames(tmp_path, range(1, 6))
    assert data["frame_ids"].tolist() == [1, 2, 3, 4, 5]
    assert data["rgb"].shape == (5, 3, 228, 304)
    assert data["sparse"].shape == (5, 228, 304)
    assert data["gt"].shape == (5, 228, 304)
    assert data["valid"].shape == (5, 228, 304)
    assert data["cspn_pred_raw"].shape == (5, 228, 304)
    assert all(np.count_nonzero(x) == 500 for x in data["sparse"])
    assert len(data["input_digest"]) == 64


def test_load_canonical_frames_rejects_changed_sparse_coordinates(tmp_path):
    masks = []
    for index in range(5):
        mask = np.zeros((228, 304), dtype=bool)
        mask.flat[index:index + 500] = True
        masks.append(mask)
    write_canonical(tmp_path, masks)
    with pytest.raises(ValueError, match="shared sparse coordinates"):
        sequence_io.load_canonical_frames(tmp_path, range(1, 6))
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run:

```bash
python -m pytest -q tests/test_spn_sequence_io.py -k canonical
```

Expected: collection fails because `scripts.spn_sequence_io` does not exist.

- [ ] **Step 3: Implement canonical loading and stable input hashing**

Create `scripts/spn_sequence_io.py` with constants and these public functions:

```python
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path

import numpy as np


HEIGHT = 228
WIDTH = 304
SPARSE_COUNT = 500
MAX_DEPTH = 10.0
FRAME_IDS = tuple(range(1, 6))


def _update_array_digest(digest, name, value):
    value = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))


def canonical_input_digest(frame_ids, rgb, sparse, gt, valid):
    digest = hashlib.sha256()
    for name, value in (
            ("frame_ids", np.asarray(frame_ids, dtype=np.int64)),
            ("rgb", rgb), ("sparse", sparse), ("gt", gt),
            ("valid", np.asarray(valid, dtype=np.uint8))):
        _update_array_digest(digest, name, value)
    return digest.hexdigest()


def load_canonical_frames(root, frame_ids=FRAME_IDS):
    rows = []
    for frame_id in frame_ids:
        path = Path(root) / ("frame_%04d.npz" % int(frame_id))
        if not path.is_file():
            raise FileNotFoundError(str(path))
        with np.load(str(path), allow_pickle=False) as item:
            required = {"frame_id", "rgb", "sparse", "gt", "valid",
                        "pred_raw", "pred_clamped"}
            missing = sorted(required - set(item.files))
            if missing:
                raise ValueError("canonical NPZ missing keys: %s" % missing)
            row = {key: np.asarray(item[key]).copy() for key in required}
        if int(row["frame_id"]) != int(frame_id):
            raise ValueError("canonical frame ID mismatch")
        rows.append(row)
    data = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "rgb": np.stack([x["rgb"] for x in rows]).astype(np.float32),
        "sparse": np.stack([x["sparse"] for x in rows]).astype(np.float32),
        "gt": np.stack([x["gt"] for x in rows]).astype(np.float32),
        "valid": np.stack([x["valid"] for x in rows]).astype(bool),
        "cspn_pred_raw": np.stack([x["pred_raw"] for x in rows]).astype(np.float32),
        "cspn_pred_clamped": np.stack([x["pred_clamped"] for x in rows]).astype(np.float32),
    }
    if data["rgb"].shape != (5, 3, HEIGHT, WIDTH):
        raise ValueError("canonical RGB shape must be 5x3x228x304")
    for key in ("sparse", "gt", "valid", "cspn_pred_raw", "cspn_pred_clamped"):
        if data[key].shape != (5, HEIGHT, WIDTH):
            raise ValueError("canonical %s shape must be 5x228x304" % key)
    masks = data["sparse"] > 0.0
    if any(int(mask.sum()) != SPARSE_COUNT for mask in masks):
        raise ValueError("each frame must contain exactly 500 sparse values")
    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise ValueError("frames do not share sparse coordinates")
    if not np.isfinite(data["rgb"]).all():
        raise ValueError("canonical RGB contains non-finite values")
    if not np.isfinite(data["sparse"]).all():
        raise ValueError("canonical sparse depth contains non-finite values")
    if not np.isfinite(data["gt"][data["valid"]]).all():
        raise ValueError("canonical valid ground truth contains non-finite values")
    data["input_digest"] = canonical_input_digest(
        data["frame_ids"], data["rgb"], data["sparse"], data["gt"], data["valid"])
    return data
```

- [ ] **Step 4: Add failing result-schema, digest, and metric tests**

Append tests that exercise `file_sha256`, atomic worker result IO, frame metrics,
and unregistered temporal residuals:

```python
def test_worker_result_round_trip_and_digest_validation(tmp_path):
    write_canonical(tmp_path / "canonical")
    data = sequence_io.load_canonical_frames(tmp_path / "canonical")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "predictions.npz"
    pred = np.full((5, 228, 304), 2.0, dtype=np.float32)
    sequence_io.write_worker_result(
        output, "dyspn", data["frame_ids"], pred, data["input_digest"],
        sequence_io.file_sha256(checkpoint), {"iteration": 6}, 1.25)
    result = sequence_io.load_worker_result(
        output, "dyspn", data["frame_ids"], data["input_digest"],
        sequence_io.file_sha256(checkpoint))
    np.testing.assert_array_equal(result["pred_raw"], pred)
    assert result["metadata"]["iteration"] == 6


def test_frame_and_temporal_metrics_use_valid_pixels():
    gt = np.asarray([[[1.0, 2.0]], [[2.0, 4.0]]], dtype=np.float32)
    pred = np.asarray([[[1.0, 2.0]], [[3.0, 5.0]]], dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    frame = sequence_io.frame_metrics(gt[1], pred[1], valid[1])
    assert frame["rmse"] == pytest.approx(1.0)
    temporal, maps = sequence_io.temporal_metrics(gt, pred, valid, [1, 2])
    assert temporal[0]["pair"] == "0001->0002"
    assert temporal[0]["rmse"] == pytest.approx(1.0)
    np.testing.assert_array_equal(maps[0]["residual"], [[1.0, 1.0]])
```

- [ ] **Step 5: Implement result IO and metrics**

Add `file_sha256(path)`, `write_worker_result(...)`,
`load_worker_result(...)`, `frame_metrics(...)`, and
`temporal_metrics(...)`. `write_worker_result` must clamp a separate copy to
`[1e-6, 10]`, serialize JSON metadata as a Unicode NumPy scalar, write through
an opened `*.tmp` file, then call `os.replace`. `load_worker_result` must reject
model/frame/digest mismatches, non-finite values, or any shape other than
`5 x 228 x 304`. Metrics use float64 reductions and valid ground-truth masks.

- [ ] **Step 6: Run Task 1 tests and commit**

Run:

```bash
python -m pytest -q tests/test_spn_sequence_io.py
```

Expected: all tests pass.

Commit:

```bash
git add scripts/spn_sequence_io.py tests/test_spn_sequence_io.py
git commit -m "feat: add SPN sequence IO contracts"
```

### Task 2: External-model inference worker

**Files:**
- Create: `scripts/run_spn_sequence_worker.py`
- Create: `tests/test_run_spn_sequence_worker.py`

- [ ] **Step 1: Write failing configuration and adapter tests**

Create `tests/test_run_spn_sequence_worker.py`:

```python
import argparse
import json

import numpy as np
import torch

from scripts import run_spn_sequence_worker as worker


def test_model_namespace_matches_converged_baseline_arguments():
    args = {
        "iteration": 18, "from_scratch": False, "lr": 0.001,
        "nlspn_network": "resnet34",
        "completionformer_model": "CompletionFormer",
    }
    nlspn = worker.nlspn_namespace(args)
    assert (nlspn.network, nlspn.prop_time, nlspn.prop_kernel) == ("resnet34", 18, 3)
    assert (nlspn.conf_prop, nlspn.affinity, nlspn.affinity_gamma) == (True, "TGASS", 0.5)
    assert nlspn.preserve_input is False
    completionformer = worker.completionformer_namespace(args)
    assert completionformer.model == "CompletionFormer"
    assert completionformer.max_depth == 10.0


class DySPNToy(torch.nn.Module):
    def forward(self, rgb, dep):
        return dep + rgb[:, :1]


class DictToy(torch.nn.Module):
    def forward(self, sample):
        return {"pred": sample["dep"] + sample["rgb"][:, :1]}


def test_predict_frames_adapts_dyspn_and_dict_model_signatures():
    rgb = np.ones((2, 3, 4, 6), dtype=np.float32)
    dep = np.full((2, 4, 6), 2.0, dtype=np.float32)
    for model_name, model in (("dyspn", DySPNToy()), ("nlspn", DictToy()),
                              ("completionformer", DictToy())):
        pred = worker.predict_frames(model_name, model, rgb, dep, torch.device("cpu"))
        assert pred.shape == (2, 4, 6)
        np.testing.assert_allclose(pred, 3.0)
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
python -m pytest -q tests/test_run_spn_sequence_worker.py
```

Expected: collection fails because `scripts.run_spn_sequence_worker` does not
exist.

- [ ] **Step 3: Implement builders and prediction adapters**

Create a Python-3.7-compatible worker with:

```python
def nlspn_namespace(args):
    return argparse.Namespace(
        network=args["nlspn_network"], from_scratch=args["from_scratch"],
        prop_time=args["iteration"], prop_kernel=3, conf_prop=True,
        affinity="TGASS", affinity_gamma=0.5, preserve_input=False,
        legacy=False, lr=args["lr"])


def completionformer_namespace(args):
    return argparse.Namespace(
        model=args["completionformer_model"],
        from_scratch=args["from_scratch"], prop_time=args["iteration"],
        prop_kernel=3, conf_prop=True, affinity="TGASS",
        affinity_gamma=0.5, preserve_input=False, legacy=False,
        max_depth=10.0)


def predict_frames(model_name, model, rgb, sparse, device):
    predictions = []
    with torch.no_grad():
        for index in range(rgb.shape[0]):
            rgb_tensor = torch.from_numpy(rgb[index:index + 1]).to(device)
            dep_tensor = torch.from_numpy(sparse[index:index + 1, None]).to(device)
            if model_name == "dyspn":
                output = model(rgb_tensor, dep_tensor)
            else:
                output = model({"rgb": rgb_tensor, "dep": dep_tensor})["pred"]
            predictions.append(output.detach().cpu().numpy()[0, 0])
    result = np.stack(predictions).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("model prediction contains non-finite values")
    return result
```

Implement `build_model(model_name, args, device)` using exactly the constructor
arguments from `train_nyu_iteration_sweep.py`: DySPN `Model(... mode="dyspn",
norm_depth=[0.0, 10.0])`, NLSPN `NLSPNModel(nlspn_namespace(args))`, and
CompletionFormer `CompletionFormer(completionformer_namespace(args))`. External
repository paths are received by CLI and inserted into `sys.path`; NLSPN and
CompletionFormer construction temporarily changes cwd to their `src` directory.

- [ ] **Step 4: Add failing strict-checkpoint and CLI contract tests**

Add a toy strict-load test and parser test:

```python
def test_load_checkpoint_rejects_partial_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "best.pt"
    torch.save({"net": {"weight": model.weight.detach().clone()}}, path)
    try:
        worker.load_checkpoint_strict(model, path)
    except RuntimeError as error:
        assert "bias" in str(error)
    else:
        raise AssertionError("partial checkpoint was accepted")


def test_parser_requires_external_model_and_paths():
    parser = worker.make_parser()
    parsed = parser.parse_args([
        "--model", "dyspn", "--canonical-dir", "/input",
        "--checkpoint", "/weights/best.pt", "--args-json", "/weights/args.json",
        "--output", "/output/predictions.npz", "--device", "cuda:0",
    ])
    assert parsed.model == "dyspn"
```

- [ ] **Step 5: Implement strict loading and worker CLI**

Implement `load_checkpoint_strict(model, path)` as `torch.load(...,
map_location="cpu")`, require a dictionary-valued `net`, and call
`model.load_state_dict(checkpoint["net"], strict=True)`. Implement `main()` to:

1. load and validate canonical frames with `spn_sequence_io`;
2. parse `args.json` and require its `model` matches `--model`;
3. construct the requested network and strictly load `best.pt`;
4. move to the requested CUDA device and set evaluation mode;
5. time five-frame inference;
6. call `write_worker_result` with model/checkpoint/input digests and architecture
   metadata;
7. print one JSON completion record to stdout.

- [ ] **Step 6: Run Task 2 tests and commit**

Run:

```bash
python -m pytest -q tests/test_run_spn_sequence_worker.py
```

Expected: all tests pass.

Commit:

```bash
git add scripts/run_spn_sequence_worker.py tests/test_run_spn_sequence_worker.py
git commit -m "feat: add external SPN inference worker"
```

### Task 3: Cross-environment orchestration and result reuse

**Files:**
- Create: `scripts/compare_spn_sequence_models.py`
- Create: `tests/test_compare_spn_sequence_models.py`

- [ ] **Step 1: Write failing worker-command and cache tests**

Create `tests/test_compare_spn_sequence_models.py`:

```python
from pathlib import Path

from scripts import compare_spn_sequence_models as comparison


def test_worker_specs_use_native_conda_environments():
    specs = comparison.default_worker_specs(Path("/repo/scripts/run_spn_sequence_worker.py"))
    assert specs["dyspn"]["environment"] == "pointkan"
    assert specs["nlspn"]["environment"] == "completionformer-py37"
    assert specs["completionformer"]["environment"] == "completionformer-py37"
    assert "/workspace/external_depth_completion_models/DySPN" in specs["dyspn"]["pythonpath"]
    assert "/workspace/CompletionFormer/src/model/deformconv" in specs["completionformer"]["pythonpath"]


def test_build_worker_command_is_explicit_and_deterministic(tmp_path):
    spec = comparison.default_worker_specs(Path("/repo/worker.py"))["dyspn"]
    command, env = comparison.build_worker_command(
        "dyspn", spec, Path("/canonical"), tmp_path / "predictions.npz", "cuda:0")
    assert command[:5] == ["conda", "run", "-n", "pointkan", "python"]
    assert command[-2:] == ["--device", "cuda:0"]
    assert env["PYTHONPATH"].startswith(spec["pythonpath"])
```

- [ ] **Step 2: Run focused tests and verify import failure**

Run:

```bash
python -m pytest -q tests/test_compare_spn_sequence_models.py -k 'worker or command'
```

Expected: collection fails because `scripts.compare_spn_sequence_models` does
not exist.

- [ ] **Step 3: Implement worker specifications, command construction, and logs**

Create `default_worker_specs(worker_path)` with exact checkpoint, `args.json`,
Conda environment, and `PYTHONPATH` values from the design. Implement
`build_worker_command(...)` as a list beginning with `conda run -n ENV python
WORKER` and all explicit CLI arguments. Implement `run_or_reuse_worker(...)`:

- validate an existing `predictions.npz` against current canonical-input and
  checkpoint digests and reuse only when it passes;
- otherwise invoke `subprocess.run(..., check=False, capture_output=True,
  text=True, env=env)`;
- atomically write stdout/stderr to `<model>/worker.log`;
- raise `RuntimeError` including the exit code and log path on failure;
- load and validate the newly written result before returning it.

- [ ] **Step 4: Add failing aggregation and expected-artifact tests**

Append:

```python
def test_expected_artifacts_lists_twenty_panels_and_combined_outputs(tmp_path):
    paths = comparison.expected_artifacts(tmp_path, range(1, 6))
    panels = [path for path in paths if path.name.startswith("frame_") and path.suffix == ".png"]
    frame_npz = [path for path in paths if path.name.startswith("frame_") and path.suffix == ".npz"]
    worker_logs = [path for path in paths if path.name == "worker.log"]
    worker_results = [path for path in paths if path.name == "predictions.npz"]
    assert len(panels) == 20
    assert len(frame_npz) == 20
    assert len(worker_logs) == 3
    assert len(worker_results) == 3
    assert tmp_path / "four_model_depth_comparison.png" in paths
    assert tmp_path / "four_model_error_comparison.png" in paths
    assert tmp_path / "four_model_temporal_comparison_unregistered.png" in paths
    assert tmp_path / "four_model_frame_metrics.csv" in paths
    assert tmp_path / "four_model_temporal_metrics.csv" in paths
    assert tmp_path / "run_metadata.json" in paths


def test_collect_metrics_has_twenty_frame_and_sixteen_temporal_rows():
    import numpy as np
    frame_ids = np.arange(1, 6)
    gt = np.ones((5, 2, 3), dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    predictions = {name: gt.copy() for name in comparison.MODEL_ORDER}
    frame_rows, temporal_rows, temporal_maps = comparison.collect_metrics(
        frame_ids, gt, valid, predictions)
    assert len(frame_rows) == 20
    assert len(temporal_rows) == 16
    assert set(temporal_maps) == set(comparison.MODEL_ORDER)
```

- [ ] **Step 5: Implement metric aggregation and CLI skeleton**

Define `MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")`.
Implement `collect_metrics(...)` by calling the Task 1 common metrics for every
model/frame and every adjacent pair, attaching `model`, `frame_id`, or `pair`
to each row. Implement argument parsing with defaults for canonical input,
output directory, device, and `--force`. The CLI loads canonical data, validates
the existing CSPN prediction, runs/reuses the three workers, and holds all four
prediction arrays for rendering.

- [ ] **Step 6: Run Task 3 tests and commit**

Run:

```bash
python -m pytest -q tests/test_compare_spn_sequence_models.py
```

Expected: all current tests pass.

Commit:

```bash
git add scripts/compare_spn_sequence_models.py tests/test_compare_spn_sequence_models.py
git commit -m "feat: orchestrate four SPN sequence models"
```

### Task 4: Common visualizations, CSVs, and completion manifest

**Files:**
- Modify: `scripts/compare_spn_sequence_models.py`
- Modify: `tests/test_compare_spn_sequence_models.py`

- [ ] **Step 1: Write failing synthetic rendering test**

Add a test using `matplotlib`'s noninteractive backend:

```python
def test_write_artifacts_creates_complete_manifest(tmp_path):
    import numpy as np
    for name in comparison.EXTERNAL_MODELS:
        model_dir = tmp_path / name
        model_dir.mkdir(parents=True)
        (model_dir / "worker.log").write_text("ok\n", encoding="utf-8")
        (model_dir / "predictions.npz").write_bytes(b"validated-worker-result")
    frame_ids = np.arange(1, 6)
    rgb = np.zeros((5, 3, 228, 304), dtype=np.float32)
    sparse = np.zeros((5, 228, 304), dtype=np.float32)
    sparse.reshape(5, -1)[:, :500] = 1.0
    gt = np.full((5, 228, 304), 2.0, dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    predictions = {
        name: gt + 0.1 * index
        for index, name in enumerate(comparison.MODEL_ORDER)
    }
    metadata = {"input_geometry": [228, 304], "temporal_alignment": "unregistered"}
    comparison.write_all_artifacts(
        tmp_path, frame_ids, rgb, sparse, gt, valid, predictions, metadata)
    missing = [path for path in comparison.expected_artifacts(tmp_path, frame_ids)
               if not path.is_file() or path.stat().st_size == 0]
    assert missing == []
```

- [ ] **Step 2: Run rendering test and verify failure**

Run:

```bash
python -m pytest -q tests/test_compare_spn_sequence_models.py::test_write_artifacts_creates_complete_manifest
```

Expected: failure reports missing `write_all_artifacts`.

- [ ] **Step 3: Implement model/frame artifacts and CSV output**

For each model/frame, save `<model>/frame_XXXX.npz` containing frame ID, RGB,
sparse, GT, validity, raw/clamped prediction, and absolute error, then render a
five-column `<model>/frame_XXXX.png`. Write frame CSV fields
`model,frame_id,rmse,mae,abs_rel,valid_pixels,valid_coverage,sparse_count` and
temporal CSV fields
`model,pair,from_frame,to_frame,rmse,mae,valid_pixels,valid_coverage,alignment`.

- [ ] **Step 4: Implement the three combined figures and metadata**

Render shared-scale figures with these exact layouts:

- depth comparison: five frame rows; columns GT, CSPN, DySPN, NLSPN,
  CompletionFormer; common `viridis` range `0..10 m`;
- error comparison: five frame rows; four model columns; common `magma` range
  from zero to the 99th percentile of all valid absolute errors;
- temporal comparison: four adjacent-pair rows; four model columns; symmetric
  `coolwarm` range from the 99th percentile of absolute valid residuals; title
  and footer include `UNREGISTERED IMAGE-SPACE TEMPORAL RESIDUAL`.

Write `run_metadata.json` atomically with model configuration/checkpoint/input
digests, source and network geometry, sparse count, worker runtimes/logs,
`temporal_alignment: "unregistered"`, and a final `complete: true` only after
every path returned by `expected_artifacts` exists and is non-empty.

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
python -m pytest -q tests/test_spn_sequence_io.py tests/test_run_spn_sequence_worker.py tests/test_compare_spn_sequence_models.py
python -m pytest -q
```

Expected: focused tests pass and the full suite remains at least `208 passed`
plus the new tests, with zero failures.

- [ ] **Step 6: Commit Task 4**

```bash
git add scripts/compare_spn_sequence_models.py tests/test_compare_spn_sequence_models.py
git commit -m "feat: render four-model depth comparisons"
```

### Task 5: Real five-frame inference and verification

**Files:**
- Modify only if a real-run defect is proven: the smallest relevant script/test
- Generate: `/workspace/VoxelNet/spn_model_comparison/BeachApartmentInterior_My_ir/frames_0001_0005/`

- [ ] **Step 1: Run preflight contract and runtime checks**

Run:

```bash
python -c "from scripts.spn_sequence_io import load_canonical_frames; d=load_canonical_frames('/workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005'); print(d['rgb'].shape, d['sparse'].shape, d['input_digest'])"
PYTHONPATH=/workspace/external_depth_completion_models/DySPN conda run -n pointkan python -c "import torch; from DySPN.base import Model; print(torch.__version__, torch.cuda.is_available(), Model.__name__)"
PYTHONPATH=/workspace/external_depth_completion_models/NLSPN_ECCV20/src:/workspace/external_depth_completion_models/NLSPN_ECCV20/src/model/deformconv conda run -n completionformer-py37 python -c "import torch; from model.nlspnmodel import NLSPNModel; from functions.modulated_deform_conv_func import ModulatedDeformConvFunction; print(torch.__version__, torch.cuda.is_available(), NLSPNModel.__name__)"
PYTHONPATH=/workspace/CompletionFormer/src:/workspace/CompletionFormer/src/model/deformconv conda run -n completionformer-py37 python -c "import torch; from model.completionformer import CompletionFormer; from functions.modulated_deform_conv_func import ModulatedDeformConvFunction; print(torch.__version__, torch.cuda.is_available(), CompletionFormer.__name__)"
```

Expected: canonical shapes are `(5, 3, 228, 304)` and `(5, 228, 304)`; all
three runtimes report CUDA available and import their model/DCN classes.

- [ ] **Step 2: Run all three external models and render the comparison**

Run:

```bash
python scripts/compare_spn_sequence_models.py \
  --canonical-dir /workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005 \
  --output-dir /workspace/VoxelNet/spn_model_comparison/BeachApartmentInterior_My_ir/frames_0001_0005 \
  --device cuda:0
```

Expected: DySPN, NLSPN, and CompletionFormer workers each complete or validate
a digest-matching cached result; the orchestrator reports `complete: true`.

- [ ] **Step 3: Validate artifact counts, metadata, and finite metrics**

Run a Python validation command that loads `run_metadata.json`, both CSV files,
and every model/frame NPZ; assert:

```python
assert metadata["complete"] is True
assert metadata["input_geometry"] == [228, 304]
assert metadata["source_geometry"] == [480, 640]
assert metadata["sparse_count"] == 500
assert metadata["temporal_alignment"] == "unregistered"
assert len(frame_rows) == 20
assert len(temporal_rows) == 16
assert np.isfinite([float(row["rmse"]) for row in frame_rows]).all()
assert np.isfinite([float(row["rmse"]) for row in temporal_rows]).all()
```

Expected: all assertions pass, all 20 frame panels and three combined figures
exist, and no output NPZ contains non-finite predictions.

- [ ] **Step 4: Visually inspect all three combined figures**

Open the depth, error, and temporal figures at original detail. Check labels,
common colour scales, crop orientation, sparse overlay, and obvious metre-scale
or tensor-layout errors. If a defect is found, first add a failing automated
test that reproduces it, then make the smallest correction and rerun Tasks 4-5.

- [ ] **Step 5: Run final verification and commit any proven correction**

Run:

```bash
python -m pytest -q
git status --short
```

Expected: all tests pass; only intentional plan/code/test changes are tracked,
and generated comparison artifacts remain outside the repository.

If Task 5 required no code correction, do not create an empty commit. Otherwise
commit only the corrected scripts/tests with:

```bash
git add scripts tests
git commit -m "fix: validate real four-model sequence inference"
```
