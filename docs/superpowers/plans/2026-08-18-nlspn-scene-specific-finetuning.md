# NLSPN Scene-Specific Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune a separate, inference-compatible NLSPN checkpoint that lowers absolute Full NLSPN RMSE on scene-disjoint `room3` and `room7` test data without changing or overwriting the generic model.

**Architecture:** A current-environment launcher creates immutable scene manifests, then starts one Python 3.7 GPU worker that strictly loads the generic NLSPN checkpoint, performs two-stage fine-tuning, selects by `room6` validation RMSE, and evaluates both checkpoints once on the held-out test set. Focused data/training modules make preprocessing, sparse sampling, parameter policies, metrics, and checkpoint contracts independently testable; current-environment artifact code recomputes results, renders comparisons, validates staging, and publishes atomically.

**Tech Stack:** Python 3.7/3.11, PyTorch 1.10.1, official NLSPN ResNet-34 and deformable convolution, OpenCV EXR, Pillow, torchvision, NumPy, Matplotlib Agg, CSV/JSON/NPZ, pytest, CUDA A100.

---

## File structure

- Create `scripts/nlspn_scene_finetune_data.py` for manifests, JPG/EXR preprocessing, augmentation, and sparse sampling.
- Create `scripts/nlspn_scene_finetune_core.py` for phase policies, loss, metrics, convergence, checkpoints, and memory probing.
- Create `scripts/run_nlspn_scene_finetune_worker.py` for legacy-environment training and paired held-out evaluation.
- Create `scripts/nlspn_scene_finetune_artifacts.py` for independent aggregation, plots, reports, and final-tree validation.
- Create `scripts/run_nlspn_scene_finetune.py` for preflight, worker launch, staging, validation, and atomic publication.
- Create one matching test module for each production module.

### Task 1: Build exact scene-disjoint manifests

**Files:**
- Create: `scripts/nlspn_scene_finetune_data.py`
- Create: `tests/test_nlspn_scene_finetune_data.py`

- [ ] **Step 1: Write failing split tests**

```python
@pytest.fixture
def fake_data_root(tmp_path):
    for scenes in data.SPLIT_SCENES.values():
        for scene in scenes:
            (tmp_path / scene / "rgb").mkdir(parents=True)
            (tmp_path / scene / "depth").mkdir()
            for frame_id in range(1, data.SCENE_FRAME_COUNTS[scene] + 1):
                (tmp_path / scene / "rgb" / f"{frame_id:04d}.jpg").touch()
                (tmp_path / scene / "depth" /
                 f"Image{frame_id:04d}.exr").touch()
    return tmp_path


def test_build_manifests_has_exact_scene_disjoint_geometry(fake_data_root):
    manifests = data.build_manifests(fake_data_root)
    assert tuple(manifests) == ("train", "val", "test")
    assert [len(manifests[name]) for name in manifests] == [10000, 2000, 8000]
    assert {row["scene"] for row in manifests["train"]} == {
        "BeachApartmentInterior_My_ir", "bedroom_ir",
        "livingroom_ir", "room4"}
    assert {row["scene"] for row in manifests["val"]} == {"room6"}
    assert {row["scene"] for row in manifests["test"]} == {"room3", "room7"}


def test_build_manifests_rejects_missing_pair(fake_data_root):
    path = fake_data_root / "room6/depth/Image0007.exr"
    path.unlink()
    with pytest.raises(ValueError, match="room6.*0007.*pair"):
        data.build_manifests(fake_data_root)
```

Add explicit tests rejecting extra IDs, duplicate identities, split overlap,
path traversal, noncanonical order, wrong row count, changed CSV fields, and
changed canonical digest.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_data.py -x
```

Expected: import failure because the data module does not exist.

- [ ] **Step 3: Implement manifest constants and validation**

```python
SPLIT_SCENES = {
    "train": ("BeachApartmentInterior_My_ir", "bedroom_ir",
              "livingroom_ir", "room4"),
    "val": ("room6",),
    "test": ("room3", "room7"),
}
SCENE_FRAME_COUNTS = {
    "BeachApartmentInterior_My_ir": 2000, "bedroom_ir": 2000,
    "livingroom_ir": 4000, "room3": 4000, "room4": 2000,
    "room6": 2000, "room7": 4000,
}
MANIFEST_FIELDS = ("split", "scene", "frame_id", "rgb_path", "depth_path")


def canonical_row(split, scene, frame_id):
    return {
        "split": split, "scene": scene, "frame_id": int(frame_id),
        "rgb_path": f"{scene}/rgb/{frame_id:04d}.jpg",
        "depth_path": f"{scene}/depth/Image{frame_id:04d}.exr",
    }
```

`build_manifests` must reconstruct all expected rows, require both files, and
call `validate_manifest` plus `validate_split_isolation`. Implement atomic CSV
read/write and `canonical_manifest_sha256` from compact canonical JSON. Loaded
paths are always reconstructed relative to the approved resolved data root.

- [ ] **Step 4: Verify and commit manifests**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_data.py
git diff --check
git add scripts/nlspn_scene_finetune_data.py tests/test_nlspn_scene_finetune_data.py
git commit -m "feat: define scene-disjoint NLSPN manifests"
```

### Task 2: Add reproducible JPG/EXR samples

**Files:**
- Modify: `scripts/nlspn_scene_finetune_data.py`
- Modify: `tests/test_nlspn_scene_finetune_data.py`

- [ ] **Step 1: Write failing preprocessing tests**

```python
def test_preprocess_matches_inference_geometry_and_rgb_scale():
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth = np.full((480, 640), 2.5, dtype=np.float32)
    rgb_out, gt, valid = data.preprocess_arrays(rgb, depth)
    assert rgb_out.shape == (3, 228, 304)
    assert gt.shape == valid.shape == (228, 304)
    assert rgb_out.min() == pytest.approx(128.0 / 255.0)
    assert rgb_out.max() == pytest.approx(128.0 / 255.0)
    assert np.all(gt == 2.5) and valid.all()


def test_validation_sparse_is_fixed_and_training_sparse_changes_by_epoch():
    valid = np.ones((228, 304), dtype=bool)
    gt = np.arange(valid.size, dtype=np.float32).reshape(valid.shape) + 1.0
    va = data.build_sparse(gt, valid, "val", "room6", 7, 0, 2026)
    vb = data.build_sparse(gt, valid, "val", "room6", 7, 9, 2026)
    ta = data.build_sparse(gt, valid, "train", "room4", 7, 0, 2026)
    tb = data.build_sparse(gt, valid, "train", "room4", 7, 1, 2026)
    assert np.array_equal(va, vb)
    assert not np.array_equal(ta, tb)
    assert all(np.count_nonzero(x) == 500 for x in (va, ta, tb))
```

Add tests for equal three-channel EXR, malformed/unequal EXR rejection,
sanitization of nonfinite/nonpositive/>10 m depth, fewer than 500 valid pixels,
deterministic flip/jitter, returned identities, and absence of ImageNet
normalization.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_data.py -x
```

Expected: failures for missing preprocessing and dataset functions.

- [ ] **Step 3: Implement preprocessing and sparse sampling**

Set `OPENCV_IO_ENABLE_OPENEXR=1` before importing OpenCV. Use bilinear RGB
resize, nearest depth/valid resize, height 240, center crop `(228,304)`, and
float32 RGB `[0,1]` with no mean/std normalization.

```python
def sample_seed(split, scene, frame_id, epoch, seed):
    effective_epoch = int(epoch) if split == "train" else 0
    payload = f"{seed}|{split}|{scene}|{frame_id}|{effective_epoch}"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def build_sparse(gt, valid, split, scene, frame_id, epoch, seed):
    candidates = np.flatnonzero(valid)
    if candidates.size < 500:
        raise ValueError("preprocessed sample has fewer than 500 valid pixels")
    rng = np.random.default_rng(sample_seed(
        split, scene, frame_id, epoch, seed))
    chosen = rng.choice(candidates, 500, replace=False)
    sparse = np.zeros(gt.size, dtype=np.float32)
    sparse[chosen] = gt.reshape(-1)[chosen]
    return sparse.reshape(gt.shape)
```

`SceneDepthDataset.set_epoch` controls training augmentation/masks. Return
`rgb`, `dep`, `gt`, `valid`, `scene`, and `frame_id` with channel-first tensors.
Apply train-only horizontal flip jointly to RGB/depth/valid and deterministic
RGB-only brightness, contrast, and saturation jitter.

- [ ] **Step 4: Verify both environments and commit**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_data.py
conda run -n completionformer-py37 python -m pytest -q tests/test_nlspn_scene_finetune_data.py
git diff --check
git add scripts/nlspn_scene_finetune_data.py tests/test_nlspn_scene_finetune_data.py
git commit -m "feat: load reproducible NLSPN scene samples"
```

### Task 3: Implement phase policies, objective, and metrics

**Files:**
- Create: `scripts/nlspn_scene_finetune_core.py`
- Create: `tests/test_nlspn_scene_finetune_core.py`

- [ ] **Step 1: Write failing parameter-policy tests**

```python
class ToyNLSPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("conv1_rgb", "conv1_dep", "conv2", "conv3", "conv4",
                     "conv5", "conv6", "dec5", "dec4", "dec3", "dec2",
                     "id_dec1", "id_dec0", "gd_dec1", "gd_dec0",
                     "cf_dec1", "cf_dec0", "prop_layer"):
            setattr(self, name, torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, 1), torch.nn.BatchNorm2d(1)))


@pytest.fixture
def toy_nlspn():
    return ToyNLSPN()


def test_stage_one_freezes_encoder_and_batch_norm(toy_nlspn):
    groups = core.configure_stage(toy_nlspn, stage=1)
    names = core.trainable_parameter_names(toy_nlspn)
    assert all(not name.startswith((
        "conv1_rgb.", "conv2.", "conv3.", "conv4.", "conv5.", "conv6."))
        for name in names)
    assert any(name.startswith("conv1_dep.") for name in names)
    assert {group["lr"] for group in groups} == {1e-4}


def test_stage_two_uses_differential_rates(toy_nlspn):
    groups = core.configure_stage(toy_nlspn, stage=2)
    assert [group["name"] for group in groups] == ["encoder", "adaptation"]
    assert [group["lr"] for group in groups] == [5e-6, 2e-5]
    grouped = {n for group in groups for n in group["parameter_names"]}
    assert grouped == set(core.trainable_parameter_names(toy_nlspn))
```

Add tests rejecting unknown trainable prefixes, empty groups, trainable
BatchNorm parameters, and any declared/actual trainable-set mismatch.

- [ ] **Step 2: Write failing objective and metric tests**

The formal supervised objective is exactly `1.0 * L1 + 1.0 * L2` on valid
final-prediction pixels. It does not include intermediate or temporal losses.

```python
def test_l1_l2_objective_uses_only_valid_target_pixels():
    pred = torch.tensor([[[[1.0, 4.0], [9.0, 7.0]]]])
    gt = torch.tensor([[[[2.0, 2.0], [0.0, 0.0]]]])
    assert core.masked_l1_l2(pred, gt, 10.0).item() == pytest.approx(4.0)


def test_metric_accumulator_recomputes_pooled_rmse():
    acc = core.MetricAccumulator()
    acc.add("room3", 8.0, 4.0, 2.0, 2)
    acc.add("room7", 9.0, 3.0, 1.0, 1)
    result = acc.finalize()
    assert result["pooled_rmse"] == pytest.approx((17.0 / 3.0) ** 0.5)
    assert result["pooled_mae"] == pytest.approx(7.0 / 3.0)
```

The loss expectation is mean L1 plus mean squared error after clamping to
`[0,10]`: errors 1 and 2 give `1.5 + 2.5 = 4.0`. Add tests for all five
depth bands, empty masks, nonfinite inputs, and float64 raw-sum accumulation.

- [ ] **Step 3: Run tests and verify RED**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_core.py -x
```

Expected: import failure because the core module does not exist.

- [ ] **Step 4: Implement exact phase policies**

```python
ENCODER_PREFIXES = (
    "conv1_rgb.", "conv2.", "conv3.", "conv4.", "conv5.", "conv6.")
ADAPTATION_PREFIXES = (
    "conv1_dep.", "dec5.", "dec4.", "dec3.", "dec2.",
    "id_dec1.", "id_dec0.", "gd_dec1.", "gd_dec0.",
    "cf_dec1.", "cf_dec0.", "prop_layer.")


def configure_stage(model, stage):
    eligible = original_trainable_names(model)
    freeze_batch_norm(model)
    if stage == 1:
        spec = (("adaptation", ADAPTATION_PREFIXES, 1e-4),)
    elif stage == 2:
        spec = (("encoder", ENCODER_PREFIXES, 5e-6),
                ("adaptation", ADAPTATION_PREFIXES, 2e-5))
    else:
        raise ValueError("stage must be 1 or 2")
    return build_exact_groups(model, eligible, spec)
```

Capture original trainability before changing flags. Exclude all BatchNorm
affine parameters and the already-fixed propagation dummy weights. Reject any
other eligible prefix. Call `keep_batch_norm_frozen` after every
`model.train()`.

- [ ] **Step 5: Implement loss and aggregates, then verify**

Implement `masked_l1_l2`, per-frame raw sums, `MetricAccumulator`, scene macro,
pooled RMSE/MAE/AbsRel, and bands `(0,2]`, `(2,4]`, `(4,6]`, `(6,8]`,
`(8,10]`. Use flat CSV keys `band_0_2`, `band_2_4`, `band_4_6`,
`band_6_8`, and `band_8_10`; each carries squared-error sum, absolute-error
sum, and valid-pixel count. Their raw sums must equal the full-frame raw sums.
Run:

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_core.py
git diff --check
git add scripts/nlspn_scene_finetune_core.py tests/test_nlspn_scene_finetune_core.py
git commit -m "feat: define staged NLSPN fine-tuning core"
```

### Task 4: Add convergence, checkpoints, resume, and memory probing

**Files:**
- Modify: `scripts/nlspn_scene_finetune_core.py`
- Modify: `tests/test_nlspn_scene_finetune_core.py`

- [ ] **Step 1: Write failing convergence/checkpoint tests**

```python
def test_tracker_stops_after_four_nonsignificant_epochs():
    tracker = core.ValidationTracker(patience=4, min_relative_gain=0.001)
    assert tracker.update(1, 1.0000)["save_best"]
    assert tracker.update(2, 0.9995)["save_best"]
    tracker.update(3, 0.9994)
    tracker.update(4, 0.9993)
    assert tracker.update(5, 0.9992)["stop"]
    assert tracker.best_epoch == 5
    assert tracker.best_rmse == pytest.approx(0.9992)


def test_checkpoint_strictly_loads_into_identical_model(toy_nlspn):
    meta = {name: name for name in (
        "source_checkpoint_sha256", "train_manifest_sha256",
        "val_manifest_sha256", "test_manifest_sha256",
        "preprocessing_sha256", "split_seed", "model_state_schema_sha256",
        "stage_configuration_sha256")}
    checkpoint = core.build_checkpoint(
        toy_nlspn, epoch=4, stage=2, optimizer_state={},
        tracker_state={}, val_metrics={"pooled_rmse": 0.2},
        args={"seed": 2026}, meta=meta)
    assert set(checkpoint) == {
        "net", "epoch", "optimizer", "scheduler", "tracker", "val",
        "args", "meta"}
    core.load_net_strict(ToyNLSPN(), checkpoint)
```

Add tests rejecting resume on changed source, any manifest, preprocessing,
seed, model schema, or stage configuration; accepting exact resume; atomic
checkpoint replacement; and refusal to overwrite the generic path.

- [ ] **Step 2: Write failing batch-probe tests**

```python
def test_probe_chooses_largest_effective_batch_divisor_that_fits():
    calls = []
    def attempt(batch):
        calls.append(batch)
        if batch > 3:
            raise core.ProbeOutOfMemory()
    assert core.probe_physical_batch(attempt, 12) == {
        "physical_batch_size": 3, "accumulation_steps": 4}
    assert calls == [12, 6, 4, 3]


def test_probe_does_not_hide_non_oom_errors():
    with pytest.raises(ValueError, match="bad sample"):
        core.probe_physical_batch(
            lambda _: (_ for _ in ()).throw(ValueError("bad sample")), 12)
```

- [ ] **Step 3: Run tests and verify RED**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_core.py -x
```

Expected: failures for missing tracker, checkpoint, resume, and probe symbols.

- [ ] **Step 4: Implement convergence and immutable resume metadata**

```python
RESUME_META_FIELDS = (
    "source_checkpoint_sha256", "train_manifest_sha256",
    "val_manifest_sha256", "test_manifest_sha256",
    "preprocessing_sha256", "split_seed", "model_state_schema_sha256",
    "stage_configuration_sha256")


def validate_resume(checkpoint, expected):
    actual = checkpoint.get("meta", {})
    for field in RESUME_META_FIELDS:
        if actual.get(field) != expected[field]:
            raise RuntimeError(f"resume metadata mismatch for {field}")
```

Track absolute-best RMSE separately from significant-best RMSE. Save every
finite absolute improvement, reset patience only on at least 0.1% relative
gain, and stop Stage 2 after four nonsignificant epochs. Store the explicit
two-stage controller in `scheduler`; learning rates remain constant per stage.

- [ ] **Step 5: Implement preformal memory probing**

Try divisors `(12,6,4,3,2,1)`. Each attempt constructs a fresh model, strictly
loads the source, runs one real forward/backward/step, destroys the model, and
clears CUDA cache. Convert only PyTorch CUDA OOM to `ProbeOutOfMemory`. Once
selected, physical batch and accumulation factor are immutable.

- [ ] **Step 6: Verify and commit contracts**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_core.py
git diff --check
git add scripts/nlspn_scene_finetune_core.py tests/test_nlspn_scene_finetune_core.py
git commit -m "feat: checkpoint resumable NLSPN fine-tuning"
```

### Task 5: Implement the Python 3.7 training worker

**Files:**
- Create: `scripts/run_nlspn_scene_finetune_worker.py`
- Create: `tests/test_run_nlspn_scene_finetune_worker.py`

- [ ] **Step 1: Write failing lifecycle tests with injected boundaries**

```python
def test_worker_never_reads_test_before_best_selection(tmp_path):
    calls = []
    result = worker.run_training(_cli(tmp_path), **fake_dependencies(calls))
    assert calls.index("select_best") < calls.index("load_test")
    assert calls.count("evaluate_test_pair") == 1
    assert result["stage1_epochs"] == 3


def test_worker_applies_stage_one_then_stage_two(tmp_path):
    calls = []
    worker.run_training(_cli(tmp_path), **fake_dependencies(calls))
    stages = [x for x in calls if x.startswith("configure_stage")]
    assert stages == ["configure_stage:1", "configure_stage:2"]
    assert calls.count("train_epoch:stage1") == 3
    assert 1 <= calls.count("train_epoch:stage2") <= 15


def test_training_failure_never_starts_test(tmp_path):
    deps = fake_dependencies([])
    deps["train_epoch_fn"] = lambda *args: (_ for _ in ()).throw(
        RuntimeError("nonfinite loss"))
    with pytest.raises(RuntimeError, match="nonfinite loss"):
        worker.run_training(_cli(tmp_path), **deps)
    assert not (tmp_path / "raw/test_frame_metrics.csv").exists()
```

Add tests for fixed three-epoch Stage 1, Stage 2 patience, exact gradient
accumulation including a partial final group, clipping, BatchNorm freeze after
`train()`, best-by-validation selection, atomic latest/best, strict resume,
paired generic/specialized inputs, exactly 16,000 test rows, 30 approved
held-out windows, and fixed CLI policies.

- [ ] **Step 2: Run tests and verify RED**

```bash
python -m pytest -q tests/test_run_nlspn_scene_finetune_worker.py -x
```

Expected: import failure because the worker does not exist.

- [ ] **Step 3: Implement construction and epoch execution**

Reuse `run_spn_sequence_worker.build_model("nlspn", ...)` and
`load_checkpoint_strict`. Build validated datasets from the three manifests.
The stage loop must follow this exact sequence:

```python
def execute_stage(model, stage, epochs, train_loader_factory, val_loader,
                  tracker, output_dir, context):
    groups = core.configure_stage(model, stage)
    optimizer = torch.optim.Adam([
        {"params": group["params"], "lr": group["lr"]}
        for group in groups])
    for epoch in epochs:
        train_metrics = train_epoch(
            model, train_loader_factory(epoch), optimizer,
            context["accumulation_steps"], 1.0)
        val_metrics = evaluate_model(model, val_loader)
        decision = tracker.update(epoch, val_metrics["pooled_rmse"])
        append_epoch_metrics(output_dir, stage, epoch,
                             train_metrics, val_metrics, groups)
        save_latest(model, optimizer, tracker, context,
                    stage, epoch, val_metrics, output_dir)
        if decision["save_best"]:
            save_best(model, optimizer, tracker, context,
                      stage, epoch, val_metrics, output_dir)
        if stage == 2 and decision["stop"]:
            break
```

Run Stage 1 at epochs 1–3 without early stop and Stage 2 at epochs 4–18.
Divide microbatch loss by accumulation steps; before each step require finite
loss/gradients, clip global norm at 1.0, and correctly rescale/step a nonempty
partial final group.

- [ ] **Step 4: Implement baseline validation and paired test evaluation**

Before training, evaluate the generic model only on validation. After choosing
`best.pt`, construct the test dataset for the first time. Load fresh generic
and specialized models, and run both on each identical test batch before
advancing. Write raw squared/absolute/AbsRel sums, valid counts, derived
metrics, and five depth-band sums for every `(variant,scene,frame_id)`; require
exactly 16,000 unique rows.

Filter the existing 90-window manifest to `room3/room7`, requiring exactly 30
windows. Store the 150 fixed inputs plus generic/specialized predictions in
compressed NPZ data. Require 500 sparse samples and identical inputs across
variants.

- [ ] **Step 5: Implement worker outputs and CLI**

```python
RAW_ARTIFACTS = (
    "best.pt", "latest.pt", "specialized_args.json",
    "epoch_metrics.csv", "baseline_val_frame_metrics.csv",
    "test_frame_metrics.csv", "window_predictions.npz",
    "worker_metadata.json")
```

Write metadata first with `complete=false`, then set true only after all raw
files validate and immutable digests still match. Stream progress after every
epoch. CLI requires data root, three manifests, source checkpoint/args,
previous selected-window manifest, raw output, device, seed, and optional exact
resume checkpoint.

- [ ] **Step 6: Verify both environments and commit**

```bash
python -m pytest -q tests/test_run_nlspn_scene_finetune_worker.py
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_nlspn_scene_finetune_data.py \
  tests/test_nlspn_scene_finetune_core.py \
  tests/test_run_nlspn_scene_finetune_worker.py
git diff --check
git add scripts/run_nlspn_scene_finetune_worker.py tests/test_run_nlspn_scene_finetune_worker.py
git commit -m "feat: train scene-specialized NLSPN weights"
```

### Task 6: Recompute metrics and build validated artifacts

**Files:**
- Create: `scripts/nlspn_scene_finetune_artifacts.py`
- Create: `tests/test_nlspn_scene_finetune_artifacts.py`

- [ ] **Step 1: Write failing aggregate and success-gate tests**

```python
def summary(overall, room3, room7):
    return {
        "pooled_rmse": overall,
        "scenes": {"room3": {"rmse": room3}, "room7": {"rmse": room7}},
    }


def minimal_valid_test_rows():
    rows = []
    for variant, scale in (("generic", 1.0), ("specialized", 0.9)):
        for frame_id, scene in enumerate(("room3", "room7"), start=1):
            row = {
                "variant": variant, "scene": scene, "frame_id": frame_id,
                "squared_error_sum": 5.0 * scale,
                "absolute_error_sum": 5.0 * scale,
                "abs_rel_sum": 0.5 * scale, "valid_pixels": 5,
            }
            for band in ("0_2", "2_4", "4_6", "6_8", "8_10"):
                row[f"band_{band}_squared_error_sum"] = 1.0 * scale
                row[f"band_{band}_absolute_error_sum"] = 1.0 * scale
                row[f"band_{band}_valid_pixels"] = 1
            rows.append(row)
    return rows


def test_aggregate_recomputes_pooled_scene_macro_and_bands():
    rows = minimal_valid_test_rows()
    result = artifacts.aggregate_test_rows(rows, exact_geometry=False)
    baseline = result["variants"]["generic"]
    expected = np.sqrt(sum(float(r["squared_error_sum"]) for r in rows
                           if r["variant"] == "generic") /
                       sum(int(r["valid_pixels"]) for r in rows
                           if r["variant"] == "generic"))
    assert baseline["pooled_rmse"] == pytest.approx(expected)
    assert tuple(baseline["depth_bands"]) == artifacts.DEPTH_BAND_NAMES


def test_gate_requires_five_percent_and_no_scene_regression():
    baseline = summary(overall=1.0, room3=1.0, room7=1.0)
    specialized = summary(overall=0.94, room3=0.95, room7=0.93)
    assert artifacts.calculate_success_gate(baseline, specialized)["passed"]
    specialized = summary(overall=0.94, room3=1.02, room7=0.86)
    assert not artifacts.calculate_success_gate(
        baseline, specialized)["passed"]
```

Add tests rejecting 15,999/16,001 rows, duplicate keys, wrong scenes,
nonfinite or inconsistent raw sums, missing bands, and gate calculations based
on rounded values.

- [ ] **Step 2: Write failing final-tree tests**

```python
def test_write_artifacts_creates_exact_root_and_thirty_windows(tmp_path):
    result = artifacts.write_final_artifacts(
        tmp_path, fake_raw(), fake_manifests(), source_digests(),
        checkpoint_loader=fake_checkpoint_loader)
    assert result["window_count"] == 30
    assert result["test_frame_count"] == 8000
    assert result["test_metric_row_count"] == 16000
    assert len(list((tmp_path / "windows").rglob("depth_comparison.png"))) == 30
    assert len(list((tmp_path / "windows").rglob("error_comparison.png"))) == 30
```

Add corruption tests for extra/missing files, malformed PNG, incompatible
checkpoint, changed digest, wrong sparse count, mismatched variant inputs,
changed aggregate/gate, `complete=false`, and blank panels.

- [ ] **Step 3: Run tests and verify RED**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_artifacts.py -x
```

Expected: import failure because the artifact module does not exist.

- [ ] **Step 4: Implement independent recomputation**

Recompute all metrics from float64 raw sums. Calculate:

```python
relative_improvement = (
    baseline["pooled_rmse"] - specialized["pooled_rmse"]
) / baseline["pooled_rmse"]
```

Pass only if improvement is at least 5% and each scene ratio is at most 1.01.
Store unrounded values and every Boolean sub-gate. Gate failure remains a valid
publishable result and cannot initiate test-driven retuning.

- [ ] **Step 5: Implement plots, report, and exact validation**

Render training/validation curves, pooled/scene RMSE bars, depth-band bars,
and 30 depth plus 30 common-scale error comparisons with columns `GT`,
`Generic Full NLSPN`, `Specialized Full NLSPN`. Copy the specialized checkpoint
only after strict loading. Validate exact files, manifests, raw/recomputed CSV,
report, every PNG through Pillow, checkpoint key/shape schema, finite held-out
inference, and all source/input digests.

The final root has exactly two directories, `raw` and `windows`, plus these
root files:

```python
ROOT_ARTIFACTS = (
    "best.pt", "args.json", "train_manifest.csv", "val_manifest.csv",
    "test_manifest.csv", "epoch_metrics.csv",
    "baseline_val_frame_metrics.csv", "test_frame_metrics.csv",
    "aggregate_metrics.csv", "depth_band_metrics.csv",
    "training_curves.png", "rmse_comparison.png",
    "depth_band_comparison.png", "run_metadata.json", "report.md",
    "worker.log")
```

`raw` contains exactly `RAW_ARTIFACTS`; `windows` contains exactly 30 canonical
window directories, each with `depth_comparison.png`, `error_comparison.png`,
`predictions.npz`, and `metrics.csv`. The root `best.pt` is byte-identical to
`raw/best.pt`; root CSV files are validated copies or independently derived
tables as appropriate.

- [ ] **Step 6: Verify and commit artifacts**

```bash
python -m pytest -q tests/test_nlspn_scene_finetune_artifacts.py
git diff --check
git add scripts/nlspn_scene_finetune_artifacts.py tests/test_nlspn_scene_finetune_artifacts.py
git commit -m "feat: validate scene-specific NLSPN results"
```

### Task 7: Add staged orchestration and atomic publication

**Files:**
- Create: `scripts/run_nlspn_scene_finetune.py`
- Create: `tests/test_run_nlspn_scene_finetune.py`

- [ ] **Step 1: Write failing command and preflight tests**

```python
def _cli(tmp_path):
    return argparse.Namespace(
        data_root=str(tmp_path / "data"),
        source_checkpoint=str(tmp_path / "best.pt"),
        source_args=str(tmp_path / "args.json"),
        motion_root=str(tmp_path / "motion"),
        target_root=str(tmp_path / "target"),
        staging_root=str(tmp_path / "staging"),
        device="cuda:2", seed=2026, resume=None)


def test_worker_command_uses_legacy_environment(tmp_path):
    command, environment = launcher.build_worker_command(_cli(tmp_path))
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--seed") + 1] == "2026"
    assert str(launcher.REPO_ROOT) in environment["PYTHONPATH"]


@pytest.mark.parametrize("name", ("target_root", "staging_root"))
def test_existing_output_is_rejected_before_scan(tmp_path, name):
    cli = _cli(tmp_path)
    Path(getattr(cli, name)).mkdir()
    calls = []
    with pytest.raises(FileExistsError):
        launcher.run(cli, manifest_builder=lambda *_: calls.append("scan"))
    assert calls == []
```

Add tests for missing source/args/data/motion manifest, nonsibling output
paths, source resolving inside output, fewer than 20 GB free, unavailable CUDA,
seed other than 2026, resume outside staging, and immutable changes before
promotion.

- [ ] **Step 2: Write failing orchestration tests**

```python
assert calls == [
    "preflight", "build_manifests", "write_manifests", "snapshot_inputs",
    "worker", "replace_log", "finalize_artifacts", "validate_staging",
    "recheck_inputs", "promote", "validate_target", "recheck_inputs"]
```

Worker/finalizer/validator/recheck failure must keep target absent and never
call promotion. Worker failure leaves staging and streamed log intact.

- [ ] **Step 3: Run tests and verify RED**

```bash
python -m pytest -q tests/test_run_nlspn_scene_finetune.py -x
```

Expected: import failure because the launcher does not exist.

- [ ] **Step 4: Implement defaults and immutable preflight**

```python
DEFAULT_TARGET = Path(
    "/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1")
DEFAULT_STAGING = DEFAULT_TARGET.with_name(
    "full_rmse_scene_disjoint_v1_staging")
DEFAULT_SOURCE = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS = DEFAULT_SOURCE.with_name("args.json")
DEFAULT_MOTION_ROOT = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "cross_scene_motion_stratified_5x3")
```

Preflight requires absent sibling outputs, exact source schema, complete data,
valid 90-window manifest, 20 GB free disk, requested CUDA, and seed 2026.
Snapshot SHA-256 for source checkpoint/args, three manifests, previous motion
manifest, and preprocessing/stage contracts.

- [ ] **Step 5: Implement streaming worker and promotion**

Create staging, write manifests, and start one legacy worker. Stream combined
stdout/stderr directly to `worker.log`; do not retain a multi-hour log in
memory. On success, finalize artifacts, validate staging, rehash inputs,
promote with `promote_new_validated_output`, validate target, and rehash again.
Return compact JSON with path, selected epoch, baseline/specialized RMSE,
improvement, gate, batch sizes, timings, checkpoint digest, report, and plots.

- [ ] **Step 6: Verify focused path and commit**

```bash
python -m pytest -q \
  tests/test_nlspn_scene_finetune_data.py \
  tests/test_nlspn_scene_finetune_core.py \
  tests/test_run_nlspn_scene_finetune_worker.py \
  tests/test_nlspn_scene_finetune_artifacts.py \
  tests/test_run_nlspn_scene_finetune.py
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_nlspn_scene_finetune_data.py \
  tests/test_nlspn_scene_finetune_core.py \
  tests/test_run_nlspn_scene_finetune_worker.py
git diff --check
git add scripts/run_nlspn_scene_finetune.py tests/test_run_nlspn_scene_finetune.py
git commit -m "feat: orchestrate scene-specific NLSPN fine-tuning"
```

### Task 8: Run and verify the formal experiment

**Files:**
- Create through launcher: `/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1`
- Verify: the complete output and specialized `best.pt`

- [ ] **Step 1: Run complete tests before GPU work**

```bash
python -m pytest -q
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_nlspn_scene_finetune_data.py \
  tests/test_nlspn_scene_finetune_core.py \
  tests/test_run_nlspn_scene_finetune_worker.py \
  tests/test_run_spn_sequence_worker.py
```

Expected: all current and legacy tests pass.

- [ ] **Step 2: Verify immutable inputs, disk, and GPU**

```bash
test -d /workspace/VoxelNet/train
test -f /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18/best.pt
test -f /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18/args.json
test -f /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_stratified_5x3/selected_windows.csv
test ! -e /workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1
test ! -e /workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1_staging
df -h /workspace/VoxelNet
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

Expected: inputs exist, outputs are absent, 20 GB is free, and one A100 can
run the formal batch probe.

- [ ] **Step 3: Start the observable formal job**

Use the least occupied verified GPU. For GPU 2:

```bash
python scripts/run_nlspn_scene_finetune.py \
  --target-root /workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1 \
  --device cuda:2 --seed 2026
```

Monitor `worker.log`, process state, GPU utilization, disk, epoch loss,
validation RMSE, and checkpoint updates without changing formal settings.

- [ ] **Step 4: Revalidate promoted results from disk**

Freshly recompute source, manifest, preprocessing, and checkpoint digests and
call the read-only validator. Require split counts 10,000/2,000/8,000, 16,000
test rows, 30 window bundles, 60 window PNGs, strict checkpoint loading, finite
held-out inference, target present, and staging absent.

- [ ] **Step 5: Report results without test-driven retuning**

Report generic/specialized pooled, scene-macro, `room3`, and `room7` RMSE;
improvement; MAE/AbsRel; all depth bands; selected epoch; validation curve;
and gate subresults. If the gate fails, report it and stop without changing
loss, split, learning rates, or epochs based on test outcomes.

- [ ] **Step 6: Inspect visuals and inference compatibility**

Decode all PNGs. Inspect curves, aggregate plots, and one low/medium/high
window per test scene at original detail. Require correct columns/frames,
shared scales, nonblank panels, identical inputs, and no crop/colorbar defect.
Load specialized `best.pt` through the unchanged five-frame worker and require
a finite `(5,228,304)` prediction.

- [ ] **Step 7: Run final verification**

```bash
python -m pytest -q
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_nlspn_scene_finetune_data.py \
  tests/test_nlspn_scene_finetune_core.py \
  tests/test_run_nlspn_scene_finetune_worker.py \
  tests/test_run_spn_sequence_worker.py
git diff --check
git status --short
```

Expected: all tests pass and the feature worktree is clean.

- [ ] **Step 8: Finish the branch**

Invoke `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch`. Present merge, push/PR, keep, and
discard options and take no branch action without the user's choice.
