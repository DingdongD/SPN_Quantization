# NLSPN Cross-Scene RAFT-GOP2 Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strictly causal official RAFT-Small GOP2 prediction as the fifth method in the existing six-scene NLSPN visualizations, validate it independently, and safely promote it over the current output while retaining a recoverable backup.

**Architecture:** Keep the current four-method pipeline as the compatibility default, but make visualization and aggregation helpers accept either the exact four-method order or the exact five-method order ending in `raft_gop2`. A new legacy worker loads one NLSPN and one RAFT-Small, reuses the existing frame-difference and in-memory GOP2 engines, and writes a staging tree; a new current-environment launcher validates that tree and delegates recoverable promotion to a small filesystem-only module.

**Tech Stack:** Python 3.7/3.11, NumPy, PyTorch 1.10.1, official RAFT-Small weights, frozen NLSPN, Matplotlib Agg, Pillow, pytest, CSV/JSON/NPZ/PNG.

---

## File structure

- Modify `scripts/nlspn_frame_difference_visualization.py` to define the exact five-method schema and render/write either approved method set.
- Modify `scripts/generate_nlspn_frame_difference_visualization.py` so scene validation derives the approved method order from metadata while retaining four-method compatibility.
- Modify `tests/test_nlspn_frame_difference_visualization.py` and `tests/test_generate_nlspn_frame_difference_visualization.py` for five-column/five-method behavior and legacy behavior.
- Modify `scripts/nlspn_cross_scene_motion_windows.py` so pooling, report rendering, metadata, and root CSVs support exactly 24 or 30 rows.
- Modify `tests/test_nlspn_cross_scene_motion_windows.py` for five-method pooling and root artifacts.
- Modify `scripts/run_nlspn_frame_difference_visualization_worker.py` to expose a focused RAFT-GOP2 inference helper.
- Modify `tests/test_run_nlspn_frame_difference_visualization_worker.py` and `tests/test_nlspn_in_memory_gop2.py` to verify schedule, RAFT argument direction, flow updates, and strict warped-state use.
- Create `scripts/run_nlspn_cross_scene_raft_visualization_worker.py` for one-load NLSPN/RAFT six-scene inference.
- Create `tests/test_run_nlspn_cross_scene_raft_visualization_worker.py` for model lifetime, method combination, and metadata.
- Create `scripts/nlspn_validated_output_promotion.py` for recoverable same-filesystem staging promotion.
- Create `tests/test_nlspn_validated_output_promotion.py` for preflight, success, and restoration behavior.
- Create `scripts/run_nlspn_cross_scene_raft_visualization.py` for recorded-window loading, worker launch, independent five-method validation, and promotion.
- Create `tests/test_run_nlspn_cross_scene_raft_visualization.py` for command construction, exact 30-row validation, and launcher sequencing.

### Task 1: Generalize per-scene artifacts to the exact five-method schema

**Files:**
- Modify: `scripts/nlspn_frame_difference_visualization.py:22-343`
- Modify: `scripts/generate_nlspn_frame_difference_visualization.py:68-149`
- Modify: `tests/test_nlspn_frame_difference_visualization.py`
- Modify: `tests/test_generate_nlspn_frame_difference_visualization.py`

- [ ] **Step 1: Write failing schema and geometry tests**

Add a helper in the test module that appends a finite RAFT prediction and five
latency rows, then specify both approved schemas:

```python
def make_raft_predictions(payload):
    predictions = make_predictions(payload)
    predictions["raft_gop2"] = payload["gt"] + np.float32(0.4)
    return predictions


def test_five_method_metrics_and_grid_geometry(tmp_path):
    payload = make_payload(range(282, 287))
    predictions = make_raft_predictions(payload)
    latency = make_latency_rows(range(282, 287)) + [
        {"method": "raft_gop2", "frame_id": frame_id, "latency_ms": 3.0}
        for frame_id in range(282, 287)
    ]
    rows = visual.collect_frame_metrics(payload, predictions, latency)
    assert len(rows) == 25
    assert [row["method"] for row in rows[20:]] == ["raft_gop2"] * 5
    depth = visual.render_depth_comparison(tmp_path / "depth.png", payload,
                                           predictions)
    error = visual.render_error_comparison(tmp_path / "error.png", payload,
                                           predictions)
    assert len(depth.axes) == 31
    assert len(error.axes) == 26


def test_prediction_schema_rejects_partial_raft_method():
    predictions = make_predictions()
    predictions["unknown_flow"] = predictions["full"].copy()
    with pytest.raises(ValueError, match="approved method schema"):
        visual._validate_predictions(predictions)
```

Extend the artifact validator test so metadata contains
`"method_order": list(visual.RAFT_METHOD_ORDER)`, the archive contains
`raft_gop2`, and the CSV contains 25 rows. Keep the existing test with no
`method_order` field to prove the four-method default still validates.

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_nlspn_frame_difference_visualization.py \
  tests/test_generate_nlspn_frame_difference_visualization.py
```

Expected: failures because `RAFT_METHOD_ORDER` does not exist and the current
validators require exactly four predictions, 20 rows, and the 5x5/5x4 grids.

- [ ] **Step 3: Implement exact schema resolution and dynamic rendering**

In `nlspn_frame_difference_visualization.py`, retain the old public name as an
alias and add one resolver:

```python
BASE_METHOD_ORDER = ("full", "zero_flow", "rgb_diff", "global_diff")
RAFT_METHOD_ORDER = BASE_METHOD_ORDER + ("raft_gop2",)
METHOD_ORDER = BASE_METHOD_ORDER
METHOD_NAMES["raft_gop2"] = "RAFT-GOP2"


def prediction_method_order(predictions):
    keys = set(predictions)
    if keys == set(BASE_METHOD_ORDER):
        return BASE_METHOD_ORDER
    if keys == set(RAFT_METHOD_ORDER):
        return RAFT_METHOD_ORDER
    raise ValueError("predictions do not match an approved method schema")
```

Make `_validate_predictions` return the resolved order. In metrics, error
range, renderers, NPZ writing, and artifact writing, iterate over that returned
order. Derive panel count, figure width, expected metric count, and method keys
from it:

```python
methods = _validate_predictions(predictions)
columns = ("gt",) + methods
figure, axes = plt.subplots(
    5, len(columns), figsize=(3.1 * len(columns), 12.25), squeeze=False)
# error grid uses len(methods), figsize=(3.1 * len(methods), 12.25)
expected_metric_count = 5 * len(methods)
metadata["method_order"] = list(methods)
```

The four-method image geometry and API remain unchanged because the default
order still has four elements.

- [ ] **Step 4: Generalize independent scene validation**

In `validate_final_artifacts`, resolve metadata methods as follows:

```python
method_order = tuple(metadata.get(
    "method_order", visual.BASE_METHOD_ORDER))
if method_order not in (
        visual.BASE_METHOD_ORDER, visual.RAFT_METHOD_ORDER):
    raise RuntimeError("visualization method order is invalid")
```

Use `method_order` for expected CSV keys, expected row count, required archive
keys, shape checks, and finite checks. Return it in the validation result. Do
not require RAFT metadata in this generic validator; the dedicated launcher in
Task 6 owns that policy.

- [ ] **Step 5: Verify both schemas and commit**

Run:

```bash
python -m pytest -q \
  tests/test_nlspn_frame_difference_visualization.py \
  tests/test_generate_nlspn_frame_difference_visualization.py
git diff --check
git add scripts/nlspn_frame_difference_visualization.py \
  scripts/generate_nlspn_frame_difference_visualization.py \
  tests/test_nlspn_frame_difference_visualization.py \
  tests/test_generate_nlspn_frame_difference_visualization.py
git commit -m "feat: render five-method NLSPN comparisons"
```

Expected: all focused tests pass, including the unchanged four-method tests.

### Task 2: Generalize scene and root aggregation to 30 rows

**Files:**
- Modify: `scripts/nlspn_cross_scene_motion_windows.py:27-404`
- Modify: `tests/test_nlspn_cross_scene_motion_windows.py`

- [ ] **Step 1: Write failing five-method summary tests**

Extend the fake metric factory with an optional method order and add:

```python
def test_build_scene_summary_accepts_exact_raft_method_order():
    rows = _fake_frame_metrics(
        methods=motion.RAFT_METHOD_ORDER,
        rmse_by_method={"raft_gop2": 1.005})
    summary = motion.build_scene_summary("room3", rows)
    assert [row["method"] for row in summary] == \
        list(motion.RAFT_METHOD_ORDER)
    raft = summary[-1]
    assert raft["rmse_ratio"] == pytest.approx(1.005)
    assert raft["passes_1pct"] is True


def test_write_root_artifacts_accepts_thirty_rows(tmp_path):
    summaries = _summaries_for_six_scenes(motion.RAFT_METHOD_ORDER)
    completed = motion.write_root_artifacts(
        tmp_path, _six_windows(), summaries,
        {"method_order": list(motion.RAFT_METHOD_ORDER)})
    assert completed["summary_row_count"] == 30
    with (tmp_path / "cross_scene_summary.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 30
```

Also test that a 29-row set and an unknown sixth method are rejected.

- [ ] **Step 2: Run RED verification**

Run `python -m pytest -q tests/test_nlspn_cross_scene_motion_windows.py`.

Expected: failure because `RAFT_METHOD_ORDER` is absent and aggregation is
hard-coded to four methods and 24 rows.

- [ ] **Step 3: Implement exact four-or-five method aggregation**

Add:

```python
BASE_METHOD_ORDER = ("full", "zero_flow", "rgb_diff", "global_diff")
RAFT_METHOD_ORDER = BASE_METHOD_ORDER + ("raft_gop2",)
METHOD_ORDER = BASE_METHOD_ORDER


def resolve_summary_method_order(rows):
    methods = {str(row.get("method")) for row in rows}
    if methods == set(BASE_METHOD_ORDER):
        return BASE_METHOD_ORDER
    if methods == set(RAFT_METHOD_ORDER):
        return RAFT_METHOD_ORDER
    raise ValueError("summary rows do not match an approved method schema")
```

Make `build_scene_summary`, `_validated_summaries`, `_all_scene_summary`, and
`_render_report` use the resolved order. Require exactly
`len(SCENES) * len(method_order)` unique root rows. Store the resolved
`method_order` and dynamic `summary_row_count` in completed metadata. Preserve
identical output for four-method callers.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q tests/test_nlspn_cross_scene_motion_windows.py
git diff --check
git add scripts/nlspn_cross_scene_motion_windows.py \
  tests/test_nlspn_cross_scene_motion_windows.py
git commit -m "feat: pool five-method cross-scene metrics"
```

### Task 3: Add the strict RAFT-GOP2 inference path

**Files:**
- Modify: `scripts/run_nlspn_frame_difference_visualization_worker.py:20-72`
- Modify: `tests/test_run_nlspn_frame_difference_visualization_worker.py`
- Modify: `tests/test_nlspn_in_memory_gop2.py`

- [ ] **Step 1: Write the failing RAFT schedule test**

Add a recording online engine and verify exactly one clean causal pass:

```python
class RecordingRAFTEngine:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append(("reset", None))

    def infer_i(self, rgb, sparse, local_index):
        self.calls.append(("i", local_index))
        return online.FrameResult(
            torch.full((HEIGHT, WIDTH), 2.0), 4.0, "I")

    def infer_p(self, rgb, sparse, local_index):
        self.calls.append(("p", local_index))
        return online.FrameResult(
            torch.full((HEIGHT, WIDTH), 2.0), 7.0, "P")


def test_run_raft_inference_uses_strict_ipipi_schedule():
    engine = RecordingRAFTEngine()
    result = worker.run_raft_inference(engine, make_payload())
    assert engine.calls == [
        ("reset", None), ("i", 0), ("p", 1),
        ("i", 2), ("p", 3), ("i", 4)]
    assert result["prediction"].shape == (5, HEIGHT, WIDTH)
    assert [row["method"] for row in result["latency_rows"]] == \
        ["raft_gop2"] * 5
```

- [ ] **Step 2: Strengthen existing engine tests for strict flow semantics**

In `test_nlspn_in_memory_gop2.py`, make a recording RAFT receive a current RGB
filled with `0.25` and previous RGB filled with `0.0`. Assert its forward call
observes the padded and normalized current tensor first, previous tensor
second, and `num_flow_updates == 12`. Add a propagation layer that captures
its guidance and confidence arguments; use a one-pixel backward flow and
spatially varying I-frame state, then compare the captured tensors with direct
`residual.backward_warp` results. This fixes the direction and all three warped
state fields in regression tests instead of merely checking output shape.

- [ ] **Step 3: Run RED verification**

```bash
python -m pytest -q \
  tests/test_run_nlspn_frame_difference_visualization_worker.py \
  tests/test_nlspn_in_memory_gop2.py
```

Expected: the worker test fails because `run_raft_inference` does not exist;
the core engine tests establish the already-implemented strict flow contract.

- [ ] **Step 4: Implement the focused RAFT runner**

Add to the visualization worker:

```python
def run_raft_inference(engine, payload):
    visual._validate_payload(payload)
    engine.reset()
    predictions = []
    latency_rows = []
    for local_index, frame_id in enumerate(payload["frame_ids"]):
        if online.frame_kind(local_index) == "I":
            result = engine.infer_i(
                payload["rgb"][local_index],
                payload["sparse"][local_index], local_index)
        else:
            result = engine.infer_p(
                payload["rgb"][local_index],
                payload["sparse"][local_index], local_index)
        predictions.append(_prediction_array(result))
        latency_rows.append({
            "method": "raft_gop2",
            "frame_id": int(frame_id),
            "latency_ms": float(result.latency_ms),
        })
    return {
        "prediction": np.stack(predictions).astype(np.float32),
        "latency_rows": latency_rows,
    }
```

Do not alter the existing `run_inference` four-method behavior.

- [ ] **Step 5: Verify current and legacy environments, then commit**

```bash
python -m pytest -q \
  tests/test_run_nlspn_frame_difference_visualization_worker.py \
  tests/test_nlspn_in_memory_gop2.py
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_run_nlspn_frame_difference_visualization_worker.py \
  tests/test_nlspn_in_memory_gop2.py
git diff --check
git add scripts/run_nlspn_frame_difference_visualization_worker.py \
  tests/test_run_nlspn_frame_difference_visualization_worker.py \
  tests/test_nlspn_in_memory_gop2.py
git commit -m "feat: expose strict RAFT-GOP2 visualization inference"
```

### Task 4: Add the one-load six-scene RAFT worker

**Files:**
- Create: `scripts/run_nlspn_cross_scene_raft_visualization_worker.py`
- Create: `tests/test_run_nlspn_cross_scene_raft_visualization_worker.py`

- [ ] **Step 1: Write failing one-load and metadata tests**

Create test fakes for the six existing windows and payloads. Inject a bundle
builder returning `(nlspn, raft, metadata)` and assert:

```python
def test_run_batch_loads_each_model_once_and_emits_thirty_rows(tmp_path):
    result = worker.run_batch(
        cli(tmp_path), bundle_builder=recording_bundle_builder,
        payload_loader=fake_payload_loader,
        cache_engine_factory=recording_cache_engine_factory,
        raft_engine_factory=recording_raft_engine_factory,
        cache_runner=fake_four_method_runner,
        raft_runner=fake_raft_runner,
        artifact_writer=fake_artifact_writer,
        root_writer=fake_root_writer,
        config_loader=fixed_config_loader,
        directory_digest_fn=unchanged_digests,
        file_digest_fn=known_digests,
        input_digest_fn=lambda *args: "input")
    assert calls["bundle_builder"] == 1
    assert calls["cache_engine_factory"] == 1
    assert calls["raft_engine_factory"] == 1
    assert calls["scenes"] == list(motion.SCENES)
    assert result["summary_row_count"] == 30
    assert captured_root_metadata["nlspn_model_load_count"] == 1
    assert captured_root_metadata["raft_model_load_count"] == 1
```

Also assert the CLI requires `--raft-weights`, exposes no calibration or
fallback arguments, rejects a non-official digest, and combines prediction
keys in exact `visual.RAFT_METHOD_ORDER` order.

- [ ] **Step 2: Run RED verification**

Run:

```bash
python -m pytest -q tests/test_run_nlspn_cross_scene_raft_visualization_worker.py
```

Expected: collection fails because the new worker module does not exist.

- [ ] **Step 3: Implement manifest validation and one-load setup**

Reuse `load_manifest` from `run_nlspn_cross_scene_motion_worker`. Default the
bundle builder to `run_nlspn_in_memory_gop2_worker.build_models`, which already
strict-loads NLSPN and official RAFT-Small. Before building, require:

```python
raft_digest = file_digest_fn(cli.raft_weights)
if raft_digest != raft_small_compat.EXPECTED_WEIGHT_SHA256:
    raise RuntimeError("official RAFT-Small weight digest mismatch")
nlspn, raft, model_metadata = bundle_builder(
    cli.checkpoint, cli.args_json, cli.raft_weights, cli.device)
cache_engine = cache.FrameDifferenceGOP2Engine(nlspn, cli.device)
raft_engine = online.InMemoryGOP2Engine(nlspn, raft, cli.device)
```

Build these objects once before the six-scene loop.

- [ ] **Step 4: Implement exact five-method scene inference**

For each selected payload, run the existing four-method runner, then the RAFT
runner. Combine without overwriting:

```python
four = cache_runner(cache_engine, payload, configs)
raft_result = raft_runner(raft_engine, payload)
predictions = dict(four["predictions"])
predictions["raft_gop2"] = raft_result["prediction"]
latency_rows = list(four["latency_rows"]) + raft_result["latency_rows"]
if tuple(predictions) != visual.RAFT_METHOD_ORDER:
    raise RuntimeError("five-method prediction order is invalid")
metrics = visual.collect_frame_metrics(payload, predictions, latency_rows)
```

Write scene metadata with exact method order, weight path/digest, implementation
`raft_small_compat`, 12 updates, `current_to_previous`, and both model load
counts. Pool five rows per method and require 30 root rows. Snapshot and verify
the formal pilot directory before and after all scenes.

- [ ] **Step 5: Implement CLI and response**

Require data root, manifest, checkpoint, args JSON, formal directory, staging
output root, RAFT weights, device, and seed. Print a JSON response containing
`complete`, `scene_count=6`, `summary_row_count=30`, and both load counts.

- [ ] **Step 6: Verify both environments and commit**

```bash
python -m pytest -q \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py \
  tests/test_run_nlspn_frame_difference_visualization_worker.py
git diff --check
git add scripts/run_nlspn_cross_scene_raft_visualization_worker.py \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py
git commit -m "feat: run one-load cross-scene RAFT-GOP2 inference"
```

### Task 5: Add recoverable validated-output promotion

**Files:**
- Create: `scripts/nlspn_validated_output_promotion.py`
- Create: `tests/test_nlspn_validated_output_promotion.py`

- [ ] **Step 1: Write failing promotion tests**

Use three sibling temporary directories and a validator that checks a marker:

```python
def test_promote_keeps_validated_new_target_and_old_backup(tmp_path):
    target = make_tree(tmp_path / "cross_scene_motion_windows", "old")
    staging = make_tree(tmp_path / "cross_scene_motion_windows_raft_staging",
                        "new")
    backup = tmp_path / "cross_scene_motion_windows_pre_raft_backup"
    promotion.promote_validated_output(
        staging, target, backup,
        lambda path: (path / "marker").read_text(encoding="utf-8"))
    assert (target / "marker").read_text(encoding="utf-8") == "new"
    assert (backup / "marker").read_text(encoding="utf-8") == "old"


def test_failed_post_promotion_validation_restores_old_target(tmp_path):
    target, staging, backup = make_three_paths(tmp_path)
    calls = []
    def validator(path):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("post validation failed")
    with pytest.raises(RuntimeError, match="post validation"):
        promotion.promote_validated_output(
            staging, target, backup, validator)
    assert target.is_dir()
    assert (target / "marker").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
```

Also test: missing target/staging, pre-existing backup, non-sibling paths, and
pre-validation failure all leave the target untouched.

- [ ] **Step 2: Run RED verification**

Run `python -m pytest -q tests/test_nlspn_validated_output_promotion.py`.

Expected: collection fails because the promotion module does not exist.

- [ ] **Step 3: Implement preflight and promotion**

Implement one public function using explicit resolved paths and `os.replace`:

```python
def promote_validated_output(staging, target, backup, validator):
    staging, target, backup = map(lambda value: Path(value).resolve(),
                                  (staging, target, backup))
    if not staging.is_dir() or not target.is_dir():
        raise ValueError("staging and target must be existing directories")
    if len({staging, target, backup}) != 3 or not (
            staging.parent == target.parent == backup.parent):
        raise ValueError("promotion paths must be distinct siblings")
    if backup.exists():
        raise FileExistsError("backup path already exists")
    validator(staging)
    os.replace(str(target), str(backup))
    try:
        os.replace(str(staging), str(target))
        validator(target)
    except Exception:
        if target.exists() and not staging.exists():
            os.replace(str(target), str(staging))
        if backup.exists() and not target.exists():
            os.replace(str(backup), str(target))
        raise
```

Do not delete the backup after success and do not accept a force flag.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q tests/test_nlspn_validated_output_promotion.py
git diff --check
git add scripts/nlspn_validated_output_promotion.py \
  tests/test_nlspn_validated_output_promotion.py
git commit -m "feat: promote validated NLSPN outputs recoverably"
```

### Task 6: Add the five-method launcher and independent validator

**Files:**
- Create: `scripts/run_nlspn_cross_scene_raft_visualization.py`
- Create: `tests/test_run_nlspn_cross_scene_raft_visualization.py`

- [ ] **Step 1: Write failing recorded-window and command tests**

Specify that the launcher reads the existing CSV rather than rescanning RGB:

```python
def test_load_recorded_windows_preserves_exact_six_windows(tmp_path):
    write_selected_windows(tmp_path / "selected_windows.csv", SIX_WINDOWS)
    result = launcher.load_recorded_windows(tmp_path)
    assert result == SIX_WINDOWS


def test_worker_command_requires_official_raft_weights(tmp_path):
    command, environment = launcher.build_worker_command(
        data_root=tmp_path / "data", manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "best.pt", args_json=tmp_path / "args.json",
        formal_dir=tmp_path / "formal", output_root=tmp_path / "staging",
        raft_weights=tmp_path / "raft-small.pt", device="cuda:0", seed=2026)
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert command[command.index("--raft-weights") + 1] == \
        str(tmp_path / "raft-small.pt")
```

- [ ] **Step 2: Write failing exact-tree validation tests**

Build a fake exact tree with six scene validators returning 25 rows and assert:

```python
def test_validate_raft_tree_recomputes_thirty_rows(tmp_path):
    result = launcher.validate_raft_final_tree(
        tmp_path, SIX_WINDOWS, "checkpoint", "sweep", "raft-digest",
        scene_validator=fake_five_method_scene_validator)
    assert result["scene_count"] == 6
    assert result["summary_row_count"] == 30
    assert result["nlspn_model_load_count"] == 1
    assert result["raft_model_load_count"] == 1
```

Corrupt one RAFT archive key, one RAFT metadata direction, and one root summary
value in separate tests; each must fail independently.

- [ ] **Step 3: Run RED verification**

Run:

```bash
python -m pytest -q tests/test_run_nlspn_cross_scene_raft_visualization.py
```

Expected: collection fails because the launcher module does not exist.

- [ ] **Step 4: Implement recorded-window loading and worker launch**

Parse `selected_windows.csv`, JSON-decode frame IDs and pair scores, validate
the approved scene order and five consecutive IDs, and write the exact values
to a temporary manifest. Define defaults:

```python
DEFAULT_TARGET = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "cross_scene_motion_windows")
DEFAULT_STAGING = DEFAULT_TARGET.with_name(
    "cross_scene_motion_windows_raft_staging")
DEFAULT_BACKUP = DEFAULT_TARGET.with_name(
    "cross_scene_motion_windows_pre_raft_backup")
DEFAULT_RAFT_WEIGHTS = raft_small_compat.DEFAULT_WEIGHT_PATH
```

Stop before launching if staging or backup already exists. Snapshot the formal
pilot directory and the target `selected_windows.csv` digest. Build a single
legacy worker command with `--raft-weights` and capture its combined output.
Replace all six staging `worker.log` placeholders only after worker success.

- [ ] **Step 5: Implement independent five-method validation**

Require the exact root/scene artifact scope, `method_order` equal to
`visual.RAFT_METHOD_ORDER`, 30 root summary rows, 25 scene rows, both model load
counts equal to one, official RAFT digest, 12 updates, and direction
`current_to_previous`. Call generalized `validate_final_artifacts` for each
scene, recompute scene summaries, and compare all numeric fields with
`rtol=1e-12, atol=1e-12`. Require the staging `selected_windows.csv` digest to
equal the current target digest exactly.

- [ ] **Step 6: Implement promotion sequencing**

After staging validation and formal-digest recheck, call:

```python
promotion.promote_validated_output(
    cli.staging_root, cli.target_root, cli.backup_root,
    lambda path: validate_raft_final_tree(
        path, windows, checkpoint_digest, sweep_digest, raft_digest))
```

Recheck formal pilot digests after promotion and print target, backup, selected
windows, all-scene metrics, and both load counts as JSON. The launcher never
deletes the backup.

- [ ] **Step 7: Verify launcher behavior and commit**

```bash
python -m pytest -q \
  tests/test_run_nlspn_cross_scene_raft_visualization.py \
  tests/test_nlspn_validated_output_promotion.py \
  tests/test_generate_nlspn_frame_difference_visualization.py
git diff --check
git add scripts/run_nlspn_cross_scene_raft_visualization.py \
  tests/test_run_nlspn_cross_scene_raft_visualization.py
git commit -m "feat: validate and promote cross-scene RAFT visualizations"
```

### Task 7: Run the six real scenes and finish verification

**Files:**
- Verify: `/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows`
- Create through launcher: `/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows_pre_raft_backup`
- Update through launcher: `/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows`

- [ ] **Step 1: Run all focused tests before GPU inference**

```bash
python -m pytest -q \
  tests/test_nlspn_frame_difference_visualization.py \
  tests/test_generate_nlspn_frame_difference_visualization.py \
  tests/test_nlspn_cross_scene_motion_windows.py \
  tests/test_nlspn_in_memory_gop2.py \
  tests/test_run_nlspn_frame_difference_visualization_worker.py \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py \
  tests/test_nlspn_validated_output_promotion.py \
  tests/test_run_nlspn_cross_scene_raft_visualization.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Verify immutable inputs and promotion preconditions**

```bash
test -f /root/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth
test -d /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows
test ! -e /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows_raft_staging
test ! -e /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows_pre_raft_backup
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
```

Expected: official weights and target exist, staging/backup are absent, and a
GPU has enough free memory for one NLSPN plus one RAFT-Small.

- [ ] **Step 3: Run staged inference, validation, and promotion**

```bash
python scripts/run_nlspn_cross_scene_raft_visualization.py \
  --target-root /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows \
  --device cuda:0
```

Expected JSON: `complete=true`, six scenes, 30 summary rows,
`nlspn_model_load_count=1`, `raft_model_load_count=1`, and paths to the promoted
target and preserved backup.

- [ ] **Step 4: Inspect all updated visualizations**

Open at original detail the depth and error PNG in each of the six scene
directories. Verify all five prediction labels, five correct frame labels,
common per-figure color limits, nonblank RAFT-GOP2 panels, and no crop,
alignment, or colorbar corruption. Record that P-frame differences occur only
in rows 2 and 4 under the `I,P,I,P,I` schedule.

- [ ] **Step 5: Revalidate promoted artifacts from disk**

Run the launcher's validator in a read-only Python command using the promoted
`selected_windows.csv`, checkpoint digest, sweep digest, and official RAFT
digest. Expected result: six scenes, 30 rows, and both model load counts equal
one. Confirm the backup still contains the former 24-row summary.

- [ ] **Step 6: Run full and legacy verification**

```bash
python -m pytest -q
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_nlspn_frame_difference_visualization.py \
  tests/test_nlspn_in_memory_gop2.py \
  tests/test_run_nlspn_frame_difference_visualization_worker.py \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py
git diff --check
git status --short
```

Expected: all tests pass; only the user's pre-existing main-worktree changes
remain outside this clean feature worktree.

- [ ] **Step 7: Finish the feature branch**

Invoke `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch`. Present exactly the four branch
completion options and do not merge, push, retain, or discard without the
user's choice.
