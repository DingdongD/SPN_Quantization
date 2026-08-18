# NLSPN Frame-Difference Five-Frame Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run frames 0001-0005 through full NLSPN and the three validated causal variants, then create reproducible common-scale depth and error comparison figures in a separate output directory.

**Architecture:** A Python-3.7-compatible visualization module owns configuration parsing, metrics, atomic artifacts, and plotting. A legacy worker reuses the validated NLSPN loader and causal engine to produce all four paths, while a current-environment launcher constructs the old-environment command and independently validates the six artifacts without changing the formal benchmark directory.

**Tech Stack:** Python 3.7/3.11, NumPy, PyTorch 1.10.1, Matplotlib Agg, frozen NLSPN, pytest, CSV/JSON/NPZ/PNG.

---

## File structure

- Create `scripts/nlspn_frame_difference_visualization.py` for selected-config parsing, metrics, plotting, archive writing, and artifact validation primitives.
- Create `scripts/run_nlspn_frame_difference_visualization_worker.py` for five-frame loading and four-path legacy inference.
- Create `scripts/generate_nlspn_frame_difference_visualization.py` for legacy command launch and independent final validation.
- Create `tests/test_nlspn_frame_difference_visualization.py` for pure configuration, metric, plotting, and artifact tests.
- Create `tests/test_run_nlspn_frame_difference_visualization_worker.py` for schedule and inference orchestration tests.
- Create `tests/test_generate_nlspn_frame_difference_visualization.py` for command and output-validator tests.
- Do not modify the formal `pilot_256` artifacts or the benchmark's six-file contract.

### Task 1: Add selected-configuration and metric primitives

**Files:**
- Create: `scripts/nlspn_frame_difference_visualization.py`
- Create: `tests/test_nlspn_frame_difference_visualization.py`

- [ ] **Step 1: Write failing selected-config tests**

```python
def test_load_selected_configs_requires_exact_rgb_and_global_rows(tmp_path):
    write_sweep(tmp_path / "threshold_sweep.csv", [
        sweep_row("rgb_diff", 2/255, 8, selected=True),
        sweep_row("global_diff", 2/255, 8, selected=True),
    ])
    configs = visual.load_selected_configs(tmp_path / "threshold_sweep.csv")
    assert configs["rgb_diff"] == cache.CacheConfig("rgb_diff", 2/255, 8)
    assert configs["global_diff"] == cache.CacheConfig("global_diff", 2/255, 8)

def test_load_selected_configs_rejects_ambiguous_rows(tmp_path):
    rows = [sweep_row("rgb_diff", 2/255, 8, selected=True)] * 2
    rows.append(sweep_row("global_diff", 2/255, 8, selected=True))
    write_sweep(tmp_path / "threshold_sweep.csv", rows)
    with pytest.raises(ValueError, match="exactly one"):
        visual.load_selected_configs(tmp_path / "threshold_sweep.csv")
```

- [ ] **Step 2: Run RED verification**

Run `python -m pytest -q tests/test_nlspn_frame_difference_visualization.py`.
Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement selected-config parsing**

Create constants for method order and display names. Read CSV with
`csv.DictReader`; accept selected values case-insensitively from `true` or `1`.
Require exactly one selected row for each of `rgb_diff` and `global_diff`, no
other selected variant, finite threshold, and positive integer radius. Return
`CacheConfig` instances and compute the sweep SHA-256 with
`nlspn_temporal_residual.file_sha256`.

- [ ] **Step 4: Write failing metric tests**

```python
def test_collect_frame_metrics_emits_twenty_rows():
    payload = five_frame_payload(height=2, width=3)
    predictions = {name: payload["gt"] + index * 0.1
                   for index, name in enumerate(visual.METHOD_ORDER)}
    rows = visual.collect_frame_metrics(payload, predictions, latency_rows())
    assert len(rows) == 20
    assert {(row["method"], row["frame_id"])
            for row in rows} == set(itertools.product(
                visual.METHOD_ORDER, range(1, 6)))
    assert [row["frame_kind"] for row in rows[:5]] == ["I", "P", "I", "P", "I"]
    assert all(row["sparse_count"] == 500 for row in rows)
```

- [ ] **Step 5: Implement metric collection**

Validate five frame IDs, RGB/sparse/GT/valid geometry, exactly 500 nonzero
sparse points per frame, and four finite prediction arrays. Use
`spn_sequence_io.frame_metrics` for RMSE/MAE/valid count. Map latency by
method/frame and use `nlspn_in_memory_gop2.frame_kind` for causal methods;
label full rows `FULL`.

- [ ] **Step 6: Verify and commit Task 1**

Run:

```bash
python -m pytest -q tests/test_nlspn_frame_difference_visualization.py
git diff --check
git add scripts/nlspn_frame_difference_visualization.py tests/test_nlspn_frame_difference_visualization.py
git commit -m "feat: add frame-difference visualization metrics"
```

### Task 2: Add common-scale figures and exact artifacts

**Files:**
- Modify: `scripts/nlspn_frame_difference_visualization.py`
- Modify: `tests/test_nlspn_frame_difference_visualization.py`

- [ ] **Step 1: Write failing common-scale plotting tests**

```python
def test_depth_grid_has_fixed_limits_and_external_colorbar(tmp_path):
    figure = visual.render_depth_comparison(
        tmp_path / "depth.png", payload(), predictions())
    assert len(figure.axes) == 26
    panels = figure.axes[:25]
    assert all(axis.images[0].get_clim() == (0.0, 10.0) for axis in panels)
    assert max(axis.get_position().x1 for axis in panels) + 0.01 <= \
        figure.axes[-1].get_position().x0

def test_error_grid_uses_one_global_99th_percentile(tmp_path):
    expected = visual.common_error_max(payload(), predictions())
    figure = visual.render_error_comparison(
        tmp_path / "error.png", payload(), predictions())
    assert len(figure.axes) == 21
    assert all(axis.images[0].get_clim() == (0.0, expected)
               for axis in figure.axes[:20])
```

- [ ] **Step 2: Run tests and confirm missing renderer failures**

Run the two named tests. Expected: `AttributeError` for the new renderers.

- [ ] **Step 3: Implement depth and error rendering**

Use Matplotlib Agg. Copy the proven `_masked`, `_draw_image`, and external
right-colorbar pattern from `compare_spn_sequence_models.py`. Depth columns
are GT plus four methods, `viridis`, 0-10 m. Error columns are the four
methods, `magma`, 0 to `max(percentile(valid errors, 99), 1e-3)`. Return the
figure for tests; save atomically with a PNG-format temporary file and close it
in the high-level writer.

- [ ] **Step 4: Write failing exact-artifact tests**

```python
def test_write_artifacts_creates_exact_six_files(tmp_path):
    completed = visual.write_artifacts(
        tmp_path, payload(), predictions(), metrics(), metadata())
    assert completed["complete"] is True
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        visual.FINAL_ARTIFACTS)
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as item:
        assert item["full"].shape == (5, 228, 304)
        assert item["global_diff"].shape == (5, 228, 304)
```

- [ ] **Step 5: Implement atomic writers and archive**

Write incomplete metadata first; atomically save NPZ containing frame IDs,
RGB, sparse, GT, valid, and four predictions; write the 20-row CSV; render two
PNGs; create a nonempty `worker.log` placeholder that the launcher atomically
replaces with captured worker output; then mark metadata complete with error
scale and six artifact names.
Reject pre-existing unapproved files and verify every artifact is nonempty.

- [ ] **Step 6: Verify and commit Task 2**

```bash
python -m pytest -q tests/test_nlspn_frame_difference_visualization.py
git diff --check
git add scripts/nlspn_frame_difference_visualization.py tests/test_nlspn_frame_difference_visualization.py
git commit -m "feat: render frame-difference depth comparisons"
```

### Task 3: Add the legacy five-frame inference worker

**Files:**
- Create: `scripts/run_nlspn_frame_difference_visualization_worker.py`
- Create: `tests/test_run_nlspn_frame_difference_visualization_worker.py`

- [ ] **Step 1: Write failing four-path inference test**

```python
def test_run_inference_uses_full_and_fixed_ipipi_schedule():
    engine = RecordingEngine()
    result = worker.run_inference(
        engine, one_five_frame_payload(), selected_configs())
    assert set(result["predictions"]) == set(visual.METHOD_ORDER)
    assert all(value.shape == (5, 228, 304)
               for value in result["predictions"].values())
    assert engine.reset_calls == 4
    assert engine.full_calls == 5
    assert engine.i_calls == 9
    assert [call.variant for call in engine.p_configs] == [
        "zero_flow", "zero_flow", "rgb_diff", "rgb_diff",
        "global_diff", "global_diff"]
```

- [ ] **Step 2: Run RED verification**

Run `python -m pytest -q tests/test_run_nlspn_frame_difference_visualization_worker.py`.
Expected: collection fails because the worker module does not exist.

- [ ] **Step 3: Implement inference orchestration**

Reset before every path. Call `infer_full` five times for full. For each causal
configuration, call `infer_i` on local indices 0, 2, and 4 and `infer_p` on 1
and 3. Convert predictions to float32 NumPy, retain one latency row per method
and frame, and validate exact shapes.

- [ ] **Step 4: Implement real worker CLI**

Accept data root, scene, checkpoint, args JSON, formal directory, output,
device, and seed. Load exactly clip `(1,5)` through
`run_nlspn_in_memory_gop2_worker.load_in_memory_clips`; load the model through
`run_nlspn_frame_difference_cache_worker.build_nlspn`; parse selected configs;
run inference; collect metrics; and write artifacts. Metadata includes all
digests and `formal_directory_modified=false`. The parser exposes no RAFT
argument.

- [ ] **Step 5: Verify both environments and commit Task 3**

```bash
python -m pytest -q tests/test_run_nlspn_frame_difference_visualization_worker.py
conda run -n completionformer-py37 python -m pytest -q tests/test_nlspn_frame_difference_visualization.py tests/test_run_nlspn_frame_difference_visualization_worker.py
git diff --check
git add scripts/run_nlspn_frame_difference_visualization_worker.py tests/test_run_nlspn_frame_difference_visualization_worker.py
git commit -m "feat: run four-path NLSPN visualization inference"
```

### Task 4: Add launcher and independent validator

**Files:**
- Create: `scripts/generate_nlspn_frame_difference_visualization.py`
- Create: `tests/test_generate_nlspn_frame_difference_visualization.py`

- [ ] **Step 1: Write failing command-construction test**

```python
def test_worker_command_uses_old_environment_and_no_raft(tmp_path):
    command, environment = launcher.build_worker_command(
        data_root=tmp_path / "data", scene="scene",
        checkpoint=tmp_path / "best.pt", args_json=tmp_path / "args.json",
        formal_dir=tmp_path / "pilot_256", output_dir=tmp_path / "out",
        device="cuda:0", seed=2026)
    assert command[:6] == ["conda", "run", "-n",
                           "completionformer-py37", "python",
                           str(launcher.WORKER_PATH)]
    assert all("raft" not in str(item).lower() for item in command)
    assert str(launcher.NLSPN_ROOT / "src") in environment["PYTHONPATH"]
```

- [ ] **Step 2: Implement launcher command and subprocess diagnostics**

Build the explicit argument list and NLSPN/deformconv PYTHONPATH. Run without
a shell, capture combined output, atomically save it as `worker.log`, and raise
with the log path and exit code on failure.

- [ ] **Step 3: Write failing final-validator test**

```python
def test_validator_requires_exact_figures_archive_and_metrics(tmp_path):
    build_fake_six_artifacts(tmp_path)
    result = launcher.validate_final_artifacts(
        tmp_path, checkpoint_digest="checkpoint", sweep_digest="sweep")
    assert result["metadata"]["complete"] is True
    assert len(result["metrics"]) == 20
    assert result["archive"]["full"].shape == (5, 228, 304)
    assert result["depth_image_size"][0] > 1000
```

- [ ] **Step 4: Implement independent validation and CLI**

Require the six exact files and nonzero sizes; validate metadata digests,
frame IDs, configs, common limits, and completion; parse 20 unique metric
rows; open both PNGs with PIL; load NPZ without pickle and validate all arrays.
Before launch digest every file in the formal directory, and after validation
require the same name/digest mapping. Print output paths and pooled RMSE values.

- [ ] **Step 5: Verify and commit Task 4**

```bash
python -m pytest -q tests/test_generate_nlspn_frame_difference_visualization.py
python -m pytest -q tests/test_nlspn_frame_difference_visualization.py tests/test_run_nlspn_frame_difference_visualization_worker.py tests/test_generate_nlspn_frame_difference_visualization.py
git diff --check
git add scripts/generate_nlspn_frame_difference_visualization.py tests/test_generate_nlspn_frame_difference_visualization.py
git commit -m "feat: validate NLSPN frame-difference visualizations"
```

### Task 5: Generate and inspect the real five-frame comparison

**Files:**
- Output: `/workspace/VoxelNet/nlspn_frame_difference_cache/BeachApartmentInterior_My_ir/frames_0001_0005_visualization/`

- [ ] **Step 1: Run the real command**

```bash
python scripts/generate_nlspn_frame_difference_visualization.py \
  --output-dir /workspace/VoxelNet/nlspn_frame_difference_cache/BeachApartmentInterior_My_ir/frames_0001_0005_visualization \
  --device cuda:0
```

Expected: six artifacts, 20 metric rows, four five-frame prediction arrays,
and unchanged formal-directory digests.

- [ ] **Step 2: Visually inspect both figures**

Use the local image viewer at original detail. Verify five complete rows,
correct column order/titles, readable row labels, no colorbar overlap, fixed
depth range, shared error range, and no blank/corrupted panels.

- [ ] **Step 3: Run fresh verification**

```bash
python -m pytest -q
conda run -n completionformer-py37 python -m pytest -q tests/test_nlspn_frame_difference_visualization.py tests/test_run_nlspn_frame_difference_visualization_worker.py
git diff --check
git status --short
```

- [ ] **Step 4: Use finishing-a-development-branch**

Report image paths and measured five-frame metrics, identify
`agent/model-semantic-adapters` as the base branch, and offer the four required
integration choices without automatically merging or deleting the worktree.
