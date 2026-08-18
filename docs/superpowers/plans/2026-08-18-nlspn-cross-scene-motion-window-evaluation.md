# NLSPN Cross-Scene Motion-Window Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select one deterministic motion-rich five-frame window from each of six scenes, run full NLSPN and three fixed causal variants with one model load, and produce per-scene visualizations plus cross-scene metrics.

**Architecture:** A new pure cross-scene module owns thumbnail motion scoring, selected-window manifests, pooled summaries, and root artifacts. Existing visualization helpers are generalized from hard-coded frames 0001-0005 to arbitrary consecutive five-frame IDs; a Python-3.7 worker loads NLSPN once for all scenes, and a current-environment launcher selects windows and independently validates every output.

**Tech Stack:** Python 3.7/3.11, Pillow, NumPy, PyTorch 1.10.1, Matplotlib Agg, frozen NLSPN, pytest, CSV/JSON/Markdown/NPZ/PNG.

---

## File structure

- Create `scripts/nlspn_cross_scene_motion_windows.py` for scene constants, motion scoring, fixed-config enforcement, pooled summaries, root writers, and report rendering.
- Create `tests/test_nlspn_cross_scene_motion_windows.py` for pure selection and aggregation behavior.
- Modify `scripts/nlspn_frame_difference_visualization.py` to accept arbitrary consecutive five-frame IDs.
- Modify `scripts/generate_nlspn_frame_difference_visualization.py` to validate expected frame IDs supplied by metadata.
- Modify the two existing visualization test files for non-0001 windows and legacy compatibility.
- Create `scripts/run_nlspn_cross_scene_motion_worker.py` for one-load six-scene legacy inference.
- Create `tests/test_run_nlspn_cross_scene_motion_worker.py` for batch scheduling and root output behavior.
- Create `scripts/run_nlspn_cross_scene_motion_evaluation.py` for selection, worker launch, logs, and independent validation.
- Create `tests/test_run_nlspn_cross_scene_motion_evaluation.py` for command and final tree validation.

### Task 1: Add deterministic motion-window selection

**Files:**
- Create: `scripts/nlspn_cross_scene_motion_windows.py`
- Create: `tests/test_nlspn_cross_scene_motion_windows.py`

- [ ] **Step 1: Write failing rolling-window tests**

```python
def test_select_motion_window_scores_four_pairs_and_chooses_maximum():
    ids = (1, 2, 3, 4, 5, 6)
    thumbnails = {frame_id: np.full((2, 3), value, np.float32)
                  for frame_id, value in zip(ids, (0, 0, 0, 0, 1, 1))}
    result = motion.select_motion_window(ids, thumbnails)
    assert result["frame_ids"] == [1, 2, 3, 4, 5]
    assert result["pair_scores"] == [0.0, 0.0, 0.0, 1.0]
    assert result["motion_score"] == pytest.approx(0.25)

def test_select_motion_window_rejects_gaps_and_breaks_ties_early():
    ids = (1, 2, 3, 4, 5, 10, 11, 12, 13, 14)
    thumbnails = {frame_id: np.zeros((2, 3), np.float32) for frame_id in ids}
    result = motion.select_motion_window(ids, thumbnails)
    assert result["frame_ids"] == [1, 2, 3, 4, 5]
    assert result["motion_score"] == 0.0
```

- [ ] **Step 2: Run RED verification**

Run `python -m pytest -q tests/test_nlspn_cross_scene_motion_windows.py`.
Expected: collection fails because the new module does not exist.

- [ ] **Step 3: Implement pure rolling selection**

Define `SCENES` in the approved order. Validate sorted unique positive IDs and
same-shape finite float32 grayscale thumbnails in `[0,1]`. Compute adjacent
MAD only for ID difference one. Enumerate start positions whose next four IDs
are consecutive, average their four pair scores, and choose with key
`(-motion_score, start_id)` so exact ties use the earliest start.

- [ ] **Step 4: Write failing real-layout scan tests**

```python
def test_scan_scene_intersects_rgb_and_depth_and_loads_76x57(tmp_path):
    scene = make_scene_files(tmp_path, frame_ids=range(1, 7),
                             missing_depth={6})
    result = motion.scan_scene(scene)
    assert result["frame_ids"] == [1, 2, 3, 4, 5]
    assert result["start_frame"] == 1
    assert result["end_frame"] == 5
```

- [ ] **Step 5: Implement scene scanning**

Parse exact RGB and EXR filename patterns, intersect IDs, load each required
JPEG through Pillow grayscale conversion and bilinear 76x57 resize, normalize
to float32 `[0,1]`, call `select_motion_window`, and add scene/start/end fields.
Reject missing directories, empty files, unreadable JPEGs, and fewer than five
consecutive complete frames.

- [ ] **Step 6: Verify and commit Task 1**

```bash
python -m pytest -q tests/test_nlspn_cross_scene_motion_windows.py
git diff --check
git add scripts/nlspn_cross_scene_motion_windows.py tests/test_nlspn_cross_scene_motion_windows.py
git commit -m "feat: select cross-scene motion windows"
```

### Task 2: Generalize five-frame visualization IDs and fixed configs

**Files:**
- Modify: `scripts/nlspn_frame_difference_visualization.py`
- Modify: `scripts/generate_nlspn_frame_difference_visualization.py`
- Modify: `tests/test_nlspn_frame_difference_visualization.py`
- Modify: `tests/test_generate_nlspn_frame_difference_visualization.py`
- Modify: `scripts/nlspn_cross_scene_motion_windows.py`
- Modify: `tests/test_nlspn_cross_scene_motion_windows.py`

- [ ] **Step 1: Write failing arbitrary-ID regression tests**

```python
def test_payload_accepts_any_five_consecutive_ids():
    payload = make_payload(frame_ids=range(282, 287))
    visual._validate_payload(payload)
    rows = visual.collect_frame_metrics(payload, make_predictions(payload),
                                        make_latency_rows(range(282, 287)))
    assert [row["frame_id"] for row in rows[:5]] == list(range(282, 287))

def test_payload_still_rejects_nonconsecutive_ids():
    payload = make_payload(frame_ids=(1, 2, 4, 5, 6))
    with pytest.raises(ValueError, match="consecutive"):
        visual._validate_payload(payload)
```

- [ ] **Step 2: Generalize payload, metrics, renderers, and archive validator**

Replace comparisons and loops over global `FRAME_IDS` with a helper that
returns the payload's five integer IDs after requiring `np.diff(ids) == 1`.
Keep `FRAME_IDS=(1,2,3,4,5)` only as a default. Render row labels and metric keys
from payload IDs. In `validate_final_artifacts`, derive expected IDs from
metadata and require the NPZ and 20 CSV keys to match them.

- [ ] **Step 3: Write failing fixed-config tests**

```python
def test_require_fixed_configs_accepts_only_formal_values():
    configs = fixed_configs()
    motion.require_fixed_configs(configs)
    configs["rgb_diff"] = cache.CacheConfig("rgb_diff", 4/255, 8)
    with pytest.raises(ValueError, match="2/255"):
        motion.require_fixed_configs(configs)
```

- [ ] **Step 4: Implement fixed-config enforcement**

Require exact variants `rgb_diff/global_diff`, threshold within `1e-15` of
`2/255`, and radius exactly 8. Return a JSON-serializable copy used by all
scene metadata.

- [ ] **Step 5: Verify legacy and generalized behavior, then commit**

```bash
python -m pytest -q tests/test_nlspn_frame_difference_visualization.py tests/test_generate_nlspn_frame_difference_visualization.py tests/test_nlspn_cross_scene_motion_windows.py
git diff --check
git add scripts/nlspn_frame_difference_visualization.py scripts/generate_nlspn_frame_difference_visualization.py scripts/nlspn_cross_scene_motion_windows.py tests/test_nlspn_frame_difference_visualization.py tests/test_generate_nlspn_frame_difference_visualization.py tests/test_nlspn_cross_scene_motion_windows.py
git commit -m "feat: generalize five-frame visualization windows"
```

### Task 3: Add pooled summaries and root artifacts

**Files:**
- Modify: `scripts/nlspn_cross_scene_motion_windows.py`
- Modify: `tests/test_nlspn_cross_scene_motion_windows.py`

- [ ] **Step 1: Write failing pooled-summary tests**

```python
def test_build_scene_summary_emits_four_pooled_rows_and_ratios():
    rows = fake_frame_metrics(full_rmse=1.0, zero_rmse=1.02)
    summary = motion.build_scene_summary("room3", rows)
    assert len(summary) == 4
    zero = next(row for row in summary if row["method"] == "zero_flow")
    assert zero["rmse_ratio"] == pytest.approx(1.02)
    assert zero["passes_1pct"] is False
    assert zero["valid_pixels"] == 5 * PIXELS
```

- [ ] **Step 2: Implement float64 pooling**

For each method, accumulate `rmse**2 * valid_pixels`,
`mae * valid_pixels`, valid pixels, and latency over five unique rows. Compute
pooled RMSE/MAE, then divide by the full row's RMSE. Reject missing methods,
duplicate frames, nonpositive valid counts, and non-finite inputs.

- [ ] **Step 3: Write failing root-writer tests**

```python
def test_write_root_artifacts_creates_four_files_and_twenty_four_rows(tmp_path):
    completed = motion.write_root_artifacts(
        tmp_path, six_windows(), twenty_four_summary_rows(), metadata())
    assert completed["complete"] is True
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) \
        == sorted(motion.ROOT_ARTIFACTS)
    assert csv_count(tmp_path / "selected_windows.csv") == 6
    assert csv_count(tmp_path / "cross_scene_summary.csv") == 24
```

- [ ] **Step 4: Implement root CSV/JSON/Markdown writers**

Write incomplete metadata, deterministic six-row window CSV, deterministic
24-row summary CSV, and a report with per-scene tables plus all-scene pooled
method rows. Mark complete only when exact root files are nonempty. Allow only
the six approved scene directories in addition to the four root files.

- [ ] **Step 5: Verify and commit Task 3**

```bash
python -m pytest -q tests/test_nlspn_cross_scene_motion_windows.py
git diff --check
git add scripts/nlspn_cross_scene_motion_windows.py tests/test_nlspn_cross_scene_motion_windows.py
git commit -m "feat: summarize cross-scene NLSPN metrics"
```

### Task 4: Add one-load six-scene worker

**Files:**
- Create: `scripts/run_nlspn_cross_scene_motion_worker.py`
- Create: `tests/test_run_nlspn_cross_scene_motion_worker.py`

- [ ] **Step 1: Write failing batch-scheduling test**

```python
def test_run_batch_builds_model_once_and_infers_six_scenes(tmp_path):
    result = worker.run_batch(
        cli(tmp_path), model_builder=recording_model_builder,
        payload_loader=fake_payload_loader,
        inference_runner=fake_inference_runner,
        artifact_writer=fake_artifact_writer)
    assert calls["model_builder"] == 1
    assert calls["payload_loader"] == list(motion.SCENES)
    assert calls["inference_runner"] == list(motion.SCENES)
    assert len(result["summary_rows"]) == 24
```

- [ ] **Step 2: Run RED verification**

Run `python -m pytest -q tests/test_run_nlspn_cross_scene_motion_worker.py`.
Expected: collection fails because the worker module does not exist.

- [ ] **Step 3: Implement manifest validation and batch loop**

Read JSON with exactly six unique scene entries in `SCENES` order, each five
consecutive IDs and finite motion fields. Load/enforce formal configs and
snapshot formal digests. Build NLSPN and `FrameDifferenceGOP2Engine` once.
For each window call `load_in_memory_clips` with its `(start,end)`, run existing
four-path `run_inference`, collect 20 metrics, and call existing scene artifact
writer with motion/config/digest metadata.

- [ ] **Step 4: Implement aggregate output and CLI**

Accumulate four pooled rows per scene, call `write_root_artifacts`, verify
formal digests unchanged, and print JSON response. CLI accepts data root,
manifest JSON, checkpoint, args JSON, formal dir, output root, device, and seed;
it exposes no RAFT or calibration argument.

- [ ] **Step 5: Verify both environments and commit Task 4**

```bash
python -m pytest -q tests/test_run_nlspn_cross_scene_motion_worker.py
conda run -n completionformer-py37 python -m pytest -q tests/test_nlspn_frame_difference_visualization.py tests/test_run_nlspn_cross_scene_motion_worker.py
git diff --check
git add scripts/run_nlspn_cross_scene_motion_worker.py tests/test_run_nlspn_cross_scene_motion_worker.py
git commit -m "feat: run one-load cross-scene NLSPN evaluation"
```

### Task 5: Add batch launcher and independent validation

**Files:**
- Create: `scripts/run_nlspn_cross_scene_motion_evaluation.py`
- Create: `tests/test_run_nlspn_cross_scene_motion_evaluation.py`

- [ ] **Step 1: Write failing selection and command test**

```python
def test_prepare_windows_scans_six_scenes_and_command_has_no_raft(tmp_path):
    windows = launcher.prepare_windows(tmp_path, scanner=fake_scanner)
    assert [row["scene"] for row in windows] == list(motion.SCENES)
    command, env = launcher.build_worker_command(
        tmp_path, manifest_path, checkpoint, args_json, formal, output,
        "cuda:0", 2026)
    assert command[:6] == ["conda", "run", "-n",
                           "completionformer-py37", "python",
                           str(launcher.WORKER_PATH)]
    assert all("raft" not in str(item).lower() for item in command)
```

- [ ] **Step 2: Implement CPU window preparation and command launch**

Call `scan_scene(data_root/scene)` in fixed order and add scene names. Write a
temporary manifest with `tempfile.NamedTemporaryFile`, build explicit legacy
command/PYTHONPATH, run without a shell, capture combined output, and replace
all six scene `worker.log` placeholders on success or preserve a root failure
log on error.

- [ ] **Step 3: Write failing final-tree validator test**

```python
def test_validate_tree_requires_six_scene_dirs_and_four_root_files(tmp_path):
    build_fake_cross_scene_tree(tmp_path)
    result = launcher.validate_tree(tmp_path, six_windows(), digests())
    assert len(result["windows"]) == 6
    assert len(result["summary_rows"]) == 24
    assert result["scene_results"]["room7"]["archive"]["full"].shape[0] == 5
```

- [ ] **Step 4: Implement independent tree validation and CLI**

Require exact root files/directories, validate window CSV against selection,
validate root metadata/config/checkpoint digests, call generalized existing
scene validator with each expected frame sequence, open 12 PNGs, recompute
each scene/method pooled metrics from the 120 frame CSV rows, and require exact
agreement with the 24 summary rows. Snapshot and recheck formal-pilot digests.

- [ ] **Step 5: Verify and commit Task 5**

```bash
python -m pytest -q tests/test_run_nlspn_cross_scene_motion_evaluation.py
python -m pytest -q tests/test_nlspn_cross_scene_motion_windows.py tests/test_run_nlspn_cross_scene_motion_worker.py tests/test_run_nlspn_cross_scene_motion_evaluation.py
git diff --check
git add scripts/run_nlspn_cross_scene_motion_evaluation.py tests/test_run_nlspn_cross_scene_motion_evaluation.py
git commit -m "feat: validate cross-scene NLSPN evaluation"
```

### Task 6: Run all six scenes and verify outputs

**Files:**
- Output: `/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows/`

- [ ] **Step 1: Run the approved command**

```bash
python scripts/run_nlspn_cross_scene_motion_evaluation.py \
  --output-root /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows \
  --device cuda:0
```

Expected: six selected windows, 36 per-scene files, four root artifacts, 120
frame metric rows, 24 summary rows, and unchanged formal digests.

- [ ] **Step 2: Visually inspect all figures**

Use the local image viewer at original detail for all 12 PNGs. Verify five
complete rows, correct IDs/columns, external colorbars, no blank panels, fixed
0-10 m depth scale, and one shared documented error scale per scene.

- [ ] **Step 3: Independently recompute and run fresh tests**

```bash
python -m pytest -q
conda run -n completionformer-py37 python -m pytest -q tests/test_nlspn_frame_difference_visualization.py tests/test_run_nlspn_cross_scene_motion_worker.py
git diff --check
git status --short
```

- [ ] **Step 4: Use finishing-a-development-branch**

Report selected windows, per-scene/all-scene quality ratios, image/report paths,
and test evidence. Identify `agent/model-semantic-adapters` as base and offer
the four required integration choices without automatic merge or cleanup.
