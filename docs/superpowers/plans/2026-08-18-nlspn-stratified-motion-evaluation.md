# NLSPN Stratified Motion Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic 90-window low/medium/high motion evaluation for Full NLSPN and four causal GOP2 methods, with complete predictions, statistics, visualizations, independent validation, and non-destructive staged promotion.

**Architecture:** Extend the existing motion scanner with complete five-frame candidate enumeration, then place per-scene rank-tertile selection and manifest validation in a focused sampling module. A single legacy worker loads NLSPN and official RAFT-Small once and writes 90 nested window artifact trees; current-environment statistics and launcher modules derive the 900-row P-frame table, bootstrap/correlation summaries, plots, report, independently validate everything, and atomically promote a previously absent target.

**Tech Stack:** Python 3.7/3.11, NumPy, SciPy, PyTorch 1.10.1, official RAFT-Small, frozen NLSPN, Matplotlib Agg, Pillow, CSV/JSON/NPZ/PNG, pytest.

---

## File structure

- Modify `scripts/nlspn_cross_scene_motion_windows.py` to expose all legal five-frame motion candidates while preserving the current highest-motion selector.
- Modify `tests/test_nlspn_cross_scene_motion_windows.py` to lock candidate order, scoring, and legacy compatibility.
- Create `scripts/nlspn_stratified_motion_sampling.py` for per-scene rank tertiles, seeded non-overlapping selection, exact manifest validation, scene discovery, and CSV serialization.
- Create `tests/test_nlspn_stratified_motion_sampling.py` for exact 6 x 3 x 5 selection behavior and rejection paths.
- Create `scripts/nlspn_stratified_motion_statistics.py` for P-frame derivation, macro/pooled aggregation, deterministic bootstrap intervals, correlations, plots, report, and root artifact writing.
- Create `tests/test_nlspn_stratified_motion_statistics.py` for row geometry, derived metrics, statistics, deterministic confidence intervals, and output artifacts.
- Create `scripts/run_nlspn_stratified_motion_worker.py` for one-load NLSPN/RAFT processing of the exact 90-window manifest under Python 3.7.
- Create `tests/test_run_nlspn_stratified_motion_worker.py` for manifest acceptance, model lifetime, nested output paths, five-method combination, metadata, and CLI policy.
- Modify `scripts/nlspn_validated_output_promotion.py` with a validated promotion path for a previously absent target.
- Modify `tests/test_nlspn_validated_output_promotion.py` for new-target promotion and rollback.
- Create `scripts/run_nlspn_stratified_motion_evaluation.py` for discovery, worker launch, root statistics, exact-tree validation, staging promotion, and final response.
- Create `tests/test_run_nlspn_stratified_motion_evaluation.py` for command construction, immutable inputs, independent recomputation, corruption rejection, and sequencing.

### Task 1: Expose complete motion-window candidate enumeration

**Files:**
- Modify: `scripts/nlspn_cross_scene_motion_windows.py:55-155`
- Modify: `tests/test_nlspn_cross_scene_motion_windows.py:1-70`

- [ ] **Step 1: Write failing enumeration and compatibility tests**

Add these exact tests:

```python
def _constant_thumbnails(values):
    return {
        index + 1: np.full((3, 4), value, dtype=np.float32)
        for index, value in enumerate(values)
    }


def test_enumerate_motion_windows_returns_every_consecutive_candidate():
    thumbnails = _constant_thumbnails([0.0, 0.1, 0.3, 0.6, 0.7, 1.0])
    result = motion.enumerate_motion_windows(range(1, 7), thumbnails)
    assert [row["frame_ids"] for row in result] == [
        [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
    assert result[0]["pair_scores"] == pytest.approx([0.1, 0.2, 0.3, 0.1])
    assert result[0]["motion_score"] == pytest.approx(0.175)
    assert result[1]["motion_score"] == pytest.approx(0.225)


def test_select_motion_window_still_returns_highest_candidate():
    thumbnails = _constant_thumbnails([0.0, 0.1, 0.3, 0.6, 0.7, 1.0])
    result = motion.select_motion_window(range(1, 7), thumbnails)
    assert result["frame_ids"] == [2, 3, 4, 5, 6]


def test_enumeration_skips_gaps_without_reordering_ids():
    ids = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
    thumbnails = {
        frame_id: np.zeros((2, 2), dtype=np.float32) for frame_id in ids}
    result = motion.enumerate_motion_windows(ids, thumbnails)
    assert [row["frame_ids"] for row in result] == [
        [1, 2, 3, 4, 5], [8, 9, 10, 11, 12]]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_nlspn_cross_scene_motion_windows.py
```

Expected: failure because `enumerate_motion_windows` does not exist.

- [ ] **Step 3: Extract validated enumeration from the current selector**

Move the existing thumbnail checks into `_validated_thumbnails`, then add:

```python
def enumerate_motion_windows(frame_ids, thumbnails):
    """Return all legal consecutive five-frame windows in frame order."""
    ids = _validated_ids(frame_ids)
    arrays = _validated_thumbnails(ids, thumbnails)
    candidates = []
    for start_index in range(max(0, len(ids) - 4)):
        window = ids[start_index:start_index + 5]
        if any(right != left + 1 for left, right in zip(window, window[1:])):
            continue
        pair_scores = [
            float(np.mean(np.abs(arrays[right] - arrays[left]),
                          dtype=np.float64))
            for left, right in zip(window, window[1:])]
        candidates.append({
            "frame_ids": list(window),
            "pair_scores": pair_scores,
            "motion_score": float(np.mean(pair_scores, dtype=np.float64)),
        })
    if not candidates:
        raise ValueError("no five consecutive complete frames are available")
    return candidates


def select_motion_window(frame_ids, thumbnails):
    candidates = enumerate_motion_windows(frame_ids, thumbnails)
    return min(candidates, key=lambda row: (
        -row["motion_score"], row["frame_ids"][0]))
```

Add `scan_scene_candidates(scene_root)` using the current RGB/depth
intersection and `_load_thumbnail`; update every candidate with `scene`,
`start_frame`, and `end_frame`. Reimplement `scan_scene` by choosing the
highest candidate from this new function so current callers remain compatible.

- [ ] **Step 4: Verify focused and legacy callers**

```bash
python -m pytest -q \
  tests/test_nlspn_cross_scene_motion_windows.py \
  tests/test_run_nlspn_cross_scene_motion_evaluation.py
git diff --check
```

Expected: all tests pass and the diff check prints nothing.

- [ ] **Step 5: Commit candidate enumeration**

```bash
git add scripts/nlspn_cross_scene_motion_windows.py \
  tests/test_nlspn_cross_scene_motion_windows.py
git commit -m "feat: enumerate NLSPN motion windows"
```

### Task 2: Add deterministic per-scene tertile sampling and manifest schema

**Files:**
- Create: `scripts/nlspn_stratified_motion_sampling.py`
- Create: `tests/test_nlspn_stratified_motion_sampling.py`

- [ ] **Step 1: Write failing rank-tertile tests**

```python
def _candidates(scene="room3", count=18):
    rows = []
    for index in range(count):
        start = 1 + index * 10
        score = np.float64(index + 1) / np.float64(100)
        rows.append({
            "scene": scene,
            "start_frame": start,
            "end_frame": start + 4,
            "frame_ids": list(range(start, start + 5)),
            "pair_scores": [float(score)] * 4,
            "motion_score": float(score),
        })
    return rows


def test_rank_tertiles_are_balanced_and_stable():
    groups = sampling.rank_tertiles(_candidates())
    assert tuple(groups) == sampling.STRATA
    assert [len(groups[name]) for name in sampling.STRATA] == [6, 6, 6]
    assert [row["motion_score"] for row in groups["low"]] == \
        pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert groups["medium"][0]["stratum_rank_start"] == 6
    assert groups["high"][-1]["stratum_rank_end"] == 18


def test_rank_tertiles_use_start_frame_to_break_score_ties():
    rows = _candidates(count=6)
    for row in rows:
        row["motion_score"] = 0.25
    groups = sampling.rank_tertiles(list(reversed(rows)))
    ordered = [row["start_frame"] for name in sampling.STRATA
               for row in groups[name]]
    assert ordered == sorted(ordered)
```

- [ ] **Step 2: Write failing seeded non-overlap tests**

```python
def test_select_scene_windows_is_deterministic_and_non_overlapping():
    first = sampling.select_scene_windows(
        "room3", _candidates(count=45), count_per_stratum=5, seed=2026)
    second = sampling.select_scene_windows(
        "room3", _candidates(count=45), count_per_stratum=5, seed=2026)
    assert first == second
    assert len(first) == 15
    assert [sum(row["stratum"] == name for row in first)
            for name in sampling.STRATA] == [5, 5, 5]
    used = set()
    for row in first:
        assert used.isdisjoint(row["frame_ids"])
        used.update(row["frame_ids"])


def test_select_scene_windows_rejects_insufficient_medium_windows():
    rows = _candidates(count=18)
    for row in rows[6:12]:
        row["frame_ids"] = [101, 102, 103, 104, 105]
        row["start_frame"], row["end_frame"] = 101, 105
    with pytest.raises(ValueError, match="medium.*five non-overlapping"):
        sampling.select_scene_windows(
            "room3", rows, count_per_stratum=5, seed=2026)
```

- [ ] **Step 3: Write failing exact-manifest tests**

```python
def test_validate_manifest_requires_exact_six_by_three_by_five():
    rows = _selected_for_six_scenes()
    result = sampling.validate_selected_windows(rows, seed=2026)
    assert len(result) == 90
    assert [row["scene"] for row in result[:15]] == ["bedroom_ir"] * 15
    assert result[0]["window_id"] == "bedroom_ir/low/0001_0005"
    assert result[-1]["selection_seed"] == 2026


def test_validate_manifest_rejects_shared_frames_across_strata():
    rows = _selected_for_six_scenes()
    rows[5]["frame_ids"] = list(rows[0]["frame_ids"])
    rows[5]["start_frame"] = rows[0]["start_frame"]
    rows[5]["end_frame"] = rows[0]["end_frame"]
    with pytest.raises(ValueError, match="shared frame"):
        sampling.validate_selected_windows(rows, seed=2026)
```

Add explicit tests rejecting 89 rows, unknown scene/stratum, nonconsecutive
IDs, wrong pair-score count, nonfinite scores, mismatched window mean, wrong
stratum bounds, duplicate `window_id`, and unstable ordering.

- [ ] **Step 4: Run RED verification**

```bash
python -m pytest -q tests/test_nlspn_stratified_motion_sampling.py
```

Expected: collection fails because the sampling module does not exist.

- [ ] **Step 5: Implement rank splitting and seeded selection**

```python
STRATA = ("low", "medium", "high")
WINDOWS_PER_STRATUM = 5
SELECTION_SEED = 2026


def rank_tertiles(candidates):
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (float(row["motion_score"]),
                         int(row["start_frame"])))
    if len(ordered) < 3:
        raise ValueError("at least three candidates are required")
    boundaries = [0, len(ordered) // 3, (2 * len(ordered)) // 3,
                  len(ordered)]
    result = {}
    for index, name in enumerate(STRATA):
        start, end = boundaries[index:index + 2]
        selected = []
        for row in ordered[start:end]:
            value = dict(row)
            value.update({
                "stratum": name,
                "stratum_rank_start": start,
                "stratum_rank_end": end,
                "stratum_score_min": float(ordered[start]["motion_score"]),
                "stratum_score_max": float(ordered[end - 1]["motion_score"]),
            })
            selected.append(value)
        result[name] = selected
    return result
```

In `select_scene_windows`, validate the scene, seed
`np.random.RandomState(seed + motion.SCENES.index(scene))`, shuffle candidate
indices separately per stratum, then run five round-robin passes over
`STRATA`. Accept only candidates disjoint from the scene-wide `used_frames`.
Add `selection_seed` and canonical
`window_id = "{scene}/{stratum}/{start:04d}_{end:04d}"`. Return rows in scene,
stratum, start-frame order and fail with the deficient stratum name.

- [ ] **Step 6: Implement discovery and serialization**

```python
def discover_selected_windows(data_root, seed=SELECTION_SEED):
    root = Path(data_root)
    rows = []
    for scene in motion.SCENES:
        candidates = motion.scan_scene_candidates(root / scene)
        rows.extend(select_scene_windows(
            scene, candidates, WINDOWS_PER_STRATUM, seed))
    return validate_selected_windows(rows, seed)


def manifest_object(rows, seed=SELECTION_SEED):
    return {
        "schema_version": 1,
        "selection_seed": int(seed),
        "windows": validate_selected_windows(rows, seed),
    }
```

Implement schema validation in
`validate_selected_windows(rows, seed, exact_geometry=True)`. When
`exact_geometry` is true, require the complete 6 x 3 x 5 shape, stable order,
and cross-stratum non-overlap; when false, retain all row-level, ordering, and
non-overlap checks for focused fixtures without requiring 90 rows. Implement
atomic `write_manifest_json`, `load_manifest_json`,
`write_selected_windows_csv`, and `load_selected_windows_csv`. Add
`canonical_manifest_sha256(rows, seed)` by serializing `manifest_object` with
sorted keys and compact separators before hashing; both launcher and validator
use this function so the temporary worker manifest remains reproducible. CSV
fields are:

```python
SELECTED_WINDOW_FIELDS = (
    "selection_seed", "scene", "stratum", "window_id",
    "stratum_rank_start", "stratum_rank_end",
    "stratum_score_min", "stratum_score_max",
    "start_frame", "end_frame", "frame_ids", "pair_scores",
    "motion_score")
```

Compactly JSON-encode list fields and use sibling `.tmp` files plus
`os.replace` for writes.

- [ ] **Step 7: Verify sampling and commit**

```bash
python -m pytest -q \
  tests/test_nlspn_stratified_motion_sampling.py \
  tests/test_nlspn_cross_scene_motion_windows.py
git diff --check
git add scripts/nlspn_stratified_motion_sampling.py \
  tests/test_nlspn_stratified_motion_sampling.py
git commit -m "feat: sample stratified NLSPN motion windows"
```

### Task 3: Derive the exact 900-row causal P-frame table

**Files:**
- Create: `scripts/nlspn_stratified_motion_statistics.py`
- Create: `tests/test_nlspn_stratified_motion_statistics.py`

- [ ] **Step 1: Write failing P-frame derivation tests**

Create one approved window and 25 metric rows in exact method-major order:

```python
def test_derive_p_frame_metrics_uses_adjacent_i_to_p_scores():
    window = _window(scene="room3", stratum="medium",
                     frame_ids=[101, 102, 103, 104, 105],
                     pair_scores=[0.1, 0.2, 0.3, 0.4])
    rows = _five_method_metrics(window)
    result = statistics.derive_p_frame_metrics(
        [window], rows, exact_window_count=False)
    assert len(result) == 10
    raft = [row for row in result if row["method"] == "raft_gop2"]
    assert [row["frame_id"] for row in raft] == [102, 104]
    assert [row["adjacent_motion_score"] for row in raft] == \
        pytest.approx([0.1, 0.3])
    assert raft[0]["excess_rmse"] == pytest.approx(
        raft[0]["rmse"] - raft[0]["full_rmse"])
    assert raft[0]["rmse_ratio"] == pytest.approx(
        raft[0]["rmse"] / raft[0]["full_rmse"])
    assert raft[0]["speedup"] == pytest.approx(
        raft[0]["full_latency_ms"] / raft[0]["latency_ms"])
```

Add tests where ratio `1.01` passes and `1.0100001` fails. Production geometry
must require 90 windows, two P frames per window, five methods, and 900 unique
rows. Reject missing/duplicate keys, wrong `frame_kind`, nonfinite or negative
metrics, zero Full RMSE/latency, unknown methods, and frame IDs outside the
window.

- [ ] **Step 2: Run RED verification**

```bash
python -m pytest -q \
  tests/test_nlspn_stratified_motion_statistics.py::test_derive_p_frame_metrics_uses_adjacent_i_to_p_scores
```

Expected: collection fails because the statistics module does not exist.

- [ ] **Step 3: Implement exact keyed derivation**

Define the exact row fields and derivation:

```python
P_FRAME_FIELDS = (
    "scene", "stratum", "window_id", "start_frame", "frame_id",
    "local_index", "method", "adjacent_motion_score",
    "window_motion_score", "rmse", "mae", "valid_pixels",
    "latency_ms", "full_rmse", "full_latency_ms", "excess_rmse",
    "rmse_ratio", "passes_1pct", "speedup")


def derive_p_frame_metrics(windows, frame_metrics,
                           exact_window_count=True):
    windows = sampling.validate_selected_windows(
        windows, seed=sampling.SELECTION_SEED,
        exact_geometry=exact_window_count)
    keyed = _validate_frame_metric_rows(windows, frame_metrics)
    result = []
    for window in windows:
        ids = window["frame_ids"]
        for local_index in (1, 3):
            frame_id = ids[local_index]
            full = keyed[(window["window_id"], "full", frame_id)]
            for method in visual.RAFT_METHOD_ORDER:
                row = keyed[(window["window_id"], method, frame_id)]
                ratio = row["rmse"] / full["rmse"]
                result.append({
                    "scene": window["scene"],
                    "stratum": window["stratum"],
                    "window_id": window["window_id"],
                    "start_frame": window["start_frame"],
                    "frame_id": frame_id,
                    "local_index": local_index,
                    "method": method,
                    "adjacent_motion_score":
                        window["pair_scores"][local_index - 1],
                    "window_motion_score": window["motion_score"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "valid_pixels": row["valid_pixels"],
                    "latency_ms": row["latency_ms"],
                    "full_rmse": full["rmse"],
                    "full_latency_ms": full["latency_ms"],
                    "excess_rmse": row["rmse"] - full["rmse"],
                    "rmse_ratio": ratio,
                    "passes_1pct": ratio <= 1.01,
                    "speedup": full["latency_ms"] / row["latency_ms"],
                })
    return result
```

Use sampling-module validation in production and a focused single-window
schema validator for unit fixtures. Validate I-frame arrays later from NPZ;
rounded CSV metrics are not sufficient for numerical identity.

- [ ] **Step 4: Verify derivation and commit**

```bash
python -m pytest -q tests/test_nlspn_stratified_motion_statistics.py
git diff --check
git add scripts/nlspn_stratified_motion_statistics.py \
  tests/test_nlspn_stratified_motion_statistics.py
git commit -m "feat: derive stratified NLSPN P-frame metrics"
```

### Task 4: Add macro, pooled, bootstrap, correlation, report, and plots

**Files:**
- Modify: `scripts/nlspn_stratified_motion_statistics.py`
- Modify: `tests/test_nlspn_stratified_motion_statistics.py`

- [ ] **Step 1: Write failing macro-versus-pooled tests**

```python
def test_stratified_summary_separates_scene_macro_and_pixel_pooling():
    rows = _p_rows_for_macro_test()
    result = statistics.build_stratified_summary(
        rows, bootstrap_replicates=20, seed=2026,
        exact_geometry=False)
    raft_low = next(row for row in result
                    if row["stratum"] == "low" and
                    row["method"] == "raft_gop2")
    assert raft_low["scene_macro_excess_rmse_mean"] == pytest.approx(0.3)
    expected = np.sqrt((0.4 ** 2 * 100 + 0.8 ** 2 * 900) / 1000)
    assert raft_low["pooled_rmse"] == pytest.approx(expected)
    assert raft_low["scene_count"] == 2
```

- [ ] **Step 2: Write failing deterministic bootstrap tests**

```python
def test_bootstrap_intervals_are_deterministic_and_ordered():
    rows = _complete_p_rows()
    first = statistics.build_stratified_summary(
        rows, bootstrap_replicates=2000, seed=2026)
    second = statistics.build_stratified_summary(
        rows, bootstrap_replicates=2000, seed=2026)
    assert first == second
    for row in first:
        assert row["excess_rmse_ci_low"] <= \
            row["scene_macro_excess_rmse_mean"] <= \
            row["excess_rmse_ci_high"]
        assert 0.0 <= row["scene_macro_pass_rate"] <= 1.0
```

Add a fixture proving a resampled `window_id` retains both of its P frames
together rather than independently resampling frames.

- [ ] **Step 3: Write failing correlation tests**

```python
def test_correlation_summary_reports_exact_linear_relationship():
    rows = _correlation_rows(
        motion=[0.1, 0.2, 0.3, 0.4],
        excess=[0.2, 0.4, 0.6, 0.8])
    result = statistics.build_correlation_summary(
        rows, exact_geometry=False)
    target = next(row for row in result
                  if row["method"] == "raft_gop2" and
                  row["target"] == "excess_rmse")
    assert target["sample_count"] == 4
    assert target["pearson_r"] == pytest.approx(1.0)
    assert target["spearman_rho"] == pytest.approx(1.0)
    assert 0.0 <= target["pearson_p"] <= 1.0
```

Full NLSPN has identically zero excess RMSE and unit ratio by definition, so
its correlation table contains only absolute RMSE. The four temporal methods
contain all three targets, producing 13 rows total. Reject any other constant
or nonfinite motion/target array instead of writing NaNs.

- [ ] **Step 4: Run aggregate RED verification**

```bash
python -m pytest -q tests/test_nlspn_stratified_motion_statistics.py
```

Expected: failures because summary, bootstrap, and correlation functions are
absent.

- [ ] **Step 5: Implement structured aggregation and bootstrap**

Validate exact row keys, group by `(stratum, method)` and
`(scene, stratum, method)`, and give six scene means equal macro weight. Use
pixel-weighted pooling:

```python
pooled_rmse = np.sqrt(sum(
    row["rmse"] ** 2 * row["valid_pixels"] for row in selected
) / sum(row["valid_pixels"] for row in selected))
pooled_mae = sum(
    row["mae"] * row["valid_pixels"] for row in selected
) / sum(row["valid_pixels"] for row in selected)
```

For each of 2,000 `RandomState(seed)` replicates, draw five window IDs with
replacement in every scene/stratum cell, retain both P rows for the method,
compute scene means, and macro-average the six scenes. Store 2.5th and 97.5th
percentiles for RMSE, excess RMSE, ratio, pass rate, latency, and speedup.

- [ ] **Step 6: Implement correlations using SciPy**

For Full NLSPN, evaluate only `rmse`. For every temporal method, evaluate
`("rmse", "excess_rmse", "rmse_ratio")`. Call `scipy.stats.pearsonr` and
`scipy.stats.spearmanr` using `adjacent_motion_score`. Return 13 rows ordered
by method then target. Require exactly 180 samples per method in production
and finite, nonconstant evaluated arrays.

- [ ] **Step 7: Write failing root-artifact tests**

```python
def test_write_root_artifacts_emits_exact_files_and_counts(tmp_path):
    completed = statistics.write_root_artifacts(
        tmp_path, _selected_windows(), _complete_p_rows(),
        _stratified_summary(), _correlation_summary(), _metadata())
    assert set(path.name for path in tmp_path.iterdir() if path.is_file()) == \
        set(statistics.ROOT_ARTIFACTS)
    assert _csv_count(tmp_path / "p_frame_metrics.csv") == 900
    assert _csv_count(tmp_path / "stratified_summary.csv") == 15
    assert _csv_count(tmp_path / "correlation_summary.csv") == 13
    assert completed["selected_window_count"] == 90
    assert completed["p_frame_row_count"] == 900
```

- [ ] **Step 8: Implement atomic tables, report, and plots**

Define:

```python
ROOT_ARTIFACTS = (
    "selected_windows.csv", "p_frame_metrics.csv", "stratified_summary.csv",
    "correlation_summary.csv", "run_metadata.json", "report.md",
    "motion_error_scatter.png", "stratified_error_boxplot.png")
```

Use Matplotlib Agg and atomic `.tmp` replacement. The scatter plot has one
panel per temporal method, consistent x limits, stratum colors, excess-RMSE y
axis, and least-squares trend. The two-panel box plot shows excess RMSE and
RMSE ratio grouped by method/stratum with a `1.01` reference line. The report
contains counts, score ranges, macro/pooled statistics, confidence intervals,
pass rates, correlations with p-values, latency, and low/medium/high ordering.

- [ ] **Step 9: Verify statistics and commit**

```bash
python -m pytest -q tests/test_nlspn_stratified_motion_statistics.py
git diff --check
git add scripts/nlspn_stratified_motion_statistics.py \
  tests/test_nlspn_stratified_motion_statistics.py
git commit -m "feat: summarize stratified NLSPN motion error"
```

### Task 5: Add the one-load 90-window legacy worker

**Files:**
- Create: `scripts/run_nlspn_stratified_motion_worker.py`
- Create: `tests/test_run_nlspn_stratified_motion_worker.py`

- [ ] **Step 1: Write failing manifest and CLI tests**

```python
def test_load_manifest_accepts_exact_stratified_schema(tmp_path):
    path = tmp_path / "manifest.json"
    sampling.write_manifest_json(path, _selected_windows(), seed=2026)
    assert worker.load_manifest(path) == _selected_windows()


def test_parser_requires_manifest_and_raft_weights():
    parser = worker.make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    actions = {action.dest for action in parser._actions}
    assert "raft_weights" in actions
    assert "fallback" not in actions
    assert "calibration" not in actions
```

- [ ] **Step 2: Write failing one-load and nested-output test**

```python
def test_run_batch_loads_models_once_and_processes_ninety_windows(tmp_path):
    result = worker.run_batch(
        _cli(tmp_path), bundle_builder=_bundle_builder,
        payload_loader=_payload_loader,
        cache_engine_factory=_cache_engine_factory,
        raft_engine_factory=_raft_engine_factory,
        cache_runner=_cache_runner, raft_runner=_raft_runner,
        artifact_writer=_artifact_writer,
        config_loader=_config_loader,
        directory_digest_fn=_directory_digests,
        file_digest_fn=_file_digest,
        input_digest_fn=lambda *args: "input")
    assert calls["bundle_builder"] == 1
    assert calls["cache_engine_factory"] == 1
    assert calls["raft_engine_factory"] == 1
    assert calls["windows"] == 90
    assert result["window_count"] == 90
    assert result["frame_metric_row_count"] == 2250
    assert result["nlspn_model_load_count"] == 1
    assert result["raft_model_load_count"] == 1
    assert captured_paths[0] == \
        tmp_path / "staging/bedroom_ir/low/0001_0005"
```

Assert every metadata object contains manifest identity/bounds, input and model
digests, fixed configs, official RAFT digest, 12 updates,
`current_to_previous`, `I,P,I,P,I`, and both load counts one.

- [ ] **Step 3: Run worker RED verification**

```bash
python -m pytest -q tests/test_run_nlspn_stratified_motion_worker.py
```

Expected: collection fails because the worker module does not exist.

- [ ] **Step 4: Implement the one-load worker**

Use `legacy_worker.build_models`, `visual_worker.run_inference`,
`visual_worker.run_raft_inference`, and `activate_cuda_device`. The core loop
must follow this exact shape:

```python
windows = load_manifest(cli.manifest)
raft_digest = file_digest_fn(cli.raft_weights)
if raft_digest != raft_small_compat.EXPECTED_WEIGHT_SHA256:
    raise RuntimeError("official RAFT-Small weight digest mismatch")
activate_cuda_device(cli.device)
nlspn, raft, model_metadata = bundle_builder(
    cli.checkpoint, cli.args_json, cli.raft_weights, cli.device)
cache_engine = cache_engine_factory(nlspn, cli.device)
raft_engine = raft_engine_factory(nlspn, raft, cli.device)
for window in windows:
    payload = payload_loader(
        cli.data_root, window["scene"],
        ((window["start_frame"], window["end_frame"]),),
        seed=cli.seed)[0]
    four = cache_runner(cache_engine, payload, configs)
    raft_result = raft_runner(raft_engine, payload)
    predictions = dict(four["predictions"])
    predictions["raft_gop2"] = raft_result["prediction"]
    latency_rows = list(four["latency_rows"]) + \
        list(raft_result["latency_rows"])
    metrics = visual.collect_frame_metrics(payload, predictions, latency_rows)
    output = Path(cli.output_root) / window["window_id"]
    metadata = build_window_metadata(
        window=window, payload=payload, model_metadata=model_metadata,
        checkpoint_digest=checkpoint_digest, args_digest=args_digest,
        sweep_digest=sweep_digest, raft_digest=raft_digest,
        formal_digests=formal_before,
        selected_configs=serialized_configs, device=cli.device,
        seed=cli.seed, input_digest=input_digest_fn(
            payload["frame_ids"], payload["rgb"], payload["sparse"],
            payload["gt"], payload["valid"]))
    artifact_writer(output, payload, predictions, metrics, metadata)
```

Implement `build_window_metadata` as a pure function returning all manifest
identity/bounds, checkpoint/args/formal/RAFT/input digests, fixed configs,
model metadata, device, seed, Python library versions, method order, both model
load counts, RAFT semantics, and schedule. Require exact five-method order and
25 rows per window. Snapshot and recheck the formal directory around the
complete 90-window loop.

- [ ] **Step 5: Implement Python 3.7 CLI and response**

Require manifest, checkpoint, args JSON, formal directory, output root, and
RAFT weights. Default data root, device, and seed to existing values. Call
`torch.set_num_threads(1)`, print final response JSON, and expose no fallback,
retry, resume, calibration, or partial-window flags.

- [ ] **Step 6: Verify both environments and commit**

```bash
python -m pytest -q tests/test_run_nlspn_stratified_motion_worker.py
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_run_nlspn_stratified_motion_worker.py \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py
git diff --check
git add scripts/run_nlspn_stratified_motion_worker.py \
  tests/test_run_nlspn_stratified_motion_worker.py
git commit -m "feat: run stratified NLSPN motion inference"
```

### Task 6: Add validated promotion for a previously absent target

**Files:**
- Modify: `scripts/nlspn_validated_output_promotion.py`
- Modify: `tests/test_nlspn_validated_output_promotion.py`

- [ ] **Step 1: Write failing success, refusal, and rollback tests**

```python
def test_promote_new_validated_output_renames_staging(tmp_path):
    staging = _tree(tmp_path / "result_staging", "new")
    target = tmp_path / "result"
    result = promotion.promote_new_validated_output(
        staging, target,
        lambda path: (path / "marker").read_text(encoding="utf-8"))
    assert result == {"target": str(target.resolve())}
    assert not staging.exists()
    assert (target / "marker").read_text(encoding="utf-8") == "new"


def test_promote_new_refuses_existing_target_without_mutation(tmp_path):
    staging = _tree(tmp_path / "result_staging", "new")
    target = _tree(tmp_path / "result", "old")
    with pytest.raises(FileExistsError, match="target"):
        promotion.promote_new_validated_output(staging, target, lambda path: 1)
    assert (staging / "marker").read_text(encoding="utf-8") == "new"
    assert (target / "marker").read_text(encoding="utf-8") == "old"


def test_promote_new_post_validation_failure_restores_staging(tmp_path):
    staging = _tree(tmp_path / "result_staging", "new")
    target = tmp_path / "result"
    calls = []
    def validator(path):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("post validation failed")
    with pytest.raises(RuntimeError, match="post validation"):
        promotion.promote_new_validated_output(staging, target, validator)
    assert staging.is_dir()
    assert not target.exists()
```

Also reject non-sibling or identical paths, missing staging, noncallable
validator, and pre-validation failure without mutation.

- [ ] **Step 2: Run RED verification**

```bash
python -m pytest -q tests/test_nlspn_validated_output_promotion.py
```

Expected: failures because `promote_new_validated_output` is absent.

- [ ] **Step 3: Implement new-target promotion**

```python
def promote_new_validated_output(staging, target, validator):
    staging = Path(staging).resolve()
    target = Path(target).resolve()
    if staging == target or staging.parent != target.parent:
        raise ValueError("staging and target must be distinct siblings")
    if not staging.is_dir():
        raise ValueError("staging must be an existing directory")
    if target.exists():
        raise FileExistsError("target path already exists: {}".format(target))
    if not callable(validator):
        raise TypeError("promotion validator must be callable")
    validator(staging)
    os.replace(str(staging), str(target))
    try:
        validator(target)
    except Exception:
        if target.exists() and not staging.exists():
            os.replace(str(target), str(staging))
        raise
    return {"target": str(target)}
```

- [ ] **Step 4: Verify both promotion APIs and commit**

```bash
python -m pytest -q tests/test_nlspn_validated_output_promotion.py
git diff --check
git add scripts/nlspn_validated_output_promotion.py \
  tests/test_nlspn_validated_output_promotion.py
git commit -m "feat: promote new validated evaluation outputs"
```

### Task 7: Add launcher, independent validator, and staged orchestration

**Files:**
- Create: `scripts/run_nlspn_stratified_motion_evaluation.py`
- Create: `tests/test_run_nlspn_stratified_motion_evaluation.py`

- [ ] **Step 1: Write failing worker-command and preflight tests**

```python
def test_build_worker_command_uses_one_legacy_process(tmp_path):
    command, environment = launcher.build_worker_command(
        tmp_path / "data", tmp_path / "manifest.json",
        tmp_path / "best.pt", tmp_path / "args.json",
        tmp_path / "formal", tmp_path / "staging",
        tmp_path / "raft.pt", "cuda:2", 2026)
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--seed") + 1] == "2026"
    assert str(launcher.REPO_ROOT) in environment["PYTHONPATH"]


def test_run_evaluation_refuses_existing_target(tmp_path):
    cli = _cli(tmp_path)
    Path(cli.target_root).mkdir()
    with pytest.raises(FileExistsError, match="target"):
        launcher.run_evaluation(cli)
```

Add a symmetric existing-staging refusal test and assert discovery/worker
injections are not called by either preflight failure.

- [ ] **Step 2: Write failing exact-tree validation test**

Construct an exact fake tree with eight root files and 90 nested windows. Inject
a window validator returning five methods, 25 rows, metadata, and arrays:

```python
def test_validate_final_tree_recomputes_exact_geometry(tmp_path):
    windows = _selected_windows()
    _write_exact_fake_tree(tmp_path, windows)
    result = launcher.validate_final_tree(
        tmp_path, windows, checkpoint_digest="checkpoint",
        sweep_digest="sweep", raft_digest="raft",
        manifest_digest="manifest",
        artifact_validator=_fake_window_validator)
    assert result["window_count"] == 90
    assert result["frame_count"] == 450
    assert result["p_frame_count"] == 180
    assert result["p_frame_row_count"] == 900
    assert result["nlspn_model_load_count"] == 1
    assert result["raft_model_load_count"] == 1
```

Add separate corruption tests for one missing window, an extra file, 899 P
rows, wrong stratum, shared frame, changed manifest digest, nonfinite
prediction, I-frame mismatch, wrong RAFT direction/update/digest, model load
count two, changed summary/correlation, empty PNG, and `complete=false`.

- [ ] **Step 3: Write failing orchestration-order test**

Inject all boundary functions and require:

```python
assert calls == [
    "preflight", "discover", "write_manifest", "worker",
    "replace_logs", "derive_p", "summaries", "write_root",
    "validate_staging", "formal_recheck", "promote",
    "validate_target", "formal_recheck"]
```

Worker, statistics, and staging-validation failures must never call promotion
and must leave the target absent.

- [ ] **Step 4: Run launcher RED verification**

```bash
python -m pytest -q tests/test_run_nlspn_stratified_motion_evaluation.py
```

Expected: collection fails because the launcher module does not exist.

- [ ] **Step 5: Implement defaults, discovery, and one worker launch**

```python
DEFAULT_TARGET = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "cross_scene_motion_stratified_5x3")
DEFAULT_STAGING = DEFAULT_TARGET.with_name(
    "cross_scene_motion_stratified_5x3_staging")
DEFAULT_CHECKPOINT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/"
    "nlspn_iter18/best.pt")
DEFAULT_ARGS_JSON = DEFAULT_CHECKPOINT.with_name("args.json")
DEFAULT_FORMAL_DIR = Path(
    "/workspace/VoxelNet/nlspn_frame_difference_cache/"
    "BeachApartmentInterior_My_ir/pilot_256")
```

Refuse target/staging before scanning. Discover 90 windows, create staging,
persist `selected_windows.csv`, and write the worker manifest inside a
`TemporaryDirectory`. Record its canonical digest from
`canonical_manifest_sha256`, snapshot all formal/model/config digests, run one
legacy worker, and capture combined stdout/stderr. Replace every window
`worker.log` atomically with the command and captured output.

- [ ] **Step 6: Implement root statistics after worker completion**

Read all 90 `frame_metrics.csv` files, append canonical window identity from
the manifest, derive exactly 900 P rows, build the 2,000-replicate summary and
13 correlation rows, and call `statistics.write_root_artifacts`. Root metadata
contains exact counts, method order, seed, schedule, model load counts, all
digests, bootstrap settings, formal snapshot, and library versions.

- [ ] **Step 7: Implement independent exact-tree validation**

Load `selected_windows.csv`, reconstruct and hash its canonical manifest, then
walk expected paths from those validated rows. Require eight root files plus six scene
directories, three stratum directories per scene, five windows per cell, and
exactly `visual.FINAL_ARTIFACTS` inside each window. Call the existing artifact
validator and compare metadata with manifest/digests. Load NPZ and require:

```python
for method in visual.RAFT_METHOD_ORDER[1:]:
    if not np.array_equal(archive[method][[0, 2, 4]],
                          archive["full"][[0, 2, 4]]):
        raise RuntimeError("I-frame prediction differs from Full NLSPN")
```

Recompute P rows, summaries, correlations, and report inputs; compare stored
numeric fields with `rtol=1e-12, atol=1e-12`. Validate every PNG is nonempty
and Pillow-decodable. Require official RAFT digest, direction, updates, and
both model load counts one in every window and root metadata.

- [ ] **Step 8: Implement promotion and final response**

Validate staging, recheck formal digests, call
`promote_new_validated_output`, validate target again, and recheck formal
digests. Print JSON containing target path, 90/450/180/900 counts, model load
counts, stratum score ranges, summaries, correlations, report path, and plots.

- [ ] **Step 9: Verify launcher and commit**

```bash
python -m pytest -q \
  tests/test_run_nlspn_stratified_motion_evaluation.py \
  tests/test_nlspn_stratified_motion_statistics.py \
  tests/test_nlspn_stratified_motion_sampling.py \
  tests/test_nlspn_validated_output_promotion.py
git diff --check
git add scripts/run_nlspn_stratified_motion_evaluation.py \
  tests/test_run_nlspn_stratified_motion_evaluation.py
git commit -m "feat: validate stratified NLSPN motion evaluation"
```

### Task 8: Run the real 90-window evaluation and finish verification

**Files:**
- Create through launcher: `/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_stratified_5x3`
- Verify: `/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_stratified_5x3`

- [ ] **Step 1: Run all focused tests before GPU inference**

```bash
python -m pytest -q \
  tests/test_nlspn_cross_scene_motion_windows.py \
  tests/test_nlspn_stratified_motion_sampling.py \
  tests/test_nlspn_stratified_motion_statistics.py \
  tests/test_run_nlspn_stratified_motion_worker.py \
  tests/test_nlspn_validated_output_promotion.py \
  tests/test_run_nlspn_stratified_motion_evaluation.py
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_run_nlspn_stratified_motion_worker.py \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py
```

Expected: all focused current and legacy tests pass.

- [ ] **Step 2: Verify immutable inputs, disk, and GPU**

```bash
test -d /workspace/VoxelNet/train
test -f /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18/best.pt
test -f /root/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth
test ! -e /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_stratified_5x3
test ! -e /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_stratified_5x3_staging
df -h /workspace/VoxelNet
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader
```

Expected: inputs exist, target/staging are absent, at least 1.0 GB is free,
and one GPU can hold NLSPN plus RAFT-Small.

- [ ] **Step 3: Run inference and promotion**

Use the GPU selected by Step 2; if it is GPU 2, run:

```bash
python scripts/run_nlspn_stratified_motion_evaluation.py \
  --target-root /workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_stratified_5x3 \
  --device cuda:2
```

Expected JSON: complete true, 90 windows, 450 frames, 180 P frames, 900 rows,
and NLSPN/RAFT load counts one. Substitute only the verified device index.

- [ ] **Step 4: Revalidate the promoted tree from disk**

Use a read-only Python command to load the manifest, compute checkpoint,
formal, RAFT, and manifest digests, and call `validate_final_tree`. Require
90/450/180/900 and model load counts one. Confirm staging no longer exists.

- [ ] **Step 5: Inspect statistics without overclaiming**

Read the report and both summary CSVs. Report all methods/strata, macro excess
RMSE and ratio, bootstrap intervals, 1% pass rates, Pearson/Spearman values and
p-values, latency, and nonmonotonic exceptions.

- [ ] **Step 6: Inspect representative and complete image sets**

For every six-scene by three-stratum cell, open the earliest selected window's
depth and error PNG at original detail: 36 figures. Verify correct columns,
frames, shared scales, nonblank RAFT panels, P-row changes, and no crop/colorbar
corruption. Programmatically decode all 180 PNG files with Pillow.

- [ ] **Step 7: Run complete verification**

```bash
python -m pytest -q
conda run -n completionformer-py37 python -m pytest -q \
  tests/test_nlspn_frame_difference_visualization.py \
  tests/test_nlspn_in_memory_gop2.py \
  tests/test_run_nlspn_frame_difference_visualization_worker.py \
  tests/test_run_nlspn_cross_scene_raft_visualization_worker.py \
  tests/test_run_nlspn_stratified_motion_worker.py
git diff --check
git status --short
```

Expected: all tests pass and the feature worktree is clean.

- [ ] **Step 8: Finish the branch**

Invoke `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch`. Present the four required branch
completion options and take no branch action without the user's choice.
