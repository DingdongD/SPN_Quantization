# NLSPN Frame-Difference Cache Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Implement and measure zero-flow, RGB-difference-cache, and global-translation-plus-difference-cache causal NLSPN GOP2 variants without RAFT, fine-tuning, fallback, or intermediate tensor files.

**Architecture:** A focused frame-difference module owns photometric masks, sparse-depth consistency, phase correlation, deterministic configuration selection, and a no-RAFT online engine. A legacy-environment worker loads the 256 frames into CPU memory, calibrates B/C only on the first four clips, benchmarks the frozen configurations, and writes six final report artifacts; a current-environment launcher validates the worker command and final artifacts.

**Tech Stack:** Python 3.7/3.11, NumPy, PyTorch 1.10.1, torch.fft, frozen NLSPN ResNet-34 with 18 TGASS propagation steps, pytest, CSV/JSON/Markdown.

---

## File Structure

- Create scripts/nlspn_frame_difference_cache.py for cache configurations, mask math, phase correlation, residual decoding, online state, and selection rules.
- Create scripts/run_nlspn_frame_difference_cache_worker.py for calibration, pure-memory multi-path benchmarking, metrics, and six final artifacts.
- Create scripts/run_nlspn_frame_difference_cache_pilot.py for legacy worker launch and independent artifact validation.
- Create tests/test_nlspn_frame_difference_cache.py for primitives and engine behavior.
- Create tests/test_run_nlspn_frame_difference_cache_worker.py for split, selection, timing, and artifact behavior.
- Create tests/test_run_nlspn_frame_difference_cache_pilot.py for commands and final integrity checks.
- Create docs/2026-08-18-nlspn-frame-difference-cache-pilot-results.md after the measured run.
- Do not modify the validated RAFT-GOP2 engine or its outputs.

### Task 1: Add deterministic cache configurations and mask primitives

**Files:**
- Create: scripts/nlspn_frame_difference_cache.py
- Create: tests/test_nlspn_frame_difference_cache.py

- [ ] **Step 1: Write failing configuration enumeration tests**

    from scripts import nlspn_frame_difference_cache as cache

    def test_candidate_grid_is_fixed_and_complete():
        configs = cache.candidate_configs("rgb_diff")
        assert len(configs) == 12
        assert configs[0] == cache.CacheConfig(
            variant="rgb_diff", threshold=2.0 / 255.0,
            dilation_radius=2)
        assert configs[-1] == cache.CacheConfig(
            variant="rgb_diff", threshold=16.0 / 255.0,
            dilation_radius=8)
        assert cache.candidate_configs("global_diff")[0].variant == "global_diff"

    def test_zero_flow_has_one_parameter_free_config():
        assert cache.candidate_configs("zero_flow") == (
            cache.CacheConfig(
                variant="zero_flow", threshold=None,
                dilation_radius=None),)

Run:

    python -m pytest -q tests/test_nlspn_frame_difference_cache.py

Expected: collection fails because the module does not exist.

- [ ] **Step 2: Implement CacheConfig and the fixed grid**

Use a frozen dataclass. Accept only zero_flow, rgb_diff, and global_diff.
Thresholds are exactly 2/255, 4/255, 8/255, and 16/255; radii are exactly
2, 4, and 8. Reject thresholds outside [0,1], nonpositive radii, and
parameters on zero_flow.

- [ ] **Step 3: Write failing RGB-difference and dilation tests**

    def test_rgb_delta_uses_three_by_three_blur_and_max_channel():
        previous = torch.zeros(1, 3, 5, 5)
        current = previous.clone()
        current[:, 1, 2, 2] = 0.9
        delta = cache.blurred_rgb_delta(current, previous)
        assert delta.shape == (1, 1, 5, 5)
        assert delta[0, 0, 2, 2] == pytest.approx(0.1)

    def test_threshold_is_strict_and_change_mask_is_dilated():
        delta = torch.zeros(1, 1, 9, 9)
        delta[0, 0, 4, 4] = 4.0 / 255.0
        unchanged = cache.photometric_changed(
            delta, threshold=4.0 / 255.0, radius=2)
        assert not unchanged.any()
        delta[0, 0, 4, 4] += 1e-5
        changed = cache.photometric_changed(
            delta, threshold=4.0 / 255.0, radius=2)
        assert changed.sum().item() == 25

Run the two tests and confirm failure because the functions are absent.

- [ ] **Step 4: Implement blur, max-channel delta, and dilation**

Use torch.nn.functional.avg_pool2d with kernel 3, stride 1, padding 1.
Compute channelwise absolute difference and max over channel. Dilation uses
max_pool2d on float masks with kernel 2 * radius + 1, stride 1, and matching
padding, then converts back to bool. Validate BCHW shape, equal geometry,
FP32 finite input, and RGB range [0,1].

- [ ] **Step 5: Write failing sparse consistency and blend tests**

    def test_sparse_inconsistency_uses_two_centimeter_gate():
        sparse = torch.zeros(1, 1, 9, 9)
        base = torch.ones_like(sparse)
        sparse[0, 0, 4, 4] = 1.02
        at_gate = cache.sparse_changed(sparse, base, radius=2)
        assert not at_gate.any()
        sparse[0, 0, 4, 4] = 1.021
        above_gate = cache.sparse_changed(sparse, base, radius=2)
        assert above_gate.sum().item() == 25

    def test_stable_mask_and_blend_use_previous_only_when_stable():
        photo = torch.tensor([[[[False, True]]]])
        depth = torch.tensor([[[[False, False]]]])
        stable = cache.compose_stable_mask(photo, depth)
        previous = torch.tensor([[[[2.0, 2.0]]]])
        candidate = torch.tensor([[[[3.0, 3.0]]]])
        result = cache.blend_cached_depth(previous, candidate, stable)
        torch.testing.assert_close(
            result, torch.tensor([[[[2.0, 3.0]]]]))

- [ ] **Step 6: Implement sparse consistency, stable composition, and blend**

Sparse inconsistency applies only where sparse > 0 and abs(sparse-base) >
0.02, followed by the same dilation. Stable is the complement of
photo_changed OR sparse_changed OR out_of_bounds. Blend uses torch.where and
requires identical B1HW geometry and finite depth.

- [ ] **Step 7: Verify and commit Task 1**

    python -m pytest -q tests/test_nlspn_frame_difference_cache.py
    git diff --check
    git add scripts/nlspn_frame_difference_cache.py tests/test_nlspn_frame_difference_cache.py
    git commit -m "feat: add NLSPN frame-difference cache masks"

### Task 2: Add global phase-correlation translation

**Files:**
- Modify: scripts/nlspn_frame_difference_cache.py
- Modify: tests/test_nlspn_frame_difference_cache.py

- [ ] **Step 1: Write a failing synthetic direction test**

    def test_phase_translation_returns_current_to_previous_displacement():
        previous = synthetic_texture(height=228, width=304)
        current = torch.roll(previous, shifts=(4, -8), dims=(-2, -1))
        dx, dy = cache.estimate_backward_translation(
            current, previous, downsample=4)
        assert (dx, dy) == (8.0, -4.0)

The returned displacement must be suitable for backward_warp: at a current
pixel, sample previous at current + (dx,dy). Run the test and confirm failure.

- [ ] **Step 2: Implement grayscale downsampling and phase correlation**

Use fixed grayscale coefficients 0.2989, 0.5870, 0.1140. Average-pool by four.
Compute rfft2(current_gray) and rfft2(previous_gray), form normalized cross
power with denominator clamped to 1e-12, apply irfft2, select the global
integer maximum, unwrap indices greater than half width/height, and scale by
four. Return Python floats. Reject batches other than one and non-finite or
constant inputs.

- [ ] **Step 3: Write failing constant-flow and boundary tests**

    def test_translation_flow_warps_previous_and_marks_boundaries():
        source = coordinate_image(228, 304)
        flow = cache.constant_backward_flow(
            dx=8.0, dy=-4.0, height=228, width=304,
            device=source.device, dtype=source.dtype)
        warped, in_bounds = residual.backward_warp(source, flow)
        assert flow.shape == (1, 2, 228, 304)
        assert not in_bounds[:, :, :4].any()
        assert not in_bounds[:, :, :, -8:].any()

- [ ] **Step 4: Implement constant_backward_flow and translation_state**

constant_backward_flow fills channel zero with dx and channel one with dy.
translation_state calls the existing residual.backward_warp for previous RGB,
depth, guidance, and confidence and returns their translated tensors plus the
shared in-bounds mask. Validate all geometries before warping.

- [ ] **Step 5: Verify under both Python environments and commit**

    python -m pytest -q tests/test_nlspn_frame_difference_cache.py
    conda run -n completionformer-py37 python -m pytest -q tests/test_nlspn_frame_difference_cache.py
    git add scripts/nlspn_frame_difference_cache.py tests/test_nlspn_frame_difference_cache.py
    git commit -m "feat: add global translation compensation"

### Task 3: Add the no-RAFT online engine

**Files:**
- Modify: scripts/nlspn_frame_difference_cache.py
- Modify: tests/test_nlspn_frame_difference_cache.py

- [ ] **Step 1: Write failing zero-flow engine tests**

Build FakeNLSPN with an original-style prop_layer that returns the signed seed.
Use real 228 x 304 CPU tensors with 500 sparse points.

    def test_zero_flow_p_frame_uses_previous_state_without_raft():
        engine = cache.FrameDifferenceGOP2Engine(
            FakeNLSPN(), device="cpu")
        engine.infer_i(rgb0, sparse0, local_index=0)
        result = engine.infer_p(
            rgb1, sparse1, local_index=1,
            config=cache.candidate_configs("zero_flow")[0])
        assert result.kind == "P"
        assert result.variant == "zero_flow"
        assert result.prediction.device.type == "cpu"
        assert result.mask_metrics["stable_fraction"] == 0.0
        assert not hasattr(engine, "raft")

Run and confirm failure because FrameDifferenceGOP2Engine is absent.

- [ ] **Step 2: Implement full/I delegation and shared residual decode**

Compose the existing InMemoryGOP2Engine with raft=None for infer_full and
infer_i. Store its OnlineState directly. Implement a private decode(base,
guidance,confidence,current_rgb,current_sparse) using the exact original
prop_layer signature and clamp range. Synchronize before starting timing and
after D2H output.

- [ ] **Step 3: Implement zero-flow P inference**

Require odd local index and immediately preceding state. Base/guidance/
confidence are cached without warp. Output the residual candidate and update
state with current RGB, output depth, and cached guidance/confidence. Report
stable_fraction=0, changed_fraction=1, sparse_changed_fraction, and zero
out_of_bounds_fraction.

- [ ] **Step 4: Write failing RGB-diff engine test**

Use a current RGB with one changed region and sparse inconsistency. Compare
result prediction to an independently calculated torch.where(stable,
previous,candidate), and assert mask fractions equal direct mask counts.

- [ ] **Step 5: Implement RGB-diff P inference**

Compute blurred delta, photometric changed, sparse changed, stable mask, dense
candidate, and blend within the synchronized timing boundary. Update state
with current RGB, blended depth, and previous guidance/confidence.

- [ ] **Step 6: Write failing global-diff engine test**

Use a synthetic translated RGB/state and monkeypatch only the phase estimator
to a known displacement. Assert previous depth/guidance/confidence are warped,
out-of-bounds pixels are changed, and output/state contain no non-finite data.

- [ ] **Step 7: Implement global-diff P inference**

Estimate translation, construct constant backward flow, translate state,
compute masks against translated RGB/base, decode residual, blend, and update
state. Include dx/dy and in-bounds coverage in result metrics.

- [ ] **Step 8: Verify and commit Task 3**

    python -m pytest -q tests/test_nlspn_frame_difference_cache.py
    git diff --check
    git add scripts/nlspn_frame_difference_cache.py tests/test_nlspn_frame_difference_cache.py
    git commit -m "feat: add causal frame-difference GOP2 engine"

### Task 4: Add calibration-only parameter selection

**Files:**
- Modify: scripts/nlspn_frame_difference_cache.py
- Modify: tests/test_nlspn_frame_difference_cache.py
- Create: scripts/run_nlspn_frame_difference_cache_worker.py
- Create: tests/test_run_nlspn_frame_difference_cache_worker.py

- [ ] **Step 1: Write failing selection tests**

    def sweep_row(variant, threshold, radius, *, sse=None,
                  valid=10, rmse=None):
        if sse is None:
            assert rmse is not None
            sse = (rmse ** 2) * valid
        return {
            "variant": variant,
            "threshold": threshold,
            "dilation_radius": radius,
            "sse": sse,
            "valid_pixels": valid,
        }

    def test_select_config_uses_calibration_rmse_and_conservative_tie():
        rows = [
            sweep_row("rgb_diff", 2/255, 2, sse=10.0, valid=10),
            sweep_row("rgb_diff", 2/255, 8, sse=10.0 + 1e-12, valid=10),
            sweep_row("rgb_diff", 4/255, 8, sse=9.0, valid=10),
        ]
        selected = cache.select_calibration_config(rows, "rgb_diff")
        assert selected.threshold == pytest.approx(4/255)
        assert selected.dilation_radius == 8

    def test_tie_prefers_lower_threshold_then_larger_radius():
        rows = [
            sweep_row("rgb_diff", 4/255, 8, rmse=1.0),
            sweep_row("rgb_diff", 2/255, 2, rmse=1.0 + 5e-10),
            sweep_row("rgb_diff", 2/255, 8, rmse=1.0 + 5e-10),
        ]
        selected = cache.select_calibration_config(rows, "rgb_diff")
        assert selected.threshold == pytest.approx(2/255)
        assert selected.dilation_radius == 8

The selector accepts calibration rows only and raises if any of the 12
configurations is missing.

- [ ] **Step 2: Implement deterministic selection**

Aggregate float64 squared error and valid pixels by exact config. Compute RMSE.
Find the minimum; candidates within 1e-9 m are tied. Sort ties by threshold
ascending and radius descending. Return CacheConfig and a selected=true marker
for the sweep table.

- [ ] **Step 3: Write failing calibration sweep tests**

Use two short fake payload clips and FakeFrameDifferenceEngine. Assert
run_calibration_sweep runs exactly 12 rgb_diff and 12 global_diff
configurations, never consumes held-out payloads, returns 24 rows, and resets
state at every clip/config boundary.

- [ ] **Step 4: Implement run_calibration_sweep**

The worker accepts calibration payloads separately from held-out payloads.
For each candidate, run one untimed GOP2 prediction pass, accumulate full
reference comparison SSE, valid pixels, stable/photo/sparse/out-of-bounds
coverage, and emit one row. Zero-flow is not swept.

- [ ] **Step 5: Verify and commit Task 4**

    python -m pytest -q tests/test_nlspn_frame_difference_cache.py tests/test_run_nlspn_frame_difference_cache_worker.py
    git add scripts/nlspn_frame_difference_cache.py scripts/run_nlspn_frame_difference_cache_worker.py tests/test_nlspn_frame_difference_cache.py tests/test_run_nlspn_frame_difference_cache_worker.py
    git commit -m "feat: calibrate frame-difference cache configs"

### Task 5: Add pure-memory multi-variant benchmarking and final outputs

**Files:**
- Modify: scripts/run_nlspn_frame_difference_cache_worker.py
- Modify: tests/test_run_nlspn_frame_difference_cache_worker.py
- Create: scripts/run_nlspn_frame_difference_cache_pilot.py
- Create: tests/test_run_nlspn_frame_difference_cache_pilot.py

- [ ] **Step 1: Write failing timed-path tests**

With a recording fake engine and two clips, assert run_timed_variant:

- resets state for every clip and repeat;
- runs I,P scheduling;
- returns one prediction per frame only for repeat zero;
- returns five latency rows per frame in production mode;
- records variant, kind, threshold, radius, mask fractions, dx, and dy.

- [ ] **Step 2: Implement run_timed_variant and quality splits**

Reuse load_in_memory_clips from the existing worker. Flatten predictions in
clip order. Calculate pooled quality for calibration indices 0--127,
held-out indices 128--255, and all 256 frames. Produce per-frame and per-clip
rows with variant and selected configuration. Speedup is full total latency /
variant total latency.

- [ ] **Step 3: Write failing multi-path summary tests**

Assert execute_pilot runs full once plus zero_flow and the selected rgb/global
configs; reports full/I/P/overall latency per variant; reports calibration,
held-out, and all quality; and retains the external RAFT reference
quality_ratio=1.0069332411236265 and speedup=0.5267047643822169.

- [ ] **Step 4: Implement model loading and execute_pilot**

Load only frozen NLSPN using run_spn_sequence_worker.build_model and strict
checkpoint loading. Validate ResNet-34, iteration 18, TGASS, and
preserve_input=false. Do not import raft_small_compat and do not accept a RAFT
weight argument. Measure input load, model load, one warm-up, five repeats,
and peak CUDA memory for every final path.

- [ ] **Step 5: Write failing artifact-scope tests**

Run the fake pilot into tmp_path. Require exactly:

    run_metadata.json
    summary.json
    threshold_sweep.csv
    frame_metrics.csv
    clip_summary.csv
    report.md

Reject NPZ, NPY, depth, mask, FFT, guidance, confidence, prediction, flow,
and cache files. Metadata becomes complete=true only after all six files are
nonempty and consistent.

- [ ] **Step 6: Implement atomic final writers and report**

Write threshold sweep with exactly 24 rows and selected markers. Frame CSV has
one row per path/repeat/frame: four paths x five repeats x 256 = 5,120 rows.
Attach quality SSE/valid fields only to repeat zero. Clip CSV contains each
variant/split/clip summary. Markdown states quality pass/fail and speedup
greater/less than one for all variants.

- [ ] **Step 7: Write failing launcher and validator tests**

The launcher command must use completionformer-py37, repository/NLSPN
PYTHONPATH, no RAFT argument, and the dedicated output directory. The final
validator checks six exact files, 24 sweep rows, 5,120 frame rows, 256 quality
frames per variant, calibration/held-out/all summaries, checkpoint digest,
no tensor cache, and finite metrics.

- [ ] **Step 8: Implement worker CLI and launcher**

Worker CLI accepts data root, scene, checkpoint, args JSON, device, output,
seed, clip overrides, warm-up repeats, and timed repeats. Launcher invokes it
without a shell, captures stdout for failure diagnostics, validates outputs,
and prints selected configs, all-frame ratios, and measured speedups.

- [ ] **Step 9: Verify and commit Task 5**

    python -m pytest -q tests/test_nlspn_frame_difference_cache.py tests/test_run_nlspn_frame_difference_cache_worker.py tests/test_run_nlspn_frame_difference_cache_pilot.py
    conda run -n completionformer-py37 python -m pytest -q tests/test_nlspn_frame_difference_cache.py tests/test_run_nlspn_frame_difference_cache_worker.py
    git diff --check
    git add scripts/run_nlspn_frame_difference_cache_worker.py scripts/run_nlspn_frame_difference_cache_pilot.py tests/test_run_nlspn_frame_difference_cache_worker.py tests/test_run_nlspn_frame_difference_cache_pilot.py
    git commit -m "feat: benchmark NLSPN frame-difference cache variants"

### Task 6: Run real-model smoke validation

**Files:**
- Modify only the smallest proven implementation/test file if a defect is reproduced

- [ ] **Step 1: Run a four-frame smoke pilot**

    python scripts/run_nlspn_frame_difference_cache_pilot.py \
      --clips 1:4 --calibration-clips 1:4 --heldout-clips 1:4 \
      --warmup-repeats 1 --timed-repeats 1 \
      --output-dir /workspace/VoxelNet/nlspn_frame_difference_cache/smoke_0001_0004 \
      --device cuda:0

Expected: all three variants, 24 sweep rows, full plus three timed paths,
six final files, and no RAFT/tensor cache.

- [ ] **Step 2: Diagnose any real failure before changing code**

Use systematic-debugging: reproduce the smallest failing tensor/config,
identify the first wrong boundary, add a regression test that fails for that
root cause, and implement one minimum fix. Do not change thresholds, split,
quality gate, or output count to obtain a pass.

- [ ] **Step 3: Re-run full tests**

    python -m pytest -q
    git diff --check

- [ ] **Step 4: Commit a proven integration fix if files changed**

    git add scripts tests
    git commit -m "fix: validate real frame-difference cache execution"

### Task 7: Run the formal 256-frame pilot and report it

**Files:**
- Create: docs/2026-08-18-nlspn-frame-difference-cache-pilot-results.md
- Output: /workspace/VoxelNet/nlspn_frame_difference_cache/BeachApartmentInterior_My_ir/pilot_256/

- [ ] **Step 1: Run the approved formal command**

    python scripts/run_nlspn_frame_difference_cache_pilot.py \
      --output-dir /workspace/VoxelNet/nlspn_frame_difference_cache/BeachApartmentInterior_My_ir/pilot_256 \
      --device cuda:0

Expected: calibration on the first four clips, held-out evaluation on the
last four, one warm-up, five repeats, four timed paths, and six final files.

- [ ] **Step 2: Independently recompute quality and speed**

From frame_metrics.csv repeat-zero rows, recompute float64 SSE/valid pooled
RMSE for calibration, held-out, and all frames. From all repeats, recompute
full_total_ms / variant_total_ms. Assert exact agreement with summary.json,
24 complete sweep rows, 5,120 frame rows, finite metrics, and no tensor files.

- [ ] **Step 3: Write the measured repository report**

Record selected B/C configs without post-holdout changes; calibration,
held-out, all-frame and local quality; stable/change coverage; full/I/P
latency; measured speedups; startup/warm-up/memory; comparison to RAFT-GOP2;
and whether each variant passes <=1.01 and >1.0x.

- [ ] **Step 4: Commit the report**

    git add docs/2026-08-18-nlspn-frame-difference-cache-pilot-results.md
    git commit -m "docs: report NLSPN frame-difference cache pilot"

### Task 8: Final verification and branch handoff

**Files:**
- Verify only

- [ ] **Step 1: Run fresh repository verification**

    python -m pytest -q
    git diff --check
    git status --short

- [ ] **Step 2: Re-run final artifact integrity checks**

Verify six exact nonempty files, metadata complete, 24 sweep rows, 5,120
frame rows, four paths, five repeats, calibration/held-out/all quality,
independent speedups, checkpoint identity, no RAFT construction, and no tensor
cache.

- [ ] **Step 3: Use finishing-a-development-branch**

Identify agent/model-semantic-adapters as the base branch, summarize measured
positive or negative results, and offer the four required integration choices
without automatic merge or discard.
