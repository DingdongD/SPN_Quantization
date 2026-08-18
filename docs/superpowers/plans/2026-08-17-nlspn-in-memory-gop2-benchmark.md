# NLSPN Pure-In-Memory GOP2 End-to-End Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and run a single-process, pure-in-memory NLSPN GOP=2 online pipeline that uses official RAFT-Small motion estimation and reports measured CPU-memory-to-CPU-memory speedup while keeping pooled RMSE degradation within 1%.

**Architecture:** A PyTorch-1.10-compatible RAFT-Small module and frozen NLSPN live in the same completionformer-py37 process and retain temporal tensors on one GPU. A current-environment orchestrator verifies RAFT parity through a subprocess pipe, launches the benchmark worker, and validates final artifacts; no intermediate prediction, flow, guidance, or confidence tensor is persisted.

**Tech Stack:** Python 3.7/3.11, NumPy, PyTorch 1.10.1 and 2.7.1, Torchvision RAFT-Small C_T_V2 weights, frozen NLSPN ResNet-34 with 18 TGASS propagation steps, pytest, CSV/JSON/Markdown.

---

## File Structure

- Create scripts/raft_small_compat.py for the official RAFT-Small architecture, preprocessing, strict weights, and backward-flow inference.
- Create scripts/nlspn_in_memory_gop2.py for GPU state, I/P inference, synchronized timing, and metrics.
- Create scripts/run_nlspn_in_memory_gop2_worker.py for the old-environment benchmark.
- Create scripts/run_nlspn_in_memory_gop2_benchmark.py for parity, launch, and final validation.
- Create matching tests/test_*.py files for each module.
- Create docs/2026-08-17-nlspn-in-memory-gop2-benchmark-results.md after the formal run.

### Task 1: Add the PyTorch-1.10-compatible official RAFT-Small module

**Files:**
- Create: scripts/raft_small_compat.py
- Create: tests/test_raft_small_compat.py

- [ ] **Step 1: Write failing architecture and preprocessing tests**

    import pytest
    import torch
    from scripts import raft_small_compat as compat

    def test_raft_small_matches_official_parameter_contract():
        model = compat.raft_small()
        weights = torch.load(str(compat.DEFAULT_WEIGHT_PATH), map_location="cpu")
        assert sum(p.numel() for p in model.parameters()) == 990162
        assert set(model.state_dict()) == set(weights)

    def test_prepare_pair_pads_normalizes_and_preserves_direction():
        previous = torch.zeros(1, 3, 228, 304)
        current = torch.ones(1, 3, 228, 304)
        image1, image2 = compat.prepare_backward_pair(current, previous)
        assert image1.shape == image2.shape == (1, 3, 232, 304)
        assert torch.all(image1 == 1.0)
        assert torch.all(image2 == -1.0)

    def test_load_official_weights_is_strict(tmp_path):
        bad = tmp_path / "bad.pth"
        torch.save({}, str(bad))
        with pytest.raises(RuntimeError):
            compat.load_official_weights(compat.raft_small(), bad)

- [ ] **Step 2: Run python -m pytest -q tests/test_raft_small_compat.py and verify RED**

Expected: collection fails because scripts.raft_small_compat does not exist.

- [ ] **Step 3: Implement the compatibility module**

Port only RAFT-Small inference definitions from:

    /opt/conda/lib/python3.11/site-packages/torchvision/models/optical_flow/raft.py
    /opt/conda/lib/python3.11/site-packages/torchvision/models/optical_flow/_utils.py

Include Conv2dNormActivation, BottleneckBlock, FeatureEncoder, MotionEncoder,
ConvGRU, RecurrentBlock, FlowHead, UpdateBlock, CorrBlock, RAFT, grid_sample,
make_coords_grid, upsample_flow, and raft_small. Preserve official attribute
names so C_T_V2 loads strictly. Add provenance and applicable license notice.
Do not import Torchvision optical-flow modules.

Expose:

    DEFAULT_WEIGHT_PATH = Path(
        "/root/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth")
    EXPECTED_WEIGHT_SHA256 = (
        "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27")

    def load_official_weights(model, path=DEFAULT_WEIGHT_PATH):
        if file_sha256(path) != EXPECTED_WEIGHT_SHA256:
            raise RuntimeError("official RAFT-Small weight digest mismatch")
        model.load_state_dict(
            torch.load(str(path), map_location="cpu"), strict=True)
        return model

    def prepare_backward_pair(current_rgb, previous_rgb):
        current = F.pad(current_rgb, (0, 0, 2, 2), mode="replicate")
        previous = F.pad(previous_rgb, (0, 0, 2, 2), mode="replicate")
        return (current * 2.0 - 1.0).contiguous(), (
            previous * 2.0 - 1.0).contiguous()

    def predict_backward_flow(model, current_rgb, previous_rgb):
        current, previous = prepare_backward_pair(current_rgb, previous_rgb)
        flow = model(current, previous, num_flow_updates=12)[-1][:, :, 2:-2]
        if flow.shape[-2:] != (228, 304) or not torch.isfinite(flow).all():
            raise ValueError("RAFT-Small returned invalid backward flow")
        return flow

Use a torch.meshgrid helper that tries indexing="ij" and falls back only if
PyTorch rejects the keyword.

- [ ] **Step 4: Run both environment tests and verify GREEN**

    python -m pytest -q tests/test_raft_small_compat.py
    conda run -n completionformer-py37 python -m pytest -q tests/test_raft_small_compat.py

- [ ] **Step 5: Commit**

    git add scripts/raft_small_compat.py tests/test_raft_small_compat.py
    git commit -m "feat: add compatible official RAFT-Small inference"

### Task 2: Build the pure-memory online NLSPN engine

**Files:**
- Create: scripts/nlspn_in_memory_gop2.py
- Create: tests/test_nlspn_in_memory_gop2.py

- [ ] **Step 1: Write failing state and schedule tests**

    def test_fixed_gop2_schedule():
        assert [online.frame_kind(i) for i in range(6)] == [
            "I", "P", "I", "P", "I", "P"]

    def test_p_frame_requires_live_state():
        engine = online.InMemoryGOP2Engine(FakeNLSPN(), FakeRAFT(), "cpu")
        with pytest.raises(RuntimeError, match="state"):
            engine.infer_p(torch.zeros(3, 4, 5), torch.zeros(4, 5), 1)

    def test_reset_drops_every_temporal_tensor():
        engine = online.InMemoryGOP2Engine(FakeNLSPN(), FakeRAFT(), "cpu")
        engine.state = online.OnlineState(*make_state_tensors(), local_index=1)
        engine.reset()
        assert engine.state is None

- [ ] **Step 2: Run the focused test and verify RED**

Expected: collection fails because scripts.nlspn_in_memory_gop2 is missing.

- [ ] **Step 3: Implement state and fixed scheduling**

Create OnlineState with previous_rgb, previous_depth, previous_guidance,
previous_confidence, and local_index. frame_kind returns I for even local
indices and P for odd indices. Reject invalid indices and a P call that does
not immediately follow stored state.

- [ ] **Step 4: Add a failing I-frame test, then implement I inference**

The fake-model test asserts infer_i transfers CPU RGB/sparse input, calls full
NLSPN once, returns CPU prediction, and retains GPU RGB/prediction/guidance/
confidence. Synchronize, start perf_counter, perform H2D and inference, copy
prediction to CPU, synchronize, then stop the clock.

- [ ] **Step 5: Add a failing P-frame propagation-equivalence test**

Use zero flow and a fake propagation layer returning its seed. Verify P output
equals previous_depth plus signed_sparse_seed and that state contains current
RGB, reconstructed depth, and warped guidance/confidence tensors.

- [ ] **Step 6: Implement P inference with the validated operator**

    flow = raft_small_compat.predict_backward_flow(
        self.raft, current_rgb, self.state.previous_rgb)
    base, _ = residual.backward_warp(self.state.previous_depth, flow)
    guidance, _ = residual.backward_warp(
        self.state.previous_guidance, flow)
    confidence, _ = residual.backward_warp(
        self.state.previous_confidence, flow)
    seed = torch.zeros_like(base)
    mask = current_sparse > 0.0
    seed[mask] = current_sparse[mask] - base[mask]
    dense = self.nlspn.prop_layer(
        seed, guidance, confidence, None, current_rgb)[0]
    prediction = torch.clamp(base + dense, 0.0, residual.MAX_DEPTH)

Validate 304 x 228, exactly 500 sparse points, finite values, strict sequence,
and fixed I/P method use. Apply the same synchronized CPU-to-CPU boundary.

- [ ] **Step 7: Verify and commit**

    python -m pytest -q tests/test_nlspn_in_memory_gop2.py
    git add scripts/nlspn_in_memory_gop2.py tests/test_nlspn_in_memory_gop2.py
    git commit -m "feat: add pure-memory NLSPN GOP2 engine"

### Task 3: Add timing and quality aggregation

**Files:**
- Modify: scripts/nlspn_in_memory_gop2.py
- Modify: tests/test_nlspn_in_memory_gop2.py

- [ ] **Step 1: Write a failing latency-statistics test**

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

Also test rejection of empty, negative, and non-finite latency lists. Run it
and verify failure because latency_summary is absent.

- [ ] **Step 2: Implement latency aggregation**

Use float64. Report count, total, mean, P50, P95, min, max, and FPS. Aggregate
full, GOP2, I, and P separately. Speedup equals full total ms / GOP2 total ms.

- [ ] **Step 3: Add failing pooled/local quality tests**

Construct clips whose pooled ratio passes despite one bad frame. Assert the
formal gate follows pooled squared-error accumulation and rows retain the
local failure and worst-frame identity.

- [ ] **Step 4: Implement quality aggregation**

Implement frame_quality_rows, clip_quality_rows, and benchmark_summary using
float64 errors. Production requires 256 quality frames and five complete
repeats. passes is true only when pooled GOP2 RMSE / full RMSE <= 1.01.

- [ ] **Step 5: Verify and commit**

    python -m pytest -q tests/test_nlspn_in_memory_gop2.py
    git add scripts/nlspn_in_memory_gop2.py tests/test_nlspn_in_memory_gop2.py
    git commit -m "feat: aggregate GOP2 latency and quality"

### Task 4: Add official RAFT parity through an in-memory pipe

**Files:**
- Create: scripts/run_nlspn_in_memory_gop2_benchmark.py
- Create: tests/test_run_nlspn_in_memory_gop2_benchmark.py

- [ ] **Step 1: Write failing pipe and parity tests**

    def test_raft_parity_records_hard_tolerances():
        reference = np.ones((1, 2, 228, 304), np.float32)
        result = runner.compare_raft_flows(reference, reference.copy())
        assert result["passes"] is True
        assert result["max_abs_tolerance"] == 1e-3
        assert result["rmse_tolerance"] == 1e-4

    def test_raft_parity_rejects_changed_flow():
        reference = np.zeros((1, 2, 228, 304), np.float32)
        changed = reference.copy()
        changed[..., 10, 10] = 0.01
        with pytest.raises(RuntimeError, match="parity"):
            runner.require_raft_parity(reference, changed)

Test encode_array/decode_array with float32 flow. Run and verify collection
fails because the runner is missing.

- [ ] **Step 2: Implement the pipe protocol and parity gate**

The current process loads frames 0001/0002 and runs Raft_Small_Weights.DEFAULT.
The old worker independently reads the same RGB and returns one stdout JSON
line containing a base64 NumPy byte stream, shape, dtype, weight SHA, and
environment metadata. No temporary tensor file is allowed. Require maximum
absolute difference <= 1e-3 pixels and RMSE <= 1e-4 pixels.

- [ ] **Step 3: Implement non-shell worker command construction**

    [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(WORKER_PATH), "--stage", stage,
        "--data-root", str(data_root), "--scene", scene,
        "--checkpoint", str(checkpoint), "--args-json", str(args_json),
        "--raft-weights", str(raft_weights), "--device", device,
    ]

Set PYTHONPATH to the repository, NLSPN source, and deformconv paths.

- [ ] **Step 4: Verify and commit**

    python -m pytest -q tests/test_run_nlspn_in_memory_gop2_benchmark.py
    git add scripts/run_nlspn_in_memory_gop2_benchmark.py tests/test_run_nlspn_in_memory_gop2_benchmark.py
    git commit -m "feat: verify compatible RAFT through memory pipe"

### Task 5: Implement the single-process benchmark worker

**Files:**
- Create: scripts/run_nlspn_in_memory_gop2_worker.py
- Create: tests/test_run_nlspn_in_memory_gop2_worker.py
- Modify: scripts/run_nlspn_in_memory_gop2_benchmark.py
- Modify: tests/test_run_nlspn_in_memory_gop2_benchmark.py

- [ ] **Step 1: Write failing in-memory dataset tests**

With an injected frame loader, assert eight clips match PILOT_CLIPS, every
payload validates, and each clip has its own common fixed 500-point mask.
Patch NPZ/NPY writers to raise if the loader tries to persist tensors.

- [ ] **Step 2: Implement worker stages and input loading**

Support raft-parity and benchmark. Load all RGB/EXR frames and preprocess
before timing, build sparse input with seed 2026, then strictly load both
models once. Validate NLSPN ResNet-34, 18 iterations, TGASS, and
preserve_input=False.

- [ ] **Step 3: Write failing warm-up/repeat/schedule tests**

With CPU fake models, assert warm-up metrics are excluded, five timed passes
produce 1,280 latency records per path, state resets at every clip/pass, and
only first-pass predictions feed quality.

- [ ] **Step 4: Implement formal benchmark execution**

Run one warm-up for each path, then five full and five GOP2 passes. Track peak
allocated/reserved CUDA memory separately. Retain only first-pass predictions.
Allow injected fake models/clock in tests; production synchronizes CUDA.
Measure checkpoint/model construction and each complete warm-up pass
separately from steady-state latency. Store one frame_metrics.csv row per
(path, repeat, clip, frame), so each path has exactly 1,280 timed rows; attach
RMSE and valid-pixel fields only to repeat zero to avoid counting quality five
times.

- [ ] **Step 5: Write a failing artifact-scope test**

Run fake benchmark in tmp_path and require exactly run_metadata.json,
summary.json, frame_metrics.csv, clip_summary.csv, report.md. Forbid NPZ, NPY,
prediction, flow, guidance, confidence, and cache files.

- [ ] **Step 6: Implement outputs and completion validation**

Use atomic writers. Metadata begins complete=false and becomes true only after
five non-empty, consistent files exist. The report distinguishes measured
speedup from projected 1.57x. The orchestrator validates five repeats, 1,280
records per path, 256 quality frames, digests, artifacts, and quality gate.

- [ ] **Step 7: Run all new tests and commit**

    python -m pytest -q tests/test_raft_small_compat.py tests/test_nlspn_in_memory_gop2.py tests/test_run_nlspn_in_memory_gop2_worker.py tests/test_run_nlspn_in_memory_gop2_benchmark.py
    git add scripts/run_nlspn_in_memory_gop2_benchmark.py scripts/run_nlspn_in_memory_gop2_worker.py tests/test_run_nlspn_in_memory_gop2_benchmark.py tests/test_run_nlspn_in_memory_gop2_worker.py
    git commit -m "feat: benchmark in-memory NLSPN GOP2 end to end"

### Task 6: Run real-GPU compatibility and smoke validation

**Files:**
- Modify only the smallest implementation/test file if a defect is reproduced

- [ ] **Step 1: Run official RAFT parity**

    python scripts/run_nlspn_in_memory_gop2_benchmark.py --stage raft-parity --device cuda:0

Expected: [1,2,228,304], official digest, max error <= 1e-3, RMSE <= 1e-4,
passes=true.

- [ ] **Step 2: Run a two-frame real-model smoke benchmark**

    python scripts/run_nlspn_in_memory_gop2_benchmark.py --stage all --clips 1:2 --warmup-repeats 1 --timed-repeats 1 --output-dir /workspace/VoxelNet/nlspn_in_memory_gop2/smoke_0001_0002 --device cuda:0

Expected: one I and one P complete, exactly five final files, no tensor cache.

- [ ] **Step 3: Diagnose any real failure before fixing**

Use systematic debugging: reproduce the smallest discrepancy, identify the
first wrong tensor or boundary, add a failing regression test, and make the
minimum fix. Do not relax parity or quality gates.

- [ ] **Step 4: Re-run the full suite and commit a proven fix if needed**

    python -m pytest -q
    git diff --check

If files changed, commit with message "fix: validate real in-memory GOP2 execution".

### Task 7: Run the formal benchmark and report measured results

**Files:**
- Create: docs/2026-08-17-nlspn-in-memory-gop2-benchmark-results.md
- Output: /workspace/VoxelNet/nlspn_in_memory_gop2/BeachApartmentInterior_My_ir/pilot_256/

- [ ] **Step 1: Run the approved command**

    python scripts/run_nlspn_in_memory_gop2_benchmark.py --stage all --output-dir /workspace/VoxelNet/nlspn_in_memory_gop2/BeachApartmentInterior_My_ir/pilot_256 --device cuda:0

Expected: eight clips, one warm-up, five timed passes, 256 quality frames,
1,280 timed frames per path, FP32, GOP2, no fallback/cache.

- [ ] **Step 2: Independently recompute final metrics**

From final CSV, verify pooled squared-error accumulation, quality_ratio =
RMSE_GOP2 / RMSE_full, and speedup = sum(full_latency_ms) /
sum(gop2_latency_ms). Assert agreement with summary.json, five samples per
frame, finite metrics, and ratio <= 1.01.

- [ ] **Step 3: Verify artifact scope**

    find /workspace/VoxelNet/nlspn_in_memory_gop2/BeachApartmentInterior_My_ir/pilot_256 -maxdepth 1 -type f -printf '%f\n' | sort

Expected exactly the five approved files.

- [ ] **Step 4: Write and commit the measured report**

Record identities, parity errors, full/I/P/GOP2 P50/P95/mean/FPS, measured
speedup versus projected 1.57x, pooled/local quality, worst frame, memory,
startup/warm-up, and the pooled-gate limitation.

    git add docs/2026-08-17-nlspn-in-memory-gop2-benchmark-results.md
    git commit -m "docs: report in-memory NLSPN GOP2 benchmark"

### Task 8: Final verification and handoff

**Files:**
- Verify only

- [ ] **Step 1: Run fresh verification**

    python -m pytest -q
    git diff --check
    git status --short

Expected: all tests pass, no whitespace errors, clean worktree.

- [ ] **Step 2: Recheck final artifact integrity**

Verify metadata complete, five non-empty files, independent RMSE ratio <= 1.01,
1,280 timed frames per path, and no tensor cache.

- [ ] **Step 3: Use finishing-a-development-branch**

Identify the base branch, summarize measured quality/speed, and offer the four
required integration choices without merging or discarding automatically.
