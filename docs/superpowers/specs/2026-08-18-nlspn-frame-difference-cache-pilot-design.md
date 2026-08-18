# NLSPN Frame-Difference Cache Pilot Design

## Objective

Evaluate whether adjacent-frame differences can replace RAFT-Small in the
causal frozen-NLSPN GOP2 pipeline while preserving the approved pooled RMSE
constraint and producing a real end-to-end speedup.

Three causal variants are required:

1. Zero-flow residual propagation.
2. RGB-difference-gated depth cache.
3. Global-translation compensation followed by RGB-difference-gated cache.

The frozen NLSPN architecture, weights, sparse-depth input, and original
propagation operator remain unchanged. No residual/refinement network,
fine-tuning, or automatic full-model fallback is allowed.

The primary quality gate remains:

    pooled_RMSE_variant / pooled_RMSE_full <= 1.01

## Data and Evaluation Split

Use the same 256 frames, geometry, sparse-depth policy, and clip boundaries as
the prior pure-memory benchmark.

Calibration clips:

- 0001--0032
- 0282--0313
- 0563--0594
- 0844--0875

Held-out clips:

- 1126--1157
- 1407--1438
- 1688--1719
- 1969--2000

Each split contains 128 frames. Threshold and dilation choices may use only
the calibration clips. After selection, configurations are frozen and run on
the held-out clips and all 256 frames. The formal gate is computed over all
256 frames; held-out quality is reported separately to expose overfitting.

The schedule is fixed I, P, I, P inside each 32-frame clip. Temporal state is
cleared at every clip boundary. Inputs are 304 x 228 FP32 RGB plus exactly 500
fixed sparse-depth points per clip using seed 2026.

## Shared P-Frame Residual Decoder

All three variants use the original frozen NLSPN propagation layer. Given a
base depth and cached guidance/confidence:

    seed = current_sparse - base at current sparse locations
    dense_residual = original_NLSPN_prop_layer(
        seed,
        cached_guidance,
        cached_confidence,
        fixed_depth=None,
        current_rgb,
    )
    candidate = clamp(base + dense_residual, 0, 10 m)

The variants differ only in construction of the base and final cache mask.

## Variant A: Zero-Flow Residual Propagation

Assume current pixels correspond to the same coordinates in the previous
frame.

- Base depth: previous prediction without spatial warp.
- Guidance/confidence: previous cached tensors without warp.
- Output: the dense residual candidate over the full image.

This has no tunable photometric parameters. It measures the quality lower
bound and the speed available when all motion estimation is removed.

## Variant B: RGB-Difference Cache

### Photometric mask

Current and previous RGB already reside on the GPU in the online pipeline.
For each frame:

1. Apply a 3 x 3 average blur independently to both RGB tensors.
2. Compute the maximum absolute difference across the three channels.
3. Mark pixels above the photometric threshold as changed.
4. Dilate the changed mask with max pooling.
5. At the 500 current sparse locations, mark a depth inconsistency when the
   absolute difference between current sparse depth and previous predicted
   depth exceeds 0.02 m.
6. Dilate the sparse-inconsistency mask with the same radius.
7. Stable pixels are the complement of the union of the two dilated masks.

Threshold candidates are fixed before the run:

- 2/255
- 4/255
- 8/255
- 16/255

Changed-mask dilation radii are:

- 2 pixels
- 4 pixels
- 8 pixels

The calibration sweep therefore contains 12 configurations.

### Output

Construct the same zero-flow residual candidate as Variant A, then blend:

    output = previous_depth on stable pixels
    output = residual_candidate on changed pixels

The propagation call remains dense and fixed for this pilot. The mask tests
whether cached output reuse preserves quality; it does not claim tile-sparse
propagation speed. Speed comes from eliminating RAFT-Small.

## Variant C: Global Translation plus RGB-Difference Cache

Estimate one causal two-dimensional translation between previous and current
RGB using phase correlation:

1. Convert RGB to grayscale.
2. Average-pool by a factor of four.
3. Compute normalized phase correlation with torch.fft in the legacy NLSPN
   process.
4. Select the integer peak and unwrap cyclic coordinates.
5. Scale the displacement back to 304 x 228 coordinates.
6. Construct a constant backward flow and reuse the existing backward-warp
   operator for previous RGB, depth, guidance, and confidence.
7. Mark out-of-bounds pixels as changed.

After translation compensation, compute the same photometric and
sparse-depth masks as Variant B, using the same 12 threshold/radius
candidates. Stable pixels reuse translated previous depth; changed pixels use
the translated-base residual candidate.

No learned motion model, OpenCV registration, future frame, or full-NLSPN
fallback is allowed. The estimated translation and its direction must be
validated on synthetic shifted tensors before real inference.

## Configuration Selection

Variants B and C are selected independently.

For every candidate, run one untimed causal quality pass on the four
calibration clips. Select the candidate with the lowest calibration pooled
RMSE. Ties within 1e-9 m are resolved by:

1. lower threshold;
2. larger dilation radius.

This deterministic tie rule favors conservative cache reuse. Do not select by
held-out or all-frame quality.

Variant A has no selection step.

For every selected configuration, report:

- calibration pooled RMSE and ratio;
- held-out pooled RMSE and ratio;
- all-frame pooled RMSE and ratio;
- per-clip and per-frame distributions;
- stable cache coverage over P frames;
- RGB-changed, sparse-inconsistent, and out-of-bounds coverage;
- worst frame and number of frames over the 1% local ratio.

## Pure-Memory End-to-End Benchmark

The benchmark process loads only frozen NLSPN. RAFT-Small must not be
constructed or loaded.

The timing boundary remains CPU-memory input to CPU-memory predicted depth and
includes:

- RGB/sparse H2D transfer;
- complete NLSPN for I/reference frames;
- frame difference or phase correlation for P frames;
- mask construction and dilation;
- warp for Variant C;
- original NLSPN propagation;
- state update;
- predicted-depth D2H transfer;
- CUDA synchronization.

It excludes source image decoding, preprocessing, model loading, threshold
sweep, warm-up, report generation, and disk I/O.

For the full reference and each final variant:

- run one complete unmeasured warm-up;
- run five complete timed repeats;
- reset state before each pass and clip;
- retain only one deterministic prediction pass for quality.

Report full, I, P, and overall mean/P50/P95 latency, FPS, total time, measured
speedup, startup time, warm-up time, and peak allocated/reserved CUDA memory.

The previous RAFT-GOP2 result, 0.526705x with quality ratio 1.006933, is a
reported external reference and is not rerun.

## Implementation Boundaries

Add new frame-difference cache modules and runners rather than changing the
existing validated RAFT-GOP2 scripts. Reuse:

- input loading and fixed sparse-depth creation;
- OnlineState and full-I-frame inference contracts;
- backward_warp;
- original NLSPN prop_layer;
- quality/latency aggregation and atomic final writers where their semantics
  match.

The new online engine must expose explicit reset and separate P-frame methods
for zero-flow, RGB-difference, and translated RGB-difference inference.

No intermediate depth, RGB, mask, FFT, guidance, confidence, or prediction
tensor may be persisted. Only final CSV/JSON/Markdown artifacts are allowed.

## Validation and Failure Behavior

Automated tests must cover:

- 3 x 3 RGB blur and max-channel difference;
- threshold inclusivity and changed-mask dilation;
- 0.02 m sparse inconsistency mask;
- stable-mask composition;
- stable/changed output blending;
- zero-flow residual equivalence;
- phase-correlation direction and integer translation on synthetic inputs;
- constant-flow warp and out-of-bounds invalidation;
- deterministic 12-candidate enumeration and tie breaking;
- calibration-only selection;
- clip and pass state reset;
- fixed GOP2 schedule;
- CPU-memory output and synchronized timing boundary;
- no RAFT model construction in the benchmark worker;
- final artifact scope and independent metric recomputation.

Fail immediately on non-finite data, wrong geometry, wrong sparse-point count,
invalid state order, missing mask metrics, unsupported FFT output,
incomplete threshold sweep, changed frozen-NLSPN configuration, or incomplete
final artifacts. Inputs and masks are not silently resized or repaired.

## Outputs

Write a dedicated directory:

    /workspace/VoxelNet/nlspn_frame_difference_cache/
    BeachApartmentInterior_My_ir/pilot_256/

Approved final files:

- run_metadata.json
- summary.json
- threshold_sweep.csv
- frame_metrics.csv
- clip_summary.csv
- report.md

The repository also receives a measured-results Markdown report. The report
must distinguish measured results from projections and clearly state whether
each of the three variants passes the formal quality gate and achieves
speedup greater than 1.0.

## Acceptance Criteria

The pilot is complete when:

1. All three variants run causally on the approved 256 frames.
2. Variants B/C select parameters only on the calibration clips.
3. Held-out and all-frame quality are independently reported.
4. At least one warm-up and five timed repeats complete for each final path.
5. CPU-to-CPU latency, throughput, speedup, startup, and memory are measured.
6. No intermediate tensor cache or RAFT model is used.
7. Automated tests and independent final-artifact checks pass.

Passing the engineering experiment does not require a variant to satisfy the
1% RMSE gate or exceed 1.0x speedup. A negative result is valid if measured
and reported without changing thresholds after held-out evaluation.
