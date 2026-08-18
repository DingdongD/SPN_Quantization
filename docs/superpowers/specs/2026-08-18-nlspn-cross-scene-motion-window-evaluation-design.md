# NLSPN Cross-Scene Motion-Window Evaluation Design

## Goal

Evaluate full NLSPN and the three causal frame-difference methods on one
automatically selected, motion-rich five-frame window from every other scene
in the local dataset. Preserve the formal `pilot_256` experiment and use its
fixed selected cache configurations without scene-specific tuning.

## Scope

The six evaluation scenes are:

- `bedroom_ir`
- `livingroom_ir`
- `room3`
- `room4`
- `room6`
- `room7`

`BeachApartmentInterior_My_ir` is excluded because it already has formal and
five-frame results. Every selected window contains exactly five consecutive
frame IDs and is treated as one causal `I,P,I,P,I` clip.

## Motion-window selection

For each scene, enumerate frame IDs that have both `rgb/%04d.jpg` and
`depth/Image%04d.exr`. Reject scenes without at least five consecutive complete
frames.

Decode RGB JPEGs once in frame order with Pillow, convert to grayscale, resize
to 76 x 57 with bilinear interpolation, and normalize to `[0,1]`. For each
adjacent consecutive pair, compute mean absolute grayscale difference. A
candidate five-frame window receives the mean of its four adjacent-pair
scores. Select the highest-scoring window; exact score ties choose the earliest
start ID. Nonconsecutive frame IDs cannot belong to a candidate window.

The selector records all six selected start/end IDs, window scores, and the
four pair scores before GPU inference. Window selection uses RGB only and does
not inspect model predictions or ground truth values.

## Fixed model and input protocol

- Frozen ResNet-34 NLSPN, iteration 18, TGASS, `preserve_input=false`.
- Existing checkpoint and preprocessing path at 304 x 228.
- Exactly 500 sparse-depth points per frame, seed 2026.
- Full NLSPN on all five frames.
- Parameter-free zero-flow residual path.
- RGB-difference and global-translation paths use the two rows marked selected
  in the existing formal `threshold_sweep.csv`.
- The expected fixed values are threshold `2/255` and dilation radius 8 for
  both parameterized methods. Any mismatch is recorded and rejected rather
  than silently recalibrated.
- No RAFT, fine-tuning, fallback, or per-scene parameter search.

The legacy worker loads the NLSPN model once, processes all six scenes
sequentially on one GPU, and resets temporal state before every method and
scene. Timing is one real inference pass and is reported as informational, not
as the five-repeat formal benchmark speedup.

## Reusable visualization generalization

The existing five-frame visualization helpers currently require frame IDs
0001-0005. Generalize their validation, labels, archive checks, and metrics to
accept any strictly consecutive five-frame ID sequence. Geometry, method
order, color ranges, sparse count, and `I,P,I,P,I` scheduling remain fixed.
The existing 0001-0005 command and tests must continue to work unchanged.

## Output structure

Write under:

`/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows`

Each scene subdirectory contains exactly the established six visualization
artifacts:

1. `nlspn_frame_difference_depth_comparison.png`
2. `nlspn_frame_difference_error_comparison.png`
3. `predictions.npz`
4. `frame_metrics.csv`
5. `run_metadata.json`
6. `worker.log`

Depth figures use the common 0-10 m scale. Each scene's error figure uses its
own documented common valid-pixel 99th-percentile scale across all methods and
five frames. Metadata records the selected IDs, motion score, pair scores,
checkpoint/config digests, selected configs, input digest, and completion.

The output root contains exactly four aggregate files in addition to the six
scene directories:

1. `selected_windows.csv`
2. `cross_scene_summary.csv`
3. `report.md`
4. `run_metadata.json`

`selected_windows.csv` has one row per scene. `cross_scene_summary.csv` has
one row per scene and method (24 rows), with pooled RMSE, MAE, full-NLSPN RMSE
ratio, `ratio <= 1.01`, valid pixels, mean latency, and total latency. The
Markdown report includes scene tables and all-scene pooled results.

## Data flow

1. The current-environment launcher scans all six scenes and deterministically
   selects windows.
2. It writes the selected-window manifest to a temporary JSON argument and
   snapshots every formal-pilot file digest.
3. One `completionformer-py37` worker loads the fixed model and selected
   configs, then loads and infers each scene window.
4. The worker writes each scene's predictions, metrics, figures, and metadata,
   plus aggregate CSV/JSON/Markdown files.
5. The launcher replaces scene log placeholders with captured worker output,
   independently validates all arrays, figures, metrics, windows, and digests,
   and confirms the formal directory is byte-for-byte unchanged.

No intermediate guidance, confidence, flow, FFT, or propagation tensors are
written.

## Metrics

For every scene/method pair, pool squared error and absolute error over valid
pixels across the five frames. Report:

- RMSE and MAE in meters;
- RMSE divided by the same-scene full-NLSPN RMSE;
- whether the ratio is at most 1.01;
- valid pixel count;
- mean and total per-frame latency.

The aggregate report additionally pools errors over all six scenes. Because
only one pass is timed, no claim of stable end-to-end speedup is made from this
cross-scene visualization run.

## Error handling

- Reject missing, duplicate, or nonconsecutive RGB/depth frame pairs.
- Reject fewer than five complete consecutive frames.
- Reject non-finite motion scores or unreadable JPEGs.
- Reject formal selected configs that are missing, ambiguous, or differ from
  `2/255, radius=8`.
- Reject selected EXR windows with invalid preprocessing payloads, prediction
  geometry mismatches, non-finite predictions, or sparse counts other than
  500.
- Mark scene and root metadata complete only after exact nonempty artifacts are
  present and consistent.
- On worker failure, preserve captured logs and do not mark root metadata
  complete.

## Testing and verification

Unit tests cover:

- deterministic rolling five-frame motion scores, gaps, and earliest-tie
  behavior;
- arbitrary consecutive five-frame payload validation and legacy 0001-0005
  compatibility;
- fixed selected-config enforcement;
- single model construction and six-scene scheduling;
- 24 aggregate metric rows and pooled ratio calculations;
- batch worker command construction with no RAFT argument;
- exact per-scene/root artifact scopes and independent validation.

Integration runs the six real scenes, visually inspects all 12 PNG figures,
recomputes summary metrics from the per-scene CSV/NPZ files, verifies the
formal directory digest snapshot, and runs the full repository plus legacy
Python-3.7 relevant tests.

## Success criteria

- Six deterministic motion-rich windows are selected and recorded.
- All four methods produce five finite predictions for every scene.
- Twelve readable comparison figures and six prediction archives exist.
- Per-scene and aggregate metrics independently recompute exactly.
- Fixed configs are identical across all scenes.
- Formal `pilot_256` files remain unchanged.
- The full repository and legacy-environment relevant tests pass.
