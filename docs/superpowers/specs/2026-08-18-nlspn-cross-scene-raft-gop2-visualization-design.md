# NLSPN Cross-Scene RAFT-GOP2 Visualization Design

## Goal

Extend the existing six-scene motion-window evaluation with a fifth prediction
method, `RAFT-GOP2`, using strict causal dense optical-flow compensation. Keep
the frozen NLSPN architecture and the six already-selected five-frame windows,
and update the established output path only after a complete staged run passes
independent validation.

## Scope

The evaluation keeps the existing scenes and selected windows:

- `bedroom_ir`: frames 1951-1955
- `livingroom_ir`: frames 3546-3550
- `room3`: frames 3385-3389
- `room4`: frames 1023-1027
- `room6`: frames 0276-0280
- `room7`: frames 3360-3364

The four existing methods remain unchanged:

1. Full NLSPN
2. Zero-flow
3. RGB-diff
4. Global-diff

`RAFT-GOP2` is appended as the fifth prediction method. The run uses the same
preprocessed RGB, ground truth, valid mask, 500 sparse-depth points, seed 2026,
and `I,P,I,P,I` schedule as the current comparison.

## Strict causal optical-flow method

Use the existing PyTorch-1.10-compatible RAFT-Small implementation and the
official cached RAFT-Small weights. Require the known official weight SHA-256
before inference and use exactly 12 flow updates.

For every P frame:

1. Give RAFT the current RGB and immediately preceding RGB.
2. Predict a dense backward flow whose samples map current-frame pixels into
   the preceding frame.
3. Backward-warp the preceding prediction, NLSPN guidance, and confidence into
   current-frame coordinates.
4. Form a zero dense residual seed, and at the 500 current sparse-depth
   locations insert `current_sparse_depth - warped_previous_depth`.
5. Run the frozen NLSPN propagation layer with the warped guidance and
   confidence, current RGB, and residual seed.
6. Add the propagated residual to the warped depth and clamp the result to the
   established 0-10 m interval.

For every I frame, run the full frozen NLSPN and refresh depth, guidance,
confidence, and RGB state. Reset all temporal state before each method and
scene. The method never reads a future frame and uses no RGB-difference mask,
global translation approximation, automatic fallback, scene calibration, or
fine-tuning.

## Model lifetime and inference organization

The legacy Python 3.7 worker loads the frozen NLSPN once and official
RAFT-Small once, then processes all six scenes sequentially. The existing
frame-difference engine produces the original four methods. A separate
`InMemoryGOP2Engine` shares the same frozen NLSPN instance and owns the RAFT
state for `RAFT-GOP2`.

For each scene, the worker loads the selected five-frame payload once. It runs
the original four prediction paths without changing their configurations, then
runs `RAFT-GOP2` from a clean temporal state. All five methods therefore use
identical RGB, sparse depth, ground truth, valid masks, and frame IDs.

## Visualization and per-scene artifacts

Each scene keeps the established six artifact names:

1. `nlspn_frame_difference_depth_comparison.png`
2. `nlspn_frame_difference_error_comparison.png`
3. `predictions.npz`
4. `frame_metrics.csv`
5. `run_metadata.json`
6. `worker.log`

The depth figure becomes a five-row, six-column grid:

- Ground truth
- Full NLSPN
- Zero-flow
- RGB-diff
- Global-diff
- RAFT-GOP2

All depth panels retain the common 0-10 m color scale. The absolute-error
figure becomes a five-row, five-column grid containing the five prediction
methods. Its panels share one scene-specific valid-pixel 99th-percentile error
scale across all methods and frames.

`predictions.npz` adds one finite float32 `raft_gop2` array with shape
`[5,228,304]`. Existing input and prediction arrays remain unchanged.
`frame_metrics.csv` grows from 20 to 25 unique method-frame rows.

Scene metadata additionally records:

- RAFT weight path and SHA-256;
- RAFT implementation and 12 flow updates;
- backward causal direction `current_to_previous`;
- NLSPN and RAFT model load counts;
- the unchanged fixed cache configurations;
- selected frame IDs, motion score, checkpoint and input digests.

No flow, guidance, confidence, residual, or other intermediate tensor is
written to disk.

## Root artifacts and metrics

The updated output path remains:

`/workspace/VoxelNet/nlspn_frame_difference_cache/cross_scene_motion_windows`

The four root artifact names remain unchanged:

1. `selected_windows.csv`
2. `cross_scene_summary.csv`
3. `report.md`
4. `run_metadata.json`

`selected_windows.csv` remains byte-equivalent in content. The summary grows
from 24 to 30 unique scene-method rows. For every scene and method, pool error
over valid pixels across all five frames and report RMSE, MAE, valid-pixel
count, total latency, RMSE divided by the same-scene Full NLSPN RMSE, and
whether the ratio is at most 1.01.

The root report also pools all six scenes for every method. RAFT-GOP2 latency
includes RAFT inference, three state warps, sparse residual construction, and
NLSPN propagation. Because this visualization run times only one pass and
includes cold-start effects, latency is informational and does not support a
stable speedup claim.

Root metadata records `nlspn_model_load_count=1`,
`raft_model_load_count=1`, the official RAFT weight identity, and a completion
marker written only after all 30 summary rows and six scene directories pass
validation.

## Safe replacement

Do not write the new five-method run directly into the current completed
directory. Generate it in a sibling staging directory on the same filesystem.
The staging directory must contain exactly the four approved root files and
six approved scene directories, with exactly six approved files per scene.

After inference, independently reopen every NPZ, CSV, JSON, and PNG; recompute
the 30 pooled scene-method summaries; verify the official weight digest, frame
IDs, sparse counts, method keys, finite arrays, model load counts, and formal
pilot directory digests. Only a fully valid staging tree can be promoted.

Promotion is recoverable:

1. Move the current four-method output directory to the sibling backup path
   `cross_scene_motion_windows_pre_raft_backup`.
2. Move the validated staging directory to
   `cross_scene_motion_windows` on the same filesystem.
3. Revalidate the promoted path.

If the backup path already exists, stop before moving anything. On inference
or staging validation failure, leave the current output untouched. On a
promotion failure, restore the backup before reporting failure. Preserve the
backup after success unless the user separately requests its deletion.

## Error handling

- Reject missing or digest-mismatched official RAFT weights.
- Reject any RAFT input or flow geometry other than the established batch,
  channel, 228 x 304 layout.
- Reject non-finite flow, warped state, residual, or prediction tensors.
- Reject P-frame calls without the immediately preceding live state.
- Reject method sets other than the exact five approved methods.
- Reject nonconsecutive five-frame payloads or sparse counts other than 500.
- Reject incomplete, duplicate, non-finite, or inconsistent metric rows.
- Reject unexpected staging or promoted files and directories.
- Never modify the completed four-method output until staging validation has
  succeeded.

## Testing and verification

Use test-driven development for every behavior change. Tests cover:

- the fifth method's exact `I,P,I,P,I` schedule;
- current-to-previous RAFT argument order and 12 flow updates;
- strict warping of depth, guidance, and confidence;
- reuse of one NLSPN and one RAFT load across six scenes;
- five prediction arrays and 25 per-scene metric rows;
- five-method depth/error grids and shared color limits;
- 30 root summary rows and independent metric recomputation;
- official RAFT weight identity in scene and root metadata;
- exact staging, backup, promotion, and restoration behavior;
- legacy arbitrary-five-frame and four-method behavior where retained as a
  compatibility API;
- current Python and legacy Python 3.7 relevant test suites.

The real integration run uses the six recorded windows, inspects all 12
updated PNGs, revalidates the promoted tree, confirms the formal pilot files
are unchanged, and runs the full repository test suite.

## Success criteria

- All six scenes contain finite Full, Zero-flow, RGB-diff, Global-diff, and
  RAFT-GOP2 predictions for the exact existing windows.
- The RAFT-GOP2 path is strictly causal and uses official dense RAFT-Small
  backward flow with 12 updates.
- Each scene has 25 valid metric rows and two readable comparison figures.
- The root has 30 independently reproducible scene-method summary rows.
- NLSPN and RAFT are each loaded exactly once for the six-scene worker.
- The original output remains recoverable at the documented backup path.
- The promoted output, formal pilot digests, full test suite, and legacy
  compatibility tests all pass validation.
