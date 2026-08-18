# CSPN Five-Frame Sequence Prediction Design

## Goal

Run the existing CSPN ResNet-50 checkpoint on frames `0001` through `0005`
from the `BeachApartmentInterior_My_ir` scene, then export per-frame depth
completion results and an unregistered temporal comparison. This is a pilot;
it does not include the remaining 19,995 frames or a temporal model.

## Inputs and assumptions

- Dataset root: `/workspace/VoxelNet/train`.
- Scene: `BeachApartmentInterior_My_ir`.
- RGB files: `rgb/0001.jpg` through `rgb/0005.jpg`.
- Dense depth files: `depth/Image0001.exr` through
  `depth/Image0005.exr`.
- Checkpoint: `/workspace/VoxelNet/cspn_models/best_model.pth`.
- Model: the existing CSPN ResNet-50 implementation with 24 spatial
  propagation iterations.
- Each EXR contains the same depth values in all three channels. Positive,
  finite values no greater than 10 metres are valid for this NYU-trained
  model; infinity and out-of-range values are invalid.
- The archive contains sequential Blender frames but no camera calibration or
  pose. Temporal differences are therefore image-space, unregistered
  diagnostics rather than geometry-compensated consistency measurements.

## Selected approach

Use standard depth completion: RGB plus 500 sparse depth samples predicts one
dense depth map per frame. The five frames share one deterministic pixel mask,
drawn from pixels valid in every preprocessed frame. This keeps the simulated
sampling layout constant and prevents random masks from dominating the
five-frame comparison.

Two alternatives are deliberately excluded from this pilot:

- An all-zero sparse-depth channel would turn the run into an out-of-distribution
  RGB-only stress test rather than normal CSPN depth completion.
- Feeding or warping earlier predictions into later frames would alter the
  model semantics and requires an explicit alignment method.

## Components

### Sequence input loader

The loader resolves the five RGB/EXR pairs, checks every pair exists, reads one
EXR depth channel, and verifies that the other channels agree. It converts
non-finite and out-of-range depth to invalid pixels.

RGB and depth follow the existing CSPN NYU validation geometry: resize the
short side to 240 pixels and centre-crop to `228 x 304`. RGB uses bilinear
resampling, while depth and validity use nearest-neighbour resampling. The
existing legacy CSPN RGB conversion is reused so the checkpoint sees the same
input convention used by the repository's NYU evaluation path.

### Sparse input generator

After preprocessing all five depths, intersect their validity masks. Select
exactly 500 coordinates with seed `2026`, and use those coordinates for every
frame. Each sparse map contains the current frame's depth at those locations
and zero elsewhere. The run fails clearly if the common valid region contains
fewer than 500 pixels.

### Model runner

Instantiate CSPN ResNet-50 with a 3-by-3 kernel, `8sum` affinity
normalisation, and 24 propagation iterations. Load the checkpoint safely as a
state dictionary, strip its `module.` prefix, ignore the known legacy fixed
sum kernel, and allow only the current derived unpool buffers to be absent.
Any other missing, unexpected, or shape-mismatched key is an error.

Run one frame at a time on a selected CUDA device under inference mode. Save
the raw prediction for diagnostics. Clamp a copy to `[1e-6, 10]` metres for
metrics and display; do not overwrite the raw output.

### Metrics and temporal analysis

For each frame, calculate RMSE, MAE, and absolute-relative error over valid
ground-truth pixels, plus valid-pixel coverage and sparse-point count.

For each adjacent pair (`0001->0002` through `0004->0005`), calculate:

- image-space ground-truth depth change;
- image-space predicted depth change;
- temporal residual `(prediction change - ground-truth change)`;
- RMSE and MAE of that residual where both ground-truth frames are valid.

These values must be labelled `unregistered` in generated metadata and plots.

## Outputs

Write generated artifacts beneath:

`/workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005/`

The directory will contain:

- one compressed NPZ per frame with RGB, sparse depth, ground truth, raw
  prediction, clamped prediction, validity, and absolute error;
- one per-frame PNG panel showing RGB, sparse depth, ground truth, prediction,
  and absolute error;
- `sequence_overview.png`, with the five frames arranged consistently for
  side-by-side inspection;
- `temporal_overview_unregistered.png`, showing adjacent-frame truth change,
  prediction change, and temporal residual;
- `frame_metrics.csv` and `temporal_metrics.csv`;
- `run_metadata.json` recording paths, preprocessing, seed, sparse count,
  checkpoint digest, model configuration, and the unregistered-temporal
  warning.

Generated outputs are data artifacts and remain outside the Git repository.

## Error handling

The runner exits without partial success claims when input files are missing,
EXR decoding fails, the common validity mask is too small, checkpoint keys are
incompatible, CUDA is unavailable for a requested CUDA device, or any metric
or prediction is non-finite. Output files are written only after their parent
directory exists, and a rerun deterministically replaces artifacts for the
same five-frame experiment.

## Verification

Automated tests will cover depth validity conversion, output geometry,
deterministic shared-mask construction, exact sparse count, metric formulas,
and checkpoint-key normalisation. A real-data smoke run will then verify:

- all five frame pairs load;
- the checkpoint loads with no unexplained incompatibility;
- each model input has shape `1 x 4 x 228 x 304`;
- each sparse input has exactly 500 nonzero values;
- all five NPZ files, five frame panels, two overview plots, two metric tables,
  and the metadata file are created;
- frame and temporal metric tables contain finite values;
- saved metadata identifies temporal analysis as unregistered.

The pilot is successful when these checks pass and the visual panels are
readable. Expansion to the full sequence, alternative frame intervals, or
optical-flow alignment is a separate follow-up decision based on this pilot.
