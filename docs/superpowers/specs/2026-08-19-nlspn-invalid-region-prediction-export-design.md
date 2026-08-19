# NLSPN Invalid-Region Prediction Export Design

## Goal

Export standalone depth products for the 150 frames already retained in the
30 formal test windows. The exported specialized prediction must remain visible
inside pixels where ground-truth depth is invalid. No training or inference is
rerun, and the validated formal result directory is not modified.

## Source and destination

The exporter reads only from:

`/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1/windows`

Each source window contains `predictions.npz` with aligned `scenes`,
`frame_ids`, `gt`, `valid`, and `specialized` arrays. The exporter writes a new
sibling tree:

`/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1_prediction_exports`

The source result tree is treated as immutable. Export uses a temporary sibling
directory and an atomic rename so a failed run cannot leave a partially valid
destination. A completed destination is never overwritten: if the fixed target
path already exists, the exporter stops before writing anything.

## Output layout

The destination contains one `manifest.csv` and 30 window directories under
`windows/`. Each window contains five frame directories named
`frame_<frame_id>`. Every frame directory contains exactly:

- `specialized_full_color.png`
- `specialized_depth_mm.png`
- `gt_with_prediction_fill.png`
- `invalid_mask.png`

The complete export therefore contains 150 frame directories and 600 PNGs.

## Pixel semantics

### Complete specialized prediction

`specialized_full_color.png` visualizes the unmasked specialized prediction.
It uses the `viridis` colormap with one fixed 0–10 m scale for every frame and
is saved at the native prediction resolution of 304 by 228 pixels without
axes, padding, or a per-image adaptive range. Invalid GT pixels are not removed
from this image.

### Machine-readable depth PNG

`specialized_depth_mm.png` is a single-channel 16-bit PNG. Each pixel is:

`round(clip(specialized_prediction_m, 0, 10) * 1000)`

The stored integer is therefore depth in millimetres with 1 mm resolution.
Clipping affects only this deployment image; it does not alter the source
floating-point prediction.

### GT plus predicted fill

`gt_with_prediction_fill.png` visualizes:

`where(valid, gt, specialized_prediction)`

It uses the same fixed `viridis` 0–10 m scale. This image is explicitly a
hybrid visualization, not new ground truth and not an evaluation target.

### Invalid-region mask

`invalid_mask.png` is a single-channel 8-bit PNG. A value of 255 means the GT
pixel is invalid and was filled from the specialized prediction; a value of 0
means the displayed hybrid retains valid GT.

## Manifest

`manifest.csv` has exactly 150 rows ordered by source window and frame order.
Each row records:

- window directory name
- scene
- frame ID
- invalid pixel count and fraction
- specialized prediction minimum, maximum, and mean in metres
- relative paths to all four exported PNGs

This provides provenance without duplicating the existing floating-point NPZ
payloads.

## Validation and failure handling

Before export, require exactly 30 source window directories, five aligned
frames per source NPZ, the required arrays, boolean-compatible validity masks,
304 by 228 spatial shape, and finite specialized predictions. Reject duplicate
scene/frame identities.

After export, verify:

- exactly 30 window and 150 frame directories exist;
- all 600 PNGs decode and have shape 228 by 304;
- depth PNGs are integer-valued and lie in 0–10000 mm;
- masks contain only 0 and 255;
- every hybrid image is generated from the exact `valid` selection formula;
- every manifest path resolves inside the destination;
- SHA-256 digests of all 30 source `predictions.npz` files are unchanged.

If any check fails, remove only the temporary export directory and leave any
previous completed destination untouched.

## Testing

Unit tests cover mask polarity, millimetre rounding and clipping, fixed-scale
color conversion, and the exact GT/prediction fill formula. Integration tests
use synthetic window payloads to verify naming, ordering, counts, PNG modes and
dimensions, manifest contents, duplicate rejection, and atomic publication.

The real export is accepted only after the focused tests, full project tests,
real-tree validation, and visual inspection of one low-, medium-, and
high-motion window from each test scene.
