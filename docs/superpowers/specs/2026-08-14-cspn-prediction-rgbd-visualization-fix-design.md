# CSPN Prediction RGBD Visualization Fix Design

## Problem

The prediction payload currently stores the exact official CSPN RGB model
input in the `rgb` field. The official loader applies
`Normalize -> ToPILImage -> ToTensor`, so this tensor is clipped/re-encoded and
cannot be converted back into the natural source image. The plotter incorrectly
treats it as an invertible ImageNet-normalized tensor. Sparse depth is correct
and contains exactly 500 points, but rendering all zero pixels through the
depth colormap makes those points difficult to inspect.

## Contract

- Official CSPN inference inputs, predictions, metrics, checkpoints, and
  quantization settings remain unchanged.
- Existing prediction payloads retain the exact model input as `model_rgb`.
- `rgb` becomes a display-only natural RGB image loaded from the same HDF5
  validation sample through the established `NyuHdf5Dataset` transform.
- Natural RGB and model RGB must have the same HWC shape and finite values.
- Natural RGB must lie in `[0, 1]`; invalid data is an error.
- Sparse depth remains the exact official model input. Plotting masks zero
  pixels with a neutral background and colors only valid depth points.
- The payload upgrade is explicit and atomic. It accepts the old strict schema
  and writes the new strict schema without fallback behavior.

## Data Flow

After all five configurations produce 64 payloads, the evaluator loads each
natural validation RGB once, pairs it by exact sample index, and atomically
rewrites the five corresponding payloads. The old `rgb` tensor is moved to
`model_rgb`; the natural image is written to `rgb`. A visualization manifest
records the source and field semantics.

The plotter requires the new schema. It displays `rgb` directly and renders
the exact sparse tensor with zero pixels masked. It never tries to reconstruct
natural RGB from `model_rgb`.

## Verification

Tests verify schema migration, exact preservation of model RGB, sample-index
alignment, natural RGB range validation, direct RGB display, sparse-zero
masking, and command-line plotting. Existing 320 payloads are upgraded without
rerunning inference, then all four PNG/PDF figures are regenerated and visually
inspected.
