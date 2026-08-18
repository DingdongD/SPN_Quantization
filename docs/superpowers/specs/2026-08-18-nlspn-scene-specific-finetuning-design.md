# NLSPN Scene-Specific Fine-Tuning Design

**Date:** 2026-08-18

## Goal

Create a scene-specialized NLSPN checkpoint that lowers absolute Full NLSPN
depth-completion RMSE on held-out scenes from the downloaded indoor dataset.
The existing NYUv2 checkpoint remains unchanged and available as the generic
model.  This project does not optimize temporal caching, add a temporal loss,
or change the inference architecture.

## Success criteria

The primary test metric is valid-pixel pooled RMSE over every frame in the two
held-out test scenes.  The specialized model is considered successful when:

1. pooled test RMSE is at least 5% lower than the generic checkpoint;
2. neither held-out test scene regresses in RMSE by more than 1%;
3. all predictions and metrics are finite;
4. every model input contains exactly 500 nonzero sparse-depth samples; and
5. the specialized checkpoint has exactly the same NLSPN state-dictionary key
   set and tensor geometry as the generic checkpoint.

The test set is evaluated once after training and is not used to select a
checkpoint or revise hyperparameters.  A failure to meet the success criteria
is reported as an experimental result rather than hidden through further
test-driven tuning.

## Immutable baseline

The generic model is:

`/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18/best.pt`

Its companion `args.json` specifies a ResNet-34 NLSPN with 18 propagation
iterations, a 10 m maximum depth, and 500 sparse samples.  The checkpoint was
trained on NYUv2 HDF5 lists and stores its parameters under the `net` key.
The baseline checkpoint and arguments are read-only inputs.  The fine-tuning
workflow records their SHA-256 digests and never overwrites either file.

## Dataset and scene split

The source dataset is `/workspace/VoxelNet/train`.  It contains paired
`rgb/NNNN.jpg` and `depth/ImageNNNN.exr` files in seven scenes, totaling 20,000
frames.

The split is fixed at scene level:

| Split | Scenes | Frames |
|---|---|---:|
| Train | `BeachApartmentInterior_My_ir`, `bedroom_ir`, `livingroom_ir`, `room4` | 10,000 |
| Validation | `room6` | 2,000 |
| Test | `room3`, `room7` | 8,000 |

The generated manifests contain the canonical scene, frame ID, RGB path, and
depth path for every sample.  Validation rejects duplicate identities, missing
pairs, noncanonical ordering, scene overlap, count mismatch, and any path
outside the approved data root.  The three manifests are immutable after
training starts and their digests are recorded in every checkpoint.

This scene-disjoint split prevents adjacent or near-duplicate video frames
from crossing split boundaries.  No frame from `room3` or `room7` participates
in optimization, early stopping, learning-rate selection, or checkpoint
selection.

## Input contract and preprocessing

The input contract remains identical to the existing evaluation pipeline:

- RGB input: three normalized float32 channels;
- sparse depth input: one float32 channel with exactly 500 nonzero samples;
- target: one float32 dense-depth channel plus a validity mask;
- spatial geometry: `228 x 304`;
- valid depth range: `(0, 10]` metres.

Each `640 x 480` RGB/depth pair is decoded, invalid or nonfinite depth is set
to zero, resized to height 240, and center-cropped to `228 x 304`.  RGB uses
bilinear resizing, while depth and validity use nearest-neighbor resizing.
RGB is converted to float32 in `[0, 1]` without ImageNet normalization.  This
matches both the custom NYUv2 training runner that produced the baseline
checkpoint and the current sequence-inference workers.  Introducing a new
normalization convention during fine-tuning would make the specialized and
generic inputs incompatible and is therefore forbidden.

EXR inputs must contain three numerically equal channels.  Values that are
nonfinite, nonpositive, or greater than 10 m are invalid.  A sample with fewer
than 500 valid pixels after preprocessing is rejected.

For training, 500 valid sparse coordinates are resampled independently on
every sample access.  Sampling is driven by the epoch, sample identity, and
global seed so a run can be reproduced while successive epochs see different
masks.  Validation and test use one immutable sparse mask per sample derived
from the split seed and sample identity.  Baseline and specialized models use
the exact same validation and test tensors.

Spatial augmentation is limited to horizontal flipping because it preserves
depth scale and output geometry.  RGB-only brightness, contrast, and saturation
jitter may be applied on training samples.  Rotation and scale augmentation
are excluded from the first formal run to minimize preprocessing mismatch with
the evaluation protocol.

## Model and checkpoint compatibility

The workflow constructs the same official `NLSPNModel` used by the existing
inference workers and strictly loads `checkpoint["net"]`.  It does not modify
NLSPN modules, propagation iterations, confidence propagation, affinity type,
or inference inputs.

Specialized checkpoints preserve the established top-level format:

- `net`: strict-compatible model state;
- `epoch`: selected training epoch;
- `optimizer` and `scheduler`: resumable training state;
- `tracker`: convergence and best-validation state;
- `val`: validation metrics at selection time;
- `args`: complete training configuration;
- `meta`: source checkpoint, dataset, manifest, software, and device digests.

Before publication, a fresh NLSPN instance must strictly load the new `net`
state and run a finite prediction on a held-out validation sample.

## Staged optimization

### Stage 0: baseline

Run the immutable generic checkpoint on the complete validation manifest with
fixed input tensors.  Store its per-frame and aggregate validation metrics
before training begins.  Test input files may be indexed and hashed during
preflight, but neither baseline nor specialized test inference runs until the
specialized checkpoint has been selected.

### Stage 1: depth/decoder/propagation adaptation

For three epochs, freeze `conv1_rgb` and shared encoder modules `conv2` through
`conv6`.  Train:

- `conv1_dep`;
- shared decoder modules `dec5` through `dec2`;
- initial-depth modules `id_dec1` and `id_dec0`;
- guidance modules `gd_dec1` and `gd_dec0`;
- confidence modules `cf_dec1` and `cf_dec0`; and
- trainable propagation parameters in `prop_layer`.

Use Adam with learning rate `1e-4`.  BatchNorm layers remain in evaluation
mode and their affine parameters and running statistics remain frozen.
The learning rate is constant within this three-epoch stage.

### Stage 2: full low-rate adaptation

Unfreeze convolutional weights while continuing to keep all BatchNorm state
frozen.  Use differential Adam learning rates:

- `5e-6` for `conv1_rgb` and `conv2` through `conv6`;
- `2e-5` for the depth stem, decoder, prediction heads, and propagation layer.

Train for at most 15 additional epochs.  Stop when validation RMSE has not
improved significantly for four consecutive epochs.  A significant
improvement is a relative reduction of at least `0.1%`; the lowest finite
validation RMSE is still saved even when an improvement is smaller.
Learning rates remain constant within Stage 2; the explicit stage transition
is the only planned learning-rate change.  The checkpoint `scheduler` field
stores this two-stage controller state rather than an implicit epoch-based
decay.

The effective batch size is 12.  The physical batch size is selected by a
read-only memory probe and gradient accumulation supplies the remainder.  The
selected physical batch size and accumulation factor become immutable run
metadata.  Gradients are clipped to a global norm of 1.0.  The first formal
run uses float32 training to avoid mixed-precision uncertainty in the legacy
deformable-convolution extension.

## Objective and checkpoint selection

The supervised objective is the official valid-pixel
`1.0 * L1 + 1.0 * L2` loss on the final propagated prediction.  Invalid target
pixels contribute nothing.  No loss is applied to temporal differences,
cached state, optical flow, or intermediate propagation outputs.

At the end of every epoch, evaluate every validation frame with fixed sparse
inputs.  The checkpoint with the lowest validation pooled RMSE is `best.pt`.
Training loss and MAE are diagnostic only and cannot override RMSE-based
selection.

The optimizer and scheduler can resume only when the source checkpoint digest,
manifests, split seed, preprocessing configuration, model state geometry, and
stage configuration all match.  Otherwise resume is rejected.

## Evaluation

After checkpoint selection, evaluate the generic and specialized checkpoints
on the complete 8,000-frame test manifest in the same process and with the same
preprocessed inputs.  Report:

- pooled RMSE and MAE weighted by valid pixels;
- equally weighted scene-macro RMSE and MAE;
- individual `room3` and `room7` metrics;
- AbsRel; and
- RMSE in predefined target-depth bands `(0,2]`, `(2,4]`, `(4,6]`, `(6,8]`,
  and `(8,10]` m.

The workflow also reuses the 30 previously selected low/medium/high motion
windows belonging to `room3` and `room7`.  It runs Full NLSPN only, with the
same deterministic 500-point inputs, and produces baseline-versus-specialized
depth and absolute-error comparisons.  These windows are explanatory
visualizations and do not replace the complete test-set metrics.

## Output and publication

The default final output is:

`/workspace/VoxelNet/nlspn_finetune/full_rmse_scene_disjoint_v1`

A sibling staging directory is used during execution.  The launcher refuses
to overwrite either an existing final output or staging directory.  Required
artifacts include:

- `best.pt` and `args.json` for the specialized model;
- `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv`;
- `epoch_metrics.csv` and `training_curves.png`;
- baseline and specialized per-frame test metrics;
- aggregate and depth-band comparison CSV files;
- the 30-window depth/error visualizations;
- `run_metadata.json`; and
- `report.md`.

The final directory is published atomically only after independent validation
of the exact artifact tree, all file digests, row counts, metric recomputation,
PNG decoding, checkpoint strict loading, finite inference, sparse counts, and
success-gate calculation.  Failing runs remain in staging for diagnosis and
do not replace any prior model or result.

## Error handling

Training stops immediately on:

- incomplete or overlapping manifests;
- missing or malformed JPG/EXR pairs;
- fewer than 500 valid post-crop depth pixels;
- nonfinite model inputs, loss, gradients, predictions, or metrics;
- a sparse-depth count other than 500;
- unexpected trainable/frozen parameter sets;
- incompatible checkpoint keys or tensor shapes;
- a changed baseline checkpoint or manifest digest; or
- publication validation failure.

CUDA out-of-memory during the explicit preflight probe may select a smaller
physical batch size before the formal run starts.  CUDA out-of-memory after
the formal configuration is recorded is a run failure and does not silently
change the batch size.

## Testing strategy

Unit tests cover:

- exact scene membership and manifest geometry;
- JPG/EXR decoding, sanitization, resize, crop, and normalization;
- deterministic validation/test masks and epoch-varying training masks;
- exactly 500 sparse samples;
- freeze/unfreeze and differential learning-rate parameter groups;
- BatchNorm freezing;
- masked L1/L2 and pooled metric recomputation;
- convergence tracking and strict resume rejection;
- checkpoint compatibility and immutable baseline behavior; and
- exact staged-output validation and atomic promotion.

A small synthetic-data integration test exercises baseline evaluation, both
training stages, resume, best-checkpoint selection, final evaluation, and
promotion without relying on the real GPU dataset.  Legacy-environment tests
exercise construction, strict loading, forward/backward propagation, and one
optimizer step with the official deformable-convolution extension.

Before the formal run, the complete repository test suite and focused Python
3.7 legacy suite must pass.  After training, the promoted tree is revalidated
from disk and the same test suites are run again before completion is claimed.
