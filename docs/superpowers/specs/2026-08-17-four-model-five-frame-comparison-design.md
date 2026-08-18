# Four-Model Five-Frame Depth-Completion Comparison Design

## Goal

Extend the completed CSPN five-frame pilot for frames `0001` through `0005`
of `BeachApartmentInterior_My_ir` with DySPN, NLSPN, and CompletionFormer.
All four models must receive the same preprocessed RGB, dense ground truth,
and 500-point sparse-depth observation so their predictions and errors can be
compared fairly.

This is a five-frame inference and visual-analysis experiment. It does not
retrain any model, process the full sequence, or turn the four spatial models
into temporal networks.

## Inputs and geometry

- Dataset root: `/workspace/VoxelNet/train`.
- Scene: `BeachApartmentInterior_My_ir`.
- Source RGB: `rgb/0001.jpg` through `rgb/0005.jpg`, each `640 x 480`.
- Source depth: `depth/Image0001.exr` through `depth/Image0005.exr`.
- Existing CSPN artifacts:
  `/workspace/VoxelNet/cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005/`.
- The existing CSPN frame NPZ files are the canonical experiment inputs. They
  already contain preprocessed RGB, dense depth, validity, and the deterministic
  500-point sparse depth used by CSPN.
- Network geometry is not 480p. The source is resized to `320 x 240` and centre
  cropped to `304 x 228`, so predictions have shape `228 x 304`.
- CSPN consumes a concatenated `1 x 4 x 228 x 304` RGB-plus-sparse tensor.
  DySPN, NLSPN, and CompletionFormer consume separate RGB
  `1 x 3 x 228 x 304` and sparse-depth `1 x 1 x 228 x 304` tensors.

The orchestrator must validate that the five canonical NPZ files use identical
geometry, contain exactly 500 non-zero sparse values per frame, and share the
same sparse coordinate mask. It must not silently regenerate or resample these
inputs.

## Checkpoints and runtimes

Use the converged local baselines and the arguments stored beside them:

| Model | Checkpoint | Runtime | Propagation iterations |
| --- | --- | --- | ---: |
| CSPN ResNet-50 | `/workspace/VoxelNet/cspn_models/best_model.pth` | Existing completed run | 24 |
| DySPN ResNet-34 | `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/dyspn_iter6/best.pt` | Conda `pointkan`, DySPN on `PYTHONPATH` | 6 |
| NLSPN ResNet-34 | `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18/best.pt` | Conda `completionformer-py37`, NLSPN and its deformable-convolution extension on `PYTHONPATH` | 18 |
| CompletionFormer | `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/completionformer_iter18/best.pt` | Conda `completionformer-py37`, CompletionFormer and its deformable-convolution extension on `PYTHONPATH` | 18 |

Each worker reconstructs its argument namespace from the checkpoint's saved
`args.json`, overrides only paths and inference-only settings, and loads the
checkpoint strictly enough to reject missing, unexpected, or shape-mismatched
learned parameters. A model is not reported as successful merely because a
partial state dictionary loaded.

## Selected architecture

Use a neutral orchestrator plus one model-specific worker process per external
model. The orchestrator owns input validation, launches workers in their native
Conda environments, validates their result contract, computes common metrics,
and renders comparison artifacts. Workers own only model construction,
checkpoint loading, GPU inference, and prediction serialization.

This process boundary is required because DySPN uses PyTorch 2.0 in `pointkan`,
while NLSPN and CompletionFormer use PyTorch 1.10 and locally compiled
deformable-convolution extensions in `completionformer-py37`. Importing all
three stacks into one Python process would couple incompatible Python/PyTorch/
CUDA extension ABIs.

Two alternatives are excluded:

- A single unified Python environment is fragile because of the incompatible
  PyTorch and deformable-convolution builds.
- Independent ad-hoc scripts that each preprocess the source data could change
  crop geometry, depth validity, or sparse coordinates and invalidate the
  comparison.

## Components

### Canonical-input reader

Load the five existing CSPN NPZ files and extract RGB, sparse depth, ground
truth, and validity. Validate frame IDs, dtypes, finite values on valid pixels,
`228 x 304` geometry, the exact sparse count, and coordinate equality across
frames. Record content digests so every worker result can be tied to the same
input payload.

Reuse the existing CSPN prediction and its frame-level metadata instead of
rerunning CSPN. Before combining it with the other models, verify its prediction
shape and finite values and recompute its metrics using the common evaluator.

### External-model workers

Each worker receives the canonical input directory, frame IDs, checkpoint,
output path, and CUDA device. It loads one model, performs inference in
evaluation/inference mode one frame at a time, and writes an atomic compressed
NPZ result containing:

- model name and checkpoint digest;
- canonical-input digest;
- frame IDs;
- raw predictions with shape `5 x 228 x 304`;
- clamped predictions in `[1e-6, 10]` metres;
- runtime and model configuration metadata.

The workers must preserve the model repository's expected RGB convention and
input dictionary/argument signature while feeding the exact canonical numeric
arrays. They may adapt tensor layout or key names but may not resize, crop,
resample, or redraw sparse observations.

### Common evaluator

For every model and frame, compute RMSE, MAE, and absolute-relative error on
valid ground-truth pixels. Record valid coverage and verify the sparse count.
Metrics use the clamped prediction consistently across all four models; raw
predictions remain available for diagnostics.

For every adjacent pair, compute the image-space temporal residual:

`(prediction[t+1] - prediction[t]) - (ground_truth[t+1] - ground_truth[t])`.

Report its RMSE and MAE where both ground-truth frames are valid. These are
explicitly `unregistered` diagnostics because camera pose and image alignment
are unavailable. They measure frame-to-frame image-space behaviour, not
geometry-compensated temporal consistency, and none of the four models consumes
multiple frames jointly.

### Visualizer

Use a common metre range and colour scale across models. Generate:

- one per-model, per-frame panel with RGB, sparse depth, ground truth,
  prediction, and absolute error;
- `four_model_depth_comparison.png`, with five frame rows and ground truth plus
  the four model predictions as columns;
- `four_model_error_comparison.png`, with five frame rows and one absolute-error
  column per model;
- `four_model_temporal_comparison_unregistered.png`, with four adjacent-frame
  rows and one temporal-residual column per model;
- `four_model_frame_metrics.csv` and `four_model_temporal_metrics.csv`;
- `run_metadata.json` with source/checkpoint digests, environments, model
  arguments, geometry, sparse-input checks, runtimes, and temporal caveats.

## Output layout

Write generated artifacts beneath:

`/workspace/VoxelNet/spn_model_comparison/BeachApartmentInterior_My_ir/frames_0001_0005/`

Use subdirectories `cspn/`, `dyspn/`, `nlspn/`, and
`completionformer/` for model-specific NPZ files and panels. Place the combined
plots, CSV files, and metadata in the experiment root. Generated inference data
remains outside the Git repository.

Workers write temporary files and rename them only after validation. A failed
worker may leave a diagnostic log but must not leave a result file that the
orchestrator could mistake for a completed run. A rerun may reuse a model result
only when its canonical-input and checkpoint digests match.

## Error handling

Stop the comparison with a precise error when:

- any canonical NPZ is missing or violates the shared-input contract;
- a checkpoint or required local model repository is missing;
- CUDA or a compiled deformable-convolution extension is unavailable;
- checkpoint compatibility checks fail;
- a worker exits unsuccessfully, writes malformed metadata, or returns the
  wrong frame order or prediction shape;
- a raw/clamped prediction or final metric contains a non-finite value.

Existing CSPN outputs and successful worker results are not deleted on failure.
The combined artifact set is marked complete only after all four models and all
expected files pass final validation.

## Verification

Automated tests cover canonical-input validation, shared sparse-mask checks,
worker command construction, worker-result schema validation, metric aggregation,
and comparison artifact manifests. Model construction is isolated behind narrow
worker interfaces so contract tests can use small synthetic predictions.

A real five-frame run then verifies:

- all four checkpoint digests and model configurations are recorded;
- CSPN input semantics are `RGB+sparse`, while the other three receive separate
  RGB and sparse tensors at `228 x 304`;
- all four model outputs contain frames `0001` through `0005` in order;
- all predictions and metric values are finite;
- each model was evaluated against the same dense ground truth and exact
  500-coordinate sparse mask;
- 20 model/frame panels, three combined comparison plots, two metric tables,
  worker logs, and metadata are present and readable;
- metadata and temporal plot titles clearly label temporal results as
  unregistered.

The experiment is complete only after these checks pass and the generated
comparison plots are visually inspected for legibility and obvious tensor-scale
or colour-map errors.
