# Unified Propagation-Aware W4A8 Evaluation Design

## Goal

Produce a directly comparable NYU PTQ evaluation of FP32, propagation-aware
W4A4 ablations, W4A8, and W8A8 for the official CSPN, DySPN, NLSPN, and
CompletionFormer model structures. This evaluation does not train or update
model weights.

## Immutable Model Inputs

The evaluation uses these existing converged checkpoints and propagation
settings:

| Model | Propagation iterations | Checkpoint |
| --- | ---: | --- |
| CSPN | 24 | `output/nyu_converged_baselines/cspn_iter24/best.pt` |
| DySPN | 6 | `output/nyu_converged_baselines/dyspn_iter6/best.pt` |
| NLSPN | 18 | `output/nyu_converged_baselines/nlspn_iter18/best.pt` |
| CompletionFormer | 18 | `output/nyu_converged_baselines/completionformer_iter18/best.pt` |

No forward implementation, layer, channel count, model argument, propagation
iteration, checkpoint tensor, or dataset preprocessing rule may be changed.
CSPN is loaded from the repository model source that is byte-identical to the
original `/workspace/CSPN/cspn_pytorch` source. DySPN, NLSPN, and
CompletionFormer are loaded from their pinned official Git submodules.

Checkpoint loading must reject unexpected keys and missing trainable or
persistent tensors. CSPN's fixed `post_process_layer.sum_conv.weight` is the
only allowed omitted checkpoint entry because it is reconstructed by the
official model constructor. Each result records the resolved model source,
submodule commit where applicable, and checkpoint SHA256.

## Evaluation Matrix

The propagation backend evaluates the following configurations in one clean
result root:

- `FP32`
- `PA_Generic_W4A4`
- `PA_Constraint`
- `PA_OffsetA8`
- `PA_StateA8`
- `PA_W4A8`
- `PA_W8A8`

`PA_W4A8` uses 4-bit per-output-channel symmetric weights for ordinary
Conv/Linear modules and 8-bit calibrated activations. Affinity, offsets,
propagation states, and unsigned confidence/gates use 8 bits. It preserves the
same propagation-aware integer normalization contract as `PA_W8A8`; only the
ordinary model weights differ between those two configurations.

The propagation adapter quantizes neighbor affinity before normalization,
reconstructs the center coefficient from quantized neighbors, uses signed Q13
INT16 coefficients and an integer normalization reference, and preserves
sparse-depth anchors. Sampling and propagation MAC remain the documented float
QDQ deployment reference. No claim of a bit-exact integer DCN or grid-sampling
kernel is made.

## Dataset And Reproducibility

Calibration uses the existing 128 NYU training indices selected with seed
`20260804`. Evaluation uses the existing ordered set of 64 NYU validation
indices. The runner must fail rather than append results when model identity,
checkpoint SHA256, calibration indices, or evaluation indices differ.

All seven configurations are rerun from a single code revision into a new
output root. Historical hardware-backend W4A8 results are not merged into this
result set.

## Outputs

For each model, persist sample, regional, signal, layer, state, and propagation
metrics; calibration and hardware manifests; metadata; and all 64 prediction
payloads for every comparison configuration. Regenerate prediction contact
sheets, detailed absolute-error views, propagation-step error plots, and
constraint plots with `PA_W4A8` included.

The final report compares mean RMSE and relative degradation for FP32,
`PA_W4A8`, and `PA_W8A8`, while retaining the W4A4 ablations for diagnosis.
Non-finite sample and pixel counts must be reported explicitly.

## Verification

Unit tests must first fail for the missing `PA_W4A8` configuration and model
provenance/checkpoint validation. After implementation, run the complete test
suite in both the base environment and the CompletionFormer CUDA-extension
environment. Before reporting metrics, verify that every model has exactly 64
rows and 64 prediction payloads per requested configuration, all models share
the expected calibration and evaluation indices, and metadata records the
expected official source revisions and checkpoint hashes.
