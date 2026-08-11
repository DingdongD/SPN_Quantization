# Official QDrop INT W4A4 Adaptation Design

## Objective

Implement deployment-equivalent QDrop reconstruction for the official CSPN,
DySPN, NLSPN, and CompletionFormer model adapters. The first implementation is
limited to uniform integer W4A4. FP4 and E2M1 are outside this scope.

The implementation must preserve the existing RTN, AdaRound, BRECQ,
propagation-aware, attention, and concat behavior. QDrop is exposed as a new
`qdrop_strict` method and cannot alter the meaning or artifacts of an existing
method.

## Reference Contract

The algorithm reference is the official QDrop repository branch `qdrop` at
commit `4a9ca007ce91b66620b911de97df36d5109ecae0`:

- <https://github.com/wimh966/QDrop>
- <https://github.com/wimh966/QDrop/blob/qdrop/qdrop/solver/recon.py>
- <https://github.com/wimh966/QDrop/blob/qdrop/qdrop/quantization/fake_quant.py>
- <https://openreview.net/pdf?id=ySQH0oDyp7>

The required reference semantics are:

1. Cache full-precision block inputs and outputs and quantized block inputs.
2. Mix quantized and full-precision block inputs element by element during
   reconstruction.
3. At controlled activation sites, select quantized or full-precision values
   element by element during reconstruction.
4. Jointly optimize adaptive weight-rounding parameters and learnable
   activation quantization parameters.
5. Optimize block output reconstruction loss plus adaptive-rounding
   regularization with temperature decay.
6. Harden weight rounding and enable activation quantization for every value
   before deterministic evaluation.

The official field named `drop_prob` is the probability of selecting the
quantized value. The new implementation names this value
`quant_probability` so the contract is unambiguous. A value of `1.0` means
fully quantized activation inference, not complete dropping.

## Architecture

### QDrop Reconstruction

`spn_quant/qdrop_reconstruction.py` owns the block reconstruction algorithm.
It reuses the existing `AdaptiveRoundingController` and strict folded-graph
weight contract. It does not modify `StrictBlockReconstructor`.

`QDropBlockReconstructor` receives an explicit block, explicit semantic
activation sites, cached teacher and student records, and a complete
`QDropReconstructionConfig`. The configuration has no embedded method
fallbacks. Every optimization field is required from the experiment config:

- reconstruction steps and mini-batch size;
- weight and activation learning rates;
- rounding regularization weight;
- warm-up fraction and beta endpoints;
- reconstruction loss and exponent;
- quantization probability;
- random seed;
- activation scale lower bound.

The reconstructor freezes ordinary model parameters. Gradients are permitted
only for adaptive weight-rounding parameters and owned activation scale or
zero-point parameters. The original model weights, biases, BatchNorm values,
and propagation parameters remain unchanged.

### Activation Quantization

`spn_quant/qdrop_activation.py` owns learnable A4 fake quantization and its
export contract. Each site has one explicit integer format:

- unsigned A4 for nonnegative ReLU outputs;
- signed symmetric A4 for signed feature tensors;
- signed or unsigned asymmetric A4 only where the existing semantic edge
  declares an asymmetric zero point.

Signed symmetric sites fix zero point to zero. Other sites may optimize a
continuous zero-point proxy during reconstruction; export rounds and clamps it
to the declared integer range. Scale is positive and finite. Quantization uses
the existing straight-through estimator and integer limits. It must not invoke
an observer after reconstruction begins.

During reconstruction, each site computes both the A4 reconstruction and the
unmodified value, then selects between them with a seeded per-element mask.
The mask is disabled during calibration target capture and forbidden during
deployment replay. Freezing a site forces full quantization and exports the
exact scale, integer zero point, integer limits, signedness, tensor ownership,
and parameter fingerprints.

### Model Target Registry

`spn_quant/qdrop_targets.py` explicitly registers reconstruction blocks and
owned activation sites for CSPN, DySPN, NLSPN, and CompletionFormer. Registry
lookup uses exact model and module names. Missing models, targets, or sites
raise errors.

QDrop may own activation edges inside convolutional, transposed-convolutional,
linear, decoder merge, and CompletionFormer attention blocks. CompletionFormer
keeps the existing attention execution contract: Q/K/V projection values use
the declared A4 sites, QK and AV use the existing integer accumulation path,
and softmax remains FP16. Concat branches retain independent pre-concat scales
and the existing merge-scale reconciliation.

The following tensors are never QDrop sites:

- sparse depth input and RGB input;
- normalized affinity and its center coefficient;
- confidence or gate logits and probabilities;
- normalization coefficients, denominators, and probability tensors;
- sparse-depth anchors and anchor masks;
- propagation state inputs, outputs, and per-iteration states;
- DCN offsets and masks owned by the propagation implementation.

These tensors continue to use the existing deterministic propagation-aware
contract. A registry collision between QDrop and propagation ownership is a
fatal configuration error.

### Runner and Deployment Contract

`scripts/run_nyu_qdrop_reconstruction.py` builds separate teacher and student
models from the original official checkpoint. Both use the deployment graph
preparation already used by strict AdaRound and BRECQ. The runner validates the
source checkpoint, Conv-BN folding graph, exclusions, and pre-reconstruction
weight fingerprints before optimization.

The runner captures teacher outputs, teacher inputs, and student quantized
inputs for every registered block. Target order, call count, nested tensor
structure, shape, and dtype family must match exactly. Reconstruction proceeds
block by block in registry order. Activation observation is completed before
optimization; learned activation parameters are then the only activation
state that may change.

The output contains two exact contracts:

1. Existing W4 integer codes and per-output-channel weight scales.
2. A4 semantic-edge scales, integer zero points, ranges, formats, and edge
   ownership.

The hardware evaluation runner accepts `qdrop_strict` as a distinct contract
method. It reconstructs the original model and replays both contracts. It must
not run another RTN pass, observer update, scale search, or calibration step.

## Optimization Protocol

Each model uses a fixed 1024-sample NYU calibration set. A deterministic split
uses 896 samples for probability search reconstruction and 128 samples for
held-out calibration validation. After probability selection, formal
reconstruction uses all 1024 calibration samples. The formal 64-sample
evaluation set is never used to select hyperparameters.

The official W2A4 configuration provides the algorithm anchor:

- quantization probability `0.5`;
- activation scale learning rate `4.0e-5`;
- weight rounding optimizer learning rate `1.0e-3`;
- warm-up fraction `0.2`;
- rounding regularization weight `0.01`;
- beta range `[20, 2]`;
- mini-batch size `32`;
- 20,000 reconstruction steps.

The probability search evaluates the explicit candidates `[0.25, 0.5, 0.75]`
and includes the official value `0.5`. Every candidate uses the same
initialization, samples, block order, step budget, and loss. Selection minimizes
held-out block output error subject to finite output, valid activation
parameters, and no deployment-contract violation. These values are recorded in
versioned experiment configuration; the runner has no defaults or hidden
candidate list.

Formal reconstruction uses 20,000 optimization steps per block and explicit
random seeds `[1005, 1006, 1007]`. Each seed exports a separate immutable
contract. Formal evaluation uses the same fixed 64 samples for FP32, RTN W4A4,
BRECQ W4A4, and QDrop W4A4.

## Outputs

Each QDrop run writes:

- reconstruction configuration and official reference commit;
- source-checkpoint and folded-graph fingerprints;
- weight and activation deployment contracts;
- block and activation ownership manifests;
- selected quantization probability and complete candidate results;
- per-step reconstruction and rounding losses;
- per-layer scale, zero point, saturation, zero ratio, and SQNR metrics;
- end-to-end per-sample RMSE, MAE, ABS_REL, and finite-value counts;
- propagation-step error diagnostics from the deterministic PA path.

The aggregate report includes per-model mean, standard deviation, best, and
worst RMSE across three seeds. It compares FP32, RTN W4A4, BRECQ W4A4, and
QDrop W4A4. Prediction figures contain RGB, sparse depth, ground truth, FP32,
RTN, BRECQ, QDrop, and aligned absolute-error maps with common depth and error
ranges.

## Failure Policy

The implementation has no fallback behavior. It raises an error when:

- a model, block, or semantic edge is absent from the explicit registry;
- teacher and student calls, structures, shapes, or ownership differ;
- a propagation-only tensor is registered as a QDrop activation;
- activation ownership is missing or duplicated;
- a scale or zero point is invalid, non-finite, or outside its contract;
- reconstruction loss or a reconstructed tensor is non-finite;
- optimization changes an ordinary model parameter;
- a random mask remains active during export or evaluation;
- checkpoint, folding graph, format, range, or parameter fingerprints differ;
- evaluation attempts a second quantization or calibration pass.

An optimized hard solution that is locally worse than its initial state is
reported as a failed QDrop reconstruction. It is not silently replaced by RTN
and is not exported as a QDrop result.

## Test Strategy

Implementation follows test-driven development. Unit tests must fail before
production behavior is added. Tests cover:

- official per-element mask and input-mixing semantics at probabilities 0, 0.5,
  and 1;
- seeded reproducibility and different-seed variation;
- gradients restricted to rounding and activation quantization parameters;
- signed symmetric and unsigned A4 integer behavior;
- asymmetric zero-point optimization, integer hardening, and clamping;
- deterministic full-quantization behavior after freeze;
- exact activation and weight contract round-trip;
- rejection of observer updates and second quantization after reconstruction;
- explicit target and exclusion registries for all four models;
- CompletionFormer Attention and Concat ownership;
- rejection of every propagation-only activation edge;
- runner metadata, manifests, fingerprints, and method validation.

GPU verification proceeds in increasing scope: synthetic unit tests, one real
block, one real NYU sample, a 64-sample smoke run, then the formal 1024-sample,
20,000-step, three-seed reconstruction and fixed 64-sample evaluation. No
formal result is reported until deterministic contract replay reproduces the
same predictions and fingerprints in a fresh model process.
