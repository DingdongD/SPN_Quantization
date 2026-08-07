# Strict W4A4 and FP4 Reconstruction Evaluation Design

## Goal

Determine whether strict AdaRound or BRECQ W4 weight reconstruction preserves
the accuracy of the official CSPN, DySPN, NLSPN, and CompletionFormer models
when ordinary CNN activations use uniform A4 or scaled E2M1. The experiment
isolates weight rounding from activation format and does not train the models or
update their source checkpoints.

## Immutable Model Inputs

The evaluation uses the existing converged NYU checkpoints and official model
implementations:

| Model | Propagation iterations | Checkpoint |
| --- | ---: | --- |
| CSPN | 24 | `output/nyu_converged_baselines/cspn_iter24/best.pt` |
| DySPN | 6 | `output/nyu_converged_baselines/dyspn_iter6/best.pt` |
| NLSPN | 18 | `output/nyu_converged_baselines/nlspn_iter18/best.pt` |
| CompletionFormer | 18 | `output/nyu_converged_baselines/completionformer_iter18/best.pt` |

No model layer, channel count, propagation implementation, iteration count,
checkpoint tensor, or data preprocessing rule may change. Every result records
the model class and module, source SHA256, checkpoint SHA256, and external model
commit where applicable.

## Experiment Isolation

The primary experiment reuses one fixed W4 contract for each weight method:

- RTN uses the deterministic per-output-channel signed-symmetric W4 contract.
- AdaRound uses the existing strict AdaRound W4 contract.
- BRECQ uses the existing strict BRECQ W4 contract.

The AdaRound and BRECQ contracts remain weight-only contracts with
`activation_bits=0`. They are not reconstructed again for each activation
format. This prevents activation calibration from changing the learned weight
rounding decisions.

Each weight contract is evaluated with the following aligned activation modes:

| Activation configuration | Ordinary CNN activation | Sensitive boundaries | Propagation signals | Bias |
| --- | --- | --- | --- | --- |
| `FP4V_W4A8` | uniform A8 | uniform A8 | A8 | FP32 |
| `FP4V_W4A4` | uniform A4 | uniform A8 | A8 | FP32 |
| `FP4V_W4E2M1` | scaled E2M1 | uniform A8 | A8 | FP32 |

The sensitive boundaries are the existing model-specific sparse-depth,
initial-depth, guidance, confidence, and combined-propagation boundaries. The
same boundary ownership and bit overrides apply to uniform A4 and E2M1. The
existing per-channel activation-input rules are also identical between these
two formats.

`FP4V_W4E2M1` is a `float_e2m1_qdq_reference`. It is not reported as a bit-exact
integer backend. FP32 bias isolation is used by all three primary activation
configurations so that the A4 versus E2M1 comparison changes only the ordinary
activation format.

## Integer Stress Baseline

Each weight method is also evaluated with `HW_W4A4_full`. This configuration
uses the standard hardware-aligned integer contract, including int32 bias with
`scale = sx * sw[o]`, and does not inherit the FP4 validation group's semantic
A8 islands.

`HW_W4A4_full` is an all-A4 deployment stress baseline. It must be displayed
separately and must not be used to attribute a difference specifically to
uniform A4 versus E2M1 because its bias and semantic-boundary contracts differ
from the primary FP4 validation group.

## Dataset And Reproducibility

Activation calibration uses the same ordered set of 64 NYU training samples for
every model, weight method, and activation configuration. Evaluation uses the
same ordered set of 64 NYU validation samples. The seeds remain fixed at the
values used by the strict W4A8 evaluation.

All new evaluations are written to a clean result root. Historical metrics may
be used only to locate the immutable sample indices; historical predictions or
aggregated metrics are not merged into the new result set.

The runner rejects a result root when any model identity, source hash,
checkpoint hash, reconstruction contract, calibration index, evaluation index,
activation ownership rule, or activation configuration differs across methods.

## Metrics And Acceptance

The aggregate report records, for every model, weight method, and activation
configuration:

- mean RMSE, MAE, and ABS_REL;
- delta and relative degradation versus the same model's FP32 result;
- delta versus RTN with the same activation configuration;
- nonfinite sample and pixel ratios;
- activation SQNR, zero ratio, and saturation ratio by semantic group;
- propagation-step RMSE where the model exposes propagation states.

A W4A4 or W4-E2M1 result preserves performance only when all of the following
hold:

1. No evaluated sample or prediction pixel is nonfinite.
2. Mean RMSE degradation relative to FP32 is at most 10 percent.
3. Mean RMSE is no worse than RTN under the same activation configuration.

Paired sample-level bootstrap with 10,000 resamples and a fixed seed reports a
95 percent confidence interval for:

- AdaRound minus RTN RMSE under each activation format;
- BRECQ minus RTN RMSE under each activation format;
- E2M1 minus uniform A4 RMSE within each weight method.

The pass or fail decision uses the observed means and nonfinite checks. The
confidence intervals describe uncertainty and must be shown next to the
decision; they do not silently replace the stated threshold.

## Outputs

The experiment writes one self-contained directory per weight method and model,
including metadata, calibration and quantization manifests, sample metrics,
layer and signal diagnostics, and all 64 prediction payloads for every requested
configuration.

The aggregate output contains:

1. A grouped log-scale RMSE chart for RTN, AdaRound, and BRECQ across A8,
   uniform A4, and E2M1.
2. A performance-retention heatmap showing relative RMSE degradation and the
   acceptance status.
3. Paired A4-to-E2M1 RMSE-difference distributions with bootstrap intervals.
4. One prediction comparison per model containing GT, FP32, RTN A4/E2M1,
   AdaRound A4/E2M1, and BRECQ A4/E2M1.
5. A matching absolute-error comparison using one common error scale.
6. Layer-group activation diagnostics and propagation-step error plots.
7. A separate integer-stress table and plot for `HW_W4A4_full`.

Sample selection for prediction figures is deterministic. For each model, the
figure uses the sample with the largest finite RMSE spread across the six
primary A4/E2M1 predictions. Nonfinite predictions are rendered with the
existing explicit nonfinite color and remain included in aggregate counts.

## Validation

Before aggregation, validation must prove that:

- all four official model identities and checkpoint hashes match the approved
  inputs;
- every AdaRound and BRECQ result references an exact strict weight contract;
- all compared methods use identical calibration and evaluation indices;
- every model-method-configuration combination has 64 metric rows and 64
  prediction payloads;
- primary A4 and E2M1 configurations have identical semantic A8 ownership,
  propagation bits, per-channel activation-input rules, and FP32 bias policy;
- E2M1 metadata reports `float_e2m1_qdq_reference`;
- the integer stress baseline reports the expected hardware-aligned execution
  and int32 bias contract;
- no stale or partially completed result is accepted.

Unit tests cover contract composition, strict-manifest rejection, configuration
fairness, bootstrap calculations, acceptance decisions, sample alignment, and
missing prediction rejection. The complete test suite must pass before the
experiment is reported.

## Follow-Up Boundary

This experiment evaluates standard weight-only AdaRound and BRECQ. If neither
method preserves A4 or E2M1 performance, activation-scale optimization and
joint activation-aware reconstruction are separate follow-up methods. Their
results require new method names and cannot overwrite or be reported as the
standard AdaRound/BRECQ results from this experiment.
