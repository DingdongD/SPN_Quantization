# CSPN Dynamic Group-A4 Design

## Objective

Evaluate whether per-sample online activation ranges reduce the zero-code
collapse observed with static MinMax Group-A4 on the official CSPN model.
The experiment isolates activation range selection from weight quantization,
SmoothQuant, and propagation quantization changes.

## Quantization Contract

Weights use the existing static signed symmetric W4 path with one scale per
output channel. Activation ranges are computed from the current inference
sample immediately before QDQ. No calibration observer range is read by a
dynamic activation quantizer.

For an activation with batch dimension `N`, channel dimension `C`, and
contiguous groups of eight channels, the signed range is:

```text
M[n, g] = max(abs(x[n, c, ...])) for c in group g
scale[n, g] = M[n, g] / 7
code[n, c, ...] = clamp(round(x[n, c, ...] / scale[n, g]), -7, 7)
```

For a nonnegative activation, including ReLU outputs, the unsigned range is:

```text
M[n, g] = max(x[n, c, ...]) for c in group g
scale[n, g] = M[n, g] / 15
code[n, c, ...] = clamp(round(x[n, c, ...] / scale[n, g]), 0, 15)
```

The maximum reduction covers the eight channels and every non-batch,
non-channel dimension. Scale shape is therefore `[N, C / 8]`, not one scale
per batch and not one scale per spatial tile. A zero-range group maps exactly
to zero without dividing by zero. Non-finite inputs are an error.

The existing hybrid Group-8 site policy is retained. Sites whose channel count
is divisible by eight use dynamic Group-8. Existing tensor-granularity sites
use one dynamic scale per sample over all non-batch dimensions. The dynamic
and static configurations must cover exactly the same activation owners.

## Ownership Boundaries

Dynamic A4 applies only to ordinary encoder, decoder, and depth-head activation
owners already controlled by `HardwareAlignedInstrumentor`.

The following remain unchanged:

- guidance and affinity outputs remain FP;
- confidence, offsets, anchors, and propagation state are not assigned generic
  Dynamic A4;
- propagation uses A8 activation, signed INT16 Q13 coefficients, and INT32
  accumulation;
- ReLU outputs use unsigned A4 and signed sites use symmetric A4;
- Conv-BN folding occurs before weight quantization;
- bias follows the existing hardware-aligned integer bias contract.

SmoothQuant, activation permutation, OCI, rotation, BRECQ, and QDrop are not
enabled in this experiment.

## Components

### Dynamic Activation Quantizers

Add dedicated dynamic tensor and grouped activation quantizers. Each invocation
derives its scale from its input, returns QDQ output and integer codes, and
exposes runtime scale statistics. The implementation must not create or update
observers during evaluation.

### QuantSpec Integration

`QuantSpec.dynamic=True` selects the dynamic quantizer path. Static specs keep
their current behavior. Dynamic specs reject externally supplied activation
maxima because a static override would violate the online contract.

### Instrumentation

The hardware instrumentor builds dynamic quantizers for every declared dynamic
owner and records the same SQNR, zero-code, saturation, and block-output metrics
as the static path. It additionally records the number of online scales and
the reduction element count needed to derive them.

### Evaluation Runner

A focused CSPN runner uses the official converged checkpoint and the fixed
64-sample NYU validation subset. The static configurations use the existing
random-128 MinMax ranges. Dynamic configurations do not use calibration ranges,
but use the same prepared model and evaluation samples for paired comparison.

## Experiment Matrix

- `FP32`
- `W4_ONLY`
- `A4_ONLY_G8_STATIC`
- `A4_ONLY_G8_DYNAMIC`
- `W4A4_G8_STATIC`
- `W4A4_G8_DYNAMIC`

The A4-only pair isolates activation-range effects. The W4A4 pair measures the
deployed combination. FP32 and W4-only verify that the reference model and
static per-output-channel weight path remain unchanged.

## Metrics and Outputs

Record:

- RMSE, MAE, AbsRel, iRMSE, flat RMSE, and boundary RMSE;
- paired per-sample RMSE delta and better/worse sample counts;
- activation SQNR, new-zero rate, zero-collapse error, rounding error,
  clipping error, saturation, and non-finite ratio;
- block-output MSE/SQNR through encoder, decoder, initial depth, and
  propagation;
- online scale count and reduction element count per owner.

Runtime artifacts are written under:

```text
/workspace/SPN_Quantization/profile_logs/nyu_cspn_dynamic_group_a4/
```

Runtime artifacts remain outside Git. The repository stores the implementation,
tests, and a concise result report.

## Tests

Unit tests verify signed and unsigned code domains, per-sample independence,
Group-8 reduction dimensions, zero-range behavior, tensor-site behavior,
non-finite rejection, dynamic-spec validation, and static-path invariance.

Integration tests verify exact owner coverage, the six-configuration manifest,
paired 64-sample coverage, and unchanged propagation/guidance ownership.

## Acceptance Criteria

- Dynamic scales depend only on the current sample and never on calibration
  observer extrema.
- One sample's outlier cannot alter another sample's scale in a batched input.
- Dynamic and static Group-8 configurations quantize the same ordinary owners.
- Guidance remains FP and propagation remains A8/Q13/INT32.
- All six configurations produce 64 finite predictions.
- Results report accuracy changes together with zero-code reduction and online
  scale overhead; lower zero-code rate alone is not treated as success.
