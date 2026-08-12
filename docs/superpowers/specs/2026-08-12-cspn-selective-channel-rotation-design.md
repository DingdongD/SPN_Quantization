# CSPN Selective Channel Rotation Design

## Objective

Validate whether selective orthogonal channel rotation reduces signed activation
outliers and improves CSPN depth-completion accuracy under integer W4A4
quantization. The first stage isolates encoder/decoder activation effects by
keeping CSPN propagation at a fixed propagation-aware A8/Q13 contract.

This stage covers CSPN only. It does not add FP4, Learned Rotation, BRECQ, or
QDrop integration until Random or Hadamard Rotation demonstrates a useful
calibration or end-to-end improvement.

## Constraints

- Use the pinned official CSPN model architecture, converged checkpoint, real
  NYU calibration data, and the existing 64-sample evaluation subset.
- Use 128 fixed calibration samples. Evaluation samples must not select rotation
  matrices, activation scales, group sizes, or candidate sites.
- Do not rotate activations produced by ReLU. Those sites retain unsigned A4.
- Do not rotate sparse depth, the depth head, the guidance head, affinity,
  propagation coefficients, propagation state, or the final prediction.
- Do not add fallback behavior. Missing data, checkpoint, CUDA support, or an
  invalid configured site must fail directly.
- Do not add hashes, compatibility layers, feature flags, migration scaffolding,
  or persistent smoke-test outputs.

## Quantization Contract

The common contract for every quantized comparison is:

- CNN weights: per-output-channel signed symmetric W4.
- Ordinary signed CNN activations: signed symmetric A4.
- ReLU outputs: unsigned A4.
- Guidance head weights and its own input/output QDQ: disabled. The guidance
  head still consumes the shared decoder feature produced by the quantized
  backbone.
- Raw CSPN affinity at the propagation boundary: A8.
- Normalized CSPN coefficients: signed INT16 Q13, where `1.0` is represented by
  `8192`.
- Center coefficient: derived as
  `8192 - sum(neighbor_coefficient_codes)` and never independently quantized.
- Propagation state: A8 at each iteration.
- Propagation accumulation: INT32 semantics.

The shorthand for this contract is:

`CNN W4A4 + guidance head FP + affinity A8 + state A8 + coefficient INT16-Q13`.

## Eligible Rotation Boundaries

The CSPN adapter declares exactly two logical rotation boundaries. Calibration
must also confirm that each declared activation contains negative values.

### Decoder Entry

The signed output of `bn2(conv2)` feeds `gud_up_proj_layer1`. The block unpools
the feature and sends it to both `conv1` and `sc_conv1`. One rotation matrix is
shared by this fanout. Both consumer weights absorb the inverse basis:

```text
x' = R x
W_conv1' = W_conv1 R^T
W_sc_conv1' = W_sc_conv1 R^T
```

Channel rotation commutes with the channel-independent unpool operation, so the
rotation is represented once at the logical block input.

### Layer4 Signed Skip Branch

`skip4` is the signed output of the initial `conv1_1`, captured before BN and
ReLU. It is concatenated with a ReLU-produced decoder branch at
`gud_up_proj_layer4.conv1_1`.

Only the `skip4` channel slice is rotated. The ReLU branch remains unchanged.
The consumer weight absorbs the equivalent block-diagonal transform
`diag(I, R)` by transforming only the input-channel slice corresponding to
`skip4`.

All other CSPN convolution inputs are excluded because they are ReLU outputs or
belong to protected sparse-depth, depth-head, guidance, or propagation paths.

## Rotation Methods

### Random Rotation

Generate a deterministic orthogonal matrix with QR decomposition from the
experiment seed. Apply the same stored matrix to calibration and evaluation.
This is the dense research reference rather than the preferred deployment
method.

### Hadamard Rotation

Use normalized `H D`, where `H` is a Walsh-Hadamard matrix and `D` is a fixed
random sign diagonal. Both CSPN candidate channel counts must be powers of two;
otherwise configuration fails. The implementation exposes the fast transform
semantics and does not silently pad or split channels.

### Group-A4

Use one signed A4 scale per contiguous channel group at an eligible boundary.
Select one group size from `{16, 32, 64}` using calibration block-output error.
The selected group size must divide the configured channel count.

Learned Rotation is deferred. If the first-stage results justify it, a separate
design will define its orthogonal parameterization and optimization objective.

## Experiment Matrix

Run these CSPN configurations with identical data, checkpoint, weight
quantization, propagation contract, and evaluation order:

1. FP32.
2. RTN W4A4 without rotation.
3. Group-A4 without rotation.
4. Random Rotation-A4.
5. Hadamard Rotation-A4.
6. Hadamard Rotation with Group-A4.
7. Each eligible boundary enabled independently.
8. Both eligible boundaries enabled together.

Random and Hadamard comparisons use the same seed. Group size is selected only
from calibration block-output metrics. Evaluation reports all configured
variants rather than using evaluation RMSE to choose a hidden winner.

## Data Flow

1. Load the official CSPN model and compatible checkpoint.
2. Install the existing semantic and propagation adapters.
3. Capture the two declared boundary activations and corresponding FP block
   outputs over 128 calibration samples.
4. Confirm signed activation support and compute the fixed Random and Hadamard
   matrices.
5. Transform each consumer weight along its input-channel axis.
6. Verify FP block and end-to-end equivalence before enabling QDQ.
7. Calibrate W4A4 activation ranges and Group-A4 candidates.
8. Run the fixed 64-sample evaluation for every experiment configuration.
9. Write one manifest, site metrics, end-to-end metrics, and requested analysis
   plots under `profile_logs/nyu_cspn_rotation_w4a4/`.

Rotation is owned by a focused rotation component. CSPN topology and eligible
boundaries remain owned by the CSPN semantic adapter. The experiment runner
coordinates existing model loading, calibration, propagation, metric, and
prediction-export APIs without duplicating them.

## Metrics

For each eligible boundary, report before and after rotation:

- maximum absolute activation;
- `p75`, `p99`, `p99.9`, and `p99.99` absolute activation;
- channel imbalance, defined as maximum channel L2 norm divided by mean channel
  L2 norm;
- kurtosis;
- signed A4 SQNR;
- zero-code ratio;
- saturation ratio;
- FP-referenced block-output MSE and SQNR.

For each end-to-end configuration, report:

- RMSE;
- MAE;
- AbsRel;
- iRMSE;
- flat-region error;
- boundary-region error;
- NaN/Inf ratio.

Generate concise CSV outputs, activation before/after plots, an RMSE comparison,
and a prediction comparison containing GT, FP32, RTN W4A4, and the best reported
Rotation configuration. The analysis must identify the selected configuration
explicitly rather than overwriting other results.

## Correctness and Failure Handling

Before quantization, rotated block outputs and the final prediction must have
normalized RMS error at most `5e-4` and maximum error divided by reference
maximum at most `1e-3`. These scale-aware bounds account for the additional
FP32 matrix multiply and changed CUDA convolution reduction order while still
detecting an incorrect matrix orientation, fanout transform, or concat weight
slice. A failure stops the experiment.

The runner also fails directly when:

- a declared module is absent from the official model;
- a declared candidate is not signed on calibration data;
- a consumer convolution uses unsupported groups;
- Hadamard channel count is not a power of two;
- Group-A4 group size does not divide channel count;
- calibration or prediction contains non-finite values;
- checkpoint, NYU data, or CUDA execution is unavailable.

There is no automatic site skipping, bit-width promotion, method substitution,
or FP fallback.

## Tests

Focused tests cover:

- dense rotation and convolution weight absorption;
- one shared rotation absorbed by both decoder-entry fanout consumers;
- branch-local concat rotation and the corresponding consumer weight slice;
- normalized Hadamard orthogonality and transform equivalence;
- Group-A4 divisibility and scale ownership;
- the CSPN adapter declaring exactly the two approved boundaries;
- guidance, ReLU, depth-head, sparse-depth, and propagation paths remaining
  outside rotation ownership;
- guidance head remaining outside generic CNN QDQ;
- FP block and end-to-end equivalence;
- output schema and metric calculations.

Run the focused tests and the existing tests covering hardware-aligned
quantization, CSPN adapters, propagation-aware quantization, and NYU evaluation.
The real calibration/evaluation job exercises the complete runtime path, so no
separate persistent smoke experiment is retained.

## Artifact Cleanup

- Delete `profile_logs/nyu_brecq_cspn_pa_w4a4/smoke` and
  `profile_logs/nyu_brecq_cspn_pa_w4a4/smoke_v2`.
- Remove
  `profile_logs/nyu_strict_w4a4_fp4_evaluation/stress/brecq/cspn` from active
  comparisons and plots because generic A4 guidance quantization created
  zero-normalization denominators and non-finite predictions there.
- Correct comparison code so that CSPN guidance ownership is enforced by the
  quantization contract.
- Regenerate the CSPN baseline under the corrected contract.
- Store the new rotation study only under
  `profile_logs/nyu_cspn_rotation_w4a4/` without cache indexes, hashes, or
  compatibility directories.

## Follow-up Gate

After Random, Hadamard, Group-A4, and their approved combinations are reported,
judge whether rotation improves activation distribution, block-output error, or
end-to-end depth metrics. Only a demonstrated benefit proceeds to a separate
Learned Rotation and Rotation+BReCQ/QDrop design. A negative result is retained
as the CSPN conclusion rather than hidden by changing the quantization boundary.
