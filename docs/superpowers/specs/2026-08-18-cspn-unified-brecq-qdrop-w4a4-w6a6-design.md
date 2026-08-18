# CSPN Unified BRECQ and QDrop W4A4/W6A6 Evaluation Design

## Goal

Re-evaluate CSPN BRECQ and QDrop under the current task-sensitive evaluation
protocol at W4A4 and W6A6. The implementation modifies the existing BRECQ,
QDrop, evaluation, and plotting paths. It does not add a parallel quantization
framework or a new Python runner.

## Scope

The experiment contains four reconstructed configurations:

- BRECQ W4A4;
- BRECQ W6A6;
- QDrop W4A4;
- QDrop-style W6A6 adaptation.

The comparison also contains FP32, RTN W4A4, RTN W6A6, and P3/T3. QDrop W6A6
uses the same reconstruction algorithm as the official-aligned W4A4 path but
is reported as an extension rather than an official QDrop configuration.

The experiment is CSPN-only. It does not retrain the depth-completion model and
does not change the P3/T3 allocation.

## Unified Data Protocol

Every method uses the official CSPN checkpoint whose SHA256 is
`482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`.
The runner reads the existing persisted protocol files:

- `profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json`;
- `profile_logs/nyu_cspn_stratified_calibration_128/metadata.json`;
- the evaluation protocol already consumed by the task-sensitive runner.

The ordered 128 train indices are fixed. The first 112 samples form the
reconstruction subset and the final 16 form the reconstruction-validation
subset. All methods use the same 64 validation indices, seed, deterministic
crop, and order. QDrop seeds change optimization randomness only; they do not
change the calibration or evaluation samples.

The implementation validates checkpoint, train-list, calibration-index,
evaluation-index, and protocol hashes before reconstruction. Missing or
inconsistent inputs raise errors. There is no random-sample fallback.

## Quantization Contract

Ordinary CNN weights and activations use the named precision of each
configuration. Weights use the existing per-output-channel symmetric integer
contract. Activations use the existing static contiguous Group-8 MinMax integer
contract. Conv-BN folding occurs before calibration and bias remains FP32.

The semantic exclusions are identical for all methods and bit widths:

- guidance remains FP32;
- sparse-depth and anchor boundaries retain their current protected contract;
- CSPN propagation uses A8 state, INT16 Q13 coefficients, and INT32
  accumulation;
- propagation coefficients and center-neighbor constraints are not owned by
  generic CNN activation quantization.

BRECQ generates independent strict W4 and W6 weight contracts with the existing
strict reconstruction implementation. The matching A4 or A6 activation
contract is calibrated from the unified 128-sample set.

QDrop uses the existing activation bank, semantic edge ownership, block
reconstructor, and strict contract exporter. Both weight and activation bit
widths become explicit required configuration values restricted to `(4, 6)`.
The W4A4 configuration preserves official probability `0.5`. W6A6 uses the
same probability and optimizer as a controlled bit-width extension. Each
precision runs formal seeds `1005`, `1006`, and `1007` for 20,000 steps.

No failed reconstruction is converted to RTN, BRECQ, another seed, or a legacy
contract.

## Existing-Code Changes

`spn_quant/qdrop_config.py` validates explicit matched W4A4 or W6A6 settings
instead of requiring only W4A4. The configuration remains strict: required
keys use direct attribute or dictionary access and unknown values fail.

`scripts/run_nyu_qdrop_reconstruction.py` reads the persisted calibration and
evaluation protocols instead of drawing random indices. Its strict manifest
records the actual bit widths, source hashes, split identities, seed, and
QDrop variant.

`scripts/run_nyu_strict_reconstruction.py` accepts the same persisted
calibration protocol for BRECQ and records it in the deployment contract. Its
existing `--w-bits` path is used for W4 and W6; no separate BRECQ runner is
created.

`scripts/run_nyu_qdrop_w4a4.py` remains the existing orchestration entry even
though it now evaluates both named precisions. Its CLI and output metadata make
the matrix explicit and reject incomplete method/precision/seed coverage.

`scripts/run_nyu_edge_quantization.py` replays strict BRECQ and QDrop contracts
with the matching A4 or A6 edge contract under the existing propagation-aware
ownership rules.

`scripts/plot_nyu_qdrop_w4a4.py` verifies sample identity and renders the new
comparison outputs. Existing module names are retained to avoid adding a
second framework; user-facing labels and metadata state the actual precision.

## Evaluation and Figures

Each of the 64 evaluation samples records RMSE, MAE, AbsRel, iRMSE, non-finite
pixel count, and method/precision/seed identity. Aggregates report each BRECQ
configuration once and each QDrop configuration by three-seed mean, standard
deviation, minimum, and maximum.

Two prediction figures avoid an unreadable single wide sheet:

- W4A4: `GT | FP32 | RTN | BRECQ | QDrop`;
- W6A6: `GT | FP32 | RTN | BRECQ | QDrop | P3/T3`.

The displayed QDrop seed is selected by median reconstruction-validation RMSE
before evaluation. Test RMSE is never used for seed selection. Figures use the
same sample rows and color limits for every method.

The final summary table compares all eight configurations on the same 64
samples and includes bit-width and reconstruction metadata. BRECQ/QDrop are
not described as better unless paired metrics under this protocol support the
claim.

## Parallel Execution

The four available A100 GPUs execute independent reconstruction jobs. QDrop
seeds run as separate jobs; BRECQ W4 and W6 use the remaining device in
sequence. Evaluation starts only after every required contract passes source
hash, bit-width, target-coverage, and finite-parameter checks.

Partial outputs remain in the new experiment root and are never interpreted as
formal results. Resume accepts only a contract whose manifest matches the full
protocol identity.

## Historical Artifact Cleanup

The new experiment writes to a distinct current-protocol root. Cleanup occurs
only after all four reconstructed configurations, baselines, 64-sample metrics,
figures, and artifact hashes pass validation.

The following old incompatible-protocol roots are then deleted:

- `profile_logs/nyu_qdrop_w4a4`;
- `profile_logs/nyu_brecq_cspn_pa_w4a4`.

Training data, checkpoints, stratified calibration artifacts, P3/T3, uniform
W6A6, and other reproducible experiments remain unchanged. Existing reports
that cite the deleted results are marked as superseded rather than silently
retaining those numbers as current results.

## Tests and Acceptance

Tests are added before implementation changes and cover:

- QDrop accepts exactly matched W4A4 and W6A6 and rejects mixed or unsupported
  bit widths;
- BRECQ and QDrop consume the exact persisted 112/16 split and 64 evaluation
  identities;
- optimization seeds do not alter sample identities;
- strict manifests preserve source hashes, bit widths, propagation ownership,
  and activation-contract coverage;
- no RTN or legacy-contract fallback is possible;
- aggregation rejects missing methods, precisions, seeds, duplicate samples,
  or mismatched GT/FP32 arrays and reports non-finite predictions as measured
  failures instead of dropping them;
- figures contain the declared columns and use only aligned predictions;
- cleanup refuses to run before the new result manifest is complete.

Acceptance requires all focused tests and the complete repository test suite to
pass, all methods to cover the same 64 samples, every non-finite result to be
accounted for explicitly, and the final artifact manifest to reproduce its
recorded hashes. A finite-output requirement is applied when judging a
quantized configuration as accuracy-preserving, not when deciding whether a
negative experiment completed correctly.
