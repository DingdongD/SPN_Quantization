# Quantization Framework Inventory

This document is the source of truth for supported quantization workflows in
this repository. Results use metres for depth RMSE. Generated experiment data
under `profile_logs/` is not tracked by Git.

## Active Deployment Frameworks

| Framework | Active precision | Scope |
| --- | --- | --- |
| RTN | W8A8, W4A8 | Per-output-channel symmetric weights and uniform integer activations |
| Hardware-aligned QDQ | W8A8, W4A8 | Conv-BN folding, signed/unsigned activation contracts, INT32 bias, Add/Concat requantization |
| Propagation-aware QDQ | W8A8, W4A8, diagnostic W4A4 | Quantize-then-normalize affinity, Q13 coefficients, A8 confidence/offset/state, sparse-anchor restoration |
| Mixed precision | P3/T3 and task-sensitive assignments | Explicit per-layer weight and activation bits in `{2,4,6,8}` |
| AdaRound | W4A8 | Strict weight contract reconstructed against the original model graph |
| BRECQ | W4A8, W6A6 | Strict weight reconstruction at W4A8 and deployment-aligned joint block reconstruction at W6A6 |
| QDrop | W6A6 | Official-style joint weight/activation reconstruction with propagation boundaries excluded |
| Group8 QAT | Static-G8 W4A4 and Dynamic-G8 W4A4 | Hard-forward STE training and fresh hard-path evaluation |
| Mixed task-aware QAT | P3/T3 W4/W8 weights and A4/A6/A8 activations | Budget-constrained activation search, propagation-aware task loss, and canonical hard deployment |
| CompletionFormer joint integer | W4A4 research reference, W4A8 comparison | Explicit attention Q/K/V, QK/AV, softmax and concat scale contracts |
| Three-model selected matrix | FP32 plus nine exact selected methods | Explicit DySPN/NLSPN/CompletionFormer DAG, immutable artifacts, fixed-64 pooled evaluation |

W4A4 entries above are research or training configurations, not accepted
deployment configurations. W8A8 is the stable low-risk baseline across the
four official model structures. CSPN mixed task-aware QAT is the strongest
measured configuration under an activation-element-weighted average budget of
at most 6 bits: fixed-64 RMSE is `0.171399 m` at average A `5.917515` bits.

## Active Research And Diagnostics

- MinMax and zero-aware calibration.
- The fixed 128-sample CSPN stratified calibration set.
- Contiguous per-tensor and Group8 activation quantization.
- Encoder, decoder, stem, depth-head and propagation sensitivity scans.
- Activation p75/p99/p99.9/p99.99/max, zero-code, saturation, kurtosis and
  channel-imbalance diagnostics.
- Integer W4A4 activation histograms using `PA_W4A4_PROP_A8`.
- PA-W8A8 im2col channel, kernel-offset and spatial-token diagnostics.
- CompletionFormer attention and concat scale audits.

## Retired Methods

These methods have no active implementation, CLI option or compatibility
alias. The source values below were checked against the corresponding CSV
files before generated artifacts were removed.

| Method | Reference RMSE | Method RMSE | Decision |
| --- | ---: | ---: | --- |
| SmoothQuant, W4-only | 0.204621 | 0.204136 | Retired: 0.24% gain is too small and does not transfer to W4A4 |
| SmoothQuant, W4A4 Group8 | 0.347124 | 0.359289 | Retired: worse than baseline |
| SmoothQuant, W4A4 Group16 | 0.417303 | 0.407825 | Retired: small isolated gain, still poor absolute accuracy |
| Channel rotation, Group W4A4 | 0.426802 | 0.430580 | Retired: best rotation result is worse |
| Scale-aware Group8 | 0.313318 | 0.316218 | Retired: worse than contiguous Group8 |
| Percentile P99.9 | 0.313318 | 1.354118 | Retired: severe regression |
| Percentile P99.99 | 0.313318 | 0.490240 | Retired: severe regression |
| Histogram-MSE calibration | 0.313318 | 0.988717 | Retired: severe regression |

LogNP/selective LogNP, random/Hadamard rotation, AWQ-style CNN clipping,
RMS/Max clustering, Snake spread, scale-aware permutation, and outlier-channel
isolation/splitting are also retired. Their block-level improvements did not
produce stable end-to-end gains.

Strict CSPN RTN W4A4 and BRECQ W4A4 both produced 64/64 invalid samples in the
unified strict study. No four-model W4-E2M1 configuration met the retention
criterion. FP4 E2M1 and its evaluation workflow are therefore retired.
AdaRound/BRECQ W4A8 results remain accepted for CSPN, NLSPN and
CompletionFormer; BRECQ W4A8 also remains accepted for DySPN.

## Retired Configurations In Active Frameworks

- RTN, AdaRound and BRECQ W4A4 are not deployment candidates.
- QDrop W4A4 is not retained as an accepted result; QDrop W6A6 remains active.
- BRECQ W6A6 uses the deployment-aligned `brecq_joint_strict` contract. Its
  fixed-64 RMSE is `0.194704 m` with zero invalid predictions.
- Propagation-aware W4A4 remains only for error-source and invariant analysis.
- CompletionFormer joint W4A4 remains a correctness reference, not a native
  integer-kernel performance claim.

## Retained Artifact Roots

- `profile_logs/nyu_propagation_aware_quantization_unified`
- `profile_logs/nyu_strict_w4a8_evaluation`
- `profile_logs/nyu_strict_w4a8_reconstruction_current`
- `profile_logs/nyu_cspn_qdrop_w6a6_64`
- `profile_logs/nyu_cspn_brecq_w6a6_aligned_64`
- `profile_logs/nyu_cspn_group_a4_qat`
- `profile_logs/nyu_cspn_mixed_task_aware_qat`
- `profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64`
- `profile_logs/nyu_cspn_stratified_calibration_128`
- `profile_logs/nyu_completionformer_joint_integer_64`
- `profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64`
- `profile_logs/nyu_w4a4_activation_histograms_64`
- `profile_logs/nyu_cspn_w8a8_im2col_64`

## Rerun Commands

All commands run from the repository root. Paths in angle brackets are
required experiment inputs and must refer to the same checkpoint and sample
protocol when results are compared.

### DySPN, NLSPN, And CompletionFormer Selected Matrix

The immutable method order is FP32, RTN W8A8, RTN W4A4, QDrop W6A6, BRECQ
W6A6, HAWQ mixed<=6, LSQ++ W6A6, LSQ++ W4A4, mixed task-aware QAT, and
model-relative P3/T3 mixed PTQ. The exact official checkpoints, 128-sample
calibration metadata paths, 64 validation identities, interpreters, and CUDA
devices are in `configs/three_model_selected_quantization.json`. The exact
operational environments and hyperparameters are in
`configs/three_model_quantization_launch.json`. That launch contract explicitly
uses the reviewed `--skip-conv-bn-fold` policy for all three models. It is an
exact matrix setting and is never selected dynamically after a failed fold.

The per-model static producer writes these five inputs at the configured formal
root from the official train and validation splits:

| Input | Contract |
| --- | --- |
| `calibration_metadata.json` | Ordered 128 train identities and stratified selection evidence |
| `calibration_indices.json` | QDrop/BRECQ calibration identity contract |
| `evaluation_protocol.json` | Exact ordered 64 validation identities |
| `weight_cost_rows.csv` | Complete `module,macs` rows for contract-owned weights |
| `activation_cost_rows.csv` | Complete `site,role,elements` rows for activation traffic |

The exact producer commands are the three `prepare_static_inputs` jobs in the
generated plan. Each names its absolute model interpreter, indexed CUDA device,
full replacement environment, seeds (`20260824` and `20260812`), 256 candidate
train samples, 32 tail samples, and all five destinations. Each corresponding
`validate_static_inputs` job checks exact JSON/CSV schemas, ordered 128/64
identities, dataset-list and checkpoint hashes, descriptor schema, positive
costs, complete cost coverage, and all cross-file identities before P3/T3,
HAWQ, PTQ, or QAT can run.

Planning performs no search, reconstruction, training, or evaluation:

```bash
/opt/conda/bin/python \
  scripts/launch_nyu_three_model_quantization.py plan \
  --config "$PWD/configs/three_model_selected_quantization.json" \
  --launch-spec "$PWD/configs/three_model_quantization_launch.json"
```

Review the generated `launch/launch_plan.json` and all 70 job manifests before
starting the formal run. Formal execution has no idle-GPU, interpreter,
environment, precision, or backend substitution:

```bash
/opt/conda/bin/python \
  scripts/launch_nyu_three_model_quantization.py execute \
  --config "$PWD/configs/three_model_selected_quantization.json" \
  --launch-spec "$PWD/configs/three_model_quantization_launch.json" \
  --plan /workspace/SPN_Quantization/profile_logs/nyu_three_model_selected_quantization/launch/launch_plan.json
```

The per-model artifact order is:

1. Static-input production, then semantic validation before method artifacts.
2. P3/T3 search before selected PTQ and mixed task-aware QAT.
3. HAWQ trace under the model interpreter, allocation under the explicitly
   declared orchestrator interpreter, then HAWQ QAT.
4. LSQ++ W4A4 and W6A6 QAT from the same official checkpoint.
5. `formal/formal_artifacts.json` only after the selected PTQ matrix and all
   four terminal QAT checkpoints exist and validate.
6. Ten serial fixed-64 evaluations, aggregate tables, and prediction figures.
7. `cross_model_summary.json` and `.csv` only after all three model plots and
   exact ten-row model summaries exist.

The official one-sample smoke harness covers the exact FP32, RTN W8A8/W4A4,
QDrop/BRECQ W6A6 hard, LSQ++ W4A4 step, HAWQ probe, and P3/T3 candidate matrix.
It asserts native extension and official propagation calls, finite
`[1,1,228,304]` output, and model propagation invariants. The HAWQ curvature
mode is explicit and persisted: all three models use central finite-difference
generalized Gauss-Newton block traces with `epsilon=0.001`.

Formal accuracy uses pooled RMSE from global squared-error sum and global valid
pixel count. Mean per-sample RMSE is a separately labeled diagnostic. Bit cost
uses MAC-weighted average weight bits and activation-traffic-weighted average
activation bits; the HAWQ weight and activation limits are independent and no
greater than six bits. P3/T3 normalized costs divide bit-weighted MACs or
activation elements by the corresponding uniform W4 denominator. Every final
row remains bound to the official checkpoint, calibration/evaluation
identities, P3/T3 or HAWQ assignment, hard-deployment manifest, terminal QAT
checkpoint, 64 prediction files, metrics, diagnostics, and artifact hashes.

### RTN W8A8 And W4A8

```bash
python scripts/run_nyu_rtn_quantization.py \
  --run-dir <run-dir> --checkpoint best.pt \
  --sample-metrics <fixed64-sample-metrics.csv> \
  --data-root <nyu-workspace> --out-dir <output-root> \
  --device cuda:0 --seed 20260804 --calibration-samples 128 \
  --max-eval-samples 64 --quant-backend rtn \
  --config-names FP32 W8A8_full W4A8_full \
  --export-prediction-configs FP32 W8A8_full W4A8_full
```

Use `--quant-backend hardware` with `HW_W8A8_full` and `HW_W4A8_full`, or
`--quant-backend propagation` with `PA_W8A8` and `PA_W4A8`, for the matching
hardware and propagation-aware contracts.

### P3/T3 Mixed Precision

```bash
python scripts/run_nyu_cspn_task_sensitive_bits.py \
  --run-dir <cspn-run-dir> --checkpoint <cspn-checkpoint> \
  --data-root <nyu-workspace> \
  --calibration-indices <stratified-root>/calibration_indices.json \
  --calibration-metadata <stratified-root>/metadata.json \
  --evaluation-protocol <activation-resolution-root>/cspn/metadata.json \
  --out-dir <output-root> --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --seed 20260812 --fold-max-error 0.05 --beam-width 512 \
  --joint-measured-limit 128 --local-round-limit 3 \
  --refinement-block-limit 4 --refinement-width 128 \
  --refinement-measured-limit 128
```

### AdaRound And BRECQ W4A8

Run once with each method:

```bash
python scripts/run_nyu_strict_reconstruction.py \
  --run-dir <run-dir> --checkpoint best.pt --data-root <nyu-workspace> \
  --calibration-indices <calibration-indices.json> \
  --calibration-metadata <calibration-metadata.json> \
  --evaluation-protocol <evaluation-metadata.json> \
  --method adaround_strict --w-bits 4 --device cuda:0 \
  --out-dir <adaround-contract-root>

python scripts/run_nyu_strict_reconstruction.py \
  --run-dir <run-dir> --checkpoint best.pt --data-root <nyu-workspace> \
  --calibration-indices <calibration-indices.json> \
  --calibration-metadata <calibration-metadata.json> \
  --evaluation-protocol <evaluation-metadata.json> \
  --method brecq_strict --w-bits 4 --device cuda:0 \
  --out-dir <brecq-contract-root>
```

Evaluate each generated deployment contract with
`scripts/run_nyu_edge_quantization.py` using `HW_W4A8_full`.

### QDrop W6A6

```bash
python scripts/run_nyu_qdrop_reconstruction.py \
  --config configs/qdrop_w4a4_official.json \
  --run-dir <run-dir> --checkpoint <checkpoint> \
  --data-root <nyu-workspace> --model cspn --precision W6A6 \
  --phase formal --seed 1006 \
  --calibration-indices <calibration-indices.json> \
  --calibration-metadata <calibration-metadata.json> \
  --evaluation-protocol <evaluation-metadata.json> \
  --out-dir <qdrop-w6a6-root>
```

### Static And Dynamic Group8 QAT

Run the following command twice with `MODE=static` and `MODE=dynamic`, then
evaluate both hard contracts together:

```bash
python scripts/train_nyu_cspn_group_a4_qat.py \
  --mode "$MODE" --checkpoint <cspn-checkpoint> \
  --data-root <nyu-workspace> \
  --calibration-metadata <stratified-metadata.json> \
  --output-root <qat-output-root> --device cuda:0 \
  --epochs 30 --patience 6 --min-relative-improvement 0.001 \
  --batch-size 4 --val-batch-size 1 --workers 2 \
  --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 \
  --max-gradient-norm 10 --seed 20260812 \
  --max-train-samples 0 --max-val-samples 0 \
  --fold-max-error 0.05 --log-interval 50
```

```bash
python scripts/evaluate_nyu_cspn_group_a4_qat.py \
  --device cuda:0 --fp32-checkpoint <fp32.pt> \
  --static-checkpoint <static-best.pt> \
  --dynamic-checkpoint <dynamic-best.pt> \
  --data-root <nyu-workspace> \
  --calibration-metadata <stratified-metadata.json> \
  --output-root <qat-evaluation-root> --batch-size 1 --workers 4 \
  --seed 20260812 --fold-max-error 0.05 --sample-capacity 1000000
```

### Mixed Task-Aware QAT

Search and train with `configs/cspn_mixed_task_aware_qat.json`, then evaluate
the canonical best checkpoint with the strict mixed protocol:

```bash
python scripts/evaluate_nyu_cspn_group_a4_qat.py \
  --mixed-protocol \
  --precision-config configs/cspn_mixed_task_aware_qat.json \
  --device cuda:0 --fp32-checkpoint <fp32.pt> \
  --mixed-checkpoint <mixed-static-best.pt> \
  --assignment <search-root>/selected_assignment.json \
  --cost-basis <search-root>/cost_basis.json \
  --data-root <nyu-workspace> \
  --calibration-metadata <combined-calibration-metadata.json> \
  --output-root <mixed-evaluation-root> --batch-size 1 --workers 4 \
  --seed 20260812 --fold-max-error 0.05 --sample-capacity 1000000
```

Measured details, hashes, and fixed-64 acceptance evidence are recorded in
`docs/2026-08-21-cspn-mixed-task-aware-qat-results.md`.

### CompletionFormer Attention And Concat

```bash
COMPLETIONFORMER_RUN_DIR=<completionformer-run-dir> \
COMPLETIONFORMER_REFERENCE_METRICS=<fixed64-sample-metrics.csv> \
SPN_DATA_ROOT=<nyu-workspace> \
COMPLETIONFORMER_ROOT="$PWD/external/CompletionFormer" \
COMPLETIONFORMER_PYTHON=<completionformer-python> \
COMPLETIONFORMER_DEVICE=cuda:0 \
scripts/run_completionformer_joint_quantization.sh
```

### Activation Histograms

Set the required model Python and GPU environment variables documented by the
launcher, then run:

```bash
W4A4_HISTOGRAM_OUTPUT_ROOT=<new-output-root> \
scripts/run_w4a4_activation_histograms.sh full
```

### PA-W8A8 Im2col Diagnostics

```bash
python scripts/run_nyu_cspn_w8a8_im2col.py \
  --device cuda:0 --checkpoint <cspn-checkpoint> \
  --data-root <nyu-workspace> \
  --stratified-metadata <stratified-metadata.json> \
  --output-dir <new-output-root> --percentile-capacity 1000000 \
  --token-topk 64 --token-chunk 4096 --plot-layer-count 8 \
  --plot-sample-count 8 --fold-max-error 0.05
```

## Artifact Cleanup

| Measurement | Bytes |
| --- | ---: |
| Before cleanup | 15,100,345,202 |
| After cleanup | 9,765,169,301 |
| Reclaimed | 5,335,175,901 |

The older `nyu_propagation_aware_quantization` root was retained. All 1,592
old files had a relative counterpart in the unified root, but 44 metric or
metadata files had different sizes, so exact supersession was not proven.

The retained `nyu_cspn_qdrop_w6a6_64` root contains three reconstruction
contracts, 192 evaluation predictions, 192 filtered sample rows, and the W6A6
prediction figures. Its 259 retained files are covered by the root SHA-256
manifest.
