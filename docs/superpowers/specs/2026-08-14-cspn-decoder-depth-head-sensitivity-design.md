# CSPN Decoder and Initial-Depth Sensitivity Design

## Objective

Identify which official CSPN decoder blocks, activation boundaries, and Conv
weights are responsible for the remaining static Group-8 W4A4 degradation.
The search distinguishes activation sensitivity (`W4A8`), weight sensitivity
(`W8A4`), and their interaction (`W8A8`) before constructing a measured
accuracy-versus-precision Pareto set.

## Immutable Evaluation Contract

- Use the official CSPN ResNet-18 architecture with 24 propagation steps.
- Load `cspn_iter24/best.pt` independently for every evaluated configuration.
- Use the persisted stratified 128-sample NYU train calibration identities and
  fixed 64-sample NYU validation identities with seed `20260812`.
- Keep the encoder, non-selected decoder sites, and depth head at static
  contiguous Group-8 W4A4 MinMax.
- Keep guidance in FP32 and propagation at A8/INT16-Q13/INT32.
- Do not train or enable QAT, AdaRound, BRECQ, QDrop, SmoothQuant, rotation,
  clipping search, dynamic activation ranges, or evaluation-driven calibration.
- Do not modify the checkpoint, official forward path, model dimensions, or
  propagation iterations.

## Candidate Model

The block search covers exactly:

| Block | Official module |
|---|---|
| `decoder_layer1` | `gud_up_proj_layer1` |
| `decoder_layer2` | `gud_up_proj_layer2` |
| `decoder_layer3` | `gud_up_proj_layer3` |
| `decoder_layer4` | `gud_up_proj_layer4` |
| `initial_depth` | `gud_up_proj_layer5` |

An activation belongs to a block through the existing strict activation owner
and `owner_block` mapping. A Conv weight belongs to a block through its exact
official module prefix. Only modules and owners observed during calibration are
eligible. The candidate registry is explicit and is validated against the
executed official model; there is no prefix fallback at evaluation time.

The initial-depth output entering propagation is already A8 under the fixed
propagation-aware contract. The initial-depth block activation candidate is
therefore the input of `gud_up_proj_layer5.conv1`; its output precision is not
silently changed.

## Stage 1: Block Search

Start with `STRICT_W4A4`, then evaluate three configurations for each of the
five blocks:

- `W4A8`: promote every activation owner in the block from A4 to A8 while all
  block weights remain W4.
- `W8A4`: promote every executed Conv weight in the block from W4 to W8 while
  all block activations remain A4.
- `W8A8`: apply both promotions to the same block.

The block search therefore contains 16 configurations including the baseline.
Each configuration receives a fresh model and a fresh calibration pass.

## Stage 2: Site Search

Rank the five blocks by the best primary RMSE achieved by any of their three
Stage-1 configurations. Select the best two blocks. If no block improves on
`STRICT_W4A4`, select the two blocks with the smallest RMSE regression so that
site-level error cancellation can still be tested. Ties are resolved by block
order from decoder layer 1 through initial depth.

Within each selected block:

- evaluate one `W4A8` configuration for each individual activation owner;
- evaluate one `W8A4` configuration for each individual Conv weight;
- evaluate one `W8A8` configuration for each Conv weight together with the
  activation edges that feed that Conv.

The Conv-to-edge dependency table is explicit. Ordinary Conv inputs use their
strict input owner. `gud_up_proj_layer1.conv1` and
`gud_up_proj_layer1.sc_conv1` use `rotation.decoder_entry`.
`gud_up_proj_layer4.conv1_1` uses both the decoder branch at
`gud_up_proj_layer4.relu#0` and `rotation.layer4_signed_skip`, because its
input is a concatenation. A missing or extra dependency is an error.

## Cumulative Search and Pareto Set

Site-level promotions that individually reduce RMSE are sorted by RMSE delta,
then by added precision cost, then by stable candidate name. Starting from
`STRICT_W4A4`, apply these atomic promotions cumulatively in that order and
rerun the full 64-sample evaluation after every addition. Duplicate weight or
activation promotions are rejected rather than counted twice.

The reported Pareto set is the non-dominated subset of all measured block,
site, and cumulative configurations under:

- primary objective: lower aggregate RMSE;
- cost objective: lower normalized added bit-element cost.

For executed Conv weight elements `W_m` and quantized activation-edge elements
`A_e`, the cost is

`C = (sum_m((b_w,m - 4) W_m) + sum_e((b_a,e - 4) A_e)) /
     (4 (sum_m W_m + sum_e A_e))`.

The strict W4A4 baseline therefore has zero added cost. This logical
bit-element measure covers standalone ReLU, skip, and concat boundaries without
falsely assigning their work to one Conv. W8 weight-MAC fraction and A8
activation-element fraction are reported separately. None of these logical
counts is presented as measured kernel latency.

## Metrics and Diagnostics

For every configuration, record:

- sample and aggregate RMSE, MAE, AbsRel, iRMSE, flat RMSE, and boundary RMSE;
- per-block output MSE and SQNR against the same FP32 reference;
- initial-depth and final propagation MSE/SQNR;
- per-module MAC, weight-element, and input-element counts with effective
  weight and activation bits;
- normalized added bit-element cost, W8 weight-MAC fraction, W8
  weight-element fraction, and A8 activation-element fraction;
- paired wins and RMSE delta against `STRICT_W4A4`;
- all 64 prediction payloads for the strict baseline, the best block candidate,
  the best site candidate, and every cumulative Pareto candidate.

The primary sensitivity ranking uses aggregate sample RMSE. Local SQNR is
diagnostic only because the stem experiment demonstrated that SQNR and depth
RMSE are not monotonic.

## Failure Contract

The runner terminates on checkpoint mismatch, model-source mismatch, missing
or unexpected checkpoint tensors, changed calibration/evaluation identities,
unobserved candidate sites, duplicate quantization ownership, incomplete
sample coverage, non-finite predictions, inconsistent operation shapes, or an
existing output directory. It does not skip candidates or substitute another
site.

## Outputs

Write a new immutable result root containing:

- `stage1_block_metrics.csv`;
- `stage2_site_metrics.csv`;
- `cumulative_metrics.csv` and `pareto_metrics.csv`;
- sample, regional, block-error, operation-count, and precision-coverage CSVs;
- selected prediction payloads;
- `manifest.json` with source hashes, exact candidate registries, selected
  blocks, configuration contracts, and artifact hashes;
- one concise Markdown report and a Pareto plot generated from measured CSVs.

## Verification

Tests first cover exact candidate discovery, block ownership, Conv input-edge
dependencies, all three precision modes, per-module bit overrides, selected
block ranking, cumulative de-duplication, Pareto filtering, and immutable
protocol validation. Focused tests and the complete repository suite must pass
before CUDA evaluation. Final auditing requires exact row counts, 64 unique
sample identities per evaluated configuration, finite predictions and metrics,
prediction coverage for selected configurations, and matching artifact hashes.
