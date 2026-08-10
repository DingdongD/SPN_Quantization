# CompletionFormer Front-Encoder W8A8 Pareto Search Design

## Goal

Find a small set of early CompletionFormer encoder blocks that should remain
W8A8 while the rest of the current joint reference remains W4A4. Report the
accuracy-cost Pareto frontier rather than selecting layers against a fixed
RMSE threshold. This experiment performs post-training quantization only and
does not retrain or update the checkpoint.

## Immutable Baseline

The experiment starts from `JIQ_Joint_W4A4` and uses the same official full
CompletionFormer source, strict converged checkpoint, NYU preprocessing,
two-pass quantization calibration, integer Attention/concat contract, and
propagation-aware A8 settings as the completed fixed-64 evaluation.

The following behavior remains unchanged outside the promoted front-encoder
units:

- ordinary Conv/Linear weights and activations remain W4A4;
- Q, K, and V remain independently quantized W4A4 in INT8 lanes;
- QK and probability-V products use INT32 accumulation;
- softmax remains FP16 and its probability output remains unsigned A8;
- joint `concat_conv` modules remain W4A4 with separate branch scales;
- affinity, confidence, offsets, and propagation state remain on the existing
  propagation-aware A8 contract.

FP32 and the existing full-activation `JIQ_W4A8` result are retained as
accuracy references. They are not candidates in the W8A8 layer search.

## Atomic Promotion Units

A selected unit promotes every ordinary quantized Conv/Linear weight, input,
output, and associated ReLU activation site in that unit to 8 bits. A unit is
never partially promoted during the search. Conv-BN folding is completed
before calibration, as in the existing backend.

The nine units follow official forward order:

1. `Stem`: `backbone.conv1_rgb`, `backbone.conv1_dep`, the merged input of
   `backbone.conv1`, and `backbone.conv1`, including associated ReLU sites.
2. `Embed1.0`: the complete `backbone.former.embed_layer1.0` residual block.
3. `Embed1.1`: the complete `backbone.former.embed_layer1.1` residual block.
4. `Embed1.2`: the complete `backbone.former.embed_layer1.2` residual block.
5. `Embed2.0`: the complete `backbone.former.embed_layer2.0` residual block,
   including its downsample path.
6. `Embed2.1`: the complete `backbone.former.embed_layer2.1` residual block.
7. `Embed2.2`: the complete `backbone.former.embed_layer2.2` residual block.
8. `Embed2.3`: the complete `backbone.former.embed_layer2.3` residual block.
9. `PatchEmbed1`: `backbone.former.patch_embed1.proj` and its LayerNorm output
   activation boundary.

The stem is one atomic unit because independently retaining one RGB/depth
branch at A8 while requantizing their merged consumer input to A4 does not
preserve that branch's information. Attention blocks, their CNN residual
branches, and their concat fusion are outside this front-encoder search.

Module membership is resolved against the official model and validated
strictly. Unknown modules, empty units, overlapping weight ownership, missing
activation sites, or a model structure different from the declared nine-unit
contract must fail before evaluation.

## Bit-Override Semantics

The hardware-aligned instrumentor gains explicit per-module weight-bit
overrides in addition to its existing activation-bit overrides. Configuration
uses direct dictionary indexing for required fields. A promoted unit declares
8-bit weight overrides for all owned Conv/Linear modules and 8-bit activation
overrides for their input, output, fused LayerNorm, and associated ReLU sites.

The default remains per-output-channel signed symmetric weight quantization.
Nonnegative ReLU/input activations remain unsigned uniform activations; other
activations remain signed uniform. Bias uses the selected input and weight
scales through the existing INT32 bias contract. No hidden fallback changes an
unresolved override to the default bit width.

## Dataset Isolation

Three deterministic and disjoint roles are used:

- **Calibration:** the existing 64 NYU training indices selected with seed
  `20260804`; these determine quantization ranges.
- **Search:** 32 NYU training indices selected with seed `20260810`, excluding
  all calibration indices; these rank candidate W8A8 sets.
- **Final evaluation:** the existing ordered 64 NYU validation indices; these
  are never used to select a block or quantization scale.

The output metadata records all three index lists and asserts that calibration
and search indices are disjoint. Final evaluation must not begin if model,
checkpoint, source revision, index identity, or calibration observations differ
from the strict reference contract.

## Search Procedure

The experiment evaluates two complementary paths.

### Cost-aware greedy path

Start with no promoted unit. At each of nine rounds, evaluate the addition of
every remaining unit on the fixed search samples. Choose the addition with the
largest positive search-RMSE reduction divided by its incremental whole-model
W8A8 MAC share. If every addition increases RMSE, choose the addition with the
smallest RMSE increase so that the full nested path is still measured. Ties are
resolved by lower incremental MAC share and then official unit order.

This produces ten nested configurations, including the W4A4 baseline and the
configuration containing all nine front-encoder units. Every candidate
evaluated during the greedy search is persisted, not only each winning step.

### Strict-prefix path

Evaluate the nine official-order prefixes beginning with `Stem`. These points
provide a deployment-friendly contiguous alternative and reveal whether the
non-contiguous greedy set is materially better than retaining the first N
blocks.

Only the greedy winners, strict prefixes, FP32, and `JIQ_W4A8` proceed to the
fixed 64-sample final evaluation. Duplicate sets are evaluated once.

## Cost Accounting

Costs are measured from one FP32 forward with runtime tensor shapes and count
every invocation of ordinary quantized `Conv2d`, `ConvTranspose2d`, and
`Linear` modules:

- Conv MACs are
  `N * Hout * Wout * Cout * (Cin / groups) * Kh * Kw`;
- Linear MACs are the number of output vectors multiplied by
  `in_features * out_features`;
- parameter cost is the number of quantized weight elements;
- operator count is the number of unique ordinary quantized modules.

The primary Pareto x-axis is promoted W8A8 MACs divided by all ordinary
quantized Conv/ConvTranspose/Linear MACs in the complete model. Secondary
tables and plots report whole-model parameter and operator-count shares, plus
front-encoder-only shares. Custom QK/AV and propagation MACs are excluded from
these denominators and are labeled separately because they do not represent
promoted W8A8 weight layers.

## Pareto And Selection Rules

A final-evaluation point is non-dominated when no other measured configuration
has both lower or equal W8A8 MAC share and lower or equal mean RMSE, with at
least one strict improvement. The report preserves all measured points and
marks the non-dominated frontier.

The report identifies:

- the lowest-cost point that improves mean RMSE over `JIQ_Joint_W4A4`;
- the Pareto knee, defined as the point with maximum perpendicular distance
  from the normalized line joining the lowest-cost and lowest-RMSE frontier
  endpoints;
- the lowest-RMSE measured front-encoder W8A8 set.

No fixed RMSE threshold is used to alter the search.

## Outputs

Generate under a new result root without modifying the completed joint result:

```text
profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64/
```

Persist:

- unit-to-module and activation-site manifest;
- runtime MAC, parameter, and operator-count cost manifest;
- every greedy search candidate and its 32-sample aggregate metrics;
- chosen greedy path and strict-prefix configurations;
- 64-sample per-sample and aggregate final metrics;
- metadata with source/checkpoint hashes and all index lists;
- prediction payloads for FP32, W4A4, W4A8, the Pareto knee, and the
  lowest-RMSE front-encoder set;
- a mean-RMSE versus W8A8 MAC-share Pareto plot;
- secondary parameter-share and operator-count-share Pareto plots;
- a GT/FP32/W4A4/W4A8/knee/best prediction comparison sheet.

All plots use Arial-compatible styling, non-rotated labels, grid lines below
artists, and explicit labels for promoted unit sets.

## Verification

Tests must be written before implementation and cover exact unit discovery,
weight and activation override ownership, per-module weight bits, deterministic
disjoint search indices, MAC accounting, greedy tie-breaking, Pareto filtering,
and output-table completeness.

Before reporting results:

- run the complete test suite and CUDA integer tests;
- verify every final configuration has 64 unique finite sample rows;
- verify each exported prediction configuration has 64 payloads;
- verify calibration/search/evaluation index identity and disjointness;
- verify every promoted unit is W8A8 at all declared sites and every
  non-promoted front-encoder site remains W4A4;
- verify source commit and checkpoint SHA256 match the completed official
  CompletionFormer reference;
- visually inspect the Pareto and prediction figures.
