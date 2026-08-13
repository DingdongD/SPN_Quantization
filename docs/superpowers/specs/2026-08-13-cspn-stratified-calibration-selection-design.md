# CSPN Stratified Calibration Selection Design

## Goal

Construct a deterministic 128-sample CSPN calibration set from the NYU train
split that jointly covers raw RGB/depth/sparse-depth variation and selected
FP32 activation variation. Compare its train-only coverage against the current
uniform random 128-sample set without using validation samples, endpoint RMSE,
or quantized evaluation outputs during selection.

## Data isolation

The source is the official 6,700-sample NYU train split used by the converged
CSPN run. The current seed-`20260812` random-128 indices are materialized first.
Audit seed `20260813` then reserves 512 unique train indices from their
complement. This makes the audit split disjoint from both the current baseline
and the new selector. Audit indices do not participate in normalization,
quantile thresholds, candidate selection, medoid fitting, or configuration.

The remaining 6,188 train indices are selection-eligible. Their dataset access
uses the official CSPN training preprocessing. Every sample is deterministic
through the existing `seed + sample_index` RNG contract and contains one
augmented RGB/depth view and up to 500 valid sparse-depth points. The official
sampler returns fewer points when an augmented sample has fewer than 500 valid
GT pixels; this count is recorded and never repaired.

The fixed NYU validation indices, validation tensors, ground truth, prediction
metrics, and RMSE are prohibited inputs to the selector. They remain available
only to a later, separately invoked quantization evaluation.

## Two-stage selection

The selection has two explicit stages:

1. Scan raw descriptors for all 6,188 eligible samples. Select 1,024 candidates
   using raw tail coverage followed by deterministic greedy k-center coverage.
2. Run the official FP32 CSPN checkpoint on those 1,024 candidates, append
   activation descriptors, and select the final 128 samples as 32 tail-cover
   samples plus 96 k-medoids.

All index sets are unique and sorted only when serialized. Selection order and
selection reason are preserved separately. A missing descriptor, non-finite
value, duplicate index, wrong set size, zero-IQR feature, missing activation
owner, uncovered tail condition, or empty medoid cluster is an error. The
implementation does not infer defaults, relax thresholds, substitute random
samples, or fall back to the current random calibration set.

## Raw descriptors

Depth descriptors are computed over valid GT pixels:

- mean;
- p50;
- p95;
- maximum;
- valid-pixel ratio as a diagnostic field. It is excluded from distance because
  at least 75% of the official augmented train samples are fully valid.

RGB descriptors are computed on the selected augmented view before CSPN model
normalization. The descriptor loader returns this RGB tensor together with the
official CSPN input and verifies model-input parity against the existing
dataset for fixed sample seeds:

- luminance mean;
- luminance standard deviation;
- mean per-channel RGB standard deviation as contrast;
- Sobel luminance edge density, defined as the fraction of pixels whose
  gradient magnitude exceeds the fixed normalized-luminance threshold `0.1`.

Sparse-depth descriptors are:

- valid point count as a diagnostic contract, excluded from distance because
  the requested budget is fixed at 500;
- valid occupancy for each of four image quadrants;
- occupied-cell ratio on a 16 by 16 spatial grid. The finer grid prevents the
  500-point sampler from saturating the occupancy statistic;
- normalized RMS distance of valid coordinates from their centroid.

The valid sparse count must be in `[1, 500]` and must equal
`min(500, valid_depth_pixels)` for every selected and audit sample.

## Activation descriptors

The official converged CSPN ResNet-18 with 24 propagation iterations and
checkpoint `cspn_iter24/best.pt` runs in FP32 evaluation mode. Activation hooks
capture outputs at these fixed modules:

- encoder stem: `conv1_1` and `relu#0`;
- encoder sensitive ReLU: `layer1.0.relu#1`;
- decoder fusion projections: `gud_up_proj_layer2.sc_conv1` and
  `gud_up_proj_layer4.sc_conv1`;
- decoder sensitive ReLU: `gud_up_proj_layer4.relu#0`.

For every sample and owner, the selector records:

- tensor absolute p99;
- tensor absolute maximum;
- p99 divided by maximum;
- channel RMS imbalance,
  `max(channel_rms) / (mean(channel_rms) + epsilon)`.

Reused ReLU modules are identified by the existing call-index convention.
Each declared owner must produce exactly one tensor per model forward. The
selector records FP32 activations only; no QDQ, propagation quantization, or
quantized weights are active.

## Feature normalization and distance

Features are divided into four groups: depth, RGB, sparse depth, and
activations. Stage-one raw features are normalized from the 6,188 eligible
population. Stage-two combined features are independently normalized from the
1,024-candidate population. Each non-diagnostic scalar dimension uses median
and IQR. The stage-two normalizer is frozen and reused for audit and baseline
comparison. Zero or non-finite IQR is an error.

Within each group, squared standardized differences are averaged across that
group's dimensions. The total sample distance is the sum of the four group
distances divided by four. This keeps the activation group from dominating
only because it has more dimensions.

## Candidate and final algorithms

For the 1,024 raw candidate pool, low and high tails are defined by p5 and p95
for every non-diagnostic raw descriptor. Deterministic greedy set cover first
chooses 256 samples that cover the greatest number of uncovered tail
conditions. After every condition is represented, unused tail slots are filled
by descending aggregate absolute tail distance. The remaining 768 slots are
filled by greedy k-center using the raw grouped distance. Ties use sample index
ascending.

For the final 128 samples, tails are recomputed on the 1,024-candidate combined
raw and activation descriptors. Greedy set cover must cover every low and high
tail condition within 32 unique samples. Unused tail slots are filled by
descending aggregate absolute tail distance. The remaining 96 samples are
selected by deterministic weighted k-medoids while the 32 tail samples remain
fixed centers. Each candidate is weighted by the number of eligible train
samples in its stage-one raw-feature Voronoi cell. Initialization greedily
minimizes weighted distance to the fixed and selected centers; assignment and
weighted within-cluster medoid updates repeat until medoids stop changing. Ties
use sample index ascending. An empty cluster is an error.

The final selection records whether each sample was selected for raw tail,
activation tail, both, or a medoid cluster. Evaluation metrics never break a
tie or choose a budget.

## Independent audit

After final selection, raw and FP32 activation descriptors are collected for
the 512 reserved audit samples. The audit compares stratified-128 with the
existing random-128 and with 16 deterministic uniform random 128-sample
baselines drawn from the same 6,188 eligible pool.

For every scalar descriptor, report:

- calibration minimum and maximum;
- audit p01, p50, p99, and maximum;
- fraction of audit values inside the calibration range;
- calibration-to-audit quantile ratios where the denominator is nonzero;
- one-dimensional Wasserstein distance in the standardized domain.

For the grouped feature space, report audit nearest-calibration distance p50
and p95. For activation owners, report whether the audit maximum exceeds the
calibration maximum and the exceedance ratio.

The run is accepted only when:

- all raw and activation low/high tail conditions are represented;
- stratified p95 nearest-calibration distance is lower than the mean of the 16
  random baselines;
- its count of uncovered activation maxima is no greater than the current
  random-128 set;
- all 128 calibration and 512 audit indices are unique and disjoint;
- all descriptors and distances are finite.

Failure is reported directly. There is no threshold relaxation or alternate
selection path.

## Outputs

The selector writes no images. Its output directory contains:

- `calibration_indices.json`: ordered final indices and immutable selection
  metadata;
- `audit_indices.json`: ordered reserved audit indices;
- `raw_descriptors.csv`: raw descriptors for eligible and audit samples;
- `activation_descriptors.csv`: activation descriptors for candidates and
  audit samples;
- `candidate_selection.csv`: 1,024 candidate reasons and order;
- `calibration_selection.csv`: 128 final reasons, tail conditions, and medoid
  cluster identifiers;
- `descriptor_coverage.csv`: per-descriptor stratified/random audit metrics;
- `distance_coverage.csv`: grouped nearest-distance metrics;
- `activation_range_coverage.csv`: per-owner maximum coverage;
- `metadata.json`: checkpoint, architecture, seeds, dataset lists, counts,
  feature schema, algorithm parameters, and acceptance result;
- `coverage_report.md`: concise numeric findings without plots.

## Integration boundary

Existing quantization runners continue to accept explicit calibration indices.
The new selector does not silently replace their current sampling behavior.
A later evaluation must pass `calibration_indices.json` explicitly, making the
comparison between random-128 and stratified-128 visible and reproducible.

## Verification

Unit tests cover descriptor formulas, exact sparse count, robust
normalization, grouped distance, tail set cover, deterministic k-center,
deterministic k-medoids, tie handling, disjoint split construction, strict
owner coverage, acceptance checks, and serialization. A real CUDA run audits
all expected sample and owner counts before the calibration set is used in any
W4A4 evaluation.
