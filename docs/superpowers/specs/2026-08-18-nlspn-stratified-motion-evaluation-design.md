# NLSPN Stratified Motion Evaluation Design

## Goal

Evaluate how causal depth-completion error changes with adjacent-frame motion
using substantially more data than the current six high-motion windows. The
evaluation keeps the frozen NLSPN and official RAFT-Small models unchanged,
uses the existing `I,P,I,P,I` schedule, and compares all five existing
methods:

1. Full NLSPN
2. Zero-flow GOP2
3. RGB-difference GOP2
4. Global-difference GOP2
5. RAFT-GOP2

The primary question is whether P-frame error relative to Full NLSPN increases
from low to medium to high motion after balancing scene contributions.

## Scope

The evaluation covers the same six approved scenes:

- `bedroom_ir`
- `livingroom_ir`
- `room3`
- `room4`
- `room6`
- `room7`

For each scene, it selects five non-overlapping five-frame windows in each of
three motion strata. The final manifest therefore contains exactly 90 windows,
450 frames, and 180 causal P frames. Every window retains all predictions,
per-frame metrics, and visualizations. The work does not retrain or modify any
model, change sparse-depth generation, calibrate thresholds, or overwrite the
existing six-window results.

## Motion Measurement and Candidate Enumeration

Candidate enumeration follows the existing data and scoring path. RGB frames
are converted to the same finite grayscale `float32` thumbnails in `[0, 1]`.
For every legal sequence of five consecutive complete frames, the four
adjacent-pair scores are:

```text
pair_score(t, t+1) = mean(abs(gray_rgb[t+1] - gray_rgb[t]))
```

The window score is the arithmetic mean of its four pair scores. Candidates
with missing RGB, depth, or required consecutive frame IDs are not legal.

All candidates are enumerated once per scene. A deterministic record stores
the scene, start/end IDs, five frame IDs, four pair scores, and window motion
score before stratification begins.

## Per-Scene Stratification and Selection

Stratification happens independently inside each scene so every scene
contributes equally to every motion level. Candidates are sorted by
`(motion_score, start_frame)`, then split by rank into three groups whose sizes
differ by at most one:

- lower third: `low`
- middle third: `medium`
- upper third: `high`

Rank-based splitting gives deterministic, balanced groups even when multiple
candidates have identical scores. The selected manifest records each scene's
actual stratum score range as well as its rank boundaries; no fixed global
motion thresholds are implied.

Selection uses seed `2026`. Candidate order is shuffled deterministically
within each scene and stratum, and strata are considered round-robin so no one
stratum receives systematic priority. A candidate is accepted only when none
of its five frames is already used by another selected window from the same
scene. Adjacent, non-overlapping windows are allowed. Exactly five windows
must be selected in every scene/stratum cell. If any cell cannot meet that
requirement, selection fails explicitly and does not silently reduce the
sample count.

The resulting manifest has these invariant counts:

- 6 scenes
- 3 strata per scene
- 5 windows per scene/stratum
- 15 windows per scene
- 30 windows per stratum
- 90 windows total

## Inference Architecture

A current-environment launcher performs candidate discovery, writes the fixed
manifest, prepares a same-parent staging directory, and starts one legacy
Python 3.7 worker. The worker loads the frozen NLSPN checkpoint once and the
official RAFT-Small checkpoint once before processing any window.

For each manifest row, the worker loads one five-frame payload using the
existing deterministic 500-point sparse-depth rule. The same per-frame RGB,
sparse depth, ground truth, and valid mask are provided to all five methods.
It runs the existing four-method cache engine and strict RAFT-GOP2 engine, then
writes the exact five-method artifacts for that window. Model and engine
objects are reused for all 90 windows.

The schedule remains strictly causal:

```text
local frame:  0 1 2 3 4
frame kind:   I P I P I
```

I frames execute Full NLSPN. P frames use only current inputs and past cached
state. RAFT estimates current-to-previous flow with 12 updates and warps the
past depth, guidance, and confidence state before the existing NLSPN
propagation layer. There is no future-frame access, model fallback,
calibration, or skipped-window behavior.

## Metrics and Statistical Analysis

All 450 frames retain the existing RMSE, MAE, absolute-relative error, valid
pixel count, sparse count, frame kind, and latency fields. The primary
statistical table contains all five methods for the 180 P frames, yielding
exactly 900 rows.

For every temporal method and P frame, analysis derives:

- absolute RMSE and MAE;
- `excess_rmse = method_rmse - full_rmse`;
- `rmse_ratio = method_rmse / full_rmse`;
- whether `rmse_ratio <= 1.01`;
- method latency and speedup relative to Full NLSPN.

The primary endpoint is P-frame excess RMSE because absolute RMSE is
confounded by scene and frame difficulty. RMSE ratio and the 1% pass rate are
co-primary interpretability metrics. Absolute RMSE and MAE remain visible.

For every method and motion stratum, the report includes:

- count, mean, median, and standard deviation;
- 95% bootstrap confidence intervals;
- the fraction satisfying the 1% RMSE constraint;
- latency and speedup summaries.

Bootstrap resampling preserves the experiment structure: scenes are retained
as six equal contributors, and windows are resampled within scene/stratum
cells with seed `2026`. The report uses 2,000 bootstrap replicates. It reports
both scene-macro summaries (primary) and valid-pixel pooled summaries
(secondary). Macro summaries first aggregate within each scene/stratum and
then give each scene equal weight.

Continuous-motion analysis uses the actual I-to-P adjacent-pair score for each
P frame rather than the five-frame mean. It reports Pearson and Spearman
correlations against absolute RMSE, excess RMSE, and RMSE ratio. It also
reports the observed low-to-medium-to-high ordering but does not label a trend
"statistically significant" solely from a median split. Confidence intervals,
sample counts, and p-values accompany inferential statistics.

I frames are a numerical sanity check rather than part of the motion-error
endpoint. Each temporal method's I-frame prediction must equal the Full NLSPN
prediction within the existing strict numerical tolerance.

## Artifact Layout

The new formal output root is:

```text
/workspace/VoxelNet/nlspn_frame_difference_cache/
  cross_scene_motion_stratified_5x3
```

Its layout is:

```text
cross_scene_motion_stratified_5x3/
├── selected_windows.csv
├── p_frame_metrics.csv
├── stratified_summary.csv
├── correlation_summary.csv
├── run_metadata.json
├── report.md
├── motion_error_scatter.png
├── stratified_error_boxplot.png
└── <scene>/<low|medium|high>/<start_end>/
    ├── predictions.npz
    ├── frame_metrics.csv
    ├── nlspn_frame_difference_depth_comparison.png
    ├── nlspn_frame_difference_error_comparison.png
    └── run_metadata.json
```

`selected_windows.csv` records selection seed, scene, stratum, stratum rank
and score bounds, start/end frames, frame IDs, pair scores, and window score.
Every `predictions.npz` stores frame IDs, RGB, sparse depth, ground truth,
valid masks, and the five finite `5 x 228 x 304` prediction arrays.

The scatter plot shows P-frame adjacent-pair motion versus excess RMSE with
method-specific points and trend summaries. The box plot compares the P-frame
excess RMSE and RMSE ratio distributions across low, medium, and high strata.
All plot data are also present in CSV form.

Expected additional disk use is approximately 0.8 to 1.0 GB.

## Validation and Failure Handling

The launcher writes only to the sibling staging root
`cross_scene_motion_stratified_5x3_staging` during inference. It refuses to
start if either staging or the formal target already exists. The existing
`cross_scene_motion_windows` target and its pre-RAFT backup are immutable
inputs outside this output scope.

After worker completion, an independent current-environment validator checks:

1. exact root, scene, stratum, and window directory scope;
2. exact 6/3/5/90/450/180 selection counts;
3. no shared frame IDs among a scene's selected windows;
4. exact manifest values and stable ordering;
5. exactly 25 five-method metric rows per window;
6. exactly 900 root P-frame metric rows;
7. finite `5 x 228 x 304` prediction tensors for all methods;
8. complete, nonempty PNG/CSV/JSON/NPZ artifacts;
9. NLSPN and RAFT model load counts both equal to one;
10. checkpoint, args, formal threshold sweep, input, manifest, and RAFT weight
    digests;
11. official RAFT implementation, 12 updates, and
    `current_to_previous` direction;
12. numerical I-frame identity with Full NLSPN;
13. recomputed per-window, stratum, macro, pooled, correlation, and pass-rate
    summaries matching the stored tables.

Only a fully validated staging tree is atomically renamed to the formal
target. No existing target is overwritten. On failure, the formal target
remains absent and the staging tree plus logs remain available for diagnosis.
There is no automatic retry, fallback, partial promotion, or silent cleanup.

## Testing Strategy

Implementation follows test-driven development. Unit tests cover:

- complete candidate enumeration and motion-score calculation;
- deterministic rank tertiles, stable tie handling, and fixed-seed selection;
- cross-stratum non-overlap and exact-count rejection;
- missing frames, invalid thumbnails, duplicate manifest entries, and
  insufficient candidates;
- P-frame derived metrics, scene-macro and pooled aggregation;
- deterministic 2,000-replicate bootstrap confidence intervals;
- Pearson/Spearman calculations and 1% pass rates;
- exact artifact tree and count validation;
- corrupt arrays, summaries, model metadata, flow semantics, and I-frame
  identity rejection;
- staging refusal and validated atomic promotion;
- one-load worker behavior using injected fake models and engines.

Focused tests run in the current Python environment and worker-facing tests
also run under `completionformer-py37`. Before real inference, the complete
repository suite must pass. After the real run, the formal output is validated
again from disk, all image files are checked programmatically, and one depth
and error comparison is visually inspected for every scene/stratum cell (36
figures total). Final repository and legacy-environment tests run again before
completion is reported.

## Acceptance Criteria

The evaluation is complete only when:

- the deterministic manifest contains exactly the approved 90 windows;
- all five methods produce complete artifacts for all windows;
- both models are loaded exactly once;
- independent validation and all automated tests pass;
- all root statistics are reproducible from per-window CSV/NPZ artifacts;
- the report answers whether P-frame error rises across motion strata while
  clearly documenting exceptions and uncertainty;
- the new formal output exists without modifying or deleting any prior result.
