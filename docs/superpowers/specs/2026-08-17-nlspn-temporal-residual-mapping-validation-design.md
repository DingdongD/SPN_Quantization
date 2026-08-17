# NLSPN Temporal Residual Mapping Validation Design

## Objective

Determine empirically whether adjacent inputs processed by the same frozen
NLSPN exhibit a motion-compensated output-residual relationship that the
existing NLSPN propagation operator can reconstruct. This is a validation
study, not a codec implementation. No NLSPN architecture, checkpoint, or
training procedure is changed.

The study is successful only if a causal reconstruction built from the
previous prediction, current and previous RGB, current sparse depth, and the
existing NLSPN propagation operator stays within one percent of the full
per-frame NLSPN baseline RMSE.

## Scope

The pilot uses the `BeachApartmentInterior_My_ir` sequence and the existing
NYU-compatible preprocessing:

- source geometry: 640 x 480;
- network geometry: 304 x 228;
- sparse samples: 500 fixed image coordinates per clip, selected from pixels
  valid in every frame of that clip;
- sparse seed: 2026;
- depth range: 0--10 metres;
- model: frozen NLSPN ResNet-34, 18 propagation iterations;
- checkpoint: the existing `nlspn_iter18/best.pt` baseline checkpoint.

The pilot covers 256 frames using eight non-overlapping 32-frame clips spread
over the complete 2,000-frame sequence:

| Clip | Inclusive frame range |
| --- | --- |
| 1 | 0001--0032 |
| 2 | 0282--0313 |
| 3 | 0563--0594 |
| 4 | 0844--0875 |
| 5 | 1126--1157 |
| 6 | 1407--1438 |
| 7 | 1688--1719 |
| 8 | 1969--2000 |

There are 248 evaluated adjacent-frame pairs. Clip boundaries are not treated
as temporal pairs.

## Baseline Data Flow

For every frame, run the same frozen NLSPN normally:

```text
x_t = (RGB_t, sparse_t)
d_t = NLSPN(x_t)
```

The resulting `d_t` is the full per-frame reference prediction. Ground truth
is used only for metrics. The pipeline records the full NLSPN outputs needed
for analysis: final depth, initial depth, guidance, confidence, offsets, and
affinities.

All timing measurements synchronize CUDA before and after the measured
region. Preprocessing, disk I/O, plots, and metric computation are reported
separately from model inference.

## Causal Motion Compensation

Use the official pretrained Torchvision RAFT-Small weights for the pilot.
RAFT-Small is external to the depth model and estimates motion only; it does
not predict or refine depth.

At frame `t+1`, both `RGB_t` and `RGB_t+1` are available, so backward flow
from the current frame to the previous frame is causal. Inputs are padded
from 228 x 304 to 232 x 304 for RAFT and cropped back after inference. The
weights' official preprocessing transform is used.

Backward warping produces:

```text
d_base = warp(d_t, flow_(t+1 -> t))
```

The warp also produces an in-bounds validity mask. Out-of-bounds samples use
border replication to produce a deterministic reconstruction value. They are
excluded from relationship statistics but remain included in the final GT
quality metric, so masking cannot make the acceptance result easier.
Photometric error is recorded as an occlusion and motion-quality diagnostic;
it is not used to exclude pixels or trigger a fallback. There is no dynamic
fallback.

If official RAFT-Small weights cannot be loaded, the run fails explicitly.
It must not silently substitute a different optical-flow method.

## Residual Definitions

The analysis compares unregistered and registered output residuals:

```text
r_raw     = d_(t+1) - d_t
r_aligned = d_(t+1) - d_base
```

Input-side diagnostics are:

```text
rgb_residual    = RGB_(t+1) - warp(RGB_t)
sparse_residual = sparse_(t+1) - d_base
```

`sparse_residual` is valid only at the 500 current sparse locations. It is
zero elsewhere when passed to the propagation experiment, with a separate
binary sparse mask retained so that zero residual and missing residual are
not confused in metrics.

For `r_raw` and `r_aligned`, report RMSE, MAE, median absolute residual, P95,
P99, the fraction below 1/2/5/10 cm, and residual energy. Report Pearson and
Spearman relationships between output residual magnitude and photometric
error, flow magnitude, and sparse residual at sampled locations. These
statistics describe the mapping but do not decide quality by themselves.

## Existing-Propagation Reconstruction Tests

No new depth or residual network is introduced. The experiment calls the
existing frozen `NLSPNModel.prop_layer` directly with a signed residual field
as `feat_init`.

Two tests separate expressivity from causal reuse:

1. **Current-guidance oracle.** Use current-frame guidance and confidence
   already produced by the full reference forward pass. This cannot
   accelerate inference, but determines whether the original propagation
   operator can express the required residual correction.
2. **Warped-history causal test.** Warp the previous frame's guidance and
   confidence into the current frame and use them with the same original
   propagation operator. This is the codec-feasible path because it skips the
   current NLSPN encoder and decoder.

For both tests:

```text
r_seed[sparse_mask] = sparse_(t+1) - d_base
r_seed[otherwise]   = 0
r_dense = original_NLSPN_prop_layer(
    r_seed, guidance, confidence)
d_reconstructed = clamp(d_base + r_dense, 0, 10)
```

The checkpoint's propagation configuration remains unchanged, including
`preserve_input=False`. Signed residuals are not clamped before propagation.
The reconstructed final depth alone is clamped to the valid depth range.

The oracle and causal tests are diagnostics; neither may use `d_(t+1)` or
ground truth as an input. Full current-frame outputs are used only as the
oracle guidance source and as comparison targets.

## Quality Gate and Interpretation

Let `RMSE_full` be the pooled valid-pixel GT RMSE of normal per-frame NLSPN and
`RMSE_reconstructed` the corresponding pooled RMSE of the causal
warped-history reconstruction. The sole pass gate is:

```text
RMSE_reconstructed / RMSE_full <= 1.01
```

The ratio is computed across all 256 pilot frames, excluding the first frame
of each clip because it has no causal predecessor. Per-clip ratios and the
worst frame ratio are reported to expose local failures but do not replace
the agreed pooled gate.

Interpret results as follows:

- If the warped-history causal test passes, the residual-codec hypothesis is
  supported and may proceed to a separate codec implementation design.
- If the current-guidance oracle passes but warped history fails, the original
  propagation operator can express the residual, but historical guidance is
  insufficient. Profile-guided experiments may then test partial reuse of
  original NLSPN modules; no codec implementation is approved yet.
- If the current-guidance oracle fails, the proposed residual
  reparameterization is not supported for this checkpoint and the codec path
  stops.

No automatic fallback, adaptive keyframe decision, learned residual network,
or fine-tuning is part of this validation.

## Outputs

Write pilot artifacts under:

```text
/workspace/VoxelNet/nlspn_temporal_residual_validation/
  BeachApartmentInterior_My_ir/pilot_256/
```

Required artifacts are:

- `run_metadata.json`: complete/incomplete state, frame ranges, seeds,
  checkpoint and flow-weight digests, software versions, shapes, and timing;
- `frame_metrics.csv`: full and reconstructed GT metrics;
- `pair_metrics.csv`: raw/aligned residual and mapping statistics;
- `clip_summary.csv`: pooled metrics and quality ratio per clip;
- `summary.json`: global gate result and interpretation branch;
- `residual_mapping_overview.png`: representative raw/aligned/input residuals;
- `quality_ratio_by_frame.png`: relative RMSE sequence plot;
- per-clip compressed prediction and diagnostic arrays sufficient to
  reproduce every reported metric without rerunning NLSPN.

Metadata starts with `complete=false` and becomes true only after every
required artifact has been validated and written atomically.

## Validation and Tests

Unit tests cover:

- deterministic clip selection and fixed 500-point masks;
- preprocessing geometry and depth units;
- warp direction using synthetic integer translations;
- in-bounds and valid-depth mask composition;
- signed sparse residual construction;
- propagation reconstruction composition and final clamping;
- pooled RMSE ratio calculation;
- exclusion of cross-clip pairs;
- metadata completion and artifact validation.

An integration smoke test runs two frames through frozen NLSPN and the flow
path, verifies all expected tensor shapes and finite values, and writes a
temporary complete artifact set. The 256-frame pilot begins only after unit
and integration tests pass.

## Non-Goals

- Implementing a deployable temporal codec.
- Changing, fine-tuning, or replacing NLSPN.
- Adding a learned depth residual refiner.
- Dynamic fallback, scene-change detection, or adaptive GOP selection.
- Evaluating DySPN, CSPN, or CompletionFormer before NLSPN passes the pilot.
- Claiming speedup before the residual mapping and quality gate are verified.
