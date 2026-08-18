# NLSPN Pure-In-Memory GOP2 End-to-End Benchmark Design

## Objective

Build and evaluate a causal, pure-in-memory `GOP=2` NLSPN depth-completion
pipeline. The benchmark must replace the prior component-time projection with
an end-to-end measurement while preserving the validated frozen-NLSPN
residual decoder and the official RAFT-Small motion estimator.

The formal quality gate remains:

```text
pooled_RMSE_GOP2 / pooled_RMSE_full <= 1.01
```

The benchmark reports the measured speedup without imposing a minimum speedup
gate. It does not use dynamic fallback.

## Scope

The benchmark uses the same 256 frames as the residual-mapping pilot: eight
32-frame clips from `BeachApartmentInterior_My_ir`, with NLSPN input geometry
of 304 x 228 and exactly 500 fixed sparse-depth points per frame using seed
2026. Both the full-NLSPN reference and GOP2 path use FP32 and identical
preprocessed CPU-resident inputs.

The implementation adds a new benchmark entry point and leaves the existing
NPZ-based validation scripts and their outputs unchanged. Intermediate frame
tensors must not be written to or read from disk. Only final metrics, reports,
and run metadata may be persisted.

## Compatibility Constraint

NLSPN and its deformable-convolution implementation currently run in the
`completionformer-py37` environment with PyTorch 1.10.1 and Torchvision
0.11.2. The official Torchvision RAFT-Small implementation used by the pilot
runs in the current PyTorch 2.7.1 and Torchvision 0.22.1 environment, while
Torchvision 0.11.2 does not provide the optical-flow module.

To keep all online state and compute in one process and one GPU address space,
the benchmark will add a PyTorch-1.10-compatible copy of the official
RAFT-Small inference implementation. The RAFT-Small layer structure,
parameters, learned weights, transforms, number of updates, padding, crop, and
flow convention must remain unchanged. This compatibility layer is a motion
estimator only and does not alter or replace NLSPN.

Before the end-to-end benchmark runs, the compatible implementation must be
compared against the Torchvision 0.22.1 reference on deterministic input
pairs. The check must verify output shape, finite values, flow direction, and
numerical agreement within an explicitly recorded absolute and relative
tolerance. Reference and compatible tensors are exchanged through an in-memory
subprocess pipe rather than a temporary tensor file. Failure stops the
benchmark instead of silently proceeding with a different flow model.

## Online State and Scheduling

The online state stores only:

- previous RGB;
- previous predicted depth;
- previous NLSPN guidance;
- previous NLSPN confidence;
- current local GOP position.

All state tensors remain on the GPU between frames. State is cleared at the
start of every 32-frame clip, so no clip can reference another clip.

The fixed schedule is `I, P, I, P, ...` within each clip:

### I frame

1. Accept current RGB and sparse depth from CPU memory.
2. Transfer current inputs to the GPU.
3. Run the complete frozen NLSPN.
4. Retain RGB, predicted depth, guidance, and confidence as GPU state.
5. Return the predicted depth to CPU memory.

### P frame

1. Accept current RGB and sparse depth from CPU memory.
2. Transfer current inputs to the GPU.
3. Run official RAFT-Small causally on current and previous RGB to obtain the
   current-to-previous backward flow.
4. Warp previous depth, guidance, and confidence to the current frame.
5. At current sparse-depth locations, form the signed seed residual between
   current sparse depth and warped previous prediction.
6. Pass the seed, warped guidance, and warped confidence through the original
   frozen `NLSPNModel.prop_layer`.
7. Add the propagated residual to the warped previous prediction and clamp to
   the approved depth range.
8. Return the reconstructed prediction to CPU memory and update GPU state with
   current RGB, reconstructed depth, warped guidance, and warped confidence.

No additional depth, residual, refinement, or gating network is permitted.
No automatic fallback is permitted.

## Benchmark Boundary

Steady-state per-frame timing begins when the current frame's RGB and sparse
depth already reside in CPU memory. It ends when the predicted depth resides
in CPU memory. It includes:

- CPU-to-GPU input transfer;
- complete NLSPN execution for I/reference frames;
- RAFT-Small execution for P frames;
- warp and residual construction;
- original NLSPN propagation;
- online-state update;
- GPU-to-CPU prediction transfer;
- all synchronization required for accurate timing.

It excludes disk reads, initial dataset preprocessing, visualization, final
report generation, model construction, checkpoint loading, and warm-up.
Excluded startup costs are measured and reported separately.

## Measurement Protocol

The program loads all 256 preprocessed frames into CPU memory before timing.
It then measures two paths:

1. **Full reference:** complete NLSPN independently on every frame.
2. **Online GOP2:** alternating I and P frames with state reset at clip
   boundaries.

Each path performs one unmeasured complete warm-up pass followed by five
measured complete passes. Temporal state is reset before every pass. Latency
statistics therefore cover 1,280 frames per path.

CUDA synchronization must surround every measured frame boundary. The report
must include:

- total elapsed time and FPS for each path;
- overall frame latency mean, median/P50, P95, minimum, and maximum;
- separate I-frame and P-frame latency statistics;
- measured end-to-end speedup, defined as full-reference total time divided
  by GOP2 total time;
- model load time and warm-up time;
- peak allocated and peak reserved CUDA memory;
- software, GPU, checkpoint, and weight identity metadata.

## Quality Protocol

Quality is calculated from one deterministic prediction pass per path using
the same ground truth and valid-pixel mask. The primary metric is pooled RMSE
over all valid target pixels. The GOP2 path passes only when its RMSE divided
by the full-reference RMSE is no greater than 1.01.

The final report also includes per-frame ratios, per-clip ratios, percentile
statistics, and the worst frame. These local results are diagnostic and do
not replace the approved pooled gate, but they prevent the pooled result from
being represented as a uniform per-frame guarantee.

## Components and Interfaces

The implementation will separate these responsibilities:

1. **RAFT-Small compatibility module:** constructs the official architecture
   under PyTorch 1.10.1 and strictly loads official weights.
2. **Online NLSPN engine:** owns frozen models and GPU state and exposes
   explicit state reset, I-frame inference, and P-frame inference operations.
3. **Timing and aggregation utilities:** enforce synchronization, record
   per-frame latency, calculate distribution statistics, and track memory.
4. **Benchmark runner:** loads CPU-resident inputs, performs warm-up and five
   measured passes, computes quality, and writes only final artifacts.
5. **Reference parity checker:** compares compatible RAFT output with the
   Torchvision 0.22.1 implementation before the formal run.

The online engine must not depend on NPZ cache paths or the earlier staged
orchestrator.

## Validation and Failure Behavior

Automated tests must cover:

- RAFT parameter-key and tensor-shape compatibility;
- strict official-weight loading;
- input transform, padding, crop, and backward-flow convention;
- compatible/reference RAFT numerical parity;
- state creation, update, and reset;
- deterministic I/P scheduling and clip-boundary reset;
- signed sparse residual construction through the original propagation layer;
- timing-boundary synchronization and statistics;
- CPU-memory prediction output;
- absence of intermediate disk artifacts;
- pooled-quality and per-frame/per-clip aggregation.

The run fails immediately on non-finite data, unexpected frame geometry,
incorrect sparse-point count, missing or stale state, changed NLSPN
configuration, checkpoint mismatch, RAFT parity failure, or incomplete
metrics. Inputs are not silently resized or repaired.

## Outputs

The benchmark writes a dedicated output directory containing:

- `run_metadata.json` with immutable configuration and artifact identities;
- `summary.json` with quality, latency, throughput, speedup, startup, and
  memory metrics;
- `frame_metrics.csv` with frame type, clip, latency, RMSE, and quality ratio;
- `clip_summary.csv` with per-clip quality results;
- a concise Markdown report interpreting the formal quality gate and measured
  speedup.

The report must clearly distinguish measured end-to-end results from the
earlier 1.57x component projection.

## Acceptance Criteria

The work is complete only when:

1. Compatible RAFT-Small matches the Torchvision 0.22.1 reference within the
   recorded tolerance and loads the same official weight artifact.
2. Both paths complete all 256 frames, one warm-up pass, and five timed passes
   with no intermediate tensor cache files.
3. The GOP2 pooled RMSE ratio is no greater than 1.01.
4. End-to-end latency, throughput, I/P distributions, startup cost, memory,
   and measured speedup are reported from the defined boundary.
5. Automated tests and repository verification pass.
