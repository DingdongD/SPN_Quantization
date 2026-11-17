# RK3588 compressed SPN + coarse propagation: initial validation

Date: 2026-11-17  
Platform: Orange Pi 5 Pro, RK3588S, RKNPU2 at 1 GHz  
Runtime: RKNN API 2.5.0, driver 0.9.8

## Result

The experiment validates two separate facts:

1. The standardized DySPN and CompletionFormer compressed checkpoints can be
   reconstructed, exported, compiled, and executed as complete RKNN forward
   graphs on the RK3588.
2. Packing same-feature propagation samples is algebraically exact and greatly
   reduces callback count, but **does not improve complete-forward latency** in
   its current form.

This is useful negative evidence. It rejects the hypothesis that RK3588 SPN
latency is dominated by the number of `GridSample` API invocations. The packed
rewrite keeps the same interpolation work and CPU/NPU boundary tensors; a
single wider sampling callback can also have worse cache locality and a longer
critical path.

## What was executed

The compressed graph in this experiment is the checkpoint-defined NAS and
structured-channel-pruned topology, compiled to RKNN FP16. The checkpoints
also carry mixed W4/W6/W8 policy and BF16 state, but those low-bit policies are
**not executed here**. Calling this an RK3588 NAS+prune+quant result would be
incorrect; RKNN-native calibration/QAT export remains a separate experiment.

Two exact propagation lowerings are compared:

- `legacy_grid`: one `GridSample` per original sampling group;
- `packed_grid`: concatenate grids along output width, issue one wider
  `GridSample`, then split the result before the unchanged mask, weight, and
  reduction operations.

No fixed-stencil approximation is used. ONNX Runtime gives zero elementwise
difference between legacy and packed graphs for both measured models.

## Protocol

- Fixed NYU sample 386, shapes `1x3x228x304` and `1x1x228x304`.
- Input SHA-256:
  `dd8b9d5f...b3e73ff9f` and `3682fca1...6d4e762`.
- CPU affinity: cores 6 and 7; `OMP_NUM_THREADS=2`; RKNN core mask `AUTO`.
- New candidates: 30 warmups followed by 3 rounds x 50 timed forwards in one
  controlled process. This is an initial controlled result, not yet the final
  five-process statistical closure.
- CompletionFormer vanilla reference: five independent processes with the
  same 30 + 150 contract; process-level CV is 0.131%.
- Latency scope is synchronous `rknn_run` plus `rknn_outputs_get`; loading and
  input setup are excluded.

## DySPN

| Candidate | Mean (ms) | Median (ms) | P05 / P95 (ms) | GridSample calls | callback CPU (ms) | weight memory | internal memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla FP16 | 642.830 | 642.751 | 639.566 / 645.802 | 30 | 20.752 | 55.87 MB | 209.69 MB |
| Compressed FP16, legacy | 643.304 | 642.247 | 633.854 / 661.912 | 30 | 20.954 | 33.84 MB | 209.69 MB |
| Compressed FP16, packed | 644.474 | 643.631 | 635.167 / 657.316 | 6 | 20.299 | 33.84 MB | 209.69 MB |

Structural compression reduces RKNN weight memory by 39.44% and DMA
allocation by 8.02%, while internal memory is unchanged. Relative to compressed
legacy, packing reduces callback count by 80% and callback CPU time by only
3.12%; complete latency changes by +0.18%, which is not a speedup.

The legacy and packed RKNN outputs are bit-identical (`max=mean=RMSE=0`).
Packed RKNN versus packed ONNX has maximum absolute error 0.02101, mean
absolute error 0.001640, and RMSE 0.002143.

Raw evidence:
[vanilla JSON](raw/dyspn/vanilla_controlled.json),
[compressed legacy JSON](raw/dyspn/dyspn_legacy_controlled_process0.json),
[compressed packed JSON](raw/dyspn/dyspn_packed_controlled_process0.json).

## CompletionFormer

| Candidate | Mean (ms) | Median (ms) | P05 / P95 (ms) | GridSample calls | callback CPU (ms) | weight memory | internal memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla FP16, five-process aggregate | 754.761 | 754.205 | 751.581 / 758.357 | 170 | 117.515 | 220.80 MB | 109.24 MB |
| Compressed FP16, packed | 778.700 | 776.741 | 768.563 / 796.333 | 26 | 113.650 | 145.80 MB | 109.24 MB |

Packing reduces the graph from 170 to 26 `GridSample` callbacks (84.71%) and
reduces callback CPU time by 3.29%. Structural compression reduces RKNN weight
memory by 33.97% and DMA allocation by 21.79%. Nevertheless, complete latency
is 3.17% higher than the vanilla five-process mean. This comparison is
preliminary because the new candidate currently has one controlled process,
but the direction and magnitude agree with the three-forward smoke test.

Legacy and packed ONNX outputs are identical. Packed RKNN versus packed ONNX
has maximum absolute error 0.12505, mean absolute error 0.007837, and RMSE
0.011759. A board-level legacy/packed output comparison was not claimed because
the 156 MB legacy RKNN candidate was not deployed in this initial pass.

Raw evidence:
[packed JSON](raw/completionformer/completionformer_packed_controlled_process0.json)
and vanilla runs
[1](raw/completionformer/rep1_completionformer.json),
[2](raw/completionformer/rep2_completionformer.json),
[3](raw/completionformer/rep3_completionformer.json),
[4](raw/completionformer/rep4_completionformer.json),
[5](raw/completionformer/rep5_completionformer.json).

## Interpretation and next experiment

The results separate three optimization levels:

1. **Topology compression is real but not sufficient for latency.** Weight
   memory falls by 34--39%, yet the large internal workspace is unchanged and
   end-to-end latency does not fall.
2. **Invocation packing is too shallow.** It removes scheduling calls but not
   bilinear samples, mask multiplication, weighted reduction, or boundary
   traffic. CompletionFormer additionally exposes the cache-locality cost of a
   very wide packed grid.
3. **The next useful compiler target is a fused propagation kernel.** Fuse
   offset/grid generation, sampling, mask, scalar weight, and reduction so the
   intermediate sampled tensors do not materialize or cross CPU/NPU
   boundaries. A fixed-stencil NPU lowering should be treated separately as an
   approximate algorithm that requires retraining and full accuracy testing.

CSPN is not a target for this exact GridSample-packing transform because its
propagation is fixed convolutional. NLSPN is structurally applicable, but the
repository does not yet contain a standardized standalone final compressed
checkpoint with the same reconstruction/hashing contract; no result is
fabricated for it here.

Machine-readable values and full hashes are in
[initial_metrics.json](initial_metrics.json).
