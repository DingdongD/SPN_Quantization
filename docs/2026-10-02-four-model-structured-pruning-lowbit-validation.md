# Four-model NAS, structured pruning, and low-bit validation

## Acceptance criterion

The final deployed candidate is accepted when its full-validation mean
per-image RMSE is no more than 2% above the corresponding vanilla FP32 model.
The 2% budget applies to the combined NAS, structured-pruning, and
quantization result, rather than to each transformation independently.

All results below use the complete 654-image NYU validation set. Quantization
uses the fixed 128-image calibration cohort and calibration-derived static
scales; no validation-set scale fitting or hand-tuned scale override is used.
The original selected candidates used FP32 propagation-state writeback for
CSPN and FP16 writeback for the other three models. A second complete pass
uses one shared BF16-state contract: propagation arithmetic, normalization,
and reduction accumulate in FP32, while the recurrent state is rounded to
BF16 after every iteration and restored as the next iteration's input.

## Final accuracy

| Model | Vanilla FP32 RMSE (m) | Final NAS FP32 RMSE (m) | Previous selected RMSE (m) | Unified BF16-state RMSE (m) | BF16-state increment | Final vs vanilla | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CSPN | 0.143942 | 0.142724 | 0.145618 | 0.145823 | +0.141% | +1.307% | PASS |
| DySPN | 0.106596 | 0.106451 | 0.106819 | 0.107021 | +0.189% | +0.399% | PASS |
| NLSPN | 0.116253 | 0.117500 | 0.117648 | 0.118149 | +0.425% | +1.631% | PASS |
| CompletionFormer | 0.108212 | 0.108142 | 0.108568 | 0.109132 | +0.520% | +0.850% | PASS |

CSPN's frozen U250 software cohort reports 0.142724 m for the final NAS FP32
candidate. The independent official structured-search evaluator reports
0.144578 m for the same width-320 weights. These pipelines must not be mixed
for step-to-step attribution; final acceptance uses the predeclared absolute
gate of 0.146821 m. The unified BF16-state CSPN result is 0.998 mm below that
gate.

## Final configurations

| Model | Structural search result | Selected precision assignment |
| --- | --- | --- |
| CSPN | Encoder `s64-w64-128-128-256-d2-1-0-0`, K18; decoder bottleneck 512 to 320 | W4 decoder1 and encoder bottleneck; W8/A8 elsewhere; BF16 split input and task-head outputs; BF16 propagation state |
| DySPN | Drop encoder stage 4 depth; bridge 512 to 320; stage5 hidden 512 to 320; stage4 hidden 256 to 192 | Encoder stage 4 W4A4; encoder stage 3 W6; W8A8 elsewhere; BF16 propagation state |
| NLSPN | Drop encoder stage 4 depth; bridge 512 to 320; stage5 hidden 512 to 320; stage4 hidden 256 to 192 | Encoder stage 5 W6; encoder stage 4 and tail A6; early boundary and initial depth explicit BF16 QDQ; W8A8 elsewhere; BF16 propagation state |
| CompletionFormer | Drop PVT stage3 to 3 blocks and stage4 to 2 blocks; MLP hidden 62.5%; stage4 CNN hidden 62.5%; stage3 CNN hidden 60% | Transformer fusion W4; initial depth A6; W8A8 elsewhere; BF16 propagation state |

The bridge and MLP channels are selected by joint incoming/outgoing weight
importance. External tensor shapes are preserved, so pruning does not add new
layout conversions at the model boundaries.

## Unified hardware precision contract

The recommended external tensor and weight formats are INT4, INT6, INT8, and
BF16. Integer kernels may share one signed INT8 datapath with 4/6/8 effective
bits and INT32 accumulation; packed 4/6-bit storage is a memory-format concern.
Propagation uses BF16 operands/state and BF16 writeback with FP32 internal
accumulation and normalization. INT32 and FP32 accumulators are internal
implementation types, not additional inter-region tensor formats.

This experiment directly validates BF16 recurrent-state rounding with FP32
propagation arithmetic. It does not validate pure BF16 accumulation. NLSPN's
`initial_depth` and `early_boundary` weights and activations use explicit BF16
QDQ, including the independently owned `id_dec1/id_dec0` concat branches. The
selected graph therefore exposes only INT4/6/8 and BF16 tensor/weight formats;
INT32 and FP32 remain internal accumulator formats.

## Compression

| Model | Vanilla parameters | Final parameters | Parameter reduction | Packed weights | Effective weight bits | Storage compression vs vanilla FP32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSPN | 23.609 M | 9.056 M | 61.64% | 5.923 MiB | 5.49 | 15.21x |
| DySPN | 26.798 M | 15.825 M | 40.95% | 12.455 MiB | 6.60 | 8.21x |
| NLSPN | 26.232 M | 15.390 M | 41.33% | 13.745 MiB | 7.49 | 7.28x |
| CompletionFormer | 83.505 M | 46.130 M | 44.76% | 34.206 MiB | 6.22 | 9.31x |

Packed storage counts parameter payload only. It excludes graph metadata,
alignment padding, scales, runtime buffers, and U250 region overhead.

## Findings

1. Interface-preserving hidden-width pruning is consistently lower risk than
   changing public feature widths. All selected end-to-end low-bit candidates
   remain within 1.64% of their corresponding vanilla FP32 references after
   unifying propagation-state storage to BF16.
2. CSPN benefits most because its 512-channel decoder bottleneck dominates the
   remaining NAS model. Reducing it to 320 channels yields a total 61.64%
   parameter reduction without fine-tuning.
3. DySPN tolerates a 25% stage4 hidden reduction at the robust point. A 50%
   reduction also passes at +1.863%, but leaves only 0.137 percentage points of
   margin and is retained as a compression-priority Pareto point.
4. NLSPN is more sensitive in stage4: the 25% hidden reduction passes at
   +1.200%, while 50% was rejected during screening. This is evidence against
   applying one uniform channel ratio across SPN families.
5. CompletionFormer's PVT depth reduction and MLP/CNN hidden-width pruning are
   complementary. Their combination raises parameter reduction to 44.76% and
   packed-storage compression to 9.31x.
6. CSPN W4 mixed weights add only 0.0177 mm over its W8A8 result. The current
   sensitive component is activation quantization around the encoded boundary,
   not the selected W4 weight islands.
7. The 64-image screen is a ranking device, not an accuracy claim. Full-set
   validation changed the apparent margin materially for the aggressive DySPN
   and NLSPN candidates, so every promoted point is re-evaluated on 654 images.
8. Parameter and packed-weight reductions do not prove latency improvement.
   U250 speed must be measured after lowering with identical mapped-resident
   runners, region counts, physical layouts, sample sets, and synchronization.

## Evidence

- CSPN structured search: `/workspace/SPN_Quantization/.worktrees/cspn-encoder-nas/output/cspn_encoder_nas_20260920/structured_decoder_fullval_v1/summary.json`
- CSPN W8A8: `/workspace/SPN_Quantization/.worktrees/cspn-encoder-nas/output/cspn_encoder_nas_20260920/structured_decoder_width320/agentflow_w8a8_full654_v1/report.json`
- CSPN selected W4: `/workspace/SPN_Quantization/.worktrees/cspn-encoder-nas/output/cspn_encoder_nas_20260920/structured_decoder_width320/agentflow_w4_d1_bn_full654_v1/report.json`
- DySPN robust point: `/workspace/SPN_Quantization/profile_logs/structured_deep_channel_nas_quant_fullval_v1/dyspn_s4hidden75/summary.json`
- DySPN maximum-compression point: `/workspace/SPN_Quantization/profile_logs/structured_deep_channel_nas_quant_fullval_v1/dyspn_s4hidden50/summary.json`
- NLSPN: `/workspace/SPN_Quantization/profile_logs/structured_deep_channel_nas_quant_fullval_v1/nlspn_s4hidden75/summary.json`
- CompletionFormer: `/workspace/SPN_Quantization/profile_logs/structured_combined_depth_nas_quant_fullval_v1/completionformer/summary.json`
- Unified BF16-state DySPN, NLSPN, and CompletionFormer: `/workspace/SPN_Quantization/profile_logs/structured_bf16_propagation_fullval_v1/`
- Unified BF16-state CSPN: `/workspace/SPN_Quantization/.worktrees/cspn-encoder-nas/output/cspn_encoder_nas_20260920/structured_decoder_width320/agentflow_w4_d1_bn_bf16prop_full654_v1/report.json`
- NLSPN explicit BF16 protected boundaries: `/workspace/SPN_Quantization/profile_logs/nlspn_explicit_bf16_float_contract_full654_v1/summary.json`
- Final storage manifests: `/workspace/SPN_Quantization/profile_logs/structured_deep_channel_nas_quant_fullval_v1/parameter_compression/` and `/workspace/SPN_Quantization/profile_logs/structured_combined_depth_nas_quant_fullval_v1/parameter_compression/`

## Additional Pareto points

DySPN's stage4-hidden-50% candidate reaches 14.128 M parameters (47.28%
reduction) and 8.78x packed-weight compression, but its selected low-bit RMSE
is 0.108582 m, or +1.863% relative to vanilla. It is useful for a compression
frontier plot, while the 15.825 M, +0.209% candidate is the safer deployment
choice. NLSPN's stage5-only point similarly provides a safer +0.700% option at
34.86% parameter reduction and 6.51x storage compression.

## Next deployment order

1. Lower and compile CSPN width-320 first because it has the largest verified
   storage reduction and an existing AgentFlow/U250 contract.
2. Lower CompletionFormer next, preserving pruned MLP and residual hidden
   dimensions through the physical layout so the 9.31x packed-weight saving is
   not lost to adapters.
3. Lower DySPN and NLSPN hidden-pruned candidates, then profile deformable
   propagation separately from the compressed backbone.
4. Report only mapped-resident end-to-end latency and board RMSE on the same
   654-image cohort used by the corresponding software reference.
