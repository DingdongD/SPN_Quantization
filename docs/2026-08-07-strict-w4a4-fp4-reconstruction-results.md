# Strict W4A4 and FP4 Reconstruction Results

## Unified CSPN BRECQ/QDrop Rerun (2026-08-19)

The current CSPN comparison uses the official converged checkpoint with SHA256
`482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`
and real NYU data. The protocol is shared by RTN, BRECQ, and QDrop:

- 128 stratified train samples, split into 112 reconstruction and 16 validation
  samples;
- 64 fixed evaluation samples selected with seed `20260812`;
- W4A4 and W6A6 convolution boundaries, FP32 bias, and CSPN propagation kept
  at A8/Q13/INT32;
- the same 16 CSPN reconstruction blocks for BRECQ and QDrop;
- 20,000 reconstruction steps, batch size 32, QDrop probability 0.5, and
  QDrop seeds 1005, 1006, and 1007;
- no retraining, test-set seed selection, output clipping, or invalid-depth
  fallback.

An evaluation sample is invalid when any valid-GT pixel has a non-finite or
non-positive prediction. Its strict RMSE is therefore infinite. The finite-only
mean below is diagnostic and is not used for acceptance.

| Method | Precision | Strict mean RMSE (m) | Invalid samples | Finite samples | Finite-only mean RMSE (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| FP32 | FP32 | 0.158092 | 0.00% | 64/64 | 0.158092 |
| P3/T3 | Mixed | 0.172321 | 0.00% | 64/64 | 0.172321 |
| RTN | W4A4 | inf | 100.00% | 0/64 | n/a |
| BRECQ | W4A4 | inf | 100.00% | 0/64 | n/a |
| QDrop | W4A4, 3 seeds | inf | 6.25%-9.38% | 58-60/64 | 0.251449-0.252601 |
| RTN | W6A6 | inf | 10.94% | 57/64 | 0.316775 |
| BRECQ | W6A6 | inf | 12.50% | 56/64 | 0.323255 |
| QDrop | W6A6, 3 seeds | 0.195805 +/- 0.000231 | 0.00% | 64/64 | 0.195805 |

QDrop W6A6 is the only uniformly finite reconstructed configuration. It is
better than the aligned RTN and BRECQ W6A6 runs, but remains 23.86% worse than
FP32 and therefore does not satisfy the 10% preservation threshold. Its three
evaluation means are 0.196088, 0.195521, and 0.195806 m.

QDrop W4A4 does not pass the strict criterion, but its failure differs from the
RTN/BRECQ collapse. Each failing QDrop sample contains only 1-4 invalid pixels,
whereas RTN and BRECQ W4A4 produce about 44,472 and 45,547 invalid pixels per
sample on average. The prediction figures consequently retain scene structure
for QDrop W4A4 even though the unmodified output contract correctly rejects it.

The encoder tail remains the strongest reconstruction bottleneck. For
`layer4.1`, BRECQ reduces block loss from 15.3980 to 1.2535 at W4 and from
0.7085 to 0.0676 at W6. Under QDrop activation perturbation, the corresponding
W4 losses remain near 21.4 after reconstruction, compared with about 1.23 at
W6. Loss scales are method-specific, but both methods independently identify
the same W4-sensitive block.

Validation-based median seed selection chose seed 1006 for W4A4 and seed 1007
for W6A6. The visual samples are 25/652 for W4A4 and 92/542 for W6A6
(median/worst). The unified artifact root is
`profile_logs/nyu_cspn_unified_brecq_qdrop_w4a4_w6a6_64` and contains:

- `sample_metrics.csv`, `seed_summary.csv`, `qdrop_summary.csv`, and
  `acceptance.csv`;
- six QDrop and two BRECQ strict deployment contracts;
- `figures/cspn_w4a4_predictions.{png,pdf}`;
- `figures/cspn_w6a6_predictions.{png,pdf}`;
- `figures/cspn_rmse_comparison.{png,pdf}`;
- a hash-checked `manifest.json` that passes the unified audit.

The superseded `profile_logs/nyu_qdrop_w4a4` and
`profile_logs/nyu_brecq_cspn_pa_w4a4` roots were removed after the new audit
passed.

## Protocol

The formal evaluation used the converged official CSPN, DySPN, NLSPN, and
CompletionFormer structures with fixed NYU sample indices:

- 64 calibration samples and 64 evaluation samples;
- RTN, strict AdaRound, and strict BRECQ W4 weights;
- `FP4V_W4A4`, `FP4V_W4E2M1`, and `FP4V_W4A8` primary configurations;
- FP32 bias and A8 sparse-depth, depth/guidance/confidence, affinity, offset,
  and propagation-state boundaries;
- `HW_W4A4_full` as a separate integer stress baseline;
- 10,000 paired bootstrap resamples with seed 20260806.

AdaRound and BRECQ load frozen strict weight contracts. This evaluation does
not retrain or further optimize them. E2M1 is calibrated float QDQ and is not a
native FP4 kernel or a latency result.

Performance is preserved only when output is finite, mean per-sample RMSE is
within 10% of FP32, and the result is no worse than RTN under the same
activation format.

## Mean RMSE

All values are metres.

| Model | Method | FP32 | W4A4 | W4-E2M1 | W4A8 |
| --- | --- | ---: | ---: | ---: | ---: |
| CSPN | RTN | 0.1669 | 0.9374 | 0.4285 | 0.2178 |
| CSPN | AdaRound | 0.1669 | 0.9948 | 0.4325 | 0.1925 |
| CSPN | BRECQ | 0.1669 | 0.9988 | 0.4335 | 0.1889 |
| DySPN | RTN | 0.1202 | 0.3032 | 0.9800 | **0.1312** |
| DySPN | AdaRound | 0.1202 | 0.4148 | 1.0369 | 0.1319 |
| DySPN | BRECQ | 0.1202 | 0.2376 | 0.6480 | **0.1304** |
| NLSPN | RTN | 0.1282 | 1.5490 | 0.9270 | 0.1731 |
| NLSPN | AdaRound | 0.1282 | 1.7278 | 1.0173 | 0.1772 |
| NLSPN | BRECQ | 0.1282 | 1.1033 | 0.9815 | 0.1854 |
| CompletionFormer | RTN | 0.1193 | 1.7039 | 1.8786 | 0.6799 |
| CompletionFormer | AdaRound | 0.1193 | 0.9086 | 0.7834 | 0.1350 |
| CompletionFormer | BRECQ | 0.1193 | 0.8969 | 0.8107 | 0.1436 |

Bold values are the only preserved quantized results. No W4A4 or W4-E2M1
configuration met the preservation criterion.

## Findings

E2M1 significantly improves W4A4 for CSPN and NLSPN. The paired 95% confidence
intervals for E2M1 minus uniform A4 are strictly below zero for every weight
method on both models. E2M1 is consistently worse for DySPN, where all three
intervals are strictly above zero. CompletionFormer E2M1 is worse under RTN
but better after AdaRound or BRECQ.

Weight reconstruction is model dependent. BRECQ improves DySPN W4A4 from
0.3032 to 0.2376 m, NLSPN W4A4 from 1.5490 to 1.1033 m, and CompletionFormer
W4A4 from 1.7039 to 0.8969 m. It worsens CSPN W4A4. AdaRound worsens W4A4 for
CSPN, DySPN, and NLSPN, while improving CompletionFormer.

The worst activation-group SQNR remains in the encoder for CSPN and DySPN, the
decoder for NLSPN, and the encoder/attention path for CompletionFormer. The
propagation-state error generally decreases from the first to the final
iteration, so the measured failure is not simple unbounded SPN accumulation.
The propagation loop cannot recover the already corrupted dense prediction,
guidance, and feature tensors.

The separate integer stress baseline makes CSPN non-finite on all 64 samples
for RTN, AdaRound, and BRECQ. This confirms that preserving semantic and
propagation boundaries is necessary even though it is not sufficient for
W4A4 accuracy preservation.

## Artifacts

The generated root is
`profile_logs/nyu_strict_w4a4_fp4_evaluation` and is intentionally ignored by
Git. It contains:

- `analysis/strict_w4a4_fp4_summary.csv`;
- `analysis/strict_w4a4_fp4_paired.csv`;
- `analysis/strict_w4a4_fp4_activation_groups.csv`;
- `analysis/strict_w4a4_fp4_propagation_steps.csv`;
- `analysis/strict_w4a4_integer_stress.csv`;
- `analysis/strict_w4a4_fp4_report.md`;
- `figures/strict_w4a4_fp4_predictions.png`;
- `figures/strict_w4a4_fp4_errors.png` and twelve aggregate figures;
- 4,608 per-sample prediction NPZ files;
- one complete log per method and model under `logs/`.

Re-running the analyzer produced byte-identical analysis files. The complete
test suite passed with 290 tests.
