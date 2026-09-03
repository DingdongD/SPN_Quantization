# Four-Model FP4/FP8 Evaluation

Protocol: official NYU checkpoints, 128 calibration samples, fixed 64-sample
evaluation set, and FP16 propagation. Propagation state, affinity, offset,
confidence, and gates are excluded from FP4/FP8. Ordinary Conv,
ConvTranspose, and Linear weights use per-output-channel floating-point
scales. Contract-owned activation sites use calibrated semantic-owner scales.
CompletionFormer attention Q/K/V and concat branches use independent scales.

FP4 is E2M1. FP8 is E4M3FN with finite range limited to 448. All reported
rows passed finite, positive, and paired-forward reproducibility checks.

| Model | Configuration | Pooled RMSE | Delta vs FP32 | Relative loss | Avg W bit | Avg A bit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CSPN | FP8W/FP8A | 0.19565 | 0.00186 | 0.96% | 8.00 | 8.00 |
| CSPN | FP4W/FP4A | 0.29692 | 0.10312 | 53.21% | 4.00 | 4.00 |
| CSPN | FP4W/FP8A | 0.23385 | 0.04005 | 20.67% | 4.00 | 8.00 |
| CSPN | FP8W/FP4A | 0.27190 | 0.07811 | 40.31% | 8.00 | 4.00 |
| DySPN | FP8W/FP8A | 0.14322 | 0.00140 | 0.99% | 8.00 | 8.00 |
| DySPN | FP4W/FP4A | 0.64293 | 0.50111 | 353.34% | 4.00 | 4.00 |
| DySPN | FP4W/FP8A | 0.14266 | 0.00084 | 0.59% | 4.00 | 8.00 |
| DySPN | FP8W/FP4A | 0.42647 | 0.28465 | 200.71% | 8.00 | 4.00 |
| NLSPN | FP8W/FP8A | 0.15896 | 0.00807 | 5.35% | 8.00 | 8.00 |
| NLSPN | FP4W/FP4A | 0.94490 | 0.79401 | 526.20% | 4.00 | 4.00 |
| NLSPN | FP4W/FP8A | 0.17946 | 0.02857 | 18.93% | 4.00 | 8.00 |
| NLSPN | FP8W/FP4A | 2.02821 | 1.87731 | 1244.13% | 8.00 | 4.00 |
| CompletionFormer | FP8W/FP8A | 0.14067 | 0.00050 | 0.36% | 8.00 | 8.00 |
| CompletionFormer | FP4W/FP4A | 0.73192 | 0.59175 | 422.17% | 4.00 | 4.00 |
| CompletionFormer | FP4W/FP8A | 0.16093 | 0.02076 | 14.81% | 4.00 | 8.00 |
| CompletionFormer | FP8W/FP4A | 0.48958 | 0.34941 | 249.28% | 8.00 | 4.00 |

The mixed rows promote the highest `A4-A8` task-sensitivity owners to FP8.
The 25/50/75% rows are retained in each model manifest for budget analysis;
they are not treated as universally optimal because owner sensitivity is
model-specific and the propagation path is nonlinear.

The main result is that FP4 weight quantization is comparatively tolerable
when activations remain FP8, while FP4 activation quantization dominates the
end-to-end degradation. The next optimization target is therefore selective
activation FP8 protection at encoder/decoder boundaries and attention/concat
sites, with propagation remaining FP16.

## Group-Budgeted Candidate

The grouped candidate starts both weights and activations at FP4 and promotes
units to FP8 independently inside each present semantic group. The configured
weighted average limit is 6.0 bits for both tensor sides in encoder, decoder,
fusion, attention, and concat. An absent group is recorded as absent and is
not assigned a default format.

| Model | Pooled RMSE | Relative loss | Avg W bit | Avg A bit | Present groups |
| --- | ---: | ---: | ---: | ---: | --- |
| CSPN | 0.308064 | 58.96% | 5.903 | 5.992 | encoder, fusion |
| DySPN | 0.244346 | 72.29% | 5.865 | 5.758 | encoder, decoder, fusion |
| NLSPN | 1.312099 | 769.55% | 5.730 | 5.799 | encoder, decoder, fusion |
| CompletionFormer | 0.341876 | 143.90% | 5.693 | 5.840 | encoder, decoder, fusion, attention, concat |

The per-group format lists, weighted fractions, and feasibility audit are
stored in each model `manifest.json` under
`assignments.GROUPED_FP4_FP8_BUDGETED.group_audit`. These results validate the
budgeted execution path; a uniform 6-bit group cap is not asserted to be the
best end-to-end operating point for every model.

## Boundary-Protected Candidate

The boundary-protected candidate keeps propagation in FP16, requires at least
50% of decoder activation cost and 50% of fusion activation cost to use FP8,
and assigns the remaining global activation budget by task sensitivity. The
global limits are 6.0 weighted average bits for weights and 6.5 for
activations. The fusion floor was selected after checking the CSPN owner-cost
granularity; a 75% floor would force nearly all CSPN fusion activation owners
to FP8 and exceed the 6.5 global limit.

| Model | Pooled RMSE | Relative loss | Avg W bit | Avg A bit |
| --- | ---: | ---: | ---: | ---: |
| CSPN | 0.275911 | 42.37% | 5.903 | 6.490 |
| DySPN | 0.165411 | 16.63% | 5.865 | 6.489 |
| NLSPN | 1.185982 | 685.97% | 5.730 | 6.491 |
| CompletionFormer | 0.327340 | 133.53% | 5.693 | 6.400 |

Compared with the previous group-budgeted candidate, pooled RMSE changes are
`-0.032152` for CSPN, `-0.078935` for DySPN, `-0.126117` for NLSPN, and
`-0.014536` for CompletionFormer. The improvement confirms that low-precision
decoder/fusion inputs before propagation were a major error source, but the
remaining NLSPN error requires finer initial-depth and propagation-entry
protection rather than changing the FP16 propagation arithmetic.
