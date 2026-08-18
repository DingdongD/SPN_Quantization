# CSPN Task-Sensitive Mixed-Bit Allocation Results

## Protocol

- Official CSPN ResNet-18 with 24 propagation iterations.
- Checkpoint: `output/nyu_converged_baselines/cspn_iter24/best.pt`.
- Calibration: persisted stratified 128-sample NYU train subset.
- Evaluation: persisted fixed 64-sample NYU validation subset, seed
  `20260812`.
- Candidate weight and activation bits: `{2, 4, 6, 8}`.
- Constraints: MAC-weighted average weight bits at most 4 and
  element-weighted average activation bits at most 4.
- Quantization: static MinMax Group-8 activations, per-output-channel weights,
  and FP32 bias.
- Guidance remains FP32. Propagation uses A8 state, affinity, and confidence,
  INT16 Q13 coefficients, and INT32 accumulation.
- Search coverage: 151 single-block probes, 128 joint candidates, 375 local
  candidates, 10 block demotions, and 128 refinement candidates.

## Validation Results

| Configuration | RMSE (m) | MAE (m) | AbsRel | iRMSE | Flat RMSE (m) | Boundary RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.158092 | 0.064308 | 0.021359 | 0.021781 | 0.107420 | 0.409986 |
| Uniform W4A4 | 0.342178 | 0.257888 | 0.113045 | 59.429073 | 0.318057 | 0.601916 |
| P3/T3 W8A8 context | 0.172321 | 0.086617 | 0.031247 | 0.027141 | 0.133216 | 0.538510 |
| Task-sensitive final | 0.320343 | 0.238003 | 0.108278 | 0.076683 | 0.295194 | 0.592262 |

The final assignment reduces RMSE by 6.38% and MAE by 7.71% relative to
uniform W4A4. It remains 0.148022 m behind the higher-cost P3/T3 context, so a
strict average-bit budget of four does not recover W8A8-context accuracy with
this PTQ search.

All final predictions are finite and positive. Coefficient-sum error,
contraction violation rate, and anchor error are zero for every validation
configuration.

## Final Allocation

The selected candidate is `LOCAL_R3_0094`, with calibration RMSE 0.321893 m.
Its exact weighted budgets are 3.708295 weight bits and 3.978706 activation
bits.

| Block | Weight bits | Activation bits |
| --- | ---: | ---: |
| Stem | 4 | 6 |
| Encoder layer1 | 4 | 4 |
| Encoder layer2 | 4 | 4 |
| Encoder layer3 | 4 | 4 |
| Encoder layer4 | 4 | 2 |
| Decoder layer1 | 2 | 6 |
| Decoder layer2 | 4 | 2 |
| Decoder layer3 | 4 | 4 |
| Decoder layer4 | 4 | 4 |
| Initial depth | 6 | 4 |

By logical cost, weights are 14.8518% W2, 84.8817% W4, and 0.2665% W6.
Activations are 8.1303% A2, 84.8042% A4, and 7.0656% A6. No W8 or A8 site is
required by the selected allocation outside the fixed propagation contract.

## Correctness Findings

Two orchestration defects were found during final audit:

1. The validation writer received `staging/predictions` although the shared
   writer already appends `predictions/`. This produced a duplicated
   `predictions/predictions/<config>` directory. The output-root contract is
   now explicit and covered by a regression test.
2. The final selector compared only refinement candidates. Every refinement
   candidate was worse than the local-search incumbent, so the old code chose
   a 1.273988 m calibration candidate and produced 1.871478 m validation RMSE.
   The selector now compares the incumbent with the full refinement set. A
   regression test prevents a worse refinement candidate from replacing it.

The official checkpoint, model architecture, quantization policy, calibration
indices, validation identities, and prediction tensors were not changed by
these fixes.

## Artifacts

Results are under
`profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64`:

- `final_assignment.json` and `final_allocation.csv`
- `calibration_metrics.csv` and `validation_metrics.csv`
- `final_block_bits.png` and `.pdf`
- `final_bit_cost_fractions.png` and `.pdf`
- `calibration_rmse_bit_budgets.png` and `.pdf`
- `prediction_comparison_64.png` and `.pdf`

The manifest records SHA-256 hashes for all result files and the audit checks
the exact 64 prediction identities across all four configurations.
