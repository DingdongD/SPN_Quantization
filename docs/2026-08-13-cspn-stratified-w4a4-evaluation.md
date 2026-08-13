# CSPN Stratified Calibration W4A4 Evaluation

## Protocol

The latest train-only `32_tail_96_kmedoids` calibration set was compared with
the seed-`20260812` random-128 baseline. Both runs use the official converged
CSPN iteration-24 checkpoint and the same 64 NYU evaluation indices. The
checkpoint, seed, calibration count, model preprocessing, quantization config,
and evaluation order are identical within every pair.

The RTN runner now accepts an explicit `--calibration-indices` JSON contract.
It validates exact count, uniqueness, and train-split bounds and records the
path, SHA256, and selection identity in metadata. There is no random fill or
fallback when an explicit index file is supplied.

## Results

| Path | Configuration | Random-128 RMSE | Stratified-128 RMSE | Change |
| --- | --- | ---: | ---: | ---: |
| Generic uniform INT | `HW_W4A4_full` | Inf | Inf | both failed |
| Propagation-aware INT | `PA_Constraint` | 2.44280 | 2.33477 | -4.42% |
| Propagation-aware INT | `PA_StateA8` | 2.46624 | 2.37138 | -3.85% |
| E2M1 activation reference | `FP4V_W4A4` | 1.12685 | 1.48735 | +31.99% |

`PA_Constraint` improves 47 of 64 paired samples; `PA_StateA8` improves 48 of
64. Generic INT W4A4 produces non-finite pixels for all 64 samples under both
calibration sets. The E2M1 row is not an INT-A4 result and serves only as a
format-sensitivity control.

For `PA_Constraint`, stratification reduces boundary RMSE by 6.86%, holes by
4.47%, and smooth-region RMSE by 4.20%. Near-range RMSE increases by 1.94% and
sparse-anchor RMSE by 5.07%. Aggregate encoder SQNR decreases from 6.98 dB to
6.23 dB, while decoder SQNR rises from 10.65 dB to 10.99 dB, depth-head SQNR
from 12.23 dB to 12.86 dB, and propagation-head SQNR from 22.96 dB to 26.36 dB.

## Conclusion

The stratified set gives a modest, statistically consistent improvement only
when CSPN uses propagation-aware integer constraints. It does not repair the
generic W4A4 instability, and its tail-heavy ranges can hurt other activation
formats. Calibration coverage is therefore complementary to, not a replacement
for, propagation-specific quantization.

Complete outputs are stored at
`/workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_w4a4_128`.
