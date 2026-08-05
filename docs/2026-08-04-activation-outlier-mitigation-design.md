# Activation Outlier and Mitigation Design

## Objective

Measure whether activation outliers explain W4A4 degradation in the official
CSPN, DySPN, NLSPN, and CompletionFormer checkpoints, distinguish encoder size
from encoder sensitivity, and evaluate whether percentile clipping,
SmoothQuant, or AWQ-style weight clipping is the appropriate mitigation.

## Distribution Profiler

Use the same deterministic 128 NYU training samples as hardware PTQ
calibration. Profile every quantized Conv2d/Linear input and output after legal
Conv-BN folding. For each call-indexed tensor, collect a bounded deterministic
sample of absolute activation values and per-input-channel absolute maxima.
Persist `p75`, `p90`, `p99`, `p99.9`, `p99.99`, and `max`, plus:

- `max / p99.99` for extreme-tail severity;
- `p99.99 / p99` for tail growth;
- fractions above each percentile threshold;
- per-channel median, p99, and maximum absmax;
- `channel_max / channel_median` for channel-localized outliers.

The bounded sample uses deterministic evenly spaced tensor positions rather
than full activation dumps. A minimum one-million-value reservoir per site
makes p99.99 represent roughly 100 retained values while bounding host memory.
Per-channel maxima are exact over all calibration samples.

## Occupancy Definition

Report three independent encoder shares from the existing hardware run:

1. quantized parameter elements;
2. activation QDQ tensor elements processed over 64 evaluation samples;
3. number of quantized boundaries.

These are footprint/traffic proxies, not CUDA latency. Existing operator
profiles do not carry module ancestry, so the report must not label them as
measured encoder latency.

## Mitigation Evaluation

Rank candidate sites by tail severity and W4A4 SQNR. Evaluate the following on
the fixed 64 validation samples:

- MinMax baseline;
- percentile activation clipping at p99, p99.9, and p99.99;
- SmoothQuant at alpha 0.25, 0.50, and 0.75 using per-input-channel activation
  absmax and per-input-channel weight absmax;
- AWQ-style W4 clipping ratios 1.0, 0.9, and 0.8 while retaining MinMax
  activations.

SmoothQuant is applied locally at Conv/Linear inputs as `x / s` and `W * s`,
where `s = a_max^alpha / w_max^(1-alpha)`. This preserves FP32 output before
QDQ and quantifies the best-case accuracy potential. The report must identify
that realizing these scales without runtime multiplies requires folding them
into a unique producer; branch merges may prevent free deployment.

AWQ is treated as a weight-side control. If activation-only clipping or
SmoothQuant materially outperforms AWQ, the conclusion is activation outliers
and coarse activation resolution rather than weight outliers.

## Outputs

Write results under `profile_logs/nyu_activation_outliers/`:

- per-site and per-group percentile CSV files;
- encoder occupancy CSV;
- outlier severity heatmaps and percentile-tail plots;
- mitigation RMSE/nonfinite comparison CSV and plot;
- a Markdown findings report.

## Acceptance Criteria

- All four models profile the same 128 calibration indices.
- Every profiled site records at least one million retained values unless the
  complete site contains fewer values.
- Percentiles are monotonic and exact channel maxima are finite.
- Occupancy shares sum to 100% per model and measure.
- Mitigation rows use the fixed 64 validation indices and explicitly report
  nonfinite rates.
- Results distinguish QDQ accuracy simulation from packed integer latency.

