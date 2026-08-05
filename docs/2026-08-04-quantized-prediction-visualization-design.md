# Quantized Prediction Visualization Design

## Goal

Export and visualize quantized NYU depth-completion predictions from the
official CSPN, DySPN, NLSPN, and CompletionFormer checkpoints. Compare ground
truth, FP32, hardware-aligned W8A8, W4A4 baseline, the best sparse-A8
allocation, and full W4A8 under one integer-backend contract.

## Scope

The evaluation uses the existing fixed 64 NYU validation samples and the same
128 calibration samples used by the activation bit-allocation study. All 64
predictions are persisted for every selected configuration. Formal figures use
four representative samples per model.

The configurations are:

- `FP32`
- `MP_W8A8_full`
- `MP_W4A4_base`
- best sparse A8: `MP_heads_A8` for CSPN, `MP_encoder_A8` for DySPN,
  `MP_site01_A8` for NLSPN, and `MP_heads_A8` for CompletionFormer
- `MP_W4A8_full`

The W8A8 result must use the same hardware-aligned model preparation as the
mixed configurations: Conv-BN folding before calibration, signed symmetric
per-output-channel weights, per-tensor activations, unsigned ReLU outputs,
integer bias scale `sx*sw[o]`, and explicit merge requantization. Existing
unfused RTN W8A8 predictions are not used in the formal comparison.

## Export Architecture

The quantization runner gains an explicit prediction-export option accepting a
set of configuration names. Export behavior is independent of naming
conventions. Existing default exports remain compatible.

The mixed-precision configuration builder gains `MP_W8A8_full`, which uses the
same hardware-aligned preparation and manifest as the existing mixed
configurations. This avoids appending results from a second backend or
overwriting the formal hardware manifest. Mixed-precision exports reuse the
already measured configurations and checkpoints.
Each exported NPZ contains:

- prediction and absolute error
- ground truth and valid-ground-truth mask
- FP32 prediction for direct comparison
- nonfinite prediction mask
- sample index, model, and configuration

The formal output root is
`profile_logs/nyu_activation_bit_allocation/prediction_comparison`. NLSPN and
CompletionFormer run in `completionformer-py37` on `cuda:0` so the official DCN
extension is used.

## Representative Sample Selection

Selection is deterministic and operates on W4A4 per-sample degradation:

1. sample nearest the median W4A4 error
2. sample nearest the P90 W4A4 error
3. sample with maximum W4A4 error
4. sample with the largest reduction from W4A4 to the model's best sparse-A8
   configuration

For CSPN, the fourth sample instead prioritizes the highest nonfinite-pixel
rate. Ties are resolved by ascending sample index. The selected IDs and reasons
are written to `representative_samples.csv`.

## Figures

One primary figure is generated per model. Each of the four selected samples
uses two rows and six columns.

The first row contains:

1. ground truth
2. FP32 prediction
3. hardware-aligned W8A8 prediction
4. W4A4 baseline prediction
5. best sparse-A8 prediction
6. full W4A8 prediction

The second row contains the corresponding absolute-error maps; the ground-truth
column carries the sample identifier and selection reason. Prediction panels
show RMSE and nonfinite percentage.

Depth maps share a fixed 0-10 m scale. Error maps share a fixed 0-3 m scale so
colors remain comparable across configurations and models. Invalid ground-truth
pixels are light gray. NaN or Inf predictions inside valid ground-truth pixels
are magenta, explicitly distinguishing numerical failure from missing ground
truth. Values above the displayed error range are clipped only for rendering;
metrics use unclipped values.

A compact cross-model overview uses one representative sample per model and the
same columns. Figures use Arial-compatible fonts, no overall title, unrotated
column labels, stable panel dimensions, and separate depth/error colorbars.

## Metrics and Validation

The visualization script recomputes per-sample RMSE, MAE, and nonfinite rate
from the NPZ arrays. These values must match `sample_metrics.csv` within a small
floating-point tolerance. Generation stops on missing configurations, shape
mismatches, duplicate sample IDs, inconsistent evaluation indices, or metric
mismatches.

Before completion, verification checks that:

- every model/configuration has exactly the same 64 sample IDs
- calibration and evaluation indices match the existing formal metadata
- all NPZ arrays have compatible shapes
- all generated PNG files decode successfully
- the four model figures and compact overview have no text or panel overlap
- relevant tests pass in both the base and `completionformer-py37` environments

## Tests

Tests cover explicit export selection, the hardware-aligned W8A8 configuration,
representative-sample selection, CSPN nonfinite priority, metric
cross-validation, and render-mask conversion for gray invalid-GT pixels and
magenta nonfinite predictions. Existing quantization tests remain unchanged in
meaning.

## Outputs

- all 64 NPZ predictions for each selected model/configuration
- `representative_samples.csv`
- `prediction_metrics.csv`
- `cspn_quantized_prediction_comparison.png`
- `dyspn_quantized_prediction_comparison.png`
- `nlspn_quantized_prediction_comparison.png`
- `completionformer_quantized_prediction_comparison.png`
- `quantized_prediction_overview.png`
