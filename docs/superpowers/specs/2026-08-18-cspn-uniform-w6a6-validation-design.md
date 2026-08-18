# CSPN Uniform W6A6 Validation Design

## Goal

Measure whether uniform W6A6 ordinary-CNN quantization approaches the existing
P3/T3 W8A8-context accuracy under the exact task-sensitive CSPN evaluation
contract.

## Experimental Contract

- Official CSPN ResNet-18 checkpoint with 24 propagation iterations.
- Persisted stratified 128-sample NYU train calibration subset.
- Persisted fixed 64-sample NYU validation subset with seed `20260812`.
- Ordinary Conv weights and activations use uniform W6A6 with the existing
  Group-8 static MinMax and per-output-channel weight policies.
- Bias remains FP32, guidance remains FP32, and propagation remains A8 with
  INT16 Q13 coefficients and INT32 accumulation.
- No training, QAT, output clipping, or fallback is introduced.

## Integration

`UNIFORM_W6A6` becomes a formal validation candidate between uniform W4A4 and
P3/T3. The result audit requires five configurations, 64 prediction payloads
per configuration, identical sample identities, finite metrics, and zero
propagation-invariant violations. The prediction comparison adds a W6A6
column while preserving the existing plotting style.

## Outputs

The existing task-sensitive result root is regenerated from its complete phase
cache. Search metrics and the selected FINAL assignment remain unchanged.
Validation metrics, prediction payloads, plots, manifest hashes, and the result
report are updated to include W6A6.

