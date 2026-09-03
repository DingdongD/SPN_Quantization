# FP4/FP8 Mixed-Precision Evaluation Design

## Goal

Evaluate whether FP4-E2M1 and FP8-E4M3FN mixed precision can reduce the
current W4A4 task-aware quantization loss on the four official SPN models.

## Fixed Protocol

- Models: official CSPN, DySPN, NLSPN, and CompletionFormer checkpoints.
- Calibration: the existing 128 train-split samples.
- Evaluation: the existing ordered 64 NYU validation samples.
- Propagation: FP16; affinity, offset, state, confidence, gates, and
  initial-depth propagation outputs are excluded from FP4/FP8 quantization.
- Ordinary Conv/ConvTranspose/Linear weights use per-output-channel scales.
- Ordinary activation owners use independent per-owner scales.
- FP4 uses the E2M1 finite codebook with a symmetric shared scale.
- FP8 uses the E4M3FN format with a symmetric shared scale.
- No exception-based fallback, alternate model implementation, or silent
  format substitution is allowed.

## Evaluation Matrix

1. FP32 reference.
2. FP8-E4M3FN W/A.
3. FP4-E2M1 W/A.
4. FP4-E2M1 weights with FP8-E4M3FN activations.
5. FP4 weights with sensitivity-selected mixed FP4/FP8 activations.
6. FP8 weights with sensitivity-selected mixed FP4/FP8 activations.

The mixed activation variants promote the existing sensitivity-ranked
ordinary owners. The selected format and weighted format/byte fractions are
recorded in the manifest; selection is not based on a hard-coded model
fallback.

## Metrics

Every model/configuration records pooled RMSE, mean sample RMSE, MAE, valid,
finite, and positive status. Quantization diagnostics record FP4 saturation,
FP4 zero-code ratio, FP8 non-finite count, per-owner SQNR, and output error
relative to the FP32 reference. The report uses activation-element and
weight-MAC weighted format fractions.

## Decision Criteria

- FP4 W + FP8 A materially better than FP4 W/A indicates activation format
  error dominates.
- Mixed FP4/FP8 activation close to all-FP8 activation indicates that
  sensitivity-guided FP8 promotion is viable.
- Any non-finite or non-positive prediction is an invalid result, even if its
  numeric RMSE is available.
- No QAT conclusion is drawn from this PTQ/fake-quant stage.
