# Unified P3/T3 Selection Design

## Goal

Apply one fair P3/T3 selection protocol to CSPN, DySPN, NLSPN, and
CompletionFormer while retaining model-specific sensitive-layer choices.

## Selection contract

- Every eligible module starts at W4A4.
- A selected sensitive module is promoted as a pair to W8A8.
- No independent W4A8 or W8A4 promotion is part of P3/T3.
- P3/T3 candidates are searched with the existing measured prefix/tail
  structure and model-specific cost tables.
- All models use the same 64 validation sample identities, the same 128
  calibration sample identities, and the same validation list.
- Candidate quality is `mean_of_per_sample_rmse`; pooled RMSE remains a
  reported secondary metric.
- A candidate is acceptable only when its paired FP32 evaluation is finite,
  propagation-valid, and has no non-positive prediction at valid GT pixels.
- The primary quality gate is a relative RMSE increase of at most 10 percent
  over the paired FP32 baseline for that model.
- Among acceptable candidates, choose the lowest W8A8 protection cost, then
  the lowest mean per-sample RMSE, then the lowest pooled RMSE.
- If no candidate passes the quality gate, publish an explicit failed search
  result rather than selecting a different method or silently relaxing the
  constraint.

## Scope

The existing P3/T3 search, launch configuration, artifact schema, and tests
are updated. Model architectures, checkpoints, quantizer math, calibration
scale rules, and non-P3/T3 methods are unchanged.

