#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-$ROOT/profile_logs/nyu_prediction_comparison_converged}"
PRED_DIR="${PRED_DIR:-$OUT_DIR/predictions}"
DEVICE="${DEVICE:-cuda:0}"
SAMPLES="${SAMPLES:-0 1 2 3}"

CSPN_RUN="${CSPN_RUN:-$ROOT/output/nyu_converged_baselines/cspn_iter24}"
DYSPN_RUN="${DYSPN_RUN:-$ROOT/output/nyu_converged_baselines/dyspn_iter6}"
NLSPN_RUN="${NLSPN_RUN:-$ROOT/output/nyu_converged_baselines/nlspn_iter18}"
COMPLETIONFORMER_RUN="${COMPLETIONFORMER_RUN:-$ROOT/output/nyu_converged_baselines/completionformer_iter18}"

rm -rf "$PRED_DIR"
mkdir -p "$PRED_DIR"

python scripts/export_nyu_predictions.py \
  --run-dir "$CSPN_RUN" \
  --label cspn_iter24 \
  --sample-indices $SAMPLES \
  --out-dir "$PRED_DIR" \
  --device "$DEVICE"

PYTHONPATH=/workspace/external_depth_completion_models/DySPN \
  conda run -n pointkan python scripts/export_nyu_predictions.py \
  --run-dir "$DYSPN_RUN" \
  --label dyspn_iter6 \
  --sample-indices $SAMPLES \
  --out-dir "$PRED_DIR" \
  --device "$DEVICE"

PYTHONPATH=/workspace/external_depth_completion_models/NLSPN_ECCV20/src:/workspace/external_depth_completion_models/NLSPN_ECCV20/src/model/deformconv \
  conda run -n completionformer-py37 python scripts/export_nyu_predictions.py \
  --run-dir "$NLSPN_RUN" \
  --label nlspn_iter18 \
  --sample-indices $SAMPLES \
  --out-dir "$PRED_DIR" \
  --device "$DEVICE"

PYTHONPATH=/workspace/CompletionFormer/src:/workspace/CompletionFormer/src/model/deformconv \
  conda run -n completionformer-py37 python scripts/export_nyu_predictions.py \
  --run-dir "$COMPLETIONFORMER_RUN" \
  --label completionformer_iter18 \
  --sample-indices $SAMPLES \
  --out-dir "$PRED_DIR" \
  --device "$DEVICE"

python scripts/visualize_nyu_prediction_comparison.py \
  --pred-dir "$PRED_DIR" \
  --out-dir "$OUT_DIR"
