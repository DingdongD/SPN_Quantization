#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${COMPLETIONFORMER_PYTHON:-python}"
OUTPUT_ROOT="${COMPLETIONFORMER_FRONT_PARETO_OUTPUT_ROOT:-$ROOT/profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64}"
MODEL_ROOT="$OUTPUT_ROOT/completionformer"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis"
DEVICE="${COMPLETIONFORMER_DEVICE:-cuda:0}"

: "${COMPLETIONFORMER_RUN_DIR:?set COMPLETIONFORMER_RUN_DIR to the converged official CompletionFormer run}"
: "${COMPLETIONFORMER_REFERENCE_METRICS:?set COMPLETIONFORMER_REFERENCE_METRICS to the fixed 64-sample metrics CSV}"
: "${SPN_DATA_ROOT:?set SPN_DATA_ROOT to the NYU training workspace}"
: "${COMPLETIONFORMER_DCN_PATH:?set COMPLETIONFORMER_DCN_PATH to the verified DCN extension directory}"

export COMPLETIONFORMER_ROOT="${COMPLETIONFORMER_ROOT:-$ROOT/external/CompletionFormer}"
export PYTHONPATH="$COMPLETIONFORMER_DCN_PATH:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

test -f "$COMPLETIONFORMER_RUN_DIR/best.pt"
test -f "$COMPLETIONFORMER_RUN_DIR/args.json"
test -f "$COMPLETIONFORMER_REFERENCE_METRICS"
test -f "$SPN_DATA_ROOT/datalist/nyudepth_hdf5_train.csv"
test -f "$SPN_DATA_ROOT/datalist/nyudepth_hdf5_val.csv"
test -f "$COMPLETIONFORMER_ROOT/src/model/completionformer.py"
test -f "$COMPLETIONFORMER_ROOT/src/model/pvt.py"
test -d "$COMPLETIONFORMER_DCN_PATH"
compgen -G "$COMPLETIONFORMER_DCN_PATH/DCN*.so" > /dev/null
git -C "$COMPLETIONFORMER_ROOT" rev-parse --verify HEAD > /dev/null

"$PYTHON_BIN" "$ROOT/scripts/run_nyu_rtn_quantization.py" \
  --run-dir "$COMPLETIONFORMER_RUN_DIR" \
  --checkpoint best.pt \
  --sample-metrics "$COMPLETIONFORMER_REFERENCE_METRICS" \
  --data-root "$SPN_DATA_ROOT" \
  --out-dir "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --seed 20260804 \
  --quant-backend completionformer_front_pareto \
  --calibration-samples 64 \
  --front-search-samples 32 \
  --front-search-seed 20260810 \
  --max-eval-samples 64

"$PYTHON_BIN" "$ROOT/scripts/plot_completionformer_front_encoder_pareto.py" \
  --root "$MODEL_ROOT" \
  --out-dir "$ANALYSIS_ROOT" \
  --expected-samples 64 \
  --dpi 120
