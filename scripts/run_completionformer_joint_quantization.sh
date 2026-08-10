#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${COMPLETIONFORMER_PYTHON:-python}"
RUN_DIR="${COMPLETIONFORMER_RUN_DIR:-/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/completionformer_iter18}"
REFERENCE_METRICS="${COMPLETIONFORMER_REFERENCE_METRICS:-/workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation/primary/rtn/completionformer/sample_metrics.csv}"
DATA_ROOT="${SPN_DATA_ROOT:-/workspace/CSPN/cspn_pytorch}"
OUTPUT_ROOT="${COMPLETIONFORMER_JOINT_OUTPUT_ROOT:-/workspace/SPN_Quantization/profile_logs/nyu_completionformer_joint_integer_64}"
MODEL_ROOT="$OUTPUT_ROOT/completionformer"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis"
DEVICE="${COMPLETIONFORMER_DEVICE:-cuda:0}"

export COMPLETIONFORMER_ROOT="${COMPLETIONFORMER_ROOT:-$ROOT/external/CompletionFormer}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

if [[ -n "${COMPLETIONFORMER_DCN_PATH:-}" ]]; then
  test -d "$COMPLETIONFORMER_DCN_PATH"
  compgen -G "$COMPLETIONFORMER_DCN_PATH/DCN*.so" > /dev/null
  export PYTHONPATH="$COMPLETIONFORMER_DCN_PATH:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

test -f "$RUN_DIR/best.pt"
test -f "$RUN_DIR/args.json"
test -f "$REFERENCE_METRICS"
test -f "$COMPLETIONFORMER_ROOT/src/model/pvt.py"

"$PYTHON_BIN" "$ROOT/scripts/run_nyu_rtn_quantization.py" \
  --run-dir "$RUN_DIR" \
  --checkpoint best.pt \
  --sample-metrics "$REFERENCE_METRICS" \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --seed 20260804 \
  --quant-backend completionformer_joint \
  --calibration-samples 64 \
  --max-eval-samples 64 \
  --config-names \
    FP32 \
    JIQ_RTN_W4A4 \
    JIQ_Attention_W4A4 \
    JIQ_Concat_W4A4 \
    JIQ_Joint_W4A4 \
    JIQ_W4A8 \
  --export-prediction-configs \
    FP32 \
    JIQ_RTN_W4A4 \
    JIQ_Attention_W4A4 \
    JIQ_Concat_W4A4 \
    JIQ_Joint_W4A4 \
    JIQ_W4A8

"$PYTHON_BIN" "$ROOT/scripts/plot_completionformer_joint_quantization.py" \
  --root "$MODEL_ROOT" \
  --out-dir "$ANALYSIS_ROOT" \
  --expected-samples 64 \
  --dpi 120
