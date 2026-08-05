#!/usr/bin/env bash
set -euo pipefail

# Run the shared quantization interface for all four model families.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/output/nyu_converged_baselines}"
SAMPLE_ROOT="${SAMPLE_ROOT:-$ROOT/profile_logs/nyu_activation_outliers}"
OUT_ROOT="${OUT_ROOT:-$ROOT/profile_logs/nyu_rtn_quantization}"
QUANT_BACKEND="${QUANT_BACKEND:-hardware}"
DEVICE="${DEVICE:-cuda:0}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
MODELS="${MODELS:-cspn dyspn nlspn completionformer}"

declare -A ITERATIONS=(
  [cspn]="${CSPN_ITERATION:-24}"
  [dyspn]="${DYSPN_ITERATION:-6}"
  [nlspn]="${NLSPN_ITERATION:-18}"
  [completionformer]="${COMPLETIONFORMER_ITERATION:-18}"
)

python_for_model() {
  case "$1" in
    cspn) echo "${CSPN_PYTHON:-python}" ;;
    dyspn) echo "${DYSPN_PYTHON:-python}" ;;
    nlspn) echo "${NLSPN_PYTHON:-python}" ;;
    completionformer) echo "${COMPLETIONFORMER_PYTHON:-python}" ;;
    *) echo "unknown model: $1" >&2; return 2 ;;
  esac
}

for model in $MODELS; do
  run_dir="$RUN_ROOT/${model}_iter${ITERATIONS[$model]}"
  sample_metrics="$SAMPLE_ROOT/$model/sample_metrics.csv"
  if [[ ! -f "$run_dir/args.json" ]]; then
    echo "missing run metadata: $run_dir/args.json" >&2
    exit 1
  fi
  if [[ ! -f "$sample_metrics" ]]; then
    echo "missing calibration metrics: $sample_metrics" >&2
    exit 1
  fi
  python_bin="$(python_for_model "$model")"
  echo "[quantize] model=$model backend=$QUANT_BACKEND python=$python_bin"
  "$python_bin" "$ROOT/scripts/run_nyu_rtn_quantization.py" \
    --run-dir "$run_dir" \
    --sample-metrics "$sample_metrics" \
    --out-dir "$OUT_ROOT" \
    --quant-backend "$QUANT_BACKEND" \
    --device "$DEVICE" \
    --calibration-samples "$CALIBRATION_SAMPLES" \
    --max-eval-samples "$MAX_EVAL_SAMPLES"
done
