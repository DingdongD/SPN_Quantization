#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 2
fi

case "$1" in
  smoke)
    CALIBRATION_SAMPLES=2
    EVALUATION_SAMPLES=2
    ;;
  full)
    CALIBRATION_SAMPLES=64
    EVALUATION_SAMPLES=64
    ;;
  *)
    echo "phase must be smoke or full" >&2
    exit 2
    ;;
esac

: "${SPN_DATA_ROOT:?}"
: "${SPN_EXTERNAL_ROOT:?}"
: "${COMPLETIONFORMER_ROOT:?}"
: "${STRICT_RECONSTRUCTION_ROOT:?}"
: "${STRICT_REFERENCE_ROOT:?}"
: "${STRICT_W4A4_FP4_OUTPUT_ROOT:?}"
: "${CSPN_PYTHON:?}"
: "${DYSPN_PYTHON:?}"
: "${NLSPN_PYTHON:?}"
: "${COMPLETIONFORMER_PYTHON:?}"
: "${CSPN_GPU:?}"
: "${DYSPN_GPU:?}"
: "${NLSPN_GPU:?}"
: "${COMPLETIONFORMER_GPU:?}"

[[ ! -e "$STRICT_W4A4_FP4_OUTPUT_ROOT" ]]
mkdir -p "$STRICT_W4A4_FP4_OUTPUT_ROOT/logs"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$SPN_DATA_ROOT/output/nyu_converged_baselines"
MODELS=(cspn dyspn nlspn completionformer)
METHODS=(rtn adaround brecq)
declare -A ITERATIONS=(
  [cspn]=24
  [dyspn]=6
  [nlspn]=18
  [completionformer]=18
)
declare -A PYTHONS=(
  [cspn]="$CSPN_PYTHON"
  [dyspn]="$DYSPN_PYTHON"
  [nlspn]="$NLSPN_PYTHON"
  [completionformer]="$COMPLETIONFORMER_PYTHON"
)
declare -A GPUS=(
  [cspn]="$CSPN_GPU"
  [dyspn]="$DYSPN_GPU"
  [nlspn]="$NLSPN_GPU"
  [completionformer]="$COMPLETIONFORMER_GPU"
)

for model in "${MODELS[@]}"; do
  run_dir="$RUN_ROOT/${model}_iter${ITERATIONS[$model]}"
  sample_metrics="$STRICT_REFERENCE_ROOT/$model/sample_metrics.csv"
  [[ -f "$run_dir/args.json" ]]
  [[ -f "$run_dir/best.pt" ]]
  [[ -f "$sample_metrics" ]]
  for method in adaround brecq; do
    manifest="$STRICT_RECONSTRUCTION_ROOT/${model}/${method}_strict/strict_reconstruction_manifest.json"
    [[ -f "$manifest" ]]
  done
done

run_model_method() {
  local method="$1"
  local model="$2"
  local run_dir="$RUN_ROOT/${model}_iter${ITERATIONS[$model]}"
  local sample_metrics="$STRICT_REFERENCE_ROOT/$model/sample_metrics.csv"
  local -a contract_args=()
  if [[ "$method" != "rtn" ]]; then
    local manifest="$STRICT_RECONSTRUCTION_ROOT/${model}/${method}_strict/strict_reconstruction_manifest.json"
    contract_args=(--reconstruction-manifest "$manifest")
  fi

  echo "[strict-fp4] method=$method model=$model gpu=${GPUS[$model]}"
  (
    cd "$SPN_DATA_ROOT"
    SPN_EXTERNAL_ROOT="$SPN_EXTERNAL_ROOT" \
    COMPLETIONFORMER_ROOT="$COMPLETIONFORMER_ROOT" \
    CUDA_VISIBLE_DEVICES="${GPUS[$model]}" \
    "${PYTHONS[$model]}" \
      "$ROOT/scripts/run_nyu_edge_quantization.py" \
      --run-dir "$run_dir" \
      --checkpoint best.pt \
      --sample-metrics "$sample_metrics" \
      --data-root "$SPN_DATA_ROOT" \
      --out-dir "$STRICT_W4A4_FP4_OUTPUT_ROOT/primary/$method" \
      --device cuda:0 \
      --seed 20260804 \
      --calibration-samples "$CALIBRATION_SAMPLES" \
      --max-eval-samples "$EVALUATION_SAMPLES" \
      --merge-policy independent \
      "${contract_args[@]}" \
      --quant-backend fp4 \
      --config-names FP32 FP4V_W4A4 FP4V_W4E2M1 FP4V_W4A8 \
      --export-prediction-configs FP32 FP4V_W4A4 FP4V_W4E2M1 FP4V_W4A8

    SPN_EXTERNAL_ROOT="$SPN_EXTERNAL_ROOT" \
    COMPLETIONFORMER_ROOT="$COMPLETIONFORMER_ROOT" \
    CUDA_VISIBLE_DEVICES="${GPUS[$model]}" \
    "${PYTHONS[$model]}" \
      "$ROOT/scripts/run_nyu_edge_quantization.py" \
      --run-dir "$run_dir" \
      --checkpoint best.pt \
      --sample-metrics "$sample_metrics" \
      --data-root "$SPN_DATA_ROOT" \
      --out-dir "$STRICT_W4A4_FP4_OUTPUT_ROOT/stress/$method" \
      --device cuda:0 \
      --seed 20260804 \
      --calibration-samples "$CALIBRATION_SAMPLES" \
      --max-eval-samples "$EVALUATION_SAMPLES" \
      --merge-policy independent \
      "${contract_args[@]}" \
      --quant-backend hardware \
      --config-names FP32 HW_W4A4_full \
      --export-prediction-configs FP32 HW_W4A4_full
  )
}

for method in "${METHODS[@]}"; do
  pids=()
  logs=()
  for model in "${MODELS[@]}"; do
    log="$STRICT_W4A4_FP4_OUTPUT_ROOT/logs/${method}_${model}.log"
    run_model_method "$method" "$model" >"$log" 2>&1 &
    pids+=("$!")
    logs+=("$log")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      failed=1
      log="${logs[$index]}"
      tail -n 80 "$log" >&2
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "strict W4A4/FP4 evaluation failed for method=$method" >&2
    exit 1
  fi
done

"$CSPN_PYTHON" "$ROOT/scripts/analyze_strict_w4a4_fp4_evaluation.py" \
  --root "$STRICT_W4A4_FP4_OUTPUT_ROOT" \
  --out-dir "$STRICT_W4A4_FP4_OUTPUT_ROOT/analysis" \
  --expected-samples "$EVALUATION_SAMPLES" \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20260806

"$CSPN_PYTHON" "$ROOT/scripts/plot_strict_w4a4_fp4_evaluation.py" \
  --root "$STRICT_W4A4_FP4_OUTPUT_ROOT" \
  --analysis-dir "$STRICT_W4A4_FP4_OUTPUT_ROOT/analysis" \
  --out-dir "$STRICT_W4A4_FP4_OUTPUT_ROOT/figures"
