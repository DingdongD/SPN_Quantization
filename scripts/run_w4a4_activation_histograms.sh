#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 2
fi

CALIBRATION_SAMPLES=64
case "$1" in
  smoke)
    PROFILE_SAMPLES=1
    ;;
  full)
    PROFILE_SAMPLES=64
    ;;
  *)
    echo "phase must be smoke or full" >&2
    exit 2
    ;;
esac

: "${SPN_DATA_ROOT:?}"
: "${SPN_EXTERNAL_ROOT:?}"
: "${COMPLETIONFORMER_ROOT:?}"
: "${STRICT_W4A4_FP4_ROOT:?}"
: "${W4A4_HISTOGRAM_OUTPUT_ROOT:?}"
: "${CSPN_PYTHON:?}"
: "${DYSPN_PYTHON:?}"
: "${NLSPN_PYTHON:?}"
: "${COMPLETIONFORMER_PYTHON:?}"
: "${CSPN_GPU:?}"
: "${DYSPN_GPU:?}"
: "${NLSPN_GPU:?}"
: "${COMPLETIONFORMER_GPU:?}"

[[ ! -e "$W4A4_HISTOGRAM_OUTPUT_ROOT" ]]
mkdir -p "$W4A4_HISTOGRAM_OUTPUT_ROOT/logs"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$SPN_DATA_ROOT/output/nyu_converged_baselines"
MODELS=(cspn dyspn nlspn completionformer)
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
  strict_dir="$STRICT_W4A4_FP4_ROOT/primary/rtn/$model"
  [[ -f "$run_dir/args.json" ]]
  [[ -f "$run_dir/best.pt" ]]
  [[ -f "$strict_dir/metadata.json" ]]
  [[ -f "$strict_dir/semantic_a8_boundaries.csv" ]]
done

run_model() {
  local model="$1"
  local run_dir="$RUN_ROOT/${model}_iter${ITERATIONS[$model]}"
  echo "[w4a4-histogram] model=$model gpu=${GPUS[$model]}"
  (
    cd "$SPN_DATA_ROOT"
    SPN_EXTERNAL_ROOT="$SPN_EXTERNAL_ROOT" \
    COMPLETIONFORMER_ROOT="$COMPLETIONFORMER_ROOT" \
    CUDA_VISIBLE_DEVICES="${GPUS[$model]}" \
    "${PYTHONS[$model]}" \
      "$ROOT/scripts/run_w4a4_activation_histograms.py" \
      --run-dir "$run_dir" \
      --checkpoint "$run_dir/best.pt" \
      --data-root "$SPN_DATA_ROOT" \
      --strict-root "$STRICT_W4A4_FP4_ROOT" \
      --out-dir "$W4A4_HISTOGRAM_OUTPUT_ROOT" \
      --device cuda:0 \
      --seed 20260804 \
      --calibration-samples "$CALIBRATION_SAMPLES" \
      --profile-samples "$PROFILE_SAMPLES" \
      --histogram-bins 128 \
      --sample-capacity 1000000 \
      --fold-max-error 0.05
  )
}

pids=()
logs=()
for model in "${MODELS[@]}"; do
  log="$W4A4_HISTOGRAM_OUTPUT_ROOT/logs/${model}.log"
  run_model "$model" >"$log" 2>&1 &
  pids+=("$!")
  logs+=("$log")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed=1
    tail -n 100 "${logs[$index]}" >&2
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "W4A4 activation histogram collection failed" >&2
  exit 1
fi

"$CSPN_PYTHON" "$ROOT/scripts/plot_w4a4_activation_histograms.py" \
  --root "$W4A4_HISTOGRAM_OUTPUT_ROOT" \
  --models cspn dyspn nlspn completionformer \
  --sites-per-page 2 \
  --critical-limit 10
