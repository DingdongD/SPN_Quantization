#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 2
fi

PHASE="$1"
case "$PHASE" in
  smoke)
    CALIBRATION_SAMPLES=2
    EVALUATION_SAMPLES=2
    VISUAL_SAMPLES=2
    ;;
  full)
    CALIBRATION_SAMPLES=128
    EVALUATION_SAMPLES=64
    VISUAL_SAMPLES=3
    ;;
  *)
    echo "phase must be smoke or full" >&2
    exit 2
    ;;
esac

: "${SPN_DATA_ROOT:?SPN_DATA_ROOT must point to the NYU training workspace}"
: "${SPN_EXTERNAL_ROOT:?SPN_EXTERNAL_ROOT must point to the official model repositories}"
: "${COMPLETIONFORMER_ROOT:?COMPLETIONFORMER_ROOT must point to the official CompletionFormer repository}"
: "${FP4_REFERENCE_ROOT:?FP4_REFERENCE_ROOT must contain the fixed 64-sample reference metrics}"
: "${FP4_OUTPUT_ROOT:?FP4_OUTPUT_ROOT must name a new experiment output directory}"
: "${CSPN_PYTHON:?CSPN_PYTHON must select the CSPN environment}"
: "${DYSPN_PYTHON:?DYSPN_PYTHON must select the DySPN environment}"
: "${NLSPN_PYTHON:?NLSPN_PYTHON must select the NLSPN CUDA-extension environment}"
: "${COMPLETIONFORMER_PYTHON:?COMPLETIONFORMER_PYTHON must select the CompletionFormer CUDA-extension environment}"
: "${CSPN_DEVICE:?CSPN_DEVICE must be explicit}"
: "${DYSPN_DEVICE:?DYSPN_DEVICE must be explicit}"
: "${NLSPN_DEVICE:?NLSPN_DEVICE must be explicit}"
: "${COMPLETIONFORMER_DEVICE:?COMPLETIONFORMER_DEVICE must be explicit}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$SPN_DATA_ROOT/output/nyu_converged_baselines"
CONFIGS=(
  FP32
  FP4V_W8A4
  FP4V_W8E2M1
  FP4V_W8A8
  FP4V_W4A4
  FP4V_W4E2M1
  FP4V_W4A8
)
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
declare -A DEVICES=(
  [cspn]="$CSPN_DEVICE"
  [dyspn]="$DYSPN_DEVICE"
  [nlspn]="$NLSPN_DEVICE"
  [completionformer]="$COMPLETIONFORMER_DEVICE"
)

for model in "${MODELS[@]}"; do
  run_dir="$RUN_ROOT/${model}_iter${ITERATIONS[$model]}"
  sample_metrics="$FP4_REFERENCE_ROOT/$model/sample_metrics.csv"
  [[ -f "$run_dir/args.json" ]]
  [[ -f "$run_dir/best.pt" ]]
  [[ -f "$sample_metrics" ]]

  echo "[fp4-$PHASE] model=$model device=${DEVICES[$model]}"
  (
    cd "$SPN_DATA_ROOT"
    SPN_EXTERNAL_ROOT="$SPN_EXTERNAL_ROOT" \
    COMPLETIONFORMER_ROOT="$COMPLETIONFORMER_ROOT" \
    "${PYTHONS[$model]}" "$ROOT/scripts/run_nyu_rtn_quantization.py" \
      --run-dir "$run_dir" \
      --checkpoint "$run_dir/best.pt" \
      --sample-metrics "$sample_metrics" \
      --data-root "$SPN_DATA_ROOT" \
      --out-dir "$FP4_OUTPUT_ROOT" \
      --device "${DEVICES[$model]}" \
      --seed 20260804 \
      --calibration-samples "$CALIBRATION_SAMPLES" \
      --max-eval-samples "$EVALUATION_SAMPLES" \
      --quant-backend fp4 \
      --config-names "${CONFIGS[@]}" \
      --export-prediction-configs "${CONFIGS[@]}"
  )
done

analysis_dir="$FP4_OUTPUT_ROOT/analysis"
figure_dir="$FP4_OUTPUT_ROOT/figures"
"$CSPN_PYTHON" "$ROOT/scripts/analyze_fp4_activation_validation.py" \
  --root "$FP4_OUTPUT_ROOT" \
  --out-dir "$analysis_dir" \
  --expected-samples "$EVALUATION_SAMPLES" \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20260806
"$CSPN_PYTHON" "$ROOT/scripts/plot_fp4_activation_validation.py" \
  --root "$FP4_OUTPUT_ROOT" \
  --analysis-dir "$analysis_dir" \
  --out-dir "$figure_dir" \
  --visual-samples "$VISUAL_SAMPLES"
