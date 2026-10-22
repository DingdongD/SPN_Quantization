#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$(realpath -m "$1")"
source_config="$repo_root/configs/four_model_unified_fp16_task_aware.json"
dse_config="$repo_root/configs/spn_array_dse_bf16.json"

if [[ -e "$output_root" ]]; then
  echo "output already exists: $output_root" >&2
  exit 1
fi

mkdir -p "$output_root/workloads"

(
  cd /workspace/CSPN/cspn_pytorch
  PYTHONPATH="$repo_root" /opt/conda/bin/python \
    "$repo_root/scripts/export_nyu_spn_workload.py" \
    --config "$source_config" --model cspn --device cuda:0 \
    --output "$output_root/workloads/cspn.json"
)

(
  cd /workspace/external_depth_completion_models/DySPN
  PYTHONPATH="/workspace/external_depth_completion_models/DySPN:$repo_root" \
    /opt/conda/bin/python "$repo_root/scripts/export_nyu_spn_workload.py" \
    --config "$source_config" --model dyspn --device cuda:2 \
    --output "$output_root/workloads/dyspn.json"
)

(
  cd /workspace/external_depth_completion_models/NLSPN_ECCV20
  LD_LIBRARY_PATH="/opt/conda/envs/completionformer-py37/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH:-}" \
  SPN_EXTERNAL_ROOT=/workspace/external_depth_completion_models \
  PYTHONPATH="/workspace/external_depth_completion_models/NLSPN_ECCV20/src/model/deformconv:/workspace/external_depth_completion_models/NLSPN_ECCV20/src:$repo_root" \
    /opt/conda/envs/completionformer-py37/bin/python \
    "$repo_root/scripts/export_nyu_spn_workload.py" \
    --config "$source_config" --model nlspn --device cuda:1 \
    --output "$output_root/workloads/nlspn.json"
)

(
  cd /workspace/CompletionFormer
  COMPLETIONFORMER_ROOT=/workspace/CompletionFormer \
  LD_LIBRARY_PATH="/opt/conda/envs/completionformer-py37/lib/python3.7/site-packages/torch/lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="/workspace/CompletionFormer/src/model/deformconv:/workspace/CompletionFormer/src:$repo_root" \
    /opt/conda/envs/completionformer-py37/bin/python \
    "$repo_root/scripts/export_nyu_spn_workload.py" \
    --config "$source_config" --model completionformer --device cuda:3 \
    --output "$output_root/workloads/completionformer.json"
)

/opt/conda/bin/python "$repo_root/scripts/run_spn_array_dse.py" \
  --config "$dse_config" \
  --workload "cspn=$output_root/workloads/cspn.json" \
  --workload "dyspn=$output_root/workloads/dyspn.json" \
  --workload "nlspn=$output_root/workloads/nlspn.json" \
  --workload "completionformer=$output_root/workloads/completionformer.json" \
  --output "$output_root/analysis"
