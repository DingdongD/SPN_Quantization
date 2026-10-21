#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT [WORKLOAD_ROOT]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$(realpath -m "$1")"
if [[ "$#" -eq 2 ]]; then
  workload_root="$(realpath -m "$2")"
else
  workload_root="$repo_root/reports/generated/spn_array_dse_bf16/workloads"
fi
memory_config="$repo_root/configs/spn_array_dse_bf16_lpddr4_4mb.json"
cacti_config="$repo_root/configs/cacti/sram_4mb_32nm_512b.json"

if [[ -e "$output_root" ]]; then
  echo "output already exists: $output_root" >&2
  exit 1
fi
for model in cspn dyspn nlspn completionformer; do
  if [[ ! -f "$workload_root/$model.json" ]]; then
    echo "missing workload profile: $workload_root/$model.json" >&2
    exit 1
  fi
done

mkdir -p "$output_root/cacti"
/opt/conda/bin/python "$repo_root/scripts/run_cacti_sram.py" \
  --config "$cacti_config" \
  --raw-output "$output_root/cacti/sram_4mb_32nm_512b.txt" \
  --json-output "$output_root/cacti/sram_4mb_32nm_512b.json"

/opt/conda/bin/python "$repo_root/scripts/run_spn_memory_dse.py" \
  --config "$memory_config" \
  --cacti-result "$output_root/cacti/sram_4mb_32nm_512b.json" \
  --workload "cspn=$workload_root/cspn.json" \
  --workload "dyspn=$workload_root/dyspn.json" \
  --workload "nlspn=$workload_root/nlspn.json" \
  --workload "completionformer=$workload_root/completionformer.json" \
  --output "$output_root/analysis"
