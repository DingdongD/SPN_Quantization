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
joint_config="$repo_root/configs/spn_joint_dse_bf16_lpddr4.json"
workers="${SPN_JOINT_DSE_WORKERS:-16}"

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
/opt/conda/bin/python "$repo_root/scripts/run_cacti_sram_sweep.py" \
  --job "$repo_root/configs/cacti/sram_2mb_32nm_256b.json" \
    "$output_root/cacti/sram_2mb_32nm_256b.txt" \
    "$output_root/cacti/sram_2mb_32nm_256b.json" \
  --job "$repo_root/configs/cacti/sram_4mb_32nm_512b.json" \
    "$output_root/cacti/sram_4mb_32nm_512b.txt" \
    "$output_root/cacti/sram_4mb_32nm_512b.json" \
  --job "$repo_root/configs/cacti/sram_8mb_32nm_1024b.json" \
    "$output_root/cacti/sram_8mb_32nm_1024b.txt" \
    "$output_root/cacti/sram_8mb_32nm_1024b.json"

/opt/conda/bin/python "$repo_root/scripts/run_spn_joint_dse.py" \
  --config "$joint_config" \
  --sram-macro "small=$output_root/cacti/sram_2mb_32nm_256b.json" \
  --sram-macro "balanced=$output_root/cacti/sram_4mb_32nm_512b.json" \
  --sram-macro "large=$output_root/cacti/sram_8mb_32nm_1024b.json" \
  --workers "$workers" \
  --workload "cspn=$workload_root/cspn.json" \
  --workload "dyspn=$workload_root/dyspn.json" \
  --workload "nlspn=$workload_root/nlspn.json" \
  --workload "completionformer=$workload_root/completionformer.json" \
  --output "$output_root/analysis"
