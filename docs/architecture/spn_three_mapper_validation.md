# SPN LLMCompass–TCM–FactorFlow validation

## Scope

This study validates three independent dense-mapping paths on the executed
BF16 workloads for CSPN, DySPN, NLSPN, and CompletionFormer. Propagation is
excluded before mapper dispatch. The validated workload contains 486 dense
GEMM instances, 115 cross-model unique GEMM signatures, and four excluded
propagation operations.

The common memory path is LPDDR4 to separate activation/output and weight
SRAMs to a two-dimensional MAC array. Operands and outputs are BF16; partial
sums are FP32. SRAM capacity, CACTI port throughput and access energy, primary
70% sustained LPDDR4 bandwidth, and BF16 MAC energy are passed to every
backend.

## Pinned backends and mapper policy

- LLMCompass revision `62321b1ee28ddbdba8a2eb7475d7caa30f75e8be`:
  the repository's capacity- and bandwidth-constrained tiling search uses the
  pinned LLMCompass cycle primitive and minimizes latency. This is not the
  complete upstream `compile_and_simulate()` compiler flow.
- AccelForge/TCM revision `7312a7d915625552bf6eb1db577941f8317b6ac7`:
  one unfused Einsum per GEMM, EDP objective, official O3 default, exact search
  of the perfectly-factorized mapspace, and `max_fused_loops=1`.
- FactorFlow revision `40bcf3109377f12a63c8e8ba3a01df2208977bf3`:
  hybrid WART mapper, padded mappings, separate activation and weight SRAM
  levels, and physical row/column fanout.

Absolute cycles and energy remain native to each backend's analytical model.
Only architecture ranks normalized within the same backend and model are
compared across backends.

## Anchor hardware

| Architecture | Array | Act./weight SRAM | SRAM port | LPDDR4 | Area proxy |
|---|---:|---:|---:|---:|---:|
| balanced 32x32 | 1,024 PEs | 4 MiB / 4 MiB | 512 bit | 1 channel | 25.445 mm2 |
| balanced 64x64 | 4,096 PEs | 4 MiB / 4 MiB | 512 bit | 1 channel | 34.788 mm2 |
| aggressive 64x64 | 4,096 PEs | 8 MiB / 8 MiB | 1024 bit | 2 channels | 62.821 mm2 |
| aggressive 128x128 | 16,384 PEs | 8 MiB / 8 MiB | 1024 bit | 2 channels | 100.192 mm2 |

The area values use CACTI SRAM area plus the pinned LLMCompass FP16-FPU array
proxy, LPDDR I/O, fixed logic, and implementation margin. They are reported
for context and are not part of the three-mapper EDP ranking objective.

## Validation result

All three backends completed 460 unique-GEMM/architecture mappings. Expansion
back to the source workload produced 5,832 layer rows and 48 model summaries.
No propagation row occurs in mapper, layer, or model outputs.

| Backend | Rank 1 | Rank 2 | Rank 3 | Rank 4 |
|---|---|---|---|---|
| LLMCompass | aggressive 128x128 | aggressive 64x64 | balanced 64x64 | balanced 32x32 |
| TCM O3 | aggressive 128x128 | balanced 64x64 | aggressive 64x64 | balanced 32x32 |
| FactorFlow | aggressive 128x128 | aggressive 64x64 | balanced 64x64 | balanced 32x32 |

The ranks minimize the equal-model geometric mean of backend-normalized EDP.
The corresponding equal-model mean effective array utilizations are:

| Backend | balanced 32x32 | balanced 64x64 | aggressive 64x64 | aggressive 128x128 |
|---|---:|---:|---:|---:|
| LLMCompass | 41.657% | 11.187% | 20.792% | 5.572% |
| TCM O3 | 14.405% | 5.746% | 6.879% | 2.952% |
| FactorFlow | 19.115% | 4.955% | 8.191% | 2.925% |

All three mappers therefore agree that the largest anchor has the lowest EDP
inside this fixed, area-unconstrained set. They also agree that most of its PEs
are idle. The 128x128 result must not be interpreted as the best chip design:
its area proxy is 2.88x balanced 64x64 and 3.94x balanced 32x32, while its mean
utilization is only 2.9% to 5.6%. For a PPA decision, balanced 64x64 and
aggressive 64x64 remain the important tradeoff: LLMCompass and FactorFlow rank
the higher-bandwidth aggressive point second, while TCM's native EDP ranks the
lower-energy balanced point second.

## TCM O3/O4 sensitivity

The full validation uses the upstream O3 default. An O4 imperfect spatial and
temporal microbenchmark was also run to expose the effect of residual tiling.

| GEMM | O3 cycles | O4 cycles | O3 cycle increase |
|---|---:|---:|---:|
| 64x64x64 | 1,536 | 1,408 | 9.1% |
| 1024x16x64 | 11,145 | 11,145 | 0.0% |
| 63x65x31 | 1,323 | 693 | 90.9% |

O4 is important for irregular tail-heavy shapes, but a full O4 run generated
7–12 GiB mapping tables per hard worker and projected to roughly five hours
for these 460 tasks. O3 is therefore the reproducible primary comparison; the
O4 microbenchmark is a sensitivity bound, not mixed into the O3 rankings.

## Reproduction and artifacts

```bash
conda create -p .venv/accelforge python=3.12
.venv/accelforge/bin/pip install -r requirements-mapper-validation.txt

PYTHONPATH=. python scripts/run_spn_mapper_validation.py \
  --config configs/spn_mapper_validation_bf16_lpddr4.json \
  --mode microbench \
  --output reports/generated/spn_three_mapper_validation_bf16_lpddr4/microbench_o3

PYTHONPATH=. python scripts/run_spn_mapper_validation.py \
  --config configs/spn_mapper_validation_bf16_lpddr4.json \
  --mode anchors

PYTHONPATH=. python scripts/run_spn_mapper_validation.py \
  --validate reports/generated/spn_three_mapper_validation_bf16_lpddr4/analysis
```

The analysis directory contains the hardware manifest, workload coverage,
backend status, unique-GEMM results, expanded layer results, model summaries,
normalized architecture rankings, validation metadata, and the ranking plot.

## Remaining modeling limits

This validation does not model cross-layer activation retention or fusion,
LPDDR bank and row-buffer timing, address-level SRAM bank conflicts, NoC and
clock-tree energy, leakage, or physical place-and-route congestion. FactorFlow
and TCM select dataflows for independent GEMMs; LLMCompass uses the local
memory-constrained adapter rather than a full-network compiler. These results
are suitable for mapper cross-checking and anchor pruning, not RTL/PPA signoff.
