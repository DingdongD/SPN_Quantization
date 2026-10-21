# SPN memory-aware array DSE requirements

Date: 2026-10-21

## Objective

Extend the four-model BF16 LLMCompass study with hard operand-buffer capacity
constraints and calibrated SRAM/LPDDR4 bandwidth ceilings. Rank the existing
array-shape candidates by memory-aware layer latency rather than compute-only
utilization.

## Requirements

| ID | Class | Priority | Requirement |
|---|---|---|---|
| MEM-F-001 | Functional | Must | Reuse the executed CSPN, DySPN, NLSPN, and CompletionFormer NYU workload profiles. |
| MEM-F-002 | Functional | Must | Preserve the existing 13 physical array-shape candidates and output-stationary dataflow. |
| MEM-F-003 | Functional | Must | Search legal M/N/K tiles and all six loop orders for every mapped GEMM. |
| MEM-F-004 | Functional | Must | Enforce independent 4 MiB activation-path and 4 MiB weight-path SRAM capacities. |
| MEM-F-005 | Functional | Must | Model BF16 A/B operands and outputs as 2 bytes and partial sums as FP32, 4 bytes. |
| MEM-F-006 | Functional | Must | Account for operand reloads, partial-sum spills, tail tiles, group/head repetitions, and single/double buffering. |
| MEM-P-001 | Performance | Must | Never report memory-aware cycles below the LLMCompass compute-cycle floor or any aggregate bandwidth floor. |
| MEM-P-002 | Performance | Must | Report DRAM, activation-SRAM, weight-SRAM, compute, startup/drain, and selected scheduled cycles per layer. |
| MEM-P-003 | Performance | Must | Report the selected tile, loop order, residency mode, capacity usage, bottleneck, and arithmetic intensity. |
| MEM-P-004 | Performance | Must | Sweep LPDDR4 sustained-bandwidth efficiencies of 50%, 70%, and 90%; use 70% as the primary ranking. |
| MEM-I-001 | Interface | Must | Derive SRAM bandwidth from a pinned official CACTI run and retain the input, raw output, parsed result, and revision. |
| MEM-I-002 | Interface | Must | Derive LPDDR4 peak bandwidth from configurable MT/s, bus width, channel count, and sustained-efficiency fields. |
| MEM-Q-001 | Quality | Must | Validate capacity feasibility, resource lower bounds, ranking arithmetic, provenance, and artifact completeness. |
| MEM-Q-002 | Quality | Must | Keep generated DSE outputs outside version control while committing scripts, configs, tests, and conclusions. |
| MEM-A-001 | Area | Should | Report CACTI SRAM area for each 4 MiB macro, without claiming full-chip area. |
| MEM-W-001 | Power | Nice | Preserve CACTI read/write energy for later energy modelling; energy is not a ranking objective in this phase. |
| MEM-S-001 | Safety/security | Nice | No safety or security requirement is specified for this research model. |

## Fixed study assumptions

- Accelerator clock: 1.0 GHz.
- One physical systolic array per candidate, one BF16 MAC per PE per cycle.
- Two independent scratchpad ports: one activation path and one weight path.
- Each scratchpad is 4 MiB, 64-byte external datapath, 16 banks, one read/write
  port, modelled as a 32 nm CACTI UCA RAM.
- The CACTI cycle-time result limits each scratchpad to at most one 64-byte
  transfer per accelerator cycle; a slower CACTI cycle time reduces that rate.
- DRAM example: one x32 Micron-style LPDDR4-4266 channel. Peak bandwidth is
  `4266e6 * 32 / 8 = 17.064 GB/s`.
- Primary sustained DRAM bandwidth is 70% of peak, or 11.9448 GB/s. The 50%
  and 90% cases bound sensitivity.
- A double-buffered mapping must fit two active A/C tile sets in the activation
  SRAM and two active B tiles in the weight SRAM. A persistent full operand is
  stored once rather than duplicated.
- The activation SRAM holds BF16 input activations and FP32 partial sums. Final
  outputs are written to DRAM in BF16.
- The memory-aware schedule is conservative: single buffering serializes
  transfer and compute; double buffering uses the maximum aggregate resource
  demand plus explicit startup/drain costs.

## Candidate classes

| Class | Representative arrays | Intent |
|---|---|---|
| Conservative | 8x8, 16x8 | Low instantaneous bandwidth demand and high utilization. |
| Balanced | 16x16, 32x16 | Moderate PE count and SRAM bandwidth demand. |
| Aggressive | 64x16, 64x64, 128x128 | High compute throughput with elevated SRAM/DRAM pressure. |

All 13 configured candidates remain in the quantitative sweep; the table only
groups them for architecture interpretation.

## Measured CACTI parameters

The pinned CACTI run completed for one 4 MiB macro with the following parsed
result. Two such macros are assumed, one per operand path.

| Quantity | One 4 MiB SRAM |
|---|---:|
| Access time | 2.10289 ns |
| Cycle time | 0.995496 ns |
| Array-facing width | 512 bits |
| Banks | 16 |
| Effective 1 GHz port ceiling | 64 B/cycle = 64 GB/s |
| Area at illustrative 32 nm | 7.054394 mm2 |
| Dynamic read energy | 0.589761 nJ/access |
| Dynamic write energy | 0.672701 nJ/access |

The two SRAM macros total 14.108788 mm2 in this illustrative CACTI result. It
does not include the systolic array, interconnect, LPDDR PHY/controller, vector
units, or control logic and is not a target-process area estimate.

## Full constrained-study results

The full run covered 486 GEMMs x 13 array shapes x three bandwidth scenarios,
or 18,954 selected layer mappings after each mapping search. The independent
artifact reader returned `SPN_MEMORY_DSE_VALIDATION_OK` after checking both
4 MiB limits, all four resource-cycle floors, traffic and summary arithmetic,
ranking completeness, and pinned tool provenance.

### Shared-array ranking

| LPDDR4 sustained efficiency | Rank 1 | Rank 2 | Rank 3 |
|---:|---:|---:|---:|
| 50% (8.532 GB/s) | 64x64, 843.460 ms | 128x128, 843.907 ms | 32x32, 857.447 ms |
| 70% (11.9448 GB/s) | 128x128, 603.287 ms | 64x64, 604.320 ms | 32x32, 644.182 ms |
| 90% (15.3576 GB/s) | 128x128, 469.253 ms | 64x64, 478.194 ms | 32x32, 538.653 ms |

At the primary bandwidth, 128x128 is only 0.171% faster than 64x64 despite
using four times as many PEs. Moving from 32x32 to 64x64 reduces latency by
6.596% while quadrupling PE count. The 32x32 penalty relative to the fastest
candidate is only 1.658% at 50% DRAM efficiency, but rises to 6.779% at 70%
and 14.789% at 90%. Thus the value of more than 1,024 PEs depends strongly on
sustained external bandwidth.

### Per-model optima at 70% LPDDR4 efficiency

| Model | Minimum-latency shape | Latency | 64x64 latency | 64x64 gap from optimum |
|---|---:|---:|---:|---:|
| CSPN | 32x32 | 47.329 ms | 47.355 ms | 0.057% |
| DySPN | 128x128 | 142.391 ms | 143.141 ms | 0.527% |
| NLSPN | 128x128 | 190.211 ms | 190.967 ms | 0.397% |
| CompletionFormer | 64x64 | 222.856 ms | 222.856 ms | 0% |

The 64x64 candidate stays within 0.53% of every per-model optimum, so it is the
best shared latency-oriented choice. The 32x32 candidate is the recommended
PPA starting point when a 6.6% aggregate latency increase is preferable to a
4x PE-count increase. The result does not justify 128x128 under the assumed
single x32 channel because its 4x PE increase over 64x64 saves only 1.033 ms
across all four mapped workloads.

### Equal-PE aspect-ratio decisions

| PE budget | Best shared shape | Four-model latency | Interpretation |
|---:|---:|---:|---|
| 128 | 8x16 | 2,315.097 ms | Effectively tied with 16x8. |
| 256 | 16x16 | 1,216.818 ms | Balanced beats both elongated alternatives. |
| 512 | 16x32 | 767.115 ms | 4.39% lower latency than 32x16. |
| 1,024 | 32x32 | 644.182 ms | 15.84% lower latency than 16x64 and 19.73% lower than 64x16. |

Memory constraints reverse the compute-only preference for tall arrays at the
512- and 1,024-PE points. Wider or square arrays reduce the operand-stream
traffic that becomes visible once the two 512-bit SRAM ports are modelled.

### Capacity and bottleneck audit

| Shape | Max activation-buffer use | Max weight-buffer use | Layer bottlenecks (compute / DRAM / act-SRAM / weight-SRAM) |
|---:|---:|---:|---:|
| 32x32 | 89.432% | 87.891% | 41 / 383 / 59 / 3 |
| 64x64 | 89.822% | 87.891% | 41 / 429 / 16 / 0 |
| 128x128 | 90.408% | 87.891% | 53 / 433 / 0 / 0 |

At 64x64, the four models move 7.175 GB from DRAM and 429 of 486 GEMMs are
DRAM-limited. Their selected mappings spill no FP32 partial sums to DRAM; the
search chooses K-inner or resident-C schedules instead. Activation-SRAM
bandwidth is material for some 32x32 layers, but the shared total is governed
primarily by the x32 LPDDR4 channel. The CACTI-derived 512-bit SRAM ports are
therefore adequate for 64x64 under this aggregate model; increasing array size
without increasing DRAM bandwidth mostly lowers utilization rather than
latency.

### Decision and confidence

- Use 32x32 as the first PPA implementation point when 1,024 BF16 MACs are an
  acceptable array budget.
- Use 64x64 only for a latency-oriented design that can afford 4,096 MACs; it
  is a robust shared choice across all four dense workloads.
- Do not select 128x128 for the assumed one-channel LPDDR4 system without an
  explicit sub-millisecond aggregate gain requirement or more DRAM bandwidth.
- Capacity and aggregate-bandwidth feasibility confidence is MEDIUM. Whole-
  model cycle accuracy remains LOW because propagation, deformable sampling,
  vector operations, inter-layer retention, fabric arbitration, and fusion are
  not modelled.
- The tile search covers array-aligned powers of two plus exact tail sizes,
  all six loop orders, residency modes, and single/double buffering. It is not
  an exhaustive search over every integer tile dimension.

### Verification evidence

- 70 workload, collector, exporter, LLMCompass adapter, compute-DSE, CACTI,
  memory-DSE, runner, and validator tests passed.
- Both persisted studies independently returned
  `SPN_ARRAY_DSE_VALIDATION_OK` and `SPN_MEMORY_DSE_VALIDATION_OK`.
- The shared exporter modules compile under the declared Python 3.7 model
  environment, and both four-model shell wrappers pass `bash -n`.
- Serial and two-worker synthetic DSE outputs are byte-identical. A real CSPN
  layer anchor also reproduces the pre-optimization mapping and all three
  scenario cycle counts after scenario-sharing was introduced.
- `tests/test_cspn_method_qat.py` retains seven unrelated baseline failures:
  its local `_config()` fixture omits `fp16_weight_modules`,
  `fp16_activation_owners`, and `propagation_mode`. No QAT source was changed
  for this study.

## Interfaces

| Interface | Protocol/model | Bandwidth |
|---|---|---|
| Activation SRAM to array | 512-bit scratchpad port | CACTI cycle-time limited |
| Weight SRAM to array | 512-bit scratchpad port | CACTI cycle-time limited |
| LPDDR4 to SRAMs | x32 LPDDR4-4266 | 17.064 GB/s peak; 8.532/11.9448/15.3576 GB/s swept |

The external SoC fabric protocol (for example AXI) is not specified and is not
assigned additional protocol overhead in this phase.

## Open questions and risks

| ID | Open item | Impact |
|---|---|---|
| OQ-1 | Target process, voltage, and signoff frequency are unspecified. | CACTI SRAM timing/area is illustrative, not implementation signoff. |
| OQ-2 | LPDDR4 controller, page policy, channel count, and measured efficiency are unspecified. | DRAM sensitivity is more reliable than the primary absolute latency. |
| OQ-3 | SRAM banking, arbitration, and array-facing port protocol are unspecified. | A 512-bit independent port per operand is an explicit modelling assumption. |
| OQ-4 | Whether accumulators are physically inside the PE array or activation SRAM is unspecified. | The model conservatively charges FP32 partial-sum storage and spills. |
| OQ-5 | Deformable sampling, propagation gather/scatter, vector ops, and layer fusion remain unmodelled. | Results rank dense-array candidates but do not equal whole-accelerator latency. |
| OQ-6 | Area, power, latency, and throughput product budgets are unspecified. | Architecture signoff is blocked even after the DSE completes. |

## Acceptance criteria

- Official CACTI builds and produces a parsed 4 MiB SRAM result from the
  committed configuration.
- Every chosen mapping satisfies both independent 4 MiB capacity constraints.
- Every layer's scheduled cycles are at least its compute, activation-SRAM,
  weight-SRAM, and DRAM lower bounds.
- Four models x 13 arrays x three DRAM efficiency scenarios are reported.
- Primary and sensitivity rankings, layer mappings, bottleneck summaries, and
  metadata validate without arithmetic or provenance mismatches.
- Existing task-scoped compute-only DSE tests remain green; the known QAT test
  fixture failure is tracked separately and is not reclassified as a DSE fault.

