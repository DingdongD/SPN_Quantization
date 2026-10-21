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

