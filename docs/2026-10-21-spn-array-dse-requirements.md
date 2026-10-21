# SPN Systolic-Array DSE Requirements

## Objective and scope

This study compares systolic-array shapes for one-image NYU inference on
CSPN, DySPN, NLSPN, and CompletionFormer. It reports layer-level GEMM shapes,
LLMCompass cycle estimates, effective PE utilization, equal-PE aspect-ratio
comparisons, and a four-model shared candidate.

The study is an architecture-screening result, not hardware sign-off.
LLMCompass models dense matrix multiplication. Dense convolution,
transposed convolution, linear projection, and CompletionFormer QK/AV products
are lowered to GEMMs. Propagation gather/scatter, deformable sampling,
normalization, activation functions, interpolation, and other vector/control
operations are retained in workload coverage but excluded from dense-array
cycle totals.

## Fixed requirements

| ID | Requirement | Priority |
|---|---|---|
| F-1 | Cover CSPN, DySPN, NLSPN, and CompletionFormer. | Must-Have |
| F-2 | Extract executed layer shapes from the official retained checkpoints. | Must-Have |
| F-3 | Use NYU batch 1 and the deployed 228x304 input/output geometry. | Must-Have |
| F-4 | Model BF16 operands, one MAC per PE per cycle, output-stationary dataflow. | Must-Have |
| F-5 | Sweep square and rectangular arrays from 8x8 through 128x128. | Must-Have |
| F-6 | Compare aspect ratios only within equal-PE groups. | Must-Have |
| F-7 | Report dense-array coverage and unsupported/irregular workloads. | Must-Have |
| F-8 | Pin LLMCompass and Scale-Sim revisions and hash all workload inputs. | Must-Have |
| P-1 | Report cycles, effective utilization, ideal-cycle overhead, and low-utilization fractions. | Must-Have |
| P-2 | Report per-model recommendations and an equal-model-weight shared recommendation. | Must-Have |
| P-3 | Do not claim whole-model latency, area, power, or throughput sign-off without hardware budgets. | Must-Have |

## Interfaces

- Input model contract: `configs/four_model_unified_fp16_task_aware.json`.
- Model execution: existing model-specific Python executable, CUDA extension,
  checkpoint, dataset root, and fixed validation identity.
- Workload interchange: schema-versioned JSON containing ordered operations,
  shapes, invocation counts, mapping eligibility, and provenance.
- Array study: BF16 configuration JSON containing candidate height/width,
  dataflow, MAC rate, and utilization threshold.
- Outputs: detailed layer CSV, model summary CSV, ranking CSV, Pareto CSV,
  coverage CSV, plot, and metadata JSON.

## Assumptions

- Batch size is one; spatial dimensions and module execution order are stable
  across NYU validation images.
- BF16 changes operand storage width but not layer dimensions or useful MACs.
- One PE performs one BF16 MAC per cycle.
- LLMCompass cycle estimates are analytical/simulator cycles, not measured
  kernel wall time.
- A grouped convolution is scheduled as one GEMM shape repeated per group.
- Transposed convolution uses input-spatial GEMM lowering; col2im/scatter is
  reported outside the dense array.
- Candidate comparisons do not silently assume that a larger array has the
  same area or power as a smaller one.

## Open hardware questions

The following inputs are required before architecture sign-off:

1. Target process node, voltage, and clock frequency.
2. Number of arrays/cores and whether array count changes with array shape.
3. Per-core SRAM capacity, banking, read/write bandwidth, and port limits.
4. Global-buffer and DRAM capacity, bandwidth, and latency.
5. Maximum per-frame latency and minimum frame throughput.
6. Area and power budgets, including the permitted array budget fraction.
7. Accumulator width, BF16 rounding mode, and supported mixed INT/BF16 modes.
8. Whether propagation/deformable sampling may use a separate sparse/vector
   datapath and its bandwidth/coherency contract.
9. Whether group/depthwise convolutions can be packed across PE columns.

## Candidate classes

- Conservative: 8x8 and 8x16/16x8 arrays, prioritizing utilization on narrow
  channel and propagation-adjacent layers.
- Balanced: equal-PE 8x32, 16x16, and 32x8 candidates plus 16x32/32x16,
  trading throughput against channel-tail waste.
- Aggressive: 32x32, 16x64, 64x16, 64x64, and 128x128 candidates,
  prioritizing dense encoder and attention throughput at greater PE cost.

No class is sign-off-ready until the open hardware questions are resolved.

## Risk register

| ID | Risk | Probability | Impact | Score | Mitigation | Owner |
|---|---|---:|---:|---:|---|---|
| R-1 | Dense GEMM cycles are mistaken for whole-model latency. | 4 | 5 | 20 | Publish coverage and excluded-operation tables beside every ranking. | Architecture team |
| R-2 | Deformable/propagation bandwidth dominates the selected array. | 4 | 4 | 16 | Add a calibrated sparse/vector and memory model before sign-off. | Architecture team |
| R-3 | Larger arrays win cycles only by consuming more area/power. | 4 | 4 | 16 | Use equal-PE aspect comparisons and a PE-count/cycle Pareto frontier. | Architecture team |
| R-4 | Upstream simulator/API drift changes results. | 2 | 4 | 8 | Pin Git and Python dependency revisions; record hashes. | Software owner |
| R-5 | One validation sample misses data-dependent operator paths. | 2 | 4 | 8 | Validate execution signatures on multiple identities if control flow differs. | Model owner |
| R-6 | Unsupported BF16 details alter accumulator/storage costs. | 3 | 3 | 9 | Keep precision assumptions explicit and calibrate against RTL/macros later. | Datapath owner |

R-1 through R-3 are HIGH risks and prevent sign-off; their stated mitigation
must be completed and assigned to named people in the hardware project.
