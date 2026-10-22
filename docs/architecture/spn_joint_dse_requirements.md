# SPN Joint Array/Memory DSE Requirements

## Scope

The study explores BF16 dense GEMMs extracted from CSPN, DySPN, NLSPN, and
CompletionFormer. It is an architecture-comparison model, not a physical
design sign-off model.

## Requirements

| ID | Priority | Class | Requirement |
|---|---|---|---|
| R1 | Must | Functional | Jointly enumerate array shape, activation SRAM macro, weight SRAM macro, and LPDDR configuration. |
| R2 | Must | Performance | Re-run the capacity-safe tile/loop/residency search for every architecture point and model all compute, SRAM, and DRAM resource floors. |
| R3 | Must | Area | Include array, both SRAM macros, per-channel LPDDR I/O proxy, fixed logic, and configurable implementation margin in area. |
| R4 | Must | Power | Estimate workload dynamic energy from MACs, CACTI SRAM accesses, and DRAM bytes; derive average dynamic power from energy/latency. |
| R5 | Must | Optimization | Report the latency-area-energy Pareto frontier without collapsing unlike units into an arbitrary scalar score. |
| R6 | Must | Fairness | Give all four SPN models equal influence through geometric-mean latency and report worst normalized model slowdown. |
| R7 | Must | Reproducibility | Record configuration, input hashes, pinned LLMCompass/CACTI revisions, assumptions, and confidence. |
| R8 | Must | Validation | Reject capacity, resource-floor, aggregation, constraint, or Pareto-consistency violations. |
| R9 | Should | Performance | Support deterministic process-level parallel execution. |
| R10 | Should | Usability | Provide conservative, balanced, and aggressive architecture profiles plus asymmetric-buffer candidates. |
| R11 | Nice | Visualization | Plot the feasible Pareto candidates in latency-area space with energy encoded by color. |

## Interface and dataflow

The modeled dataflow is:

`LPDDR4 aggregate interface -> independent activation/psum and weight SRAMs -> one output-stationary systolic array`.

SRAM macros are supplied as pinned CACTI JSON results. External memory is an
aggregate LPDDR bandwidth/energy interface. There is no AXI transaction-level
model in this phase.

## Explicit assumptions

- BF16 operands and final outputs; FP32 partial sums.
- One MAC per PE per cycle.
- LLMCompass FP16-FPU area is used as a documented conservative proxy for a
  BF16 MAC because the pinned cost model has no BF16 primitive.
- SRAM area/timing/access energy comes from CACTI at one configured node.
- MAC energy and LPDDR energy-per-byte are configuration inputs, not outputs
  of LLMCompass.
- Four models are equally weighted unless a later product workload mix is
  supplied.

## Open questions and resulting confidence limits

- Foundry node, standard-cell library, voltage, and BF16 MAC implementation
  are not specified; array area and MAC energy are LOW-confidence proxies.
- Measured LPDDR4 energy and controller/PHY area are not supplied; their
  configured values require sensitivity analysis and are not sign-off values.
- No chip area, power, or per-model latency budget is supplied; these remain
  optional hard constraints and no architecture can be signed off yet.
- Cross-layer activation retention, operator fusion, LPDDR bank/row timing,
  NoC contention, and SRAM address-level bank conflicts remain outside the
  current analytical model and must be stated in every report.

## Architecture candidate classes

| Class | Intent | Typical characteristics | Expected trade-off |
|---|---|---|---|
| Conservative | Minimize cost | Small buffers, narrow ports, one LPDDR channel | Lowest area/energy, highest latency |
| Balanced | Avoid obvious memory stalls | Mid-size buffers and ports, one channel | Likely knee of the frontier |
| Aggressive | Maximize throughput | Large buffers/ports, two LPDDR channels | Lowest latency, highest area/energy |
| Activation-heavy | Test asymmetric capacity | Larger activation than weight SRAM | Helps activation/psum-heavy layers |
| Weight-heavy | Test asymmetric capacity | Larger weight than activation SRAM | Helps resident-weight mappings |

## Measured CACTI macro inputs

All three macros were generated from pinned CACTI revision
`1ffd8dfb10303d306ecd8d215320aea07651e878` at the configured 32 nm node.

| Profile macro | Capacity / port / banks | Access / cycle (ns) | Area (mm²) | Read / write (nJ/access) |
|---|---|---:|---:|---:|
| Small | 2 MiB / 256 bit / 8 | 1.414 / 1.759 | 3.237 | 0.216 / 0.202 |
| Balanced | 4 MiB / 512 bit / 16 | 2.103 / 0.995 | 7.054 | 0.590 / 0.673 |
| Large | 8 MiB / 1024 bit / 32 | 2.660 / 1.759 | 17.485 | 1.861 / 1.943 |

The 1024-bit macro is not treated as an ideal 128-byte/cycle SRAM: its CACTI
cycle time limits it to approximately 72.75 bytes/accelerator-cycle at 1 GHz.
The small macro is similarly limited to approximately 18.19 bytes/cycle.

## Full four-model result

The validated run contains 65 architecture points, 260 model summaries, and
31,590 dense-GEMM layer mappings. Twenty-two candidates are non-dominated in
the three-dimensional latency/area/dynamic-energy space.

| Candidate | Equal-model geometric latency | Worst normalized model latency | Mean dynamic energy/model | Max model-average dynamic power | Estimated area |
|---|---:|---:|---:|---:|---:|
| Conservative 32x32 | 313.30 ms | 6.063x | 581.14 mJ | 1.57 W | 16.28 mm² |
| Balanced 32x32 | 137.99 ms | 2.265x | 607.02 mJ | 4.34 W | 25.45 mm² |
| **Balanced 64x64** | **130.33 ms** | **2.000x** | **565.45 mJ** | **4.55 W** | **34.79 mm²** |
| Aggressive 64x64 | 69.15 ms | 1.132x | 600.83 mJ | 8.51 W | 62.82 mm² |
| Aggressive 128x128 | 65.42 ms | 1.002x | 574.60 mJ | 9.12 W | 100.19 mm² |

Balanced 64x64 is the nominal knee when no product budgets are available. It
is 5.6% faster and uses 6.8% less modeled dynamic energy than balanced 32x32,
at a 36.7% area increase. Aggressive 64x64 is the latency-oriented candidate:
it is 46.9% faster than balanced 64x64, but uses 80.6% more estimated area and
6.3% more modeled energy. Moving from aggressive 64x64 to 128x128 gains only
5.4% latency while increasing area by 59.5%; it is retained on the Pareto
frontier only because modeled energy falls by 4.4%.

This recommendation is conditional, not architecture signoff. With an area
budget below 30 mm², balanced 32x32 is the practical candidate. With a dense
GEMM latency requirement below 75 ms and no 8 W-class dynamic-power limit,
aggressive 64x64 is the practical candidate.

## Risk register

| ID | Risk | Probability x impact | Class | Mitigation | Owner |
|---|---|---:|---|---|---|
| J1 | BF16 array area and MAC energy are uncalibrated proxies | 4 x 5 = 20 | HIGH | Synthesize the selected BF16 MAC/PE at the target library and replace both coefficients | Hardware implementation lead |
| J2 | Dense GEMMs exclude propagation, sampling, and vector work | 4 x 5 = 20 | HIGH | Add executable operator models and whole-graph scheduling before signoff | Performance-model owner |
| J3 | LPDDR sustained efficiency hides bank, row, refresh, and turnaround behavior | 4 x 4 = 16 | HIGH | Replay mapped traces through Ramulator/DRAMSim-class LPDDR timing model | Memory-system owner |
| J4 | SRAM aggregate bandwidth hides address-level bank conflicts | 3 x 4 = 12 | MEDIUM | Add bank mapping, multicast, port arbitration, and conflict stalls for finalists | SRAM/interconnect owner |
| J5 | No cross-layer retention or operator fusion | 3 x 4 = 12 | MEDIUM | Add tensor lifetime and fusion legality before final buffer sizing | Compiler owner |

J1-J3 remain unmitigated HIGH risks, so none of the candidates is ready for
architecture signoff.
