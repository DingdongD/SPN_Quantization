# SPN Memory-Aware Array DSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible CACTI-calibrated, LPDDR4- and 4 MiB-buffer-constrained DSE for all four SPN workloads and all existing array shapes.

**Architecture:** Keep LLMCompass as the compute-cycle backend, add an independent memory-architecture configuration and a deterministic mapping search over legal M/N/K tiles, loop orders, residency, and buffering. Calibrate two 4 MiB SRAM paths with pinned official CACTI, derive LPDDR4 bandwidth from explicit protocol fields, and rank candidates by conservative scheduled cycles while exporting all resource floors and mappings.

**Tech Stack:** Python 3.11, pytest, LLMCompass/Scale-Sim, official CACTI 7 C++, JSON/CSV, matplotlib, Bash.

---

### Task 1: Pin and calibrate CACTI

**Files:**
- Modify: `.gitmodules`
- Create: `external/cacti` submodule
- Create: `configs/cacti/sram_4mb_32nm_512b.json`
- Create: `scripts/run_cacti_sram.py`
- Create: `tests/test_cacti_sram.py`
- Create: `tool-config.json`

- [x] **Step 1: Write failing parser and provenance tests**

Test that `parse_cacti_output()` extracts capacity, access time, cycle time,
area, and read/write energy from a representative official CACTI output, and
that invalid or mismatched capacity output is rejected.

- [x] **Step 2: Run the focused test and verify RED**

Run: `/opt/conda/bin/python -m pytest -q tests/test_cacti_sram.py`

Expected: import failure because `spn_quant.hardware.cacti` does not exist.

- [x] **Step 3: Add the pinned submodule, CACTI config, parser, and runner**

Use official revision `1ffd8dfb10303d306ecd8d215320aea07651e878`.
Configure a 4 MiB, 64-byte block, 512-bit output, 16-bank, one-RW-port,
32 nm UCA scratch RAM and expose:

```python
@dataclass(frozen=True)
class CactiSramResult:
    capacity_bytes: int
    access_time_ns: float
    cycle_time_ns: float
    area_mm2: float
    read_energy_nj: float
    write_energy_nj: float
    output_width_bits: int
    bank_count: int
```

The runner must build CACTI if necessary, execute it, save raw stdout, and emit
a JSON result containing the pinned revision and input path.

- [x] **Step 4: Run CACTI and the parser tests**

Run:

```bash
/opt/conda/bin/python scripts/run_cacti_sram.py \
  --config configs/cacti/sram_4mb_32nm_512b.json \
  --raw-output reports/generated/cacti/sram_4mb_32nm_512b.txt \
  --json-output reports/generated/cacti/sram_4mb_32nm_512b.json
/opt/conda/bin/python -m pytest -q tests/test_cacti_sram.py
```

Expected: CACTI exits zero and all focused tests pass.

- [x] **Step 5: Commit**

Commit message: `build: pin CACTI for SPN memory modelling`

### Task 2: Define strict memory architecture and mapping contracts

**Files:**
- Create: `spn_quant/hardware/memory_dse.py`
- Create: `configs/spn_array_dse_bf16_lpddr4_4mb.json`
- Create: `tests/test_memory_dse.py`

- [x] **Step 1: Write failing configuration tests**

Test BF16/FP32 byte widths, independent positive capacities and bandwidths,
LPDDR peak/effective bandwidth arithmetic, 50/70/90% scenarios, and rejection
of unsupported dtype, non-positive bandwidth, efficiency outside `(0,1]`, and
missing primary scenario.

- [x] **Step 2: Run focused tests and verify RED**

Run: `/opt/conda/bin/python -m pytest -q tests/test_memory_dse.py`

Expected: import failure for the new memory-DSE types.

- [x] **Step 3: Implement immutable configuration and mapping records**

Define `MemoryArchitecture`, `MemoryScenario`, `TileMapping`, and
`MemorySimulationResult`. Compute:

```python
lpddr_peak_Bps = transfers_per_second * bus_width_bits * channels / 8
lpddr_sustained_Bps = lpddr_peak_Bps * efficiency
sram_Bps = output_width_bytes / max(cacti_cycle_time_s, accelerator_cycle_s)
```

Store all capacities in bytes and all bandwidths in bytes/second and
bytes/accelerator-cycle.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `/opt/conda/bin/python -m pytest -q tests/test_memory_dse.py`

Expected: configuration tests pass.

- [x] **Step 5: Commit**

Commit message: `feat: define constrained SPN memory architecture`

### Task 3: Implement capacity-safe tile and traffic search

**Files:**
- Modify: `spn_quant/hardware/memory_dse.py`
- Modify: `tests/test_memory_dse.py`

- [x] **Step 1: Write failing capacity and traffic tests**

Cover exact tails, independent activation/weight rejection, FP32 partial-sum
capacity, persistent A/B/C residency, the reuse-favouring `mkn`, `knm`, and
`mnk` loop orders, partial-sum spills when K is not innermost, repetitions,
and single/double-buffer capacity factors.

- [x] **Step 2: Run the new tests and verify RED**

Expected failures: missing tile enumeration, traffic accounting, and mapping
search APIs.

- [x] **Step 3: Implement candidate generation and exact resource accounting**

Generate aligned powers-of-two tile candidates plus exact dimensions and
capacity boundaries. Search six loop orders, legal residency modes, and both
buffering modes. Use LLMCompass compute cycles for each K-tile size and cache
identical queries. Calculate DRAM A/B/final-output/partial-sum bytes and
activation/weight SRAM bytes separately.

- [x] **Step 4: Implement conservative scheduling**

For double buffering:

```python
cycles = max(compute, dram, activation_sram, weight_sram) + startup + drain
```

For single buffering:

```python
cycles = compute + dram + max(activation_sram, weight_sram)
```

Assert the result never violates a resource floor and select the minimum-cycle
feasible mapping with deterministic tie-breaks.

- [x] **Step 5: Run focused and existing hardware tests**

Run:

```bash
/opt/conda/bin/python -m pytest -q \
  tests/test_memory_dse.py tests/test_llmcompass_adapter.py \
  tests/test_array_dse.py
```

Expected: all selected tests pass.

- [x] **Step 6: Commit**

Commit message: `feat: search buffer-constrained SPN tile mappings`

### Task 4: Export and validate four-model memory-aware DSE artifacts

**Files:**
- Create: `scripts/run_spn_memory_dse.py`
- Create: `scripts/run_four_model_memory_dse.sh`
- Create: `tests/test_run_spn_memory_dse.py`
- Modify: `.gitignore`

- [x] **Step 1: Write failing artifact tests**

Use small synthetic profiles to require layer mappings, per-model summaries,
scenario rankings, bottleneck summaries, Pareto rows, metadata provenance, and
validation rejection for capacity or resource-floor violations.

- [x] **Step 2: Run focused tests and verify RED**

Expected: import failure because the memory-aware runner does not exist.

- [x] **Step 3: Implement runner, CSV/JSON outputs, plots, and validator**

Emit:

- `layer_memory_results.csv`
- `model_memory_summary.csv`
- `candidate_memory_ranking.csv`
- `bottleneck_summary.csv`
- `memory_pareto_frontier.csv`
- `bandwidth_sensitivity.csv`
- `memory_shape_latency.png`
- `metadata.json`

The validator must recompute capacity limits, resource floors, totals,
rankings, scenario coverage, and provenance fields from the exported rows.

- [x] **Step 4: Run artifact tests and verify GREEN**

Run: `/opt/conda/bin/python -m pytest -q tests/test_run_spn_memory_dse.py`

Expected: all focused tests pass.

- [x] **Step 5: Commit**

Commit message: `feat: orchestrate memory-aware four-model DSE`

### Task 5: Execute the full constrained study and document conclusions

**Files:**
- Modify: `README.md`
- Modify: `docs/2026-10-21-spn-memory-aware-array-dse-requirements.md`
- Generate ignored: `reports/generated/spn_array_dse_bf16_lpddr4_4mb/`

- [x] **Step 1: Run the complete four-model, 13-shape, three-scenario study**

Run: `bash scripts/run_four_model_memory_dse.sh`

Expected: the validator prints `SPN_MEMORY_DSE_VALIDATION_OK`.

- [x] **Step 2: Inspect bottlenecks and compare against compute-only ranking**

Document the primary 70% result, equal-PE shape comparisons, per-model
exceptions, DRAM/SRAM/compute bottleneck shares, buffer occupancy, and the
50/90% sensitivity bounds. Clearly distinguish array-shape evidence from
unmodelled whole-model operations.

- [x] **Step 3: Update usage and architecture risk documentation**

Document CACTI build/run commands, the full DSE command, all assumptions,
confidence level, and the unresolved items that block architecture signoff.

- [x] **Step 4: Run fresh completion verification**

Run all hardware-DSE tests, Python 3.7 compatibility checks for shared
exporter modules, shell syntax checks, artifact validation, `git diff --check`,
and the known QAT test separately.

Expected: task-scoped tests and artifact validation pass. If the pre-existing
QAT fixture still fails on its three missing configuration arguments, report
it as an unchanged integration gate rather than a memory-DSE regression.

- [x] **Step 5: Commit**

Commit message: `docs: report constrained SPN memory DSE`

